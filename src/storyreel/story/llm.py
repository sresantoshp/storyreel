"""Local writer: llama.cpp `llama-server` with an open-weight model, fully offline.

The server is a child process (127.0.0.1, random free port) inside a Job Object / rlimit
memory cap, started for one story and stopped afterwards so its RAM is returned. Output is
constrained with a JSON schema (llama.cpp grammar), so structure is guaranteed; meaning is
validated by story.schema.

Models (licences checked from the official model cards, 2026-11-16):
  qwen2.5-1.5b  Apache-2.0  1.04 GiB  fast (measured 11 tok/s here), weak at Hindi
  phi-3.5-mini  MIT         2.23 GiB  slower (3.6-4.1 tok/s), better English, writes Hindi
Qwen2.5-3B is NOT offered: its licence (qwen-research) forbids commercial use.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import psutil

from storyreel import paths
from storyreel.errors import ExitCode, StoryReelError
from storyreel.jobs.limits import Limiter, spawn

log = logging.getLogger(__name__)
_EXE = ".exe" if sys.platform == "win32" else ""


class LLMError(StoryReelError):
    exit_code = ExitCode.ERROR


class LLMUnavailableError(LLMError):
    exit_code = ExitCode.ENV_CHECK_FAILED


@dataclass(frozen=True)
class ModelInfo:
    key: str
    file: str
    repo: str  # Hugging Face repo holding the GGUF
    licence: str
    sha256: str
    size: int
    ram_mb: int  # approx. resident size while generating (measured)
    langs: tuple[str, ...]  # languages it writes acceptably
    max_ctx: int  # context ceiling whose KV cache fits in a few GB of RAM
    template: str  # prompt format: chatml | phi3


MODELS: dict[str, ModelInfo] = {
    "qwen2.5-1.5b": ModelInfo(
        "qwen2.5-1.5b",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "Apache-2.0",
        "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        1117320736,
        1250,
        ("en",),
        8192,  # grouped-query attention: small KV cache
        "chatml",
    ),
    "phi-3.5-mini": ModelInfo(
        "phi-3.5-mini",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "MIT",
        "e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5",
        2393232672,
        2600,
        ("en",),  # it writes Devanagari, but measured Hindi was fluent-sounding nonsense
        4096,  # no grouped-query attention: 8K context alone needs ~3.2 GB of KV cache
        "phi3",
    ),
}


def build_prompt(template: str, system: str, user: str) -> tuple[str, list[str]]:
    """Prompt text in the model's own chat format, plus its end-of-turn stop strings."""
    if template == "chatml":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n",
            ["<|im_end|>"],
        )
    if template == "phi3":
        return (
            f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
            ["<|end|>"],
        )
    raise LLMError(f"internal: unknown prompt template {template!r}")


def models_dir() -> Path:
    return paths.data_dir() / "models"


def model_path(key: str) -> Path:
    return models_dir() / MODELS[key].file


def installed(key: str) -> bool:
    p = model_path(key)
    return p.is_file() and p.stat().st_size == MODELS[key].size


def pick_model(lang: str, requested: str = "auto") -> str:
    """Auto language pick: English → fast Qwen 1.5B; other languages → Phi-3.5-mini."""
    if requested != "auto":
        if requested not in MODELS:
            raise LLMError(
                f"Unknown writer {requested!r}.", fix=f"Use one of: auto, {', '.join(MODELS)}"
            )
        return requested
    order = ["qwen2.5-1.5b", "phi-3.5-mini"] if lang == "en" else ["phi-3.5-mini", "qwen2.5-1.5b"]
    for key in order:
        if installed(key):
            return key
    raise LLMUnavailableError(
        "No local writer model is installed.",
        fix=f"Put {MODELS[order[0]].file} in {models_dir()} (see docs/PROGRESS.md), "
        "or use --writer rules.",
    )


def find_server() -> Path:
    """llama-server binary: $STORYREEL_LLAMA_SERVER, else the newest under the tools dir."""
    import os

    env = os.environ.get("STORYREEL_LLAMA_SERVER")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise LLMUnavailableError(f"STORYREEL_LLAMA_SERVER points to {env}, which is not a file.")
    candidates = sorted((paths.data_dir() / "tools").glob(f"llama-*/bin/llama-server{_EXE}"))
    if candidates:
        return candidates[-1]
    raise LLMUnavailableError(
        "llama.cpp (llama-server) is not installed.",
        fix=f"Unpack a llama.cpp release into {paths.data_dir() / 'tools'}/llama-<build>/bin.",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _local_open(target: str | urllib.request.Request, timeout: float) -> Any:
    """urlopen restricted to the local writer: plain http on 127.0.0.1 only."""
    url = target if isinstance(target, str) else target.full_url
    if not url.startswith("http://127.0.0.1:"):
        raise LLMError(f"internal: refusing non-local writer URL {url!r}")
    return urllib.request.urlopen(target, timeout=timeout)  # noqa: S310  # checked above


@dataclass
class ChatResult:
    data: dict[str, Any]
    text: str
    wall_s: float
    tokens: int
    tok_per_s: float


class LocalWriter:
    """Context manager owning one llama-server process."""

    def __init__(self, key: str, *, ctx: int = 4096, cap_mb: int | None = None) -> None:
        self.info = MODELS[key]
        self.model = model_path(key)
        if not installed(key):
            raise LLMUnavailableError(
                f"Writer model {self.info.file} is missing or incomplete.",
                fix=f"Download it into {models_dir()}.",
            )
        self.ctx = min(ctx, self.info.max_ctx)
        # cap = model + KV cache + runtime margin; generous because mmap pages count as commit
        self.cap = (cap_mb or (self.info.size >> 20) + 1500) << 20
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._proc: subprocess.Popen[bytes] | None = None
        self._limiter: Limiter | None = None
        self._log: BinaryIO | None = None
        self.log_path = paths.log_dir() / "llama-server.log"
        self.peak_rss = 0
        self.load_s = 0.0

    def __enter__(self) -> LocalWriter:
        argv = [
            str(find_server()),
            "-m",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.ctx),
            "--no-webui",
        ]
        t0 = time.monotonic()
        # keep the server's own messages: a failed load must be able to say why
        self.log_path = paths.log_dir() / "llama-server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        self._proc, self._limiter = spawn(
            argv, limit_bytes=self.cap, stdout=subprocess.DEVNULL, stderr=self._log
        )
        try:
            self._wait_ready(timeout_s=300)
        except BaseException:
            self.close()
            raise
        self.load_s = time.monotonic() - t0
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(psutil.Error):
                self.peak_rss = max(self.peak_rss, psutil.Process(self._proc.pid).memory_info().rss)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._limiter is not None:
            self._limiter.close()
            self._limiter = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_tail(self, lines: int = 4) -> str:
        """Last lines of the server log that look like errors (for messages)."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        keep = [ln.strip() for ln in text.splitlines() if ln.strip()]
        errs = [ln for ln in keep if any(k in ln.lower() for k in ("error", "fail", "alloc"))]
        return " | ".join((errs or keep)[-lines:])[:500]

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise LLMError(
                    f"The writer stopped while loading (exit {self._proc.returncode}): "
                    f"{self.log_tail()}",
                    fix="Free some memory or try --writer qwen2.5-1.5b.",
                )
            try:
                with _local_open(self.url + "/health", 5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        raise LLMError(f"The writer did not start within {timeout_s:.0f} s.")

    def _sample_rss(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(psutil.Error):
                self.peak_rss = max(self.peak_rss, psutil.Process(self._proc.pid).memory_info().rss)

    def chat(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float = 0.7,
        seed: int = 7,
        timeout_s: float = 1800,
    ) -> ChatResult:
        # Raw /completion with the model's own prompt format: the chat endpoint plus a JSON
        # grammar fails on Phi-3.5 ("empty grammar stack after accepting piece |", seen
        # with llama.cpp b12063, with and without --jinja); this path works for both models.
        prompt, stops = build_prompt(self.info.template, system, user)
        body = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "json_schema": schema,
            "stop": stops,
            "cache_prompt": True,
        }
        req = urllib.request.Request(  # noqa: S310  # local URL, checked in _local_open
            self.url + "/completion",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with _local_open(req, timeout_s) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise LLMError(f"The writer rejected the request ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            if self._proc is not None and self._proc.poll() is not None:
                raise LLMError(
                    "The writer crashed (likely out of memory).",
                    fix="Free some memory, or use --writer qwen2.5-1.5b.",
                ) from exc
            raise LLMError(f"The writer did not answer: {exc}") from exc
        finally:
            self._sample_rss()
        wall = time.monotonic() - t0
        text = str(out.get("content", ""))
        timings = out.get("timings", {})
        tokens = int(timings.get("predicted_n", 0))
        if out.get("stop_type") == "limit" or (out.get("truncated") and tokens >= max_tokens):
            raise LLMError(f"The writer ran out of tokens ({max_tokens}) before finishing.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"The writer returned invalid JSON: {exc}") from exc
        return ChatResult(data, text, wall, tokens, float(timings.get("predicted_per_second", 0.0)))


@contextlib.contextmanager
def writer(key: str, **kw: Any) -> Iterator[LocalWriter]:
    with LocalWriter(key, **kw) as w:
        yield w
