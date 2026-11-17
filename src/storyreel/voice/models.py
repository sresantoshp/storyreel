"""TTS model files: where they live, where they come from, and their licences.

Every file is downloaded once, verified against the SHA-256 digest that the publisher
lists (GitHub release asset `digest`), and then used offline. Nothing here runs at
synthesis time except the presence check.

Licences (checked 2026-11-17 from the official sources):
- Kokoro-82M weights: Apache-2.0 (Hugging Face `hexgrad/Kokoro-82M`, cardData.license).
  Commercial use allowed. Voices ship inside the same repo under the same licence.
- ONNX export of those weights + `kokoro-onnx` runtime: MIT (GitHub thewh1teagle/kokoro-onnx).
- Phonemiser: `phonemizer` (GPL-3.0) driving espeak-ng (GPL-3.0, bundled by the MIT-licensed
  `espeakng-loader` wheel). Free and open source; the GPL covers redistribution of the
  software, not the audio it produces.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from storyreel import paths
from storyreel.errors import ExitCode, StoryReelError

LANGS = ("en", "hi")


class VoiceError(StoryReelError):
    """Base class for text-to-speech errors."""


class ModelMissingError(VoiceError):
    exit_code = ExitCode.ENV_CHECK_FAILED


class ModelDownloadError(VoiceError):
    exit_code = ExitCode.ENV_CHECK_FAILED


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    id: str
    files: tuple[ModelFile, ...]
    license: str
    license_source: str
    langs: tuple[str, ...]
    sample_rate: int


_GH = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"

KOKORO = ModelSpec(
    id="kokoro-v1.0",
    files=(
        ModelFile(
            "kokoro-v1.0.onnx",
            _GH + "kokoro-v1.0.onnx",
            "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
            325_505_369,
        ),
        ModelFile(
            "voices-v1.0.bin",
            _GH + "voices-v1.0.bin",
            "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
            28_214_398,
        ),
    ),
    license="Kokoro-82M Apache-2.0 (weights and voices); ONNX export MIT",
    license_source="https://huggingface.co/hexgrad/Kokoro-82M ; "
    "https://github.com/thewh1teagle/kokoro-onnx",
    langs=LANGS,
    sample_rate=24_000,
)

# espeak-ng language code and voice-id prefix per StoryReel language.
ESPEAK_LANG = {"en": "en-us", "hi": "hi"}
VOICE_PREFIXES = {"en": ("af_", "am_", "bf_", "bm_"), "hi": ("hf_", "hm_")}
DEFAULT_VOICE = {"en": "af_heart", "hi": "hf_alpha"}

_MARKER = "verified.json"


def check_lang(lang: str) -> None:
    if lang not in LANGS:
        raise VoiceError(
            f"Language {lang!r} is not supported for speech.",
            fix=f"Use one of: {', '.join(LANGS)}.",
        )


def spec_for(lang: str) -> ModelSpec:
    check_lang(lang)
    return KOKORO


def models_root() -> Path:
    override = os.environ.get("STORYREEL_MODELS_DIR")
    return Path(override) if override else paths.data_dir() / "models"


def model_dir(spec: ModelSpec) -> Path:
    return models_root() / spec.id


def download_command(lang: str) -> str:
    return f"storyreel models download --lang {lang}"


def _marker(spec: ModelSpec) -> dict[str, dict[str, int]]:
    try:
        data = json.loads((model_dir(spec) / _MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    files = data.get("files", {}) if isinstance(data, dict) else {}
    return files if isinstance(files, dict) else {}


def files_present(spec: ModelSpec) -> bool:
    """All files exist with the size and mtime recorded when their SHA-256 was verified.

    Re-hashing 140 MB on every call would cost ~0.5 s, so the full hash is checked once at
    download time and a size + mtime marker is trusted afterwards.
    """
    marker = _marker(spec)
    for f in spec.files:
        path = model_dir(spec) / f.name
        rec = marker.get(f.name)
        try:
            st = path.stat()
        except OSError:
            return False
        if rec is None or st.st_size != f.size or rec.get("mtime_ns") != st.st_mtime_ns:
            return False
    return True


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(f: ModelFile, dest: Path, progress: Callable[[str], None] | None) -> None:
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    got = 0
    last = 0.0
    try:
        # fixed https:// URLs from the table above, never user input
        req = urllib.request.Request(f.url, headers={"User-Agent": "storyreel-model-fetch"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=30) as resp, part.open("wb") as out:  # noqa: S310 (fixed https URL)
            while block := resp.read(1 << 20):
                out.write(block)
                h.update(block)
                got += len(block)
                now = time.monotonic()
                if progress and now - last > 1.0:
                    last = now
                    progress(f"{f.name}: {got / 2**20:.0f} / {f.size / 2**20:.0f} MB")
            out.flush()
            os.fsync(out.fileno())
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Could not download {f.name} ({exc}). The one-time model download needs internet.",
            fix=f"Connect to the internet and run the download again, or copy {f.name} "
            f"(SHA-256 {f.sha256}) into {dest.parent}.",
        ) from exc
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Could not save {f.name} to {dest.parent}: {exc.strerror or exc}.",
            fix="Free some disk space or check the folder's permissions.",
        ) from exc
    digest = h.hexdigest()
    if got != f.size or digest != f.sha256:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{f.name} failed verification (got {got} bytes, SHA-256 {digest}; expected "
            f"{f.size} bytes, {f.sha256}). The download was damaged or the file was changed.",
            fix="Run the download again; if it keeps failing, report it.",
        )
    part.replace(dest)


def ensure(spec: ModelSpec, progress: Callable[[str], None] | None = None) -> Path:
    """Download (once) and verify every file of `spec`. Returns the model directory."""
    d = model_dir(spec)
    if files_present(spec):
        return d
    d.mkdir(parents=True, exist_ok=True)
    marker = _marker(spec)
    for f in spec.files:
        dest = d / f.name
        ok = False
        if dest.exists() and dest.stat().st_size == f.size:
            if progress:
                progress(f"{f.name}: verifying existing file")
            ok = sha256_file(dest) == f.sha256
        if not ok:
            if progress:
                progress(f"{f.name}: downloading {f.size / 2**20:.0f} MB from {f.url}")
            _download(f, dest, progress)
        marker[f.name] = {"mtime_ns": dest.stat().st_mtime_ns, "size": f.size}
        if progress:
            progress(f"{f.name}: SHA-256 OK ({f.sha256[:16]}...)")
    info = {
        "model": spec.id,
        "license": spec.license,
        "license_source": spec.license_source,
        "files": marker,
        "sources": {f.name: {"url": f.url, "sha256": f.sha256} for f in spec.files},
        "verified_at": time.time(),
    }
    tmp = d / (_MARKER + ".part")
    tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    tmp.replace(d / _MARKER)
    return d
