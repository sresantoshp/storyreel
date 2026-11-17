"""English → Indian languages with AI4Bharat IndicTrans2 (distilled 200M, MIT), offline.

Install: `uv sync --extra translate` (torch CPU, transformers<5, sentencepiece, sacremoses,
indic-nlp-library) and download the model once (`storyreel models download --translator`,
needs a free Hugging Face login with the model's terms accepted; it is a gated repo).

Pre/post-processing follows AI4Bharat's IndicTransToolkit `IndicProcessor` (MIT) for the
English-source path: Moses punctuation normalisation + tokenisation, language tags, then
transliteration of the model's Devanagari output into the target script and Indic
detokenisation. The toolkit itself is not used because it ships only as a Cython source
package that needs a C compiler on Windows; the steps it performs for English input are
reproduced here with the same pure-Python libraries it calls. URL/e-mail placeholder
protection is not reproduced (story narration does not contain them).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from storyreel import paths
from storyreel.errors import ExitCode, StoryReelError

log = logging.getLogger(__name__)

REPO = "ai4bharat/indictrans2-en-indic-dist-200M"
REVISION = "2cfd0a20057756d756bf3e53528f492afc52f91a"  # pinned: remote code runs locally
FILES = (
    "config.json",
    "generation_config.json",
    "configuration_indictrans.py",
    "modeling_indictrans.py",
    "tokenization_indictrans.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "dict.SRC.json",
    "dict.TGT.json",
    "model.SRC",
    "model.TGT",
    "model.safetensors",
    "LICENSE",
)
TARGETS: dict[str, tuple[str, str]] = {  # our code -> (FLORES tag, indic-nlp iso)
    "hi": ("hin_Deva", "hi"),
    "ta": ("tam_Taml", "ta"),
    "te": ("tel_Telu", "te"),
    "kn": ("kan_Knda", "kn"),
    "bn": ("ben_Beng", "bn"),
    "gu": ("guj_Gujr", "gu"),
}
_SENT = re.compile(r"(?<=[.!?])\s+")


class TranslatorUnavailableError(StoryReelError):
    exit_code = ExitCode.ENV_CHECK_FAILED


def model_dir() -> Path:
    return paths.data_dir() / "models" / "indictrans2-en-indic-dist-200M"


def _deps_ok() -> bool:
    try:
        import indicnlp  # noqa: F401
        import sacremoses  # noqa: F401
        import sentencepiece  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def available() -> bool:
    d = model_dir()
    return _deps_ok() and all((d / f).is_file() for f in FILES)


def why_unavailable() -> str:
    if not _deps_ok():
        return "translator packages are not installed (uv sync --extra translate)"
    missing = [f for f in FILES if not (model_dir() / f).is_file()]
    return f"translator model files missing: {', '.join(missing[:3])}..." if missing else ""


def ensure_model(progress: Any = None) -> Path:
    """Download the pinned model files and verify the weights against the hub's SHA-256."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    say = progress or (lambda m: None)
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
        expected = {s.rfilename: s.lfs.sha256 for s in info.siblings or [] if s.lfs}
        for f in FILES:
            say(f"downloading {f}")
            hf_hub_download(REPO, f, revision=REVISION, local_dir=str(d))
    except GatedRepoError as exc:
        raise TranslatorUnavailableError(
            "Hugging Face refused the download: the model's terms are not accepted yet.",
            fix=f"Open https://huggingface.co/{REPO} while logged in and click "
            "'Agree and access repository', then retry.",
        ) from exc
    except HfHubHTTPError as exc:
        raise TranslatorUnavailableError(
            f"Download failed: {exc}", fix="Check the internet connection and `hf auth whoami`."
        ) from exc
    for f, sha in expected.items():
        if f in FILES:
            h = hashlib.sha256()
            with (d / f).open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != sha:
                (d / f).unlink(missing_ok=True)
                raise TranslatorUnavailableError(f"{f} failed its SHA-256 check and was removed.")
            say(f"verified {f}")
    return d


class Translator:
    """Loaded once per process; translate() batches sentences per call."""

    def __init__(self, threads: int = 4) -> None:
        if not available():
            raise TranslatorUnavailableError(
                f"The translator is not ready: {why_unavailable()}.",
                fix="See docs/PROGRESS.md (Phase 4, translation).",
            )
        import torch
        from sacremoses import MosesPunctNormalizer, MosesTokenizer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(threads)
        d = str(model_dir())
        # trust_remote_code runs the model's own Python files, pinned by REVISION above
        self.tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(d, trust_remote_code=True).eval()
        self.norm = MosesPunctNormalizer(lang="en")
        self.mtok = MosesTokenizer(lang="en")
        self.torch = torch

    def translate(self, texts: list[str], lang: str, *, beams: int = 4) -> list[str]:
        from indicnlp.tokenize import indic_detokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

        if lang not in TARGETS:
            raise TranslatorUnavailableError(
                f"No translation into {lang!r}.", fix=f"Supported: {', '.join(TARGETS)}"
            )
        tag, iso = TARGETS[lang]
        # translate sentence by sentence (the model is trained on sentences, max 256 tokens)
        pieces: list[tuple[int, str]] = [
            (i, s) for i, t in enumerate(texts) for s in _SENT.split(t.strip()) if s.strip()
        ]
        out: dict[int, list[str]] = {i: [] for i in range(len(texts))}
        for b in range(0, len(pieces), 8):
            chunk = pieces[b : b + 8]
            batch = [
                f"eng_Latn {tag} "
                + " ".join(self.mtok.tokenize(self.norm.normalize(s), escape=False))
                for _, s in chunk
            ]
            enc = self.tok(
                batch, padding="longest", truncation=True, max_length=256, return_tensors="pt"
            )
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, num_beams=beams, max_length=256, use_cache=True)
            decoded = self.tok.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for (i, _), text in zip(chunk, decoded, strict=True):
                if iso != "hi":  # the model writes every Indic language in Devanagari
                    text = UnicodeIndicTransliterator.transliterate(text, "hi", iso)
                out[i].append(indic_detokenize.trivial_detokenize(text, iso))
        return [" ".join(out[i]) for i in range(len(texts))]


@functools.lru_cache(maxsize=1)
def shared() -> Translator:
    return Translator()
