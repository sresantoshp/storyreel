"""CLIP image/text matching on the CPU, to pick footage that shows what the story says.

Model: OpenAI CLIP ViT-B/32 (MIT), ONNX export by Xenova (MIT), int8-quantised text and
image towers (61 + 85 MB) from a pinned Hugging Face revision, SHA-256 verified, used
offline afterwards. ~300 MB RAM while loaded; ~40 ms per image, ~15 ms per text on an
i3-1315U. English text only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from storyreel.voice import models

log = logging.getLogger(__name__)
F32 = npt.NDArray[np.float32]

_REV = "6b7c523703ae26e39be4b03774966f8571c7f523"
_HF = f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/{_REV}/"

CLIP_SPEC = models.ModelSpec(
    id="clip-vit-b32-int8",
    files=(
        models.ModelFile(
            "vision_model_int8.onnx",
            _HF + "onnx/vision_model_int8.onnx",
            "0ab0c1b3ace708e539633af1744d5a95247fe4e14d3e08ff197ef82a6cb9bd93",
            88_648_877,
        ),
        models.ModelFile(
            "text_model_int8.onnx",
            _HF + "onnx/text_model_int8.onnx",
            "18845f2ccc35223bb7fec403383a131154b11ac0918df25cf51986df5efd3a21",
            64_070_791,
        ),
        models.ModelFile(
            "tokenizer.json",
            _HF + "tokenizer.json",
            "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
            2_224_119,
        ),
    ),
    license="MIT (OpenAI CLIP weights; Xenova ONNX export)",
    license_source="huggingface.co/Xenova/clip-vit-base-patch32 cardData.license",
    langs=("en",),
    sample_rate=0,
)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
SIZE = 224
CONTEXT = 77


def preprocess(rgb: npt.NDArray[np.uint8]) -> F32:
    """CLIP input: shortest side → 224 (bicubic), centre crop, normalise; (3, 224, 224)."""
    h, w = rgb.shape[:2]
    k = SIZE / min(h, w)
    nw, nh = max(SIZE, round(w * k)), max(SIZE, round(h * k))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y, x = (nh - SIZE) // 2, (nw - SIZE) // 2
    crop = img[y : y + SIZE, x : x + SIZE].astype(np.float32) / 255.0
    out: F32 = ((crop - _MEAN) / _STD).transpose(2, 0, 1)
    return out


def _unit(x: F32) -> F32:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    out: F32 = (x / np.maximum(n, 1e-8)).astype(np.float32)
    return out


class Clip:
    """Loaded CLIP model. Create once and reuse (loading takes ~1 s)."""

    def __init__(self, progress: Callable[[str], None] | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = models.ensure(CLIP_SPEC, progress)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        cpu = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(str(d / "vision_model_int8.onnx"), opts, cpu)
        self._text = ort.InferenceSession(str(d / "text_model_int8.onnx"), opts, cpu)
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(CONTEXT)
        pad = self._tok.token_to_id("<|endoftext|>")
        self._tok.enable_padding(length=CONTEXT, pad_id=pad, pad_token="<|endoftext|>")  # noqa: S106
        self._text_inputs = {i.name for i in self._text.get_inputs()}

    def texts(self, texts: Sequence[str]) -> F32:
        """Unit-length embeddings, (n, 512)."""
        enc = self._tok.encode_batch([t.lower() for t in texts])
        feed: dict[str, Any] = {"input_ids": np.array([e.ids for e in enc], np.int64)}
        if "attention_mask" in self._text_inputs:
            feed["attention_mask"] = np.array([e.attention_mask for e in enc], np.int64)
        out = self._text.run(None, feed)
        return _unit(np.asarray(_pick(self._text, out, "text_embeds"), np.float32))

    def images(self, rgbs: Sequence[npt.NDArray[np.uint8]]) -> F32:
        """Unit-length embeddings, (n, 512); images are RGB uint8 of any size."""
        if not rgbs:
            return np.zeros((0, 512), np.float32)
        batch = np.stack([preprocess(x) for x in rgbs])
        out = self._vision.run(None, {"pixel_values": batch})
        return _unit(np.asarray(_pick(self._vision, out, "image_embeds"), np.float32))


def _pick(sess: Any, outputs: list[Any], name: str) -> Any:
    names = [o.name for o in sess.get_outputs()]
    return outputs[names.index(name)] if name in names else outputs[0]


def read_image(path: Path) -> npt.NDArray[np.uint8] | None:
    data = np.fromfile(str(path), np.uint8)  # works with non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.uint8)


def describe_spec() -> str:
    return json.dumps({"model": CLIP_SPEC.id, "license": CLIP_SPEC.license})
