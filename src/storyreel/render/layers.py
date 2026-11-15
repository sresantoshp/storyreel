"""Frame sources for the compositor.

Each `frame(c)` returns planes `(rgb, alpha)`: rgb float32 (H, W, 3) 0..1, alpha float32
(H, W) or None for fully opaque. `c` is the clip-local frame index; it may be negative or
past the end during transitions, in which case media holds its first/last frame.

MediaSource decodes through an ffmpeg pipe: rgb24 for opaque sources (most video), rgba
only when the source pixel format can carry alpha. Memory: at most 2 decoded frames
(current + previous for frame blending), or one REVERSE_BLOCK-frame block when reversed.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from storyreel.render.effects import u8_to_f
from storyreel.render.generators import Generator, Planes
from storyreel.render.plan import color_matrix
from storyreel.text.render import TextRenderer, hex_rgb

REVERSE_BLOCK = 12
MAX_DETAIL = 2.0  # never decode larger than 2x the canvas, whatever the zoom
_ALPHA_FMTS = ("rgba", "bgra", "argb", "abgr", "yuva", "gbrap", "ya8", "ya16", "pal8", "rgba64")


def ff_escape(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    p = path.replace("\\", "/").replace("'", r"'\''")
    return "'" + p.replace(":", r"\:") + "'"


def fit_size(sw: int, sh: int, cw: int, ch: int) -> tuple[int, int]:
    k = min(cw / sw, ch / sh)
    return max(2, round(sw * k / 2) * 2), max(2, round(sh * k / 2) * 2)


def has_alpha(pix_fmt: str | None, is_image: bool) -> bool:
    if pix_fmt is None:
        return is_image  # unknown: stills are checked after decoding, video assumed opaque
    return pix_fmt.startswith(_ALPHA_FMTS)


class MediaSource:
    def __init__(
        self,
        ffmpeg: Path,
        layer: dict[str, Any],
        canvas: tuple[int, int],
        proj_fps: float,
        time_of: Callable[[int], float],
        max_scale: float,
        speed: float = 1.0,
    ) -> None:
        self.ffmpeg = str(ffmpeg)
        self.path = str(layer["path"])
        a = layer["asset"]
        self.is_image = a["kind"] == "image"
        fw, fh = fit_size(a["width"], a["height"], *canvas)
        detail = max(min(max(max_scale, 1.0), a["width"] / fw, MAX_DETAIL), 1.0)
        self.fit = (fw, fh)
        self.dec = (max(2, round(fw * detail / 2) * 2), max(2, round(fh * detail / 2) * 2))
        self.matrix = color_matrix(a["height"], a.get("color_space"))
        self.alpha = has_alpha(a.get("pix_fmt"), self.is_image)
        self.channels = 4 if self.alpha else 3
        self.src_fps = (a["fps_num"] / a["fps_den"]) if a.get("fps_num") else proj_fps
        self.duration = (a["duration"] or 0) / 705_600_000
        self.time_of = time_of
        # Half an output frame, in source frames: picks the same source frame as ffmpeg's
        # fps filter (round-to-nearest), so path A and path B chunks match frame for frame.
        self.half = 0.5 * speed * self.src_fps / proj_fps
        props = layer["props"]
        self.blend = bool(props.get("blend"))
        self.reverse = bool(props.get("reverse"))
        self.lut = [e for e in layer["effects"] if e["type"] == "lut"]
        self._proc: subprocess.Popen[bytes] | None = None
        self._k = -1  # index of `_cur` in the current stream
        self._t0 = 0.0
        self._cur: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._still: Planes | None = None
        self._block: list[np.ndarray] = []
        self._block_t0 = 0.0
        self._eof = False

    # -- ffmpeg ------------------------------------------------------------------------
    def _vf(self) -> str:
        w, h = self.dec
        pix = "rgba" if self.alpha else "rgb24"
        chain = []
        if not self.is_image:
            chain.append(f"fps={self.src_fps:.6f}")
        chain.append(f"scale={w}:{h}:flags=bicubic:in_color_matrix={self.matrix}")
        for e in self.lut:
            mix = float(e["params"].get("mix", 1.0))
            lut = f"lut3d=file={ff_escape(str(e['params']['path']))}:interp=tetrahedral"
            if mix >= 0.999:
                chain.append(lut)
            else:
                return (
                    ",".join(chain) + f",format=gbrp,split[a][b];[b]{lut}[l];"
                    f"[a][l]blend=all_mode=normal:all_opacity={mix:.4f},format={pix}"
                )
        chain.append(f"format={pix}")
        return ",".join(chain)

    def _open(self, t0: float, n: int | None = None) -> subprocess.Popen[bytes]:
        argv = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if not self.is_image:
            argv += ["-ss", f"{max(t0, 0.0):.6f}"]
        argv += ["-i", self.path, "-filter_complex", f"[0:v:0]{self._vf()}[out]", "-map", "[out]"]
        if n is not None or self.is_image:
            argv += ["-frames:v", str(1 if self.is_image else n)]
        argv += ["-f", "rawvideo", "-pix_fmt", "rgba" if self.alpha else "rgb24", "pipe:1"]
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read(self, proc: subprocess.Popen[bytes]) -> np.ndarray | None:
        w, h = self.dec
        n = w * h * self.channels
        assert proc.stdout is not None
        buf = proc.stdout.read(n)
        if buf is None or len(buf) < n:
            return None
        return np.frombuffer(buf, np.uint8).reshape(h, w, self.channels)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    # -- frames ------------------------------------------------------------------------
    def _planes(self, x: np.ndarray) -> Planes:
        if self.channels == 3:
            return u8_to_f(x), None
        return u8_to_f(np.ascontiguousarray(x[..., :3])), u8_to_f(np.ascontiguousarray(x[..., 3]))

    def frame(self, c: int) -> Planes:
        if self.is_image:
            if self._still is None:
                proc = self._open(0.0)
                img = self._read(proc)
                proc.wait()
                if img is None:
                    raise RuntimeError(f"could not decode image {self.path}")
                rgb, a = self._planes(img)
                if a is not None and float(np.min(a)) >= 1.0:
                    a = None  # PNG without real transparency: take the opaque fast paths
                self._still = (rgb, a)
            return self._still
        t = min(max(self.time_of(c), 0.0), max(self.duration - 0.5 / self.src_fps, 0.0))
        if self.reverse:
            return self._planes(self._reverse_frame(t))
        return self._forward_frame(t)

    def _forward_frame(self, t: float) -> Planes:
        k_float = (t - self._t0) * self.src_fps
        k = max(int(np.ceil(k_float + self.half - 1e-6)) - 1, 0)
        if self._proc is None or k < self._k or k - self._k > 4 * self.src_fps:
            self.close()  # (re)seek: first use, backwards, or a long jump
            self._t0, self._k, self._eof = t, -1, False
            self._proc = self._open(t)
            self._cur = self._prev = None
            k_float, k = 0.0, max(int(np.ceil(self.half - 1e-6)) - 1, 0)
        while self._k < k + (1 if self.blend else 0) and not self._eof:
            nxt = self._read(self._proc)
            if nxt is None:
                self._eof = True
                break
            self._prev, self._cur = self._cur, nxt
            self._k += 1
        if self._cur is None:
            raise RuntimeError(f"no frames decoded from {self.path} at {t:.3f} s")
        if self.blend and self._prev is not None and not self._eof:
            frac = min(max(float(k_float - (self._k - 1)), 0.0), 1.0)
            prev, cur = self._planes(self._prev), self._planes(self._cur)
            rgb = np.asarray(prev[0] * (1 - frac) + cur[0] * frac, np.float32)
            if prev[1] is None or cur[1] is None:
                return rgb, None
            return rgb, np.asarray(prev[1] * (1 - frac) + cur[1] * frac, np.float32)
        if self._k > k and self._prev is not None:
            return self._planes(self._prev)
        return self._planes(self._cur)

    def _reverse_frame(self, t: float) -> np.ndarray:
        """Decode short forward blocks ending at t and serve them backwards."""
        idx = int(round((t - self._block_t0) * self.src_fps))
        if not self._block or not 0 <= idx < len(self._block):
            start = max(t - (REVERSE_BLOCK - 1) / self.src_fps, 0.0)
            proc = self._open(start, REVERSE_BLOCK)
            block: list[np.ndarray] = []
            while (img := self._read(proc)) is not None:
                block.append(img)
            proc.wait()
            if not block:
                raise RuntimeError(f"no frames decoded from {self.path} at {t:.3f} s")
            self._block, self._block_t0 = block, start
            idx = min(int(round((t - start) * self.src_fps)), len(block) - 1)
        return self._block[idx]


class ColorSource:
    def __init__(self, hex_color: str, canvas: tuple[int, int]) -> None:
        w, h = canvas
        rgb = np.empty((h, w, 3), np.float32)
        rgb[:] = hex_rgb(hex_color)
        rgb.setflags(write=False)
        self.planes: Planes = (rgb, None)
        self.fit = canvas

    def frame(self, c: int) -> Planes:
        return self.planes

    def close(self) -> None:
        return None


class TextSource:
    def __init__(self, props: dict[str, Any], fps: float) -> None:
        self.r = TextRenderer(props)
        self.fps = fps
        self.fit = self.r.size

    def frame(self, c: int) -> Planes:
        return self.r.frame(max(c, 0) / self.fps)

    def close(self) -> None:
        return None


class GeneratorSource:
    def __init__(self, props: dict[str, Any], canvas: tuple[int, int], fps: float) -> None:
        params = {k: v for k, v in props.items() if k != "type"}
        self.g = Generator(str(props["type"]), params, *canvas)
        self.fps = fps
        self.fit = canvas

    def frame(self, c: int) -> Planes:
        return self.g.frame(c / self.fps)

    def close(self) -> None:
        return None
