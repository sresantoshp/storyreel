"""Time in flicks: 1 flick = 1/705,600,000 s.

Every common frame duration (23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120 fps) and sample
duration (44.1k, 48k, 96k) is a whole number of flicks, so positions never drift.
All conversions here are integer math.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

FLICKS = 705_600_000


@dataclass(frozen=True)
class Rate:
    num: int
    den: int = 1

    def __post_init__(self) -> None:
        if self.num <= 0 or self.den <= 0:
            raise ValueError(f"invalid rate {self.num}/{self.den}")

    @property
    def fps(self) -> float:
        return self.num / self.den

    def frame_to_flicks(self, frame: int) -> int:
        q, r = divmod(frame * self.den * FLICKS, self.num)
        return q + (1 if 2 * r >= self.num else 0)

    def flicks_to_frame(self, t: int) -> int:
        """Nearest frame index (round half up)."""
        q, r = divmod(t * self.num, self.den * FLICKS)
        return q + (1 if 2 * r >= self.den * FLICKS else 0)

    def frame_to_sample(self, frame: int, sample_rate: int) -> int:
        """First audio sample of `frame`. Floor, so consecutive chunks tile exactly."""
        return frame * sample_rate * self.den // self.num

    def __str__(self) -> str:
        return f"{self.num}/{self.den}"


def seconds_to_flicks(s: float | Fraction) -> int:
    return round(Fraction(s) * FLICKS)


def flicks_to_seconds(t: int) -> float:
    return t / FLICKS


def sec_str(t: int) -> str:
    """Flicks to a decimal-seconds string for ffmpeg arguments (microsecond precision)."""
    return f"{t / FLICKS:.6f}"
