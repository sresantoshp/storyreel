"""Procedural ambience and music: exact length, no clipping, level, spectrum, determinism,
loop seam, constant memory. No downloads; loudness checks need ffmpeg."""

from __future__ import annotations

import hashlib
import tracemalloc
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import Media, needs_ffmpeg
from storyreel.audio import loudness, procedural

SR = 48_000
SECONDS = 5.3  # odd length: checks exact sample count and a partial last block
ALL = [("amb", k) for k in procedural.AMBIENCES] + [("music", m) for m in procedural.MOODS]


def _make(kind: str, name: str, out: Path, seconds: float = SECONDS, seed: int = 1) -> Path:
    if kind == "amb":
        return procedural.make_ambience(name, seconds, out, seed=seed)
    return procedural.make_music(name, seconds, out, seed=seed)


def _read(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path)) as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.uint8).reshape(-1, 3)
    b = np.zeros((raw.shape[0], 4), dtype=np.uint8)
    b[:, 1:] = raw
    x = (b.view("<i4").ravel() / 2.0**31).reshape(-1, ch)
    assert width == 3
    return x, ch, sr, n


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("proc")
    return {name: _make(kind, name, d / f"{name}.wav") for kind, name in ALL}


def _band_share(x: np.ndarray, lo: float, hi: float) -> float:
    m = x.mean(axis=1)
    f = np.fft.rfftfreq(m.size, 1 / SR)
    p = np.abs(np.fft.rfft(m)) ** 2
    return float(p[(f >= lo) & (f < hi)].sum() / p.sum())


@pytest.mark.parametrize("name", [n for _, n in ALL])
def test_exact_length_stereo_no_clipping(files: dict[str, Path], name: str) -> None:
    x, ch, sr, n = _read(files[name])
    assert (ch, sr) == (2, SR)
    assert n == round(SECONDS * SR)
    peak_db = 20 * np.log10(np.abs(x).max())
    assert peak_db < -1.0, f"{name} peaks at {peak_db:.2f} dBFS"
    rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)))
    assert rms_db > -60.0, f"{name} is nearly silent ({rms_db:.1f} dBFS)"


@pytest.mark.parametrize("name", [n for _, n in ALL])
def test_loop_seam_has_no_jump(files: dict[str, Path], name: str) -> None:
    """Wrapping from the last sample to the first is no bigger a step than the file's own
    sample-to-sample steps (99.9th percentile)."""
    x, *_ = _read(files[name])
    steps = np.abs(np.diff(x, axis=0))
    seam = np.abs(x[0] - x[-1])
    assert np.all(seam <= np.quantile(steps, 0.999, axis=0) + 1e-6)


def test_spectral_character(files: dict[str, Path]) -> None:
    rain, *_ = _read(files["rain"])
    assert _band_share(rain, 2000, 24000) > 0.6  # hiss + drops: broadband, bright
    assert _band_share(rain, 250, 2000) > 0.1
    wind, *_ = _read(files["wind"])
    assert _band_share(wind, 0, 1000) > 0.8  # low-frequency heavy
    assert _band_share(wind, 4000, 24000) < 0.02
    fire, *_ = _read(files["fire"])
    m = fire.mean(axis=1)
    assert np.abs(m).max() / np.sqrt(np.mean(m**2)) > 6  # crackles: very peaky
    night, *_ = _read(files["night"])
    assert _band_share(night, 4000, 5600) > 0.35  # crickets' carrier
    birds, *_ = _read(files["birds"])
    assert _band_share(birds, 2000, 8000) > 0.8  # chirps
    river, *_ = _read(files["river"])
    assert _band_share(river, 200, 2000) > 0.6  # babble/bubbles in the mid band
    room, *_ = _read(files["room"])
    assert 20 * np.log10(np.sqrt(np.mean(room**2))) < -38  # room tone is quiet
    for mood in procedural.MOODS:  # music is tonal: a few strong spectral lines
        x, *_ = _read(files[mood])
        p = np.abs(np.fft.rfft(x.mean(axis=1))) ** 2
        top = np.sort(p)[::-1]
        assert top[: p.size // 200].sum() / p.sum() > 0.6, mood


def test_moods_differ_in_pitch_content(files: dict[str, Path]) -> None:
    def chroma(name: str) -> np.ndarray:
        x, *_ = _read(files[name])
        p = np.abs(np.fft.rfft(x.mean(axis=1))) ** 2
        f = np.fft.rfftfreq(x.shape[0], 1 / SR)
        ok = (f > 60) & (f < 2000)
        pc = np.round(12 * np.log2(f[ok] / 440.0)).astype(int) % 12
        c = np.bincount(pc, weights=p[ok], minlength=12)
        return c / c.sum()

    calm, tense = chroma("calm"), chroma("tense")
    assert np.abs(calm - tense).sum() > 0.3


def test_deterministic_by_seed(tmp_path: Path) -> None:
    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for kind, name in (("amb", "rain"), ("amb", "birds"), ("music", "epic")):
        a = _make(kind, name, tmp_path / "a.wav", 3.0, seed=7)
        b = _make(kind, name, tmp_path / "b.wav", 3.0, seed=7)
        c = _make(kind, name, tmp_path / "c.wav", 3.0, seed=8)
        assert digest(a) == digest(b), name
        assert digest(a) != digest(c), name


def test_memory_does_not_grow_with_duration(tmp_path: Path) -> None:
    def peak(seconds: float) -> int:
        tracemalloc.start()
        procedural.make_ambience("rain", seconds, tmp_path / "m.wav")
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pk

    short, long_ = peak(8.0), peak(90.0)
    assert long_ < short + 4 * 2**20, (short, long_)
    assert long_ < 64 * 2**20


def test_smooth_control_signal_memory_is_bounded() -> None:
    s = procedural._Smooth(np.random.default_rng(1), 0.25, 0.0, 1.0)
    for i in range(2000):
        v = s.at(np.linspace(i, i + 1, 100))
        assert v.min() >= 0.0 and v.max() <= 1.0
    assert s.vals.size < 10


def test_bad_arguments_are_typed_errors(tmp_path: Path) -> None:
    with pytest.raises(procedural.ProceduralError) as e:
        procedural.make_ambience("thunder", 2, tmp_path / "x.wav")
    assert "rain" in (e.value.fix or "")
    with pytest.raises(procedural.ProceduralError):
        procedural.make_music("jazzy", 2, tmp_path / "x.wav")
    with pytest.raises(procedural.ProceduralError):
        procedural.make_music("calm", 0, tmp_path / "x.wav")
    with pytest.raises(procedural.ProceduralError):
        procedural.make_ambience("rain", 2, tmp_path / "x.mp3")
    assert not list(tmp_path.iterdir())


LOUDNESS_RANGE = {  # LUFS: beds sit well under dialogue; room tone is near-silent
    "rain": (-28, -20),
    "wind": (-31, -21),
    "fire": (-30, -22),
    "river": (-28, -20),
    "birds": (-34, -23),
    "night": (-37, -27),
    "room": (-47, -37),
    # environments: calibrated targets -24..-28 LUFS, +-5 dB because a 5 s stretch may or
    # may not hold a wave crash, a thunder roll or a rooster
    "ocean": (-30, -19),
    "beach": (-30, -19),
    "jungle": (-31, -21),
    "forest": (-32, -21),
    "thunderstorm": (-30, -19),
    "ice": (-31, -21),
    "waterfall": (-29, -19),
    "hill_station": (-33, -23),
    "village": (-33, -22),
    "stream": (-31, -21),
}


@needs_ffmpeg
@pytest.mark.parametrize("name", [n for _, n in ALL])
def test_loudness_in_sensible_range(files: dict[str, Path], media: Media, name: str) -> None:
    m = loudness.measure_file(media.ffmpeg, files[name])
    lo, hi = LOUDNESS_RANGE.get(name, (-22, -14))  # music beds around -18 LUFS
    assert lo <= m.integrated <= hi, f"{name}: {m.integrated:.1f} LUFS"
    assert m.true_peak <= -1.0, f"{name}: true peak {m.true_peak:.1f} dBTP"


ENVIRONMENTS = (
    "ocean",
    "beach",
    "jungle",
    "forest",
    "thunderstorm",
    "ice",
    "waterfall",
    "hill_station",
    "village",
    "stream",
)


def test_environment_kinds_are_listed() -> None:
    assert set(ENVIRONMENTS) <= set(procedural.AMBIENCES)
    assert set(procedural.AMBIENCES) == set(procedural._AMB_GAIN_DB)
    assert set(procedural.AMBIENCES) <= set(LOUDNESS_RANGE)


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_environment_is_finite_and_clean_at_block_boundaries(
    files: dict[str, Path], name: str
) -> None:
    """No NaN/inf, and no click where one ~1 s synthesis block hands over to the next:
    the steps across block boundaries are no bigger than the file's own large steps."""
    x, *_ = _read(files[name])
    assert np.all(np.isfinite(x))
    steps = np.abs(np.diff(x, axis=0))
    xf = min(int(procedural._XFADE_S * SR), round(SECONDS * SR) // 4)
    # output sample i is stream sample i + xf; stream blocks start at k * _BLOCK
    idx = np.arange(procedural._BLOCK, x.shape[0], procedural._BLOCK) - xf - 1
    idx = idx[(idx >= 0) & (idx < steps.shape[0])]
    assert idx.size >= 3
    assert np.all(steps[idx] <= np.quantile(steps, 0.999, axis=0) + 1e-6), name


def test_environment_spectral_character(files: dict[str, Path]) -> None:
    def x(name: str) -> np.ndarray:
        return _read(files[name])[0]

    assert _band_share(x("thunderstorm"), 2000, 24000) > 0.2  # rain hiss is always there
    assert _band_share(x("ice"), 250, 1000) > 0.5  # howling wind resonances
    assert _band_share(x("stream"), 400, 4000) > 0.6  # babble and bubbles
    assert _band_share(x("waterfall"), 0, 1000) > 0.4  # roar: weighty low end
    assert _band_share(x("waterfall"), 4000, 24000) > 0.08  # ... and mist hiss
    assert _band_share(x("jungle"), 2000, 9000) > 0.4  # insects and birdsong
    ocean = x("ocean")
    assert _band_share(ocean, 0, 1000) > 0.25 and _band_share(ocean, 2000, 24000) > 0.2


@pytest.mark.parametrize("name", ENVIRONMENTS)
def test_environment_deterministic_by_seed(tmp_path: Path, name: str) -> None:
    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    a = procedural.make_ambience(name, 2.5, tmp_path / "a.wav", seed=11)
    b = procedural.make_ambience(name, 2.5, tmp_path / "b.wav", seed=11)
    c = procedural.make_ambience(name, 2.5, tmp_path / "c.wav", seed=12)
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)


def test_environment_does_not_repeat(tmp_path: Path) -> None:
    """Events are random in time, pitch, level and pan: a 30 s ocean is not a short loop.
    The level envelope (0.25 s RMS) of one 10 s stretch does not match the next."""
    x, *_ = _read(procedural.make_ambience("ocean", 30.0, tmp_path / "o.wav", seed=3))
    hop = SR // 4
    env = np.sqrt(np.mean(x[: x.shape[0] // hop * hop].mean(axis=1).reshape(-1, hop) ** 2, axis=1))
    a, b = env[:40], env[40:80]
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr < 0.8, corr
    assert env.max() / env.min() > 2.0  # waves swell and fall back


def test_heavy_environment_memory_is_flat(tmp_path: Path) -> None:
    def peak(seconds: float) -> int:
        tracemalloc.start()
        procedural.make_ambience("thunderstorm", seconds, tmp_path / "m.wav")
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pk

    short, long_ = peak(8.0), peak(120.0)
    assert long_ < short + 4 * 2**20, (short, long_)
    assert long_ < 96 * 2**20


def test_sound_events_start_and_end_silent() -> None:
    """Every synthesised call/hit begins and ends at zero (no click when mixed in)."""
    from storyreel.audio import sfx

    rng = np.random.default_rng(5)
    calls = [sfx.songbird(rng, s) for s in sfx.SONGBIRDS]
    calls += [f(rng) for f in (sfx.gull, sfx.rooster, sfx.crow, sfx.dove, sfx.cuckoo)]
    calls += [f(rng) for f in (sfx.koel, sfx.hoot, sfx.whoop, sfx.parrot, sfx.woodpecker)]
    calls += [sfx.bell(rng), sfx.cowbell(rng)]
    calls += [sfx.thunder_crack, sfx.ice_crack, sfx.ice_creak]
    for c in calls:
        for _ in range(3):
            y = c(rng, SR)
            assert y.ndim == 1 and y.size > 100 and np.all(np.isfinite(y))
            assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
            assert 0.99 < np.abs(y).max() <= 1.0 + 1e-9
    d = sfx.drip(rng, SR, 1200.0)
    assert abs(d[0]) < 1e-3 and abs(d[-1]) < 1e-3
