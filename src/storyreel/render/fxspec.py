"""Names, parameters and defaults for everything the compositor understands.

Pure data, no numpy/cv2 import: the editor validates against this, the compositor reads
defaults from it. A parameter listed as keyframable can be animated with keyframes named
`<param>` (clip transform) or `fx<effect_id>.<param>` (effect).
"""

from __future__ import annotations

from typing import Any

from storyreel.audio.spec import KEYFRAMABLE as AUDIO_KEYFRAMABLE

# Clip transform. x/y are pixel offsets of the layer centre from the frame centre.
TRANSFORM_DEFAULTS: dict[str, float] = {
    "x": 0.0,
    "y": 0.0,
    "scale": 1.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "rotation": 0.0,
    "opacity": 1.0,
    "crop_l": 0.0,
    "crop_t": 0.0,
    "crop_r": 0.0,
    "crop_b": 0.0,
    "speed": 1.0,  # only as keyframes (ramps); constant speed uses clip.speed_num/den
}

# effect type -> {param: default}. Strings/lists are static; numbers are keyframable.
EFFECTS: dict[str, dict[str, Any]] = {
    "color": {
        "exposure": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "temperature": 0.0,
        "tint": 0.0,
        "lift_r": 0.0,
        "lift_g": 0.0,
        "lift_b": 0.0,
        "gamma_r": 1.0,
        "gamma_g": 1.0,
        "gamma_b": 1.0,
        "gain_r": 1.0,
        "gain_g": 1.0,
        "gain_b": 1.0,
        "brightness": 1.0,
    },
    "curves": {
        "master": [[0, 0], [1, 1]],
        "r": [[0, 0], [1, 1]],
        "g": [[0, 0], [1, 1]],
        "b": [[0, 0], [1, 1]],
    },
    "lut": {"path": "", "mix": 1.0},
    "colormatch": {
        "src_mean": [0.0, 0.0, 0.0],
        "src_std": [1.0, 1.0, 1.0],
        "dst_mean": [0.0, 0.0, 0.0],
        "dst_std": [1.0, 1.0, 1.0],
        "mix": 1.0,
    },
    "blur": {"radius": 4.0},
    "sharpen": {"amount": 0.5, "radius": 1.5},
    "glow": {"radius": 12.0, "strength": 0.6, "threshold": 0.6},
    "vignette": {"amount": 0.4, "softness": 0.5},
    "grain": {"amount": 0.06, "size": 1.0},
    "chromatic": {"shift": 3.0},
    "glitch": {"amount": 0.3, "seed": 1.0},
    "lens": {"k1": -0.15},
    "mask": {
        "points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
        "feather": 20.0,
        "invert": False,
    },
    "chromakey": {"key": "#00ff00", "similarity": 0.18, "smoothness": 0.08, "spill": 0.5},
}

COLOR_PRESETS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "teal-orange": [
        (
            "color",
            {
                "contrast": 1.12,
                "saturation": 1.1,
                "lift_b": 0.03,
                "lift_g": 0.01,
                "gain_r": 1.06,
                "gain_g": 1.0,
                "gain_b": 0.92,
                "temperature": 0.05,
            },
        ),
        ("vignette", {"amount": 0.25}),
    ],
    "noir": [
        ("color", {"saturation": 0.0, "contrast": 1.35, "exposure": -0.1}),
        ("curves", {"master": [[0, 0], [0.25, 0.18], [0.75, 0.85], [1, 1]]}),
        ("vignette", {"amount": 0.5}),
        ("grain", {"amount": 0.08}),
    ],
    "documentary": [
        ("color", {"contrast": 1.05, "saturation": 0.9, "temperature": 0.02}),
        ("curves", {"master": [[0, 0.02], [0.5, 0.5], [1, 0.97]]}),
    ],
    "night": [
        (
            "color",
            {
                "exposure": -0.6,
                "saturation": 0.6,
                "temperature": -0.25,
                "tint": 0.03,
                "gain_b": 1.1,
            },
        ),
        ("vignette", {"amount": 0.45}),
    ],
}

TRANSITIONS = frozenset(
    {
        "dissolve",
        "fade_black",
        "wipe_left",
        "wipe_right",
        "wipe_up",
        "wipe_down",
        "slide_left",
        "slide_right",
        "zoom",
        "whip",
    }
)

GENERATORS: dict[str, dict[str, Any]] = {
    "gradient": {"top": "#101830", "bottom": "#304060", "angle": 90.0, "drift": 0.0},
    "starfield": {"count": 900, "speed": 0.15, "color": "#ffffff", "seed": 1},
    "rain": {
        "count": 900,
        "angle": 12.0,
        "speed": 1800.0,
        "length": 38.0,
        "color": "#aabbdd",
        "opacity": 0.55,
        "seed": 1,
    },
    "snow": {"count": 500, "speed": 90.0, "size": 3.0, "sway": 30.0, "opacity": 0.9, "seed": 1},
    "fog": {"density": 0.45, "speed": 25.0, "scale": 260.0, "color": "#c8ccd4", "seed": 1},
    # width 0 = a band of flame across the whole frame; >0 = one campfire of that width
    # (fraction of frame width) centred at x, with logs, its base at `base` (fraction of height)
    "fire": {
        "height": 0.35,
        "speed": 220.0,
        "intensity": 1.0,
        "width": 0.0,
        "x": 0.5,
        "base": 0.9,
        "seed": 1,
    },
    "particles": {
        "count": 120,
        "size": 14.0,
        "speed": 30.0,
        "color": "#ffd890",
        "opacity": 0.5,
        "seed": 1,
    },
    # Illustrated landscape pieces; `horizon` (fraction of height) must match between
    # mountains and river so the valley floor lines up.
    "mountains": {
        "layers": 3,
        "far": "#a7bdd6",
        "near": "#3e5d4a",
        "height": 0.42,
        "horizon": 0.62,
        "snowcap": 0.22,
        "drift": 0.0,
        "seed": 3,
    },
    "river": {
        "horizon": 0.62,
        "bank_far": "#86a95e",
        "bank": "#4f7d35",
        "water_far": "#b4d6ee",
        "water": "#2f6fa8",
        "width_far": 0.03,
        "width_near": 0.5,
        "meander": 0.12,
        "speed": 0.12,
        "ripples": 160,
        "seed": 2,
    },
    "birds": {"count": 7, "size": 26.0, "speed": 90.0, "color": "#1d1d24", "flap": 3.0, "seed": 4},
}

# Illustrated environments (render/scenery.py) and animated characters (render/actors.py)
SCENERY_DEFAULTS: dict[str, dict[str, Any]] = {
    "sky": {
        "top": "#4f94dc",
        "bottom": "#cfe6f6",
        "body": "sun",  # sun | moon | none
        "body_x": 0.78,
        "body_y": 0.18,
        "body_size": 0.06,
        "body_color": "#fff3c4",
        "glow": 0.5,
        "clouds": 5,
        "cloud_speed": 10.0,
        "cloud_color": "#ffffff",
        "cloud_opacity": 0.92,
        "stars": 0,
        "seed": 1,
    },
    "sea": {
        "horizon": 0.58,
        "far": "#8cc2e2",
        "near": "#1b5c8a",
        "waves": 90,
        "speed": 1.0,
        "glitter": 0.6,
        "glitter_x": 0.78,
        "seed": 1,
    },
    "beach": {
        "shore": 0.74,
        "slope": 0.04,
        "sand": "#ecd6a4",
        "sand_near": "#d9b77c",
        "wet": "#b99a6a",
        "foam": "#ffffff",
        "tide": 0.025,
        "period": 7.0,
        "seed": 1,
    },
    "hills": {
        "layers": 3,
        "far": "#a9cfa4",
        "near": "#3d7a3b",
        "horizon": 0.64,
        "height": 0.26,
        "tea": 0.0,  # 0..1: terraced tea-garden rows on the nearest hill
        "houses": 0,  # little hill-station houses on the middle hill
        "wall": "#f4ead8",
        "roof": "#b4513a",
        "lit": 0.0,
        "seed": 5,
    },
    "trees": {
        "style": "pine",  # pine | round | palm | jungle | bamboo | mixed
        "count": 12,
        "rows": 2,
        "base": 0.66,
        "size": 0.3,
        "color": "#2f6b3a",
        "trunk": "#5a3d26",
        "haze": "#b8d4e6",
        "side": "both",  # both | left | right | spread
        "seed": 7,
    },
    "grass": {
        "height": 0.07,
        "density": 280,
        "color": "#4b8636",
        "tip": "#9ccc6c",
        "sway": 1.0,
        "flowers": 0,
        "flower_colors": "#ffffff,#ffd84a,#ff6f91,#b48cff",
        "seed": 9,
    },
    "ice": {
        "horizon": 0.6,
        "ice": "#e4f1f8",
        "shade": "#a3c6dc",
        "bergs": 4,
        "cracks": 16,
        "glint": 0.7,
        "seed": 11,
    },
    "houses": {
        "count": 5,
        "base": 0.68,
        "size": 0.07,
        "style": "cottage",  # cottage | hut
        "wall": "#efe2cc",
        "roof": "#a8452f",
        "lit": 0.0,  # 0 = day, 1 = windows glow (night)
        "spread": 0.8,
        "seed": 13,
    },
}

ACTOR_DEFAULTS: dict[str, Any] = {
    "species": "fox",
    "action": "walk",
    "x": 0.2,
    "y": 0.8,
    "size": 0.2,
    "speed": 0.06,
    "facing": "auto",  # auto | left | right
    "surface": 0.0,  # 0 = none; else water line (fraction of height)
    "seed": 1,
}

GENERATORS.update(SCENERY_DEFAULTS)
GENERATORS["actor"] = ACTOR_DEFAULTS

TEXT_DEFAULTS: dict[str, Any] = {
    "text": "",
    "font": "",
    "size": 72.0,
    "color": "#ffffff",
    "outline": 0.0,
    "outline_color": "#000000",
    "shadow": 0.0,
    "shadow_color": "#000000",
    "shadow_dx": 3.0,
    "shadow_dy": 3.0,
    "gradient_to": "",
    "align": "center",
    "line_spacing": 1.2,
    "max_width": 0.0,
    "box": False,
    "box_color": "#000000",
    "box_opacity": 0.6,
    "box_padding": 24.0,
    "anim": "none",
    "anim_unit": "letter",
    "anim_stagger": 0.05,
    "anim_duration": 0.4,
}
TEXT_ANIMS = frozenset({"none", "typewriter", "fade", "slide", "pop"})

EASINGS: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease_in": (0.42, 0.0, 1.0, 1.0),
    "ease_out": (0.0, 0.0, 0.58, 1.0),
    "ease_in_out": (0.42, 0.0, 0.58, 1.0),
}


def keyframable_param(name: str, effect_types: dict[int, str], *, audio: bool = False) -> bool:
    """True if `name` is a transform param or fx<id>.<numeric param> of an existing effect.

    Audio clips: `volume` (dB) and `speed` only; audio effect parameters are static.
    """
    if audio:
        return name in AUDIO_KEYFRAMABLE or name == "speed"
    if name in TRANSFORM_DEFAULTS:
        return True
    if not name.startswith("fx") or "." not in name:
        return False
    head, param = name[2:].split(".", 1)
    if not head.isdigit() or int(head) not in effect_types:
        return False
    default = EFFECTS[effect_types[int(head)]].get(param)
    return isinstance(default, int | float) and not isinstance(default, bool)
