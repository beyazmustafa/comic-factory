"""Offline editing rules fixed during development; no reference media access."""
import math
import re
from copy import deepcopy
from .api import FactoryError

STYLE = {'profile_id': 'comic-superhero-en-v2',
 'font_style': 'condensed_heavy',
 'uppercase': True,
 'font_size': 90,
 'stroke_width': 7,
 'shadow_depth': 5,
 'word_pop_ms': 70,
 'caption_x': 0.5,
 'caption_y': 0.62,
 'caption_mode': 'single',
 'caption_words': 1,
 'text_color': '#FFFF00',
 'active_color': '#FFFF00',
 'stroke_color': '#000000',
 'background_color': '#000000',
 'background': 'page_fill',
 'transition': 'whip',
 'transition_frames': 5,
 'mean_shot_seconds': 3.8,
 'zoom_amount': 0.12,
 'panel_framing': 'fill',
 'panel_max_width': 1.0,
 'panel_max_height': 1.0,
 'panel_center_y': 0.5,
 'music_present': False,
 'narrator_delivery': 'Energetic, curiosity-driven storytelling like a top comics-recap Shorts '
                      'narrator: open on the shocking moment, stress key words, build tension '
                      'with short dramatic pauses, land every sentence ending. Natural, not shouted.',
 'story_structure': 'Shocking hook (first 2 seconds) → who/why → escalating danger → verified '
                    'twist → concrete payoff; end on the strongest image, no outro.',
 'emphasis_colors': {'danger': '#FF2020', 'reveal': '#00FF40', 'turn': '#00FFFF'}}


def validate_style(style):
    for key, low, high in (
        ("caption_x", 0.25, 0.70),
        ("caption_y", 0.25, 0.86),
        ("panel_center_y", 0.3, 0.65),
        ("panel_max_width", 0.6, 1),
        ("panel_max_height", 0.4, 1),
        ("font_size", 38, 110),
        ("stroke_width", 1, 9),
        ("mean_shot_seconds", 0.7, 12),
        ("zoom_amount", 0, 0.18),
    ):
        value = style.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise FactoryError(f"Kurgu ayarı geçersiz: {key}")
    for key in ("text_color", "active_color", "stroke_color", "background_color"):
        if not re.fullmatch("#[0-9A-Fa-f]{6}", str(style.get(key, ""))):
            raise FactoryError(f"Kurgu rengi geçersiz: {key}")
    for key, choices in {
        "font_style": {"condensed_heavy", "sans_heavy"},
        "caption_mode": {"single", "current_next", "phrase"},
        "background": {"blurred_page", "solid", "page_fill"},
        "transition": {"cut", "dissolve", "slide", "whip"},
        "panel_framing": {"contain", "fill", "page"},
    }.items():
        if style.get(key) not in choices:
            raise FactoryError(f"Desteklenmeyen kurgu özelliği: {key}")
    for key, low, high in (("caption_words", 1, 5), ("transition_frames", 1, 12)):
        if type(style.get(key)) is not int or not low <= style[key] <= high:
            raise FactoryError(f"Geçersiz {key}")
    for key in ("uppercase", "music_present"):
        if type(style.get(key)) is not bool:
            raise FactoryError(f"Geçersiz {key}")
    return style



def load_style():
    return validate_style(deepcopy(STYLE))
