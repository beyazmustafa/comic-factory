import hashlib
import json
import math
import re
from urllib.parse import urlparse, parse_qs
from .config import ROOT
from .core import file_hash, save_json
from .api import FactoryError


def video_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        value = (
            parsed.path.split("/")[-1]
            if parsed.path.startswith("/shorts/")
            else parse_qs(parsed.query).get("v", [""])[0]
        )
    elif host == "youtu.be":
        value = parsed.path.lstrip("/")
    else:
        value = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError("Geçerli, herkese açık bir YouTube referansı gerekli.")
    return value


def validate_style(style):
    if style.get("video_observed") is not True:
        raise FactoryError(
            "Referans görüntülenemedi; stil tahmin edilmedi. Erişimi kontrol et veya reference.mp4 dosyasını projeye ekle."
        )
    for key, low, high in (
        ("duration_seconds", 10, 600),
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
            raise FactoryError(f"Referans ölçümü geçersiz: {key}")
    observations = style.get("observations")
    if (
        not isinstance(observations, list)
        or len(observations) < 3
        or any(
            not isinstance(row, dict) or not row.get("visual_detail")
            for row in observations
        )
    ):
        raise FactoryError("Referansın zaman damgalı gözlemleri eksik.")
    seconds = [row.get("second") for row in observations]
    if (
        any(
            type(s) not in (int, float)
            or not math.isfinite(s)
            or not 0 <= s <= style["duration_seconds"]
            for s in seconds
        )
        or len(set(seconds)) < 3
        or max(seconds) - min(seconds) < style["duration_seconds"] * 0.5
    ):
        raise FactoryError("Gözlemler referansın yeterli bölümünü kapsamıyor.")
    for key in ("text_color", "active_color", "stroke_color", "background_color"):
        if not re.fullmatch("#[0-9A-Fa-f]{6}", str(style.get(key, ""))):
            raise FactoryError(f"Referans rengi geçersiz: {key}")
    for key, choices in {
        "font_style": {"condensed_heavy", "sans_heavy"},
        "caption_mode": {"single", "current_next", "phrase"},
        "background": {"blurred_page", "solid", "page_fill"},
        "transition": {"cut", "dissolve", "slide"},
        "panel_framing": {"contain", "fill", "page"},
    }.items():
        if style.get(key) not in choices:
            raise FactoryError(f"Desteklenmeyen referans özelliği: {key}")
    for key, low, high in (("caption_words", 1, 5), ("transition_frames", 1, 12)):
        if type(style.get(key)) is not int or not low <= style[key] <= high:
            raise FactoryError(f"Geçersiz {key}")
    for key in ("uppercase", "music_present"):
        if type(style.get(key)) is not bool:
            raise FactoryError(f"Geçersiz {key}")
    return style


def analyze(api, cache):
    identifier = video_id(api.settings.reference_url)
    local = ROOT / "reference.mp4"
    key = hashlib.sha256(
        (
            (file_hash(local) if local.exists() else identifier)
            + api.settings.gemini_model
            + "reference-v3"
        ).encode()
    ).hexdigest()
    saved = cache / "reference" / (key + ".json")
    if saved.exists():
        profile = validate_style(json.loads(saved.read_text()))
    else:
        prompt = """Watch/listen to this ENTIRE reference. Extract editing grammar for ORIGINAL Turkish comic-history videos, not a transcript or a copy of the logo/music. Never infer inaccessible footage; set video_observed=false if unavailable.
Return JSON: video_observed:boolean,title:string,duration_seconds:10..600,
observations:[{second:number,visual_detail:string,caption_detail:string}] at least SIX observations across start/middle/end,
font_style:condensed_heavy|sans_heavy,uppercase:boolean,font_size:38..110 at 1080x1920,stroke_width:1..9,
caption_x:0.25..0.70,caption_y:0.25..0.86 as normalized centers,
caption_mode:single|current_next|phrase,caption_words:1..5,
text_color,active_color,stroke_color,background_color all #RRGGBB,
background:blurred_page|solid|page_fill,transition:cut|dissolve|slide,transition_frames:1..12 at 30fps (1 for cut),
mean_shot_seconds:0.7..12,zoom_amount:0..0.18,
panel_framing:contain|fill|page (complete panel / cropped panel / whole page),
panel_max_width:0.6..1,panel_max_height:0.4..1,panel_center_y:0.3..0.65,
music_present:boolean,narrator_delivery:detailed energy/pacing/pauses instructions usable in Turkish,
story_structure:opening/escalation/reveal/ending structure,framing_notes:string,
unmatched_features:[features requiring more than supported crop/pan/zoom/cut/dissolve/slide].
All values must describe actually observed video, not a generic Shorts template."""
        if local.exists():
            profile = api.video_json("Referans analizi", prompt, local)
        else:
            profile = api.json(
                "Referans analizi",
                prompt,
                video_uri=f"https://www.youtube.com/watch?v={identifier}",
            )
        profile.update(
            reference_url=f"https://www.youtube.com/watch?v={identifier}",
            analysis_model=api.settings.gemini_model,
        )
        save_json(api.directory / "reference_profile.json", profile)
        validate_style(profile)
        save_json(saved, profile)
    save_json(api.directory / "reference_profile.json", profile)
    return profile
