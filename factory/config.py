from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-10-01-archive-1"


@dataclass(frozen=True)
class Settings:
    target_seconds: int = 150
    max_events: int = 3
    max_pages: int = 28
    max_shots: int = 40
    max_api_calls: int = 120
    max_minutes: int = 100
    repair_attempts: int = 3
    chunk_words: int = 42
    alignment_threshold: float = 86
    panel_threshold: float = 90
    gemini_model: str = "gemini-3.8-flash"
    # Tried in order when the primary model answers 429/503; stable models last.
    gemini_fallback_models: str = "gemini-3.8-flash,gemini-3.7-flash,gemini-3.8-flash-lite,gemini-3.1-flash,gemini-2.5-flash"
    # Vision-capable Groq model used when every Gemini model is unavailable.
    groq_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct,meta-llama/llama-4-scout-17b-16e-instruct"
    # Free Microsoft neural voices used when Gemini TTS is unavailable.
    edge_voices: str = "en-US-ChristopherNeural,en-US-GuyNeural"
    tts_model: str = "gemini-3.1-flash-tts-preview"
    whisper_model: str = "whisper-large-v3"
    voice: str = "auto"
    music_file: str = ""
    music_gain_db: float = -25
    caption_offset: float = 0
    # "archive": complete public-domain issues from the Internet Archive (default).
    # "web": the older publisher-preview search path.
    # Narration/caption language and the channel's editorial focus.
    language: str = "en"
    channel_theme: str = "superheroes"
    # Extra Internet Archive filter applied when channel_theme is superheroes.
    archive_theme_query: str = (
        'subject:(superhero OR superheroes OR "super hero" OR "super heroes" OR heroes) OR '
        'title:(hero OR heroes OR terror OR daredevil OR samson OR yank OR flame OR beetle OR '
        '"amazing man" OR wonder OR captain OR marvel OR "black" OR "blue" OR "green" OR mask OR phantom OR atom OR "super")'
    )
    source: str = "archive"
    archive_query: str = "mediatype:texts AND collection:(comics)"
    archive_max_year: int = 1963
    archive_scan_limit: int = 72

    @classmethod
    def load(cls, path=None, **overrides):
        path = path or ROOT / "factory.json"
        values = (
            json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        )
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("factory.json bilinmeyen veya eski ayarlar içeriyor.")
        for env, key in (
            ("GEMINI_MODEL", "gemini_model"),
            ("GEMINI_FALLBACK_MODELS", "gemini_fallback_models"),
            ("GROQ_MODEL", "groq_model"),
            ("GEMINI_TTS_MODEL", "tts_model"),
            ("GEMINI_TTS_VOICE", "voice"),
            ("GROQ_WHISPER_MODEL", "whisper_model"),
        ):
            if os.getenv(env, "").strip():
                values[key] = os.environ[env].strip()
        values.update({k: v for k, v in overrides.items() if v is not None})
        result = cls(**values)
        for key, low, high in (
            ("target_seconds", 20, 165),
            ("max_events", 1, 5),
            ("max_pages", 8, 50),
            ("max_shots", 4, 60),
            ("max_api_calls", 20, 250),
            ("max_minutes", 10, 150),
            ("repair_attempts", 1, 4),
            ("chunk_words", 20, 70),
            ("archive_max_year", 1920, 1963),
            ("archive_scan_limit", 16, 120),
        ):
            value = getattr(result, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key}: {low}–{high} arasında tam sayı gerekli.")
        for key, low, high in (
            ("alignment_threshold", 85, 100),
            ("panel_threshold", 85, 100),
            ("music_gain_db", -40, -10),
            ("caption_offset", -1, 1),
        ):
            value = getattr(result, key)
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f"Geçersiz {key}")
        if result.voice not in {"auto", "Orus", "Gacrux", "Fenrir", "Puck", "Ahmet", "Emel", "Christopher", "Guy"}:
            raise ValueError("Desteklenmeyen ses.")
        for key in (
            "gemini_model",
            "tts_model",
            "whisper_model",
            "music_file",
            "archive_query",
            "archive_theme_query",
            "language",
            "channel_theme",
            "gemini_fallback_models",
            "groq_model",
            "edge_voices",
        ):
            if not isinstance(getattr(result, key), str):
                raise ValueError(f"{key} metin olmalı.")
        if result.language not in {"en", "tr"}:
            raise ValueError("language en veya tr olmalı.")
        if result.source not in {"archive", "web"}:
            raise ValueError("source archive veya web olmalı.")
        if not result.archive_query.strip():
            raise ValueError("archive_query boş olamaz.")
        return result

    def to_dict(self):
        return asdict(self)
