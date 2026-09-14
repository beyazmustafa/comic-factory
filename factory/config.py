from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-09-14-studio-3"


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
    alignment_threshold: float = 95
    panel_threshold: float = 90
    gemini_model: str = "gemini-3.8-flash"
    tts_model: str = "gemini-3.1-flash-tts-preview"
    whisper_model: str = "whisper-large-v3"
    voice: str = "auto"
    music_file: str = ""
    music_gain_db: float = -25
    caption_offset: float = 0

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
        if result.voice not in {"auto", "Orus", "Gacrux", "Fenrir", "Puck"}:
            raise ValueError("Desteklenmeyen ses.")
        for key in (
            "gemini_model",
            "tts_model",
            "whisper_model",
            "music_file",
        ):
            if not isinstance(getattr(result, key), str):
                raise ValueError(f"{key} metin olmalı.")
        return result

    def to_dict(self):
        return asdict(self)
