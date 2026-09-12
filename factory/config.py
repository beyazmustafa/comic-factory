from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    target_seconds: int = 55
    scene_count: int = 10
    max_images: int = 36
    max_events: int = 3
    max_research_rounds: int = 3
    max_api_calls: int = 150
    max_minutes: int = 90
    repair_attempts: int = 3
    ai_reconstruction: bool = False
    subtitle_offset: float = 0.0
    gemini_model: str = "gemini-3.5-flash-lite"
    tts_model: str = "gemini-3.1-flash-tts-preview"
    tts_voice: str = "Gacrux"
    image_model: str = "gemini-3.1-flash-image"
    whisper_model: str = "whisper-large-v3-turbo"
    story_threshold: float = 90.0
    visual_threshold: float = 90.0
    image_threshold: float = 82.0
    composition_threshold: float = 86.0
    final_visual_threshold: float = 90.0
    alignment_threshold: float = 95.0
    brief: str = "Enerjik, doğal Türkçe anlatım. Güçlü açılış ve somut bir final."

    @classmethod
    def load(cls, path: Path | None = None, **overrides) -> "Settings":
        path = path or ROOT / "factory.json"
        values = (
            json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        )
        if not isinstance(values, dict):
            raise ValueError("factory.json bir JSON nesnesi olmalı.")
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("Bilinmeyen ayarlar: " + ", ".join(sorted(unknown)))
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        settings = cls(**values)
        for key, low, high in (
            ("target_seconds", 20, 120),
            ("scene_count", 4, 20),
            ("max_images", 10, 100),
            ("max_events", 1, 6),
            ("max_research_rounds", 1, 6),
            ("max_api_calls", 10, 400),
            ("max_minutes", 5, 180),
            ("repair_attempts", 1, 5),
        ):
            value = getattr(settings, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key}: {low}–{high} arasında tam sayı gerekli.")
        if type(settings.ai_reconstruction) is not bool:
            raise ValueError("ai_reconstruction true veya false olmalı.")
        if (
            not isinstance(settings.subtitle_offset, (int, float))
            or not -2 <= settings.subtitle_offset <= 2
        ):
            raise ValueError("subtitle_offset -2 ile 2 arasında olmalı.")
        for key in allowed:
            if key.endswith("threshold"):
                value = getattr(settings, key)
                if not isinstance(value, (float, int)) or not 50 <= value <= 100:
                    raise ValueError(f"{key}: 50–100 arasında puan gerekli.")
        for key in (
            "gemini_model",
            "tts_model",
            "tts_voice",
            "image_model",
            "whisper_model",
            "brief",
        ):
            if (
                not isinstance(getattr(settings, key), str)
                or not getattr(settings, key).strip()
            ):
                raise ValueError(f"{key} boş olamaz.")
        return settings

    def to_dict(self) -> dict:
        return asdict(self)
