# comic_factory_v4.py
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import imageio_ffmpeg
import requests
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EVENT_DIR = DATA / "events"
RESEARCH_DIR = DATA / "research"
SCRIPT_DIR = DATA / "scripts"
AUDIO_DIR = DATA / "audio"
VIDEO_DIR = DATA / "videos"
CANDIDATE_DIR = DATA / "comic_candidates_v4"
WORK_DIR = DATA / "video_work_v4"
DIRECTOR_DIR = DATA / "director"
ASSET_DIR = ROOT / "assets" / "comic_pages_v4"

ACTIVE_EVENT_FILE = EVENT_DIR / "active_event.json"
USED_EVENTS_FILE = EVENT_DIR / "used_events.json"

LATEST_SCRIPT_FILE = SCRIPT_DIR / "latest.json"
LATEST_AUDIO_FILE = AUDIO_DIR / "latest.wav"
LATEST_WORDS_FILE = AUDIO_DIR / "latest_word_timestamps.json"
ALIGNED_WORDS_FILE = AUDIO_DIR / "aligned_subtitle_words.json"
LATEST_VIDEO_FILE = VIDEO_DIR / "latest.mp4"

CHECKPOINT_FILE = DIRECTOR_DIR / "latest_checkpoints.json"
LESSONS_FILE = DIRECTOR_DIR / "quality_lessons.json"

TZ = ZoneInfo("Europe/Istanbul")

WIDTH = 1080
HEIGHT = 1920
FPS = 30

SCENE_COUNT = 14

MAX_GLOBAL_IMAGES = 40
MAX_VISION_IMAGES = 28
MAX_SCENE_SEARCH_IMAGES = 18

STORY_THRESHOLD = 94.0
AUDIO_ALIGNMENT_THRESHOLD = 95.0
VISUAL_MATCH_THRESHOLD = 95.0
IMAGE_QUALITY_THRESHOLD = 94.0
COMPOSITION_THRESHOLD = 95.0
MOTION_THRESHOLD = 94.0
FINAL_DIRECTOR_THRESHOLD = 94.0

MAX_AI_RECONSTRUCTIONS_PER_VIDEO = int(
    os.getenv(
        "MAX_AI_RECONSTRUCTIONS",
        "6",
    )
)

ENABLE_AI_RECONSTRUCTION = (
    os.getenv(
        "ENABLE_AI_RECONSTRUCTION",
        "1",
    ).strip()
    == "1"
)

# 0 = kalite checkpoint'lerinde limitsiz repair döngüsü.
QUALITY_GATE_MAX_CYCLES = int(
    os.getenv(
        "QUALITY_GATE_MAX_CYCLES",
        "0",
    )
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.1-flash-tts-preview",
)

# Thor V2'de beğenilen ses. Kilitli.
GEMINI_TTS_VOICE = os.getenv(
    "GEMINI_TTS_VOICE",
    "Gacrux",
)

GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL",
    "gemini-3.1-flash-image",
)

GROQ_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo",
)

REQUEST_TIMEOUT = 25

SUBTITLE_MAX_WORDS = 3
SUBTITLE_BASE_FONT_SIZE = 56
SUBTITLE_MIN_FONT_SIZE = 40

SUBTITLE_MARGIN_LEFT = 125
SUBTITLE_MARGIN_RIGHT = 125
SUBTITLE_MARGIN_BOTTOM = 390

SUBTITLE_AVAILABLE_WIDTH = (
    WIDTH
    - SUBTITLE_MARGIN_LEFT
    - SUBTITLE_MARGIN_RIGHT
)

ALLOWED_MOTIONS = (
    "slow_push",
    "slow_pull",
    "pan_left",
    "pan_right",
    "vertical_scan",
    "impact",
    "hold",
)

# Gelecek sahneyi narration başlamadan göstermeyen geçişler.
ALLOWED_TRANSITIONS = (
    "cut",
    "soft_black",
    "dip_black",
    "flash_white",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
)

load_dotenv(
    ROOT / ".env"
)


class ComicFactoryError(RuntimeError):
    """Comic Factory V4 üretimi başarısız olduğunda oluşur."""


@dataclass
class CheckpointResult:
    """Tek kalite checkpoint sonucunu temsil eder."""

    name: str
    passed: bool
    score: float
    threshold: float
    cycle: int
    details: str
    created_at: str


@dataclass
class ImageCandidate:
    """Gerçek comic görsel adayı."""

    candidate_id: str
    local_file: str
    source_page: str
    image_url: str
    title: str
    width: int
    height: int
    quality_score: float
    perceptual_hash: str


@dataclass
class RankedVisual:
    """Bir sahne için Vision tarafından seçilen görsel adayı."""

    candidate_id: str
    relevance_score: int
    crop_box: list[float]
    reason: str


@dataclass
class Scene:
    """Render edilecek final sahnesi."""

    scene_number: int
    narration: str
    visual_description: str
    story_role: str
    emphasis_words: list[str]

    ranked_visuals: list[RankedVisual]

    selected_candidate_id: str
    relevance_score: int
    crop_box: list[float]

    visual_source: str
    visual_file: str

    visual_match_score: float
    image_quality_score: float
    composition_score: float

    motion: str = "slow_push"
    transition: str = "cut"
    transition_duration: float = 0.0

    audio_start: float = 0.0
    audio_end: float = 0.0


class CheckpointManager:
    """Checkpoint geçmişini ve kalite derslerini yönetir."""

    def __init__(self) -> None:
        self.results: list[CheckpointResult] = []

    def record(
        self,
        *,
        name: str,
        score: float,
        threshold: float,
        cycle: int,
        details: str,
        passed: bool | None = None,
    ) -> bool:
        if passed is None:
            passed = score >= threshold

        result = CheckpointResult(
            name=name,
            passed=bool(
                passed
            ),
            score=round(
                float(
                    score
                ),
                2,
            ),
            threshold=round(
                float(
                    threshold
                ),
                2,
            ),
            cycle=cycle,
            details=clean(
                details
            ),
            created_at=datetime.now(
                TZ
            ).isoformat(),
        )

        self.results.append(
            result
        )

        icon = (
            "✓"
            if result.passed
            else "✗"
        )

        print(
            f"{icon} CHECKPOINT "
            f"{result.name}: "
            f"{result.score:.2f}/100 "
            f"(min {result.threshold:.2f}) "
            f"| cycle={cycle}"
        )

        if result.details:
            print(
                f"  {result.details}"
            )

        self.save()

        return result.passed

    def save(self) -> None:
        save_json(
            CHECKPOINT_FILE,
            {
                "updated_at": datetime.now(
                    TZ
                ).isoformat(),
                "results": [
                    asdict(
                        result
                    )
                    for result in self.results
                ],
            },
        )

    def persist_lessons(self) -> None:
        payload = load_json(
            LESSONS_FILE,
            {
                "lessons": [],
            },
        )

        if not isinstance(
            payload,
            dict,
        ):
            payload = {
                "lessons": [],
            }

        lessons = payload.setdefault(
            "lessons",
            [],
        )

        for result in self.results:
            if result.passed:
                continue

            lessons.append(
                {
                    "checkpoint": result.name,
                    "score": result.score,
                    "threshold": result.threshold,
                    "details": result.details,
                    "recorded_at": result.created_at,
                }
            )

        payload[
            "lessons"
        ] = lessons[
            -200:
        ]

        payload[
            "updated_at"
        ] = datetime.now(
            TZ
        ).isoformat()

        save_json(
            LESSONS_FILE,
            payload,
        )


CHECKPOINTS = CheckpointManager()


def checkpoint_cycle_allowed(
    cycle: int,
) -> bool:
    """Kalite repair döngüsünün devam edip etmeyeceğini belirler."""
    return (
        QUALITY_GATE_MAX_CYCLES == 0
        or cycle <= QUALITY_GATE_MAX_CYCLES
    )


def parse_args() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""
    parser = argparse.ArgumentParser(
        description=(
            "Comic Factory V4 - "
            "repair-until-pass quality engine."
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
    )

    parser.add_argument(
        "--reuse-active",
        action="store_true",
    )

    parser.add_argument(
        "--upload",
        action="store_true",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_GLOBAL_IMAGES,
    )

    parser.add_argument(
        "--subtitle-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--disable-ai-reconstruction",
        action="store_true",
    )

    return parser.parse_args()


def ensure_dirs() -> None:
    """Gerekli klasörleri oluşturur."""
    for path in (
        EVENT_DIR,
        RESEARCH_DIR,
        SCRIPT_DIR,
        AUDIO_DIR,
        VIDEO_DIR,
        CANDIDATE_DIR,
        WORK_DIR,
        DIRECTOR_DIR,
        ASSET_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def clean(value: Any) -> str:
    """Metni normalize eder."""
    return re.sub(
        r"\s+",
        " ",
        str(
            value or ""
        ).strip(),
    )


def slug(value: str) -> str:
    """Güvenli dosya adı üretir."""
    value = (
        clean(
            value
        )
        .lower()
        .replace(
            "ı",
            "i",
        )
        .replace(
            "ğ",
            "g",
        )
        .replace(
            "ü",
            "u",
        )
        .replace(
            "ş",
            "s",
        )
        .replace(
            "ö",
            "o",
        )
        .replace(
            "ç",
            "c",
        )
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return (
        value.strip(
            "_"
        )[
            :80
        ]
        or "event"
    )


def load_json(
    path: Path,
    default: Any = None,
) -> Any:
    """JSON dosyasını okur."""
    if not path.exists():
        return default

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(
                file
            )

    except json.JSONDecodeError as error:
        raise ComicFactoryError(
            f"JSON okunamadı: {path}"
        ) from error


def save_json(
    path: Path,
    payload: Any,
) -> None:
    """JSON dosyasını atomik olarak kaydeder."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def require_env(
    name: str,
) -> str:
    """Zorunlu environment variable döndürür."""
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise ComicFactoryError(
            f"{name} bulunamadı."
        )

    return value


def load_quality_lessons() -> str:
    """Önceki başarısız checkpoint derslerini yükler."""
    payload = load_json(
        LESSONS_FILE,
        {
            "lessons": [],
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        return "Henüz kayıtlı kalite dersi yok."

    lessons = payload.get(
        "lessons",
        [],
    )

    if not isinstance(
        lessons,
        list,
    ):
        return "Henüz kayıtlı kalite dersi yok."

    recent = lessons[
        -30:
    ]

    if not recent:
        return "Henüz kayıtlı kalite dersi yok."

    return "\n".join(
        (
            f"- {clean(item.get('checkpoint'))}: "
            f"{clean(item.get('details'))}"
        )
        for item in recent
        if isinstance(
            item,
            dict,
        )
    )


def gemini_client() -> genai.Client:
    """Gemini istemcisini oluşturur."""
    return genai.Client(
        api_key=require_env(
            "GEMINI_API_KEY"
        )
    )


def parse_json_response(
    text: str,
) -> dict[str, Any]:
    """Gemini JSON cevabını ayrıştırır."""
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        payload = json.loads(
            text
        )

    except json.JSONDecodeError as error:
        raise ComicFactoryError(
            "Gemini geçerli JSON döndürmedi."
        ) from error

    if isinstance(
        payload,
        dict,
    ):
        return payload

    if isinstance(
        payload,
        list,
    ):
        if not payload:
            raise ComicFactoryError(
                "Gemini boş JSON listesi döndürdü."
            )

        first = payload[
            0
        ]

        if not isinstance(
            first,
            dict,
        ):
            raise ComicFactoryError(
                "Beklenmeyen Gemini JSON listesi."
            )

        if any(
            key in first
            for key in (
                "event_title",
                "publisher",
                "series",
            )
        ):
            return {
                "events": payload,
            }

        if (
            "scene_number"
            in first
            and "ranked_visuals"
            in first
        ):
            return {
                "scenes": payload,
            }

        return {
            "items": payload,
        }

    raise ComicFactoryError(
        "Beklenmeyen Gemini JSON yapısı."
    )


def is_retryable_gemini_error(
    error: Exception,
) -> bool:
    """Geçici Gemini API hatalarını ayırt eder."""
    message = str(
        error
    ).casefold()

    terms = (
        "503",
        "unavailable",
        "high demand",
        "temporarily unavailable",
        "429",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "service unavailable",
    )

    return any(
        term in message
        for term in terms
    )


def gemini_with_retry(
    operation: Any,
    *,
    operation_name: str,
    max_attempts: int = 5,
) -> Any:
    """Geçici Gemini hatalarında exponential backoff uygular."""
    delays = (
        10,
        20,
        40,
        60,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            if attempt > 1:
                print(
                    f"→ {operation_name}: "
                    f"{attempt}/{max_attempts}. deneme"
                )

            return operation()

        except Exception as error:
            last_error = error

            if not is_retryable_gemini_error(
                error
            ):
                raise

            if attempt >= max_attempts:
                break

            wait_seconds = delays[
                min(
                    attempt - 1,
                    len(
                        delays
                    ) - 1,
                )
            ]

            wait_seconds += random.uniform(
                0.0,
                3.0,
            )

            print(
                f"! {operation_name} geçici hata: "
                f"{error}"
            )

            print(
                f"  {wait_seconds:.1f}s sonra "
                "tekrar denenecek."
            )

            time.sleep(
                wait_seconds
            )

    raise ComicFactoryError(
        f"{operation_name} "
        f"{max_attempts} API denemesinden sonra başarısız: "
        f"{last_error}"
    )


def ask_gemini_json(
    client: genai.Client,
    prompt: str,
    *,
    temperature: float = 0.3,
    operation_name: str = "Gemini JSON",
) -> dict[str, Any]:
    """Gemini'den JSON cevap alır."""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name=operation_name,
    )

    if not response.text:
        raise ComicFactoryError(
            "Gemini boş cevap döndürdü."
        )

    return parse_json_response(
        response.text
    )


def search_web(
    queries: list[str],
    each: int = 8,
) -> list[dict[str, str]]:
    """DDGS ile web araması yapar."""
    found: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for query in queries:
        try:
            results = DDGS(
                timeout=12
            ).text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=each,
            )

        except Exception as error:
            print(
                f"! Arama atlandı: "
                f"{query}: {error}"
            )
            continue

        for item in results or []:
            url = clean(
                item.get(
                    "href"
                )
                or item.get(
                    "url"
                )
            )

            if (
                not url.startswith(
                    "http"
                )
                or url in seen
            ):
                continue

            seen.add(
                url
            )

            found.append(
                {
                    "title": clean(
                        item.get(
                            "title"
                        )
                    ),
                    "url": url,
                    "body": clean(
                        item.get(
                            "body"
                        )
                    ),
                }
            )

    return found


def event_key(
    event: dict[str, Any],
) -> str:
    """Comic event için benzersiz anahtar üretir."""
    return "|".join(
        clean(
            event.get(
                key
            )
        ).casefold()
        for key in (
            "publisher",
            "series",
            "issue",
            "event_title",
        )
    )


def used_event_keys() -> set[str]:
    """Daha önce kullanılan event'leri döndürür."""
    payload = load_json(
        USED_EVENTS_FILE,
        {
            "events": [],
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        return set()

    return {
        clean(
            item.get(
                "event_key"
            )
        )
        for item in payload.get(
            "events",
            [],
        )
        if isinstance(
            item,
            dict,
        )
        and clean(
            item.get(
                "event_key"
            )
        )
    }


def research_events(
    client: genai.Client,
    count: int,
) -> list[dict[str, Any]]:
    """Görsel potansiyeli yüksek comic event'leri araştırır."""
    count = max(
        3,
        min(
            count,
            10,
        ),
    )

    print()
    print("=" * 78)
    print("1/13 - EVENT RESEARCH")
    print("=" * 78)

    results = search_web(
        [
            "Marvel comics shocking moments specific issue review",
            "DC comics shocking moments specific issue review",
            "comic book craziest feats specific issue panels",
            "comic book transformations specific issue review",
            "Image Comics shocking moments issue review",
            "Dark Horse comics shocking moments issue review",
            "site:marvel.com comics preview issue",
            "site:dc.com comics preview issue",
        ],
        each=8,
    )

    if not results:
        raise ComicFactoryError(
            "Comic araştırması sonuç vermedi."
        )

    evidence = "\n\n".join(
        (
            f"TITLE: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"SNIPPET: {item['body']}"
        )
        for item in results[
            :48
        ]
    )

    used = (
        "\n".join(
            f"- {item}"
            for item in sorted(
                used_event_keys()
            )
        )
        or "Yok."
    )

    lessons = load_quality_lessons()

    prompt = f"""
Premium comic Shorts research editor'sın.

WEB:
{evidence}

KULLANILMIŞ EVENTLER:
{used}

ÖNCEKİ KALİTE DERSLERİ:
{lessons}

TAM {count} adet spesifik comic event seç.

Kalite mantığı:
- hook /25
- şaşırtıcılık /20
- hikâye/payoff /20
- görsel çeşitlilik potansiyeli /15
- kaynak güvenilirliği /10
- Shorts uygunluğu /10

Kurallar:
- Tek spesifik olay.
- Series, issue ve yıl doğru.
- URL uydurma.
- Kaynak URL yalnız WEB bölümünden.
- 45-60 saniyelik gerçek story arc taşımalı.
- En az 12-14 farklı görsel an çıkarılabilir olmalı.
- Gerçek comic panelleri web'de bulunma ihtimali yüksek olmalı.
- Özel isimler orijinal.
- quality_score 0-100.

SADECE JSON:
{{
  "events": [
    {{
      "event_title": "...",
      "publisher": "...",
      "series": "...",
      "issue": "...",
      "publication_year": 2000,
      "characters": ["..."],
      "hook": "...",
      "event_summary": "...",
      "power_feat": "...",
      "why_interesting": "...",
      "quality_score": 96,
      "sources": [
        {{
          "name": "...",
          "url": "https://...",
          "supports": "..."
        }}
      ]
    }}
  ]
}}
"""

    payload = ask_gemini_json(
        client,
        prompt,
        temperature=0.35,
        operation_name="Event Research",
    )

    events = payload.get(
        "events",
        [],
    )

    if not isinstance(
        events,
        list,
    ):
        raise ComicFactoryError(
            "Event listesi alınamadı."
        )

    used_keys = used_event_keys()

    eligible = [
        event
        for event in events
        if isinstance(
            event,
            dict,
        )
        and event_key(
            event
        )
        not in used_keys
        and float(
            event.get(
                "quality_score",
                0,
            )
            or 0
        )
        >= 88
    ]

    eligible.sort(
        key=lambda item: float(
            item.get(
                "quality_score",
                0,
            )
            or 0
        ),
        reverse=True,
    )

    if not eligible:
        raise ComicFactoryError(
            "88+ kalite potansiyelli event bulunamadı."
        )

    save_json(
        RESEARCH_DIR
        / (
            "research_v4_"
            + datetime.now(
                TZ
            ).strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        ),
        {
            "events": events,
        },
    )

    return eligible


def activate_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    """Event'i aktif hale getirir."""
    active = dict(
        event
    )

    active[
        "id"
    ] = slug(
        f"{clean(event.get('series'))}_"
        f"{clean(event.get('issue'))}_"
        f"{clean(event.get('event_title'))}"
    )

    active[
        "event_key"
    ] = event_key(
        event
    )

    active[
        "activated_at"
    ] = datetime.now(
        TZ
    ).isoformat()

    save_json(
        ACTIVE_EVENT_FILE,
        active,
    )

    print(
        "✓ Event: "
        + clean(
            active.get(
                "event_title"
            )
        )
    )

    return active


def load_active_event() -> dict[str, Any]:
    """Aktif event'i yükler."""
    event = load_json(
        ACTIVE_EVENT_FILE
    )

    if (
        not isinstance(
            event,
            dict,
        )
        or not clean(
            event.get(
                "id"
            )
        )
    ):
        raise ComicFactoryError(
            "active_event.json bulunamadı."
        )

    return event


def generate_storyboard(
    client: genai.Client,
    event: dict[str, Any],
    feedback: str,
) -> dict[str, Any]:
    """14 sahnelik storyboard üretir."""
    sources = "\n".join(
        (
            f"- {clean(item.get('name'))}: "
            f"{clean(item.get('supports'))} "
            f"({clean(item.get('url'))})"
        )
        for item in event.get(
            "sources",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    )

    prompt = f"""
Premium Türkçe comic Shorts Story Director'sın.

EVENT:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("publisher"))}
{clean(event.get("series"))}
{clean(event.get("issue"))}
{event.get("publication_year")}

KARAKTERLER:
{", ".join(event.get("characters", []))}

HOOK:
{clean(event.get("hook"))}

ÖZET:
{clean(event.get("event_summary"))}

FEAT:
{clean(event.get("power_feat"))}

KAYNAKLAR:
{sources}

ÖNCEKİ REPAIR FEEDBACK:
{feedback or "İlk üretim."}

TAM {SCENE_COUNT} sahne oluştur.

Kurallar:
- Toplam 120-150 Türkçe kelime.
- İlk iki sahne güçlü hook.
- Her sahne yeni bilgi getirir.
- Bir sahnede yalnızca tek ana görsel olay anlat.
- Son dört sahne payoff.
- Bilgi uydurma.
- visual_description narration sırasında ekranda TAM olarak
  görülmesi gereken spesifik anı tarif etsin.
- Narration ve visual_description birebir eşlenebilir olsun.
- Aynı görsel fikir iki kez kullanılmasın.
- Özel isimler orijinal.
- Türkçe fonetik isim yazma.
- story_role:
  hook / context / escalation / twist / payoff / cta
- emphasis_words 0-3 kelime.

SADECE JSON:
{{
  "title": "...",
  "description": "...",
  "hashtags": ["#comics", "#shorts"],
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "...",
      "visual_description": "...",
      "story_role": "hook",
      "emphasis_words": ["..."]
    }}
  ]
}}
"""

    storyboard = ask_gemini_json(
        client,
        prompt,
        temperature=0.52,
        operation_name="Story Director",
    )

    scenes = storyboard.get(
        "scenes",
        [],
    )

    if (
        not isinstance(
            scenes,
            list,
        )
        or len(
            scenes
        )
        != SCENE_COUNT
    ):
        raise ComicFactoryError(
            f"Storyboard tam "
            f"{SCENE_COUNT} sahne üretmedi."
        )

    storyboard[
        "narration"
    ] = " ".join(
        clean(
            scene.get(
                "narration"
            )
        )
        for scene in scenes
    )

    return storyboard


def score_storyboard(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> dict[str, Any]:
    """Story kalite puanı üretir."""
    prompt = f"""
Premium Shorts supervising story editor'sın.

EVENT:
{clean(event.get("event_title"))}

SCENES:
{json.dumps(
    storyboard.get("scenes", []),
    ensure_ascii=False,
    indent=2,
)}

0-100 değerlendir:
- hook
- clarity
- escalation
- payoff
- factual_discipline
- visual_storytelling
- retention
- overall

Kritik:
- Aynı bilgi tekrar etmemeli.
- Her narration tek görsel an taşımalı.
- visual_description narration ile birebir uyuşmalı.
- Final gerçek payoff taşımalı.

SADECE JSON:
{{
  "hook": 0,
  "clarity": 0,
  "escalation": 0,
  "payoff": 0,
  "factual_discipline": 0,
  "visual_storytelling": 0,
  "retention": 0,
  "overall": 0,
  "problems": ["..."],
  "revision_instruction": "..."
}}
"""

    return ask_gemini_json(
        client,
        prompt,
        temperature=0.10,
        operation_name="Story Quality Check",
    )


def build_quality_storyboard(
    client: genai.Client,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Story checkpoint geçene kadar revize eder."""
    print()
    print("=" * 78)
    print("2/13 - STORY REPAIR LOOP")
    print("=" * 78)

    feedback = ""
    cycle = 1

    while checkpoint_cycle_allowed(
        cycle
    ):
        storyboard = generate_storyboard(
            client,
            event,
            feedback,
        )

        review = score_storyboard(
            client,
            event,
            storyboard,
        )

        score = float(
            review.get(
                "overall",
                0,
            )
            or 0
        )

        problems = "; ".join(
            clean(
                value
            )
            for value in review.get(
                "problems",
                [],
            )
        )

        passed = CHECKPOINTS.record(
            name="story_quality",
            score=score,
            threshold=STORY_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        if passed:
            save_json(
                SCRIPT_DIR
                / "storyboard_v4.json",
                storyboard,
            )

            return storyboard

        feedback = clean(
            review.get(
                "revision_instruction"
            )
        )

        if not feedback:
            feedback = (
                "Önceki sürümü baştan değerlendir. "
                "En düşük puanlı alanları geliştir."
            )

        cycle += 1

    raise ComicFactoryError(
        "Story repair loop güvenlik limiti aşıldı."
    )


def create_voice_direction(
    client: genai.Client,
    narration: str,
) -> dict[str, Any]:
    """Kilitli Thor V2 voice delivery talimatını oluşturur."""
    prompt = f"""
Türkçe premium YouTube Shorts Voice Director'sın.

TRANSCRIPT:
{narration}

TRANSCRIPT kelimelerini değiştirme.

Yalnız delivery talimatı üret.

KİLİTLİ SES KARAKTERİ:
- 25-35 yaş doğal erkek YouTube storyteller.
- arkadaşına çok ilginç bir comic olayını anlatıyor gibi.
- haber spikeri değil.
- ağır belgesel değil.
- robot/TikTok sesi değil.
- enerjik ama bağırmayan.
- İngilizce özel isimleri doğal İngilizce telaffuz et.
- sonra akıcı biçimde Türkçeye dön.
- reveal öncesi kısa doğal pause.
- doğal nefes.
- doğal ritim.
- yaklaşık 1.0x tempo.

SADECE JSON:
{{
  "style_instruction": "...",
  "pronunciation_instruction": "...",
  "pace_instruction": "..."
}}
"""

    return ask_gemini_json(
        client,
        prompt,
        temperature=0.20,
        operation_name="Locked Voice Director",
    )


def write_pcm_wave(
    path: Path,
    pcm: bytes,
) -> None:
    """24 kHz mono PCM verisini WAV'a yazar."""
    with wave.open(
        str(
            path
        ),
        "wb",
    ) as file:
        file.setnchannels(
            1
        )

        file.setsampwidth(
            2
        )

        file.setframerate(
            24000
        )

        file.writeframes(
            pcm
        )


def generate_gacrux_voice(
    client: genai.Client,
    narration: str,
    cycle: int,
) -> Path:
    """Thor V2'de kullanılan kilitli Gacrux sesini üretir."""
    direction = create_voice_direction(
        client,
        narration,
    )

    prompt = f"""
Read ONLY the transcript inside <TRANSCRIPT>.

VOICE DIRECTION:
{clean(direction.get("style_instruction"))}

PRONUNCIATION:
{clean(direction.get("pronunciation_instruction"))}

PACE:
{clean(direction.get("pace_instruction"))}

Critical:
The surrounding language is Turkish.
English proper nouns must be pronounced naturally in English,
then smoothly return to Turkish.

Do not add words.
Do not remove words.
Do not paraphrase.
Do not read instructions.
Do not sound like an announcer.
Do not sound synthetic.
Use natural breaths and conversational rhythm.

<TRANSCRIPT>
{narration}
</TRANSCRIPT>
"""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=[
                    "AUDIO"
                ],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=(
                            types.PrebuiltVoiceConfig(
                                voice_name=GEMINI_TTS_VOICE,
                            )
                        )
                    )
                ),
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name="Locked Gacrux TTS",
    )

    try:
        pcm = (
            response
            .candidates[0]
            .content
            .parts[0]
            .inline_data
            .data
        )

    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as error:
        raise ComicFactoryError(
            "Gacrux audio alınamadı."
        ) from error

    if not pcm:
        raise ComicFactoryError(
            "Gacrux boş audio döndürdü."
        )

    output = (
        AUDIO_DIR
        / f"gacrux_cycle_{cycle:03d}.wav"
    )

    write_pcm_wave(
        output,
        pcm,
    )

    return output


def object_value(
    value: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Dict veya SDK nesnesinden alan okur."""
    if isinstance(
        value,
        dict,
    ):
        return value.get(
            key,
            default,
        )

    return getattr(
        value,
        key,
        default,
    )


def transcribe_words(
    audio_path: Path,
    narration: str,
) -> list[dict[str, Any]]:
    """Groq Whisper kelime timestamp'lerini çıkarır."""
    groq = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with audio_path.open(
        "rb"
    ) as audio:
        transcription = (
            groq.audio.transcriptions.create(
                file=audio,
                model=GROQ_MODEL,
                language="tr",
                response_format="verbose_json",
                timestamp_granularities=[
                    "word"
                ],
                prompt=narration[
                    :700
                ],
                temperature=0,
            )
        )

    words: list[
        dict[str, Any]
    ] = []

    for item in object_value(
        transcription,
        "words",
        [],
    ) or []:
        word = clean(
            object_value(
                item,
                "word",
                "",
            )
        )

        start = object_value(
            item,
            "start",
        )

        end = object_value(
            item,
            "end",
        )

        if (
            word
            and start is not None
            and end is not None
        ):
            words.append(
                {
                    "word": word,
                    "start": float(
                        start
                    ),
                    "end": float(
                        end
                    ),
                }
            )

    if len(
        words
    ) < 20:
        raise ComicFactoryError(
            "Groq yeterli timestamp üretmedi."
        )

    return words


def normalize_alignment_word(
    value: str,
) -> str:
    """Kelimeyi alignment karşılaştırması için normalize eder."""
    return re.sub(
        r"[^\wçğıöşü'-]",
        "",
        clean(
            value
        ).casefold(),
        flags=re.UNICODE,
    )


def narration_tokens(
    narration: str,
) -> list[str]:
    """Narration kelimelerini çıkarır."""
    return [
        token
        for token in re.findall(
            r"\S+",
            narration,
            flags=re.UNICODE,
        )
        if clean(
            token
        )
    ]


def word_similarity(
    first: str,
    second: str,
) -> float:
    """İki kelimenin fuzzy benzerliğini ölçer."""
    first = normalize_alignment_word(
        first
    )

    second = normalize_alignment_word(
        second
    )

    if (
        not first
        or not second
    ):
        return 0.0

    if first == second:
        return 1.0

    return SequenceMatcher(
        None,
        first,
        second,
    ).ratio()


def align_narration_to_audio(
    narration: str,
    whisper_words: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
]:
    """Narration kelimelerini gerçek audio timestamp'lerine hizalar."""
    script_words = narration_tokens(
        narration
    )

    if not script_words:
        raise ComicFactoryError(
            "Narration boş."
        )

    aligned: list[
        dict[str, Any]
    ] = []

    whisper_index = 0
    previous_end = 0.0

    good_matches = 0
    similarity_sum = 0.0

    for script_index, script_word in enumerate(
        script_words
    ):
        best_index: int | None = None
        best_score = 0.0

        search_end = min(
            len(
                whisper_words
            ),
            whisper_index + 8,
        )

        for candidate_index in range(
            whisper_index,
            search_end,
        ):
            candidate = whisper_words[
                candidate_index
            ]

            score = word_similarity(
                script_word,
                clean(
                    candidate.get(
                        "word"
                    )
                ),
            )

            score -= (
                candidate_index
                - whisper_index
            ) * 0.02

            if score > best_score:
                best_score = score
                best_index = candidate_index

        if (
            best_index is not None
            and best_score >= 0.46
        ):
            match = whisper_words[
                best_index
            ]

            start = float(
                match[
                    "start"
                ]
            )

            end = float(
                match[
                    "end"
                ]
            )

            whisper_index = min(
                len(
                    whisper_words
                ),
                best_index + 1,
            )

            if best_score >= 0.72:
                good_matches += 1

            similarity_sum += max(
                0.0,
                best_score,
            )

        elif whisper_index < len(
            whisper_words
        ):
            reference = whisper_words[
                whisper_index
            ]

            start = max(
                previous_end,
                float(
                    reference[
                        "start"
                    ]
                ),
            )

            duration = max(
                0.10,
                min(
                    0.42,
                    float(
                        reference[
                            "end"
                        ]
                    )
                    - float(
                        reference[
                            "start"
                        ]
                    ),
                ),
            )

            end = (
                start
                + duration
            )

        else:
            last_end = float(
                whisper_words[
                    -1
                ][
                    "end"
                ]
            )

            remaining = max(
                1,
                len(
                    script_words
                )
                - script_index,
            )

            duration = max(
                0.10,
                (
                    last_end
                    - previous_end
                )
                / remaining,
            )

            start = previous_end

            end = (
                start
                + duration
            )

        start = max(
            previous_end,
            start,
        )

        end = max(
            start + 0.05,
            end,
        )

        aligned.append(
            {
                "word": script_word,
                "start": start,
                "end": end,
            }
        )

        previous_end = end

    coverage = (
        good_matches
        / max(
            1,
            len(
                script_words
            ),
        )
        * 100.0
    )

    mean_similarity = (
        similarity_sum
        / max(
            1,
            len(
                script_words
            ),
        )
        * 100.0
    )

    combined = (
        coverage
        * 0.72
        + mean_similarity
        * 0.28
    )

    return (
        aligned,
        {
            "coverage": coverage,
            "mean_similarity": mean_similarity,
            "combined": combined,
        },
    )


def build_quality_audio(
    client: genai.Client,
    narration: str,
) -> tuple[
    Path,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Audio alignment geçene kadar Gacrux sesini yeniden üretir."""
    print()
    print("=" * 78)
    print("3/13 - GACRUX AUDIO REPAIR LOOP")
    print("=" * 78)

    cycle = 1

    while checkpoint_cycle_allowed(
        cycle
    ):
        audio = generate_gacrux_voice(
            client,
            narration,
            cycle,
        )

        whisper_words = transcribe_words(
            audio,
            narration,
        )

        aligned_words, metrics = (
            align_narration_to_audio(
                narration,
                whisper_words,
            )
        )

        score = float(
            metrics[
                "combined"
            ]
        )

        passed = CHECKPOINTS.record(
            name="audio_alignment",
            score=score,
            threshold=AUDIO_ALIGNMENT_THRESHOLD,
            cycle=cycle,
            details=(
                f"coverage={metrics['coverage']:.2f}% | "
                f"similarity="
                f"{metrics['mean_similarity']:.2f}%"
            ),
        )

        if passed:
            shutil.copy2(
                audio,
                LATEST_AUDIO_FILE,
            )

            save_json(
                LATEST_WORDS_FILE,
                {
                    "provider": "groq",
                    "model": GROQ_MODEL,
                    "words": whisper_words,
                },
            )

            save_json(
                ALIGNED_WORDS_FILE,
                {
                    "metrics": metrics,
                    "words": aligned_words,
                },
            )

            return (
                LATEST_AUDIO_FILE,
                whisper_words,
                aligned_words,
            )

        cycle += 1

    raise ComicFactoryError(
        "Audio repair loop güvenlik limiti aşıldı."
    )


def ffmpeg_path() -> str:
    """FFmpeg yolunu döndürür."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(
    command: list[str],
    message: str,
    cwd: Path | None = None,
) -> None:
    """FFmpeg komutunu çalıştırır."""
    process = subprocess.run(
        command,
        cwd=(
            str(
                cwd
            )
            if cwd
            else None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if process.returncode != 0:
        raise ComicFactoryError(
            message
            + "\n"
            + process.stderr[
                -3500:
            ]
        )


def media_duration(
    ffmpeg: str,
    media: Path,
) -> float:
    """Bir media dosyasının süresini okur."""
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                media
            ),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        process.stderr,
    )

    if not match:
        raise ComicFactoryError(
            f"Media süresi okunamadı: {media}"
        )

    return (
        int(
            match.group(
                1
            )
        )
        * 3600
        + int(
            match.group(
                2
            )
        )
        * 60
        + float(
            match.group(
                3
            )
        )
    )


def build_scene_timeline(
    storyboard: dict[str, Any],
    aligned_words: list[dict[str, Any]],
    audio_duration_seconds: float,
) -> list[
    tuple[
        float,
        float,
    ]
]:
    """Her sahne sınırını gerçek narration audio kelimelerine bağlar."""
    scenes = storyboard[
        "scenes"
    ]

    counts = [
        len(
            narration_tokens(
                clean(
                    scene.get(
                        "narration"
                    )
                )
            )
        )
        for scene in scenes
    ]

    expected = sum(
        counts
    )

    if abs(
        expected
        - len(
            aligned_words
        )
    ) > 2:
        raise ComicFactoryError(
            "Storyboard narration ile aligned audio kelimeleri "
            "uyuşmuyor."
        )

    timeline: list[
        tuple[
            float,
            float,
        ]
    ] = []

    cursor = 0

    for scene_index, count in enumerate(
        counts
    ):
        start_index = min(
            cursor,
            len(
                aligned_words
            ) - 1,
        )

        end_index = min(
            cursor
            + count
            - 1,
            len(
                aligned_words
            ) - 1,
        )

        if scene_index == 0:
            start = 0.0

        else:
            previous_end = float(
                aligned_words[
                    start_index - 1
                ][
                    "end"
                ]
            )

            current_start = float(
                aligned_words[
                    start_index
                ][
                    "start"
                ]
            )

            start = (
                previous_end
                + current_start
            ) / 2.0

        if scene_index == len(
            scenes
        ) - 1:
            end = audio_duration_seconds

        else:
            this_end = float(
                aligned_words[
                    end_index
                ][
                    "end"
                ]
            )

            next_index = min(
                end_index + 1,
                len(
                    aligned_words
                ) - 1,
            )

            next_start = float(
                aligned_words[
                    next_index
                ][
                    "start"
                ]
            )

            end = (
                this_end
                + next_start
            ) / 2.0

        if end <= start:
            end = (
                start
                + 0.20
            )

        timeline.append(
            (
                start,
                end,
            )
        )

        cursor += count

    return timeline


def search_image_queries(
    queries: list[str],
    max_results_each: int,
) -> list[dict[str, str]]:
    """DDGS image sonuçlarını normalize eder."""
    output: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for query in queries:
        try:
            results = DDGS(
                timeout=15
            ).images(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=max_results_each,
            )

        except Exception as error:
            print(
                f"! Image search atlandı: {error}"
            )
            continue

        for item in results or []:
            image_url = clean(
                item.get(
                    "image"
                )
            )

            if (
                not image_url.startswith(
                    "http"
                )
                or image_url in seen
            ):
                continue

            seen.add(
                image_url
            )

            output.append(
                {
                    "image_url": image_url,
                    "source_page": clean(
                        item.get(
                            "url"
                        )
                    ),
                    "title": clean(
                        item.get(
                            "title"
                        )
                    ),
                }
            )

    return output


def global_image_search(
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    """Event için geniş global comic görsel araştırması yapar."""
    print()
    print("=" * 78)
    print("4/13 - GLOBAL VISUAL SEARCH")
    print("=" * 78)

    series = clean(
        event.get(
            "series"
        )
    )

    issue = clean(
        event.get(
            "issue"
        )
    )

    event_title = clean(
        event.get(
            "event_title"
        )
    )

    queries = [
        f'"{series}" "{issue}" comic panels',
        f'"{series}" "{issue}" comic pages',
        f'"{series}" "{issue}" preview',
        f'"{series}" "{issue}" review panels',
        f'"{event_title}" comic panels',
        f'"{event_title}" comic pages',
    ]

    for scene in storyboard[
        "scenes"
    ]:
        visual = clean(
            scene.get(
                "visual_description"
            )
        )

        if visual:
            queries.append(
                f'"{series}" "{issue}" '
                f'{visual[:90]}'
            )

    return search_image_queries(
        queries,
        max_results_each=12,
    )


def average_hash(
    image: Image.Image,
) -> str:
    """64-bit perceptual average hash üretir."""
    tiny = image.convert(
        "L"
    ).resize(
        (
            8,
            8,
        ),
        Image.Resampling.LANCZOS,
    )

    pixels = list(
        tiny.getdata()
    )

    average = sum(
        pixels
    ) / len(
        pixels
    )

    bits = "".join(
        "1"
        if value >= average
        else "0"
        for value in pixels
    )

    return (
        f"{int(bits, 2):016x}"
    )


def hash_distance(
    first: str,
    second: str,
) -> int:
    """Perceptual hash Hamming mesafesini döndürür."""
    return bin(
        int(
            first,
            16,
        )
        ^ int(
            second,
            16,
        )
    ).count(
        "1"
    )


def source_image_quality(
    image: Image.Image,
) -> float:
    """Kaynak comic görseline teknik kalite puanı verir."""
    width, height = image.size

    megapixels = (
        width
        * height
        / 1_000_000
    )

    min_side = min(
        width,
        height,
    )

    entropy = image.convert(
        "L"
    ).entropy()

    resolution_score = min(
        50.0,
        megapixels
        / 1.50
        * 50.0,
    )

    dimension_score = min(
        30.0,
        min_side
        / 900.0
        * 30.0,
    )

    entropy_score = min(
        20.0,
        max(
            0.0,
            (
                entropy
                - 3.0
            )
            / 4.5
            * 20.0,
        ),
    )

    return max(
        0.0,
        min(
            100.0,
            resolution_score
            + dimension_score
            + entropy_score,
        ),
    )


def download_candidates(
    event: dict[str, Any],
    results: list[dict[str, str]],
    *,
    max_candidates: int,
    directory_name: str,
    start_index: int,
    known_hashes: list[str] | None = None,
) -> list[ImageCandidate]:
    """Kaliteli ve benzersiz comic görsellerini indirir."""
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    directory = (
        CANDIDATE_DIR
        / event[
            "id"
        ]
        / directory_name
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = list(
        known_hashes
        or []
    )

    accepted: list[
        ImageCandidate
    ] = []

    for result in results:
        if len(
            accepted
        ) >= max_candidates:
            break

        try:
            response = session.get(
                result[
                    "image_url"
                ],
                headers={
                    "Referer": result[
                        "source_page"
                    ],
                },
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

        except requests.RequestException:
            continue

        if len(
            response.content
        ) < 18_000:
            continue

        try:
            with Image.open(
                io.BytesIO(
                    response.content
                )
            ) as opened:
                image = ImageOps.exif_transpose(
                    opened
                ).convert(
                    "RGB"
                )

        except Exception:
            continue

        width, height = image.size

        if (
            width < 500
            or height < 500
            or width
            * height
            < 450_000
        ):
            continue

        phash = average_hash(
            image
        )

        if any(
            hash_distance(
                phash,
                existing,
            )
            <= 5
            for existing in hashes
        ):
            continue

        hashes.append(
            phash
        )

        candidate_id = (
            f"c{start_index + len(accepted):04d}"
        )

        local_file = (
            directory
            / f"{candidate_id}.jpg"
        )

        image.save(
            local_file,
            "JPEG",
            quality=96,
            optimize=True,
        )

        quality = source_image_quality(
            image
        )

        accepted.append(
            ImageCandidate(
                candidate_id=candidate_id,
                local_file=str(
                    local_file
                ),
                source_page=result[
                    "source_page"
                ],
                image_url=result[
                    "image_url"
                ],
                title=result[
                    "title"
                ],
                width=width,
                height=height,
                quality_score=round(
                    quality,
                    2,
                ),
                perceptual_hash=phash,
            )
        )

    return accepted


def thumbnail_bytes(
    path: Path,
) -> bytes:
    """Gemini Vision için thumbnail oluşturur."""
    with Image.open(
        path
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    image.thumbnail(
        (
            640,
            640,
        ),
        Image.Resampling.LANCZOS,
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        "JPEG",
        quality=82,
    )

    return buffer.getvalue()


def normalize_crop_box(
    crop_box: Any,
) -> list[float]:
    """Crop koordinatlarını 0-1 aralığına çeker."""
    default = [
        0.0,
        0.0,
        1.0,
        1.0,
    ]

    if (
        not isinstance(
            crop_box,
            list,
        )
        or len(
            crop_box
        )
        != 4
    ):
        return default

    try:
        left, top, right, bottom = [
            float(
                value
            )
            for value in crop_box
        ]

    except (
        TypeError,
        ValueError,
    ):
        return default

    left = max(
        0.0,
        min(
            0.90,
            left,
        ),
    )

    top = max(
        0.0,
        min(
            0.90,
            top,
        ),
    )

    right = max(
        left + 0.10,
        min(
            1.0,
            right,
        ),
    )

    bottom = max(
        top + 0.10,
        min(
            1.0,
            bottom,
        ),
    )

    return [
        left,
        top,
        right,
        bottom,
    ]


def rank_scene_candidates(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    candidates: list[ImageCandidate],
) -> list[RankedVisual]:
    """Tek sahne için görsel adaylarını sıralar."""
    candidates = sorted(
        candidates,
        key=lambda item: item.quality_score,
        reverse=True,
    )[
        :MAX_VISION_IMAGES
    ]

    if not candidates:
        return []

    prompt = f"""
Premium comic Visual Verification sistemisin.

EVENT:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("series"))}
{clean(event.get("issue"))}

NARRATION:
{clean(scene_data.get("narration"))}

EKRANDA TAM OLARAK GÖRÜLMESİ GEREKEN:
{clean(scene_data.get("visual_description"))}

Aday comic görsellerini incele.

En iyi 6 adayı sırala.

relevance_score:
100 = anlatılan spesifik an kesin biçimde görünür.
95 = aynı spesifik aksiyon/an açık biçimde görünür.
85 = aynı karakterler/ortam fakat tam anlatılan an değil.
70 = yalnız konu veya karakter benzer.
50 = zayıf.
0 = alakasız.

95+ puanı kolay verme.

Crop:
- doğru comic panelini seç.
- normalize [left, top, right, bottom].
- ana aksiyonu koru.
- karakteri gereksiz kesme.
- mümkünse gereksiz balon/metni dışarıda bırak.

SADECE JSON:
{{
  "ranked_visuals": [
    {{
      "candidate_id": "c0001",
      "relevance_score": 97,
      "crop_box": [0,0,1,1],
      "reason": "..."
    }}
  ]
}}
"""

    contents: list[Any] = [
        prompt
    ]

    for candidate in candidates:
        contents.append(
            (
                f"CANDIDATE {candidate.candidate_id}\n"
                f"TITLE: {candidate.title}\n"
                f"SIZE: {candidate.width}x{candidate.height}\n"
                f"TECH QUALITY: "
                f"{candidate.quality_score:.1f}"
            )
        )

        contents.append(
            types.Part.from_bytes(
                data=thumbnail_bytes(
                    Path(
                        candidate.local_file
                    )
                ),
                mime_type="image/jpeg",
            )
        )

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.05,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name=(
            "Scene Visual Ranking "
            f"{scene_data['scene_number']}"
        ),
    )

    payload = parse_json_response(
        response.text
        or "{}"
    )

    raw_items = payload.get(
        "ranked_visuals",
        payload.get(
            "items",
            [],
        ),
    )

    ranked: list[
        RankedVisual
    ] = []

    for raw in raw_items:
        if not isinstance(
            raw,
            dict,
        ):
            continue

        candidate_id = clean(
            raw.get(
                "candidate_id"
            )
        )

        if not candidate_id:
            continue

        ranked.append(
            RankedVisual(
                candidate_id=candidate_id,
                relevance_score=int(
                    raw.get(
                        "relevance_score",
                        0,
                    )
                    or 0
                ),
                crop_box=normalize_crop_box(
                    raw.get(
                        "crop_box"
                    )
                ),
                reason=clean(
                    raw.get(
                        "reason"
                    )
                ),
            )
        )

    return ranked


def restore_crop(
    candidate: ImageCandidate,
    crop_box: list[float],
    output: Path,
) -> tuple[
    Path,
    float,
]:
    """Paneli içeriği değiştirmeden crop ve restore eder."""
    with Image.open(
        candidate.local_file
    ) as opened:
        source = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    left, top, right, bottom = crop_box

    x1 = round(
        left
        * source.width
    )

    y1 = round(
        top
        * source.height
    )

    x2 = round(
        right
        * source.width
    )

    y2 = round(
        bottom
        * source.height
    )

    crop_width = max(
        1,
        x2 - x1,
    )

    crop_height = max(
        1,
        y2 - y1,
    )

    crop = source.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    megapixels = (
        crop_width
        * crop_height
        / 1_000_000
    )

    min_side = min(
        crop_width,
        crop_height,
    )

    entropy = crop.convert(
        "L"
    ).entropy()

    technical_quality = (
        min(
            50.0,
            megapixels
            / 1.25
            * 50.0,
        )
        + min(
            30.0,
            min_side
            / 850.0
            * 30.0,
        )
        + min(
            20.0,
            max(
                0.0,
                (
                    entropy
                    - 3.0
                )
                / 4.5
                * 20.0,
            ),
        )
    )

    technical_quality = max(
        0.0,
        min(
            100.0,
            technical_quality,
        ),
    )

    max_side = max(
        crop.size
    )

    if max_side < 2600:
        scale = min(
            2.5,
            2600
            / max(
                1,
                max_side,
            ),
        )

        crop = crop.resize(
            (
                round(
                    crop.width
                    * scale
                ),
                round(
                    crop.height
                    * scale
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    crop = ImageEnhance.Contrast(
        crop
    ).enhance(
        1.045
    )

    crop = ImageEnhance.Color(
        crop
    ).enhance(
        1.015
    )

    crop = crop.filter(
        ImageFilter.UnsharpMask(
            radius=1.20,
            percent=105,
            threshold=3,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        output,
        "PNG",
        optimize=True,
    )

    return (
        output,
        technical_quality,
    )


def evaluate_visual(
    client: genai.Client,
    scene_data: dict[str, Any],
    visual_file: Path,
    technical_quality: float,
) -> dict[str, Any]:
    """Final panel/crop'u narration ile doğrular."""
    prompt = f"""
Premium comic-video scene quality checker'sın.

NARRATION:
{clean(scene_data.get("narration"))}

EXPECTED VISUAL:
{clean(scene_data.get("visual_description"))}

Final görseli değerlendir.

0-100:
visual_match:
- narration'daki spesifik olay gerçekten görünüyor mu?

image_quality:
- netlik
- çözünürlük hissi
- compression artefact
- aşırı büyütme
- bozulmuş detay

composition:
- ana karakter/aksiyon açık mı?
- önemli öğe crop dışında mı?
- Shorts kadrajında okunabilir mi?
- görüntü gereksiz text/bubble karmaşası içeriyor mu?

SOURCE TECHNICAL QUALITY:
{technical_quality:.2f}

visual_match 95+ yalnız gerçekten anlatılan spesifik an
görülüyorsa ver.

SADECE JSON:
{{
  "visual_match": 0,
  "image_quality": 0,
  "composition": 0,
  "problems": ["..."],
  "repair_instruction": "..."
}}
"""

    mime_type = (
        "image/png"
        if visual_file.suffix.lower()
        == ".png"
        else "image/jpeg"
    )

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=thumbnail_bytes(
                        visual_file
                    ),
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.05,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name=(
            "Visual Verification "
            f"Scene {scene_data['scene_number']}"
        ),
    )

    return parse_json_response(
        response.text
        or "{}"
    )


def scene_specific_search(
    event: dict[str, Any],
    scene_data: dict[str, Any],
    feedback: str,
    cycle: int,
) -> list[dict[str, str]]:
    """Başarısız scene checkpoint için yeni görsel arar."""
    series = clean(
        event.get(
            "series"
        )
    )

    issue = clean(
        event.get(
            "issue"
        )
    )

    visual = clean(
        scene_data.get(
            "visual_description"
        )
    )

    narration = clean(
        scene_data.get(
            "narration"
        )
    )

    character_text = " ".join(
        clean(
            character
        )
        for character in event.get(
            "characters",
            [],
        )[
            :4
        ]
    )

    feedback_terms = clean(
        feedback
    )[
        :100
    ]

    queries = [
        f'"{series}" "{issue}" {visual[:100]}',
        f'"{series}" "{issue}" {narration[:90]} comic panel',
        f'"{series}" "{issue}" {character_text} comic page',
        f'"{series}" "{issue}" preview scan panel',
    ]

    if feedback_terms:
        queries.append(
            f'"{series}" "{issue}" '
            f'{feedback_terms}'
        )

    if cycle >= 3:
        queries.extend(
            [
                f'"{series}" "{issue}" full page review',
                f'"{series}" "{issue}" comic preview page',
            ]
        )

    return search_image_queries(
        queries,
        max_results_each=10,
    )


def generate_reconstruction(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    references: list[Path],
    output: Path,
    feedback: str,
    cycle: int,
) -> Path:
    """Eksik scene'i referans comic görsellerinden üretir."""
    prompt = f"""
Create a premium vertical American comic-book illustration.

EVENT:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("series"))}
{clean(event.get("issue"))}

EXACT NARRATED MOMENT:
{clean(scene_data.get("visual_description"))}

NARRATION:
{clean(scene_data.get("narration"))}

PREVIOUS QUALITY FEEDBACK:
{feedback or "First reconstruction."}

REPAIR CYCLE:
{cycle}

Use references only for:
- character appearance
- costumes
- comic era
- colors
- setting
- atmosphere

Create the exact narrated event.

Requirements:
- premium professional comic illustration
- cinematic
- highly detailed
- correct anatomy
- clear focal point
- strong depth
- clean 9:16 composition
- no text
- no speech bubbles
- no logo
- no watermark
- no issue number
- no UI

Do not directly copy reference composition.
Correct every problem mentioned in PREVIOUS QUALITY FEEDBACK.
"""

    interaction_input: list[
        dict[str, str]
    ] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for reference in references[
        :3
    ]:
        mime_type = (
            "image/png"
            if reference.suffix.lower()
            == ".png"
            else "image/jpeg"
        )

        interaction_input.append(
            {
                "type": "image",
                "data": base64.b64encode(
                    reference.read_bytes()
                ).decode(
                    "ascii"
                ),
                "mime_type": mime_type,
            }
        )

    def request() -> Any:
        return client.interactions.create(
            model=GEMINI_IMAGE_MODEL,
            input=interaction_input,
            response_format={
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "9:16",
                "image_size": "2K",
            },
        )

    interaction = gemini_with_retry(
        request,
        operation_name=(
            "AI Reconstruction "
            f"Scene {scene_data['scene_number']}"
        ),
    )

    if interaction.output_image is None:
        raise ComicFactoryError(
            "AI reconstruction görsel döndürmedi."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_bytes(
        base64.b64decode(
            interaction.output_image.data
        )
    )

    return output


def build_scene_object(
    scene_data: dict[str, Any],
    rankings: list[RankedVisual],
    selected_candidate_id: str,
    relevance_score: int,
    crop_box: list[float],
    visual_source: str,
    visual_file: Path,
    evaluation: dict[str, Any],
    image_quality: float,
) -> Scene:
    """Final Scene nesnesini oluşturur."""
    return Scene(
        scene_number=int(
            scene_data[
                "scene_number"
            ]
        ),
        narration=clean(
            scene_data.get(
                "narration"
            )
        ),
        visual_description=clean(
            scene_data.get(
                "visual_description"
            )
        ),
        story_role=clean(
            scene_data.get(
                "story_role"
            )
        ),
        emphasis_words=[
            clean(
                word
            )
            for word in scene_data.get(
                "emphasis_words",
                [],
            )
            if clean(
                word
            )
        ],
        ranked_visuals=rankings,
        selected_candidate_id=selected_candidate_id,
        relevance_score=relevance_score,
        crop_box=crop_box,
        visual_source=visual_source,
        visual_file=str(
            visual_file
        ),
        visual_match_score=float(
            evaluation.get(
                "visual_match",
                0,
            )
            or 0
        ),
        image_quality_score=float(
            image_quality
        ),
        composition_score=float(
            evaluation.get(
                "composition",
                0,
            )
            or 0
        ),
    )


def resolve_scene_until_pass(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    global_candidates: list[ImageCandidate],
    used_candidate_ids: set[str],
    asset_directory: Path,
    reconstruction_counter: dict[str, int],
    enable_reconstruction: bool,
    force_new_visual: bool = False,
) -> Scene:
    """Scene görselini tüm checkpoint'ler geçene kadar onarır."""
    scene_number = int(
        scene_data[
            "scene_number"
        ]
    )

    all_candidates = list(
        global_candidates
    )

    known_hashes = [
        candidate.perceptual_hash
        for candidate in all_candidates
    ]

    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in all_candidates
    }

    attempted_candidates: set[
        str
    ] = set()

    feedback = ""
    cycle = 1
    ai_cycle = 0

    rankings = rank_scene_candidates(
        client,
        event,
        scene_data,
        all_candidates,
    )

    while checkpoint_cycle_allowed(
        cycle
    ):
        print(
            f"\nSCENE {scene_number} "
            f"VISUAL REPAIR CYCLE {cycle}"
        )

        candidate_choice: RankedVisual | None = None

        for option in rankings:
            if option.candidate_id in attempted_candidates:
                continue

            if (
                force_new_visual
                and option.candidate_id
                in used_candidate_ids
            ):
                continue

            if (
                option.candidate_id
                in used_candidate_ids
                and option.relevance_score < 98
            ):
                continue

            if option.candidate_id not in candidate_map:
                continue

            candidate_choice = option
            break

        if candidate_choice is not None:
            attempted_candidates.add(
                candidate_choice.candidate_id
            )

            candidate = candidate_map[
                candidate_choice.candidate_id
            ]

            output = (
                asset_directory
                / (
                    f"scene_{scene_number:02d}_"
                    f"{candidate_choice.candidate_id}_"
                    f"cycle_{cycle:03d}.png"
                )
            )

            visual_file, technical_quality = (
                restore_crop(
                    candidate,
                    candidate_choice.crop_box,
                    output,
                )
            )

            evaluation = evaluate_visual(
                client,
                scene_data,
                visual_file,
                technical_quality,
            )

            visual_match = float(
                evaluation.get(
                    "visual_match",
                    0,
                )
                or 0
            )

            ai_quality = float(
                evaluation.get(
                    "image_quality",
                    0,
                )
                or 0
            )

            image_quality = (
                technical_quality
                * 0.50
                + ai_quality
                * 0.50
            )

            composition = float(
                evaluation.get(
                    "composition",
                    0,
                )
                or 0
            )

            problems = "; ".join(
                clean(
                    value
                )
                for value in evaluation.get(
                    "problems",
                    [],
                )
            )

            match_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "visual_match"
                ),
                score=visual_match,
                threshold=VISUAL_MATCH_THRESHOLD,
                cycle=cycle,
                details=problems,
            )

            quality_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "image_quality"
                ),
                score=image_quality,
                threshold=IMAGE_QUALITY_THRESHOLD,
                cycle=cycle,
                details=(
                    f"source={technical_quality:.1f} | "
                    f"vision={ai_quality:.1f}"
                ),
            )

            composition_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "composition"
                ),
                score=composition,
                threshold=COMPOSITION_THRESHOLD,
                cycle=cycle,
                details=problems,
            )

            if (
                match_passed
                and quality_passed
                and composition_passed
            ):
                used_candidate_ids.add(
                    candidate_choice.candidate_id
                )

                return build_scene_object(
                    scene_data=scene_data,
                    rankings=rankings,
                    selected_candidate_id=(
                        candidate_choice.candidate_id
                    ),
                    relevance_score=(
                        candidate_choice.relevance_score
                    ),
                    crop_box=(
                        candidate_choice.crop_box
                    ),
                    visual_source="real_comic",
                    visual_file=visual_file,
                    evaluation=evaluation,
                    image_quality=image_quality,
                )

            feedback = clean(
                evaluation.get(
                    "repair_instruction"
                )
            )

            if not feedback:
                feedback = problems

        # Gerçek aday kalmadıysa veya başarısızsa tekrar arama yapılır.
        search_results = scene_specific_search(
            event,
            scene_data,
            feedback,
            cycle,
        )

        supplemental = download_candidates(
            event,
            search_results,
            max_candidates=MAX_SCENE_SEARCH_IMAGES,
            directory_name=(
                f"scene_{scene_number:02d}_"
                f"cycle_{cycle:03d}"
            ),
            start_index=(
                1000
                + scene_number
                * 100
                + cycle
                * 20
            ),
            known_hashes=known_hashes,
        )

        for candidate in supplemental:
            known_hashes.append(
                candidate.perceptual_hash
            )

            candidate_map[
                candidate.candidate_id
            ] = candidate

            all_candidates.append(
                candidate
            )

        if supplemental:
            new_rankings = rank_scene_candidates(
                client,
                event,
                scene_data,
                supplemental
                + global_candidates[
                    :10
                ],
            )

            rankings = (
                new_rankings
                + rankings
            )

            cycle += 1
            continue

        # Gerçek panel bulma stratejisi sonuç üretmiyorsa AI fallback.
        if enable_reconstruction:
            references: list[
                Path
            ] = []

            for option in rankings[
                :3
            ]:
                candidate = candidate_map.get(
                    option.candidate_id
                )

                if candidate is not None:
                    references.append(
                        Path(
                            candidate.local_file
                        )
                    )

            if not references:
                references = [
                    Path(
                        candidate.local_file
                    )
                    for candidate in global_candidates[
                        :3
                    ]
                ]

            ai_cycle += 1

            output = (
                asset_directory
                / (
                    f"scene_{scene_number:02d}_"
                    f"ai_cycle_{ai_cycle:03d}.jpg"
                )
            )

            generated = generate_reconstruction(
                client,
                event,
                scene_data,
                references,
                output,
                feedback,
                ai_cycle,
            )

            evaluation = evaluate_visual(
                client,
                scene_data,
                generated,
                100.0,
            )

            visual_match = float(
                evaluation.get(
                    "visual_match",
                    0,
                )
                or 0
            )

            image_quality = float(
                evaluation.get(
                    "image_quality",
                    0,
                )
                or 0
            )

            composition = float(
                evaluation.get(
                    "composition",
                    0,
                )
                or 0
            )

            problems = "; ".join(
                clean(
                    value
                )
                for value in evaluation.get(
                    "problems",
                    [],
                )
            )

            match_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "ai_visual_match"
                ),
                score=visual_match,
                threshold=VISUAL_MATCH_THRESHOLD,
                cycle=ai_cycle,
                details=problems,
            )

            quality_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "ai_image_quality"
                ),
                score=image_quality,
                threshold=IMAGE_QUALITY_THRESHOLD,
                cycle=ai_cycle,
                details=problems,
            )

            composition_passed = CHECKPOINTS.record(
                name=(
                    f"scene_{scene_number:02d}_"
                    "ai_composition"
                ),
                score=composition,
                threshold=COMPOSITION_THRESHOLD,
                cycle=ai_cycle,
                details=problems,
            )

            if (
                match_passed
                and quality_passed
                and composition_passed
            ):
                reconstruction_counter[
                    "count"
                ] += 1

                return build_scene_object(
                    scene_data=scene_data,
                    rankings=rankings,
                    selected_candidate_id="AI",
                    relevance_score=round(
                        visual_match
                    ),
                    crop_box=[
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    visual_source="ai_reconstruction",
                    visual_file=generated,
                    evaluation=evaluation,
                    image_quality=image_quality,
                )

            feedback = clean(
                evaluation.get(
                    "repair_instruction"
                )
            )

            if not feedback:
                feedback = problems

        cycle += 1

    raise ComicFactoryError(
        f"Scene {scene_number} repair loop "
        "güvenlik limiti aşıldı."
    )


def build_quality_visuals(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    max_images: int,
    enable_reconstruction: bool,
) -> tuple[
    list[Scene],
    list[ImageCandidate],
]:
    """Tüm sahneleri checkpoint'lerden geçirir."""
    print()
    print("=" * 78)
    print("5/13 - VISUAL REPAIR ENGINE")
    print("=" * 78)

    results = global_image_search(
        event,
        storyboard,
    )

    event_candidate_directory = (
        CANDIDATE_DIR
        / event[
            "id"
        ]
    )

    shutil.rmtree(
        event_candidate_directory,
        ignore_errors=True,
    )

    global_candidates = download_candidates(
        event,
        results,
        max_candidates=max(
            20,
            max_images,
        ),
        directory_name="global",
        start_index=1,
    )

    if len(
        global_candidates
    ) < 10:
        raise ComicFactoryError(
            "Global comic görsel havuzu yetersiz."
        )

    print(
        f"✓ Global unique images: "
        f"{len(global_candidates)}"
    )

    asset_directory = (
        ASSET_DIR
        / event[
            "id"
        ]
    )

    shutil.rmtree(
        asset_directory,
        ignore_errors=True,
    )

    asset_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    used_candidate_ids: set[
        str
    ] = set()

    reconstruction_counter = {
        "count": 0,
    }

    scenes: list[
        Scene
    ] = []

    for scene_data in storyboard[
        "scenes"
    ]:
        scene = resolve_scene_until_pass(
            client=client,
            event=event,
            scene_data=scene_data,
            global_candidates=global_candidates,
            used_candidate_ids=used_candidate_ids,
            asset_directory=asset_directory,
            reconstruction_counter=reconstruction_counter,
            enable_reconstruction=enable_reconstruction,
        )

        scenes.append(
            scene
        )

    return (
        scenes,
        global_candidates,
    )


def compose_vertical(
    visual_file: Path,
) -> Image.Image:
    """Comic görselini temiz 9:16 kompozisyona yerleştirir."""
    with Image.open(
        visual_file
    ) as opened:
        source = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    ratio = (
        source.width
        / max(
            1,
            source.height,
        )
    )

    background = ImageOps.fit(
        source,
        (
            WIDTH,
            HEIGHT,
        ),
        method=Image.Resampling.LANCZOS,
        centering=(
            0.5,
            0.5,
        ),
    )

    background = background.filter(
        ImageFilter.GaussianBlur(
            radius=50
        )
    )

    background = ImageEnhance.Brightness(
        background
    ).enhance(
        0.28
    )

    background = ImageEnhance.Color(
        background
    ).enhance(
        0.78
    )

    foreground = source.copy()

    if (
        0.48
        <= ratio
        <= 0.70
    ):
        foreground.thumbnail(
            (
                WIDTH - 10,
                HEIGHT - 100,
            ),
            Image.Resampling.LANCZOS,
        )

    else:
        foreground.thumbnail(
            (
                WIDTH - 26,
                1580,
            ),
            Image.Resampling.LANCZOS,
        )

    canvas = background.copy()

    x = (
        WIDTH
        - foreground.width
    ) // 2

    y = (
        HEIGHT
        - foreground.height
    ) // 2 - 35

    y = max(
        40,
        y,
    )

    canvas.paste(
        foreground,
        (
            x,
            y,
        ),
    )

    return canvas


def build_contact_sheet(
    scenes: list[Scene],
) -> Path:
    """Final Director için contact sheet oluşturur."""
    columns = 4
    cell_width = 270
    cell_height = 480

    rows = math.ceil(
        len(
            scenes
        )
        / columns
    )

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "black",
    )

    for index, scene in enumerate(
        scenes
    ):
        frame = compose_vertical(
            Path(
                scene.visual_file
            )
        )

        frame.thumbnail(
            (
                cell_width,
                cell_height,
            ),
            Image.Resampling.LANCZOS,
        )

        x = (
            index
            % columns
        ) * cell_width

        y = (
            index
            // columns
        ) * cell_height

        sheet.paste(
            frame,
            (
                x,
                y,
            ),
        )

    output = (
        WORK_DIR
        / "contact_sheet.jpg"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output,
        "JPEG",
        quality=94,
    )

    return output


def review_full_visual_flow(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Tüm sahne akışını birlikte değerlendirir."""
    sheet = build_contact_sheet(
        scenes
    )

    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\n"
            f"NARRATION: {scene.narration}\n"
            f"EXPECTED: {scene.visual_description}\n"
            f"MATCH: {scene.visual_match_score:.1f}\n"
            f"QUALITY: {scene.image_quality_score:.1f}\n"
            f"COMPOSITION: {scene.composition_score:.1f}\n"
            f"SOURCE: {scene.visual_source}"
        )
        for scene in scenes
    )

    prompt = f"""
Premium comic Shorts final Visual Director'sın.

SCENE PLAN:
{plan}

Contact sheet'i incele.

0-100:
- visual_story_match
- visual_diversity
- image_quality
- crop_quality
- narrative_flow
- professional_feel
- rewatch_value
- overall

Thor prototype seviyesinde premium kalite hedefleniyor.

Kritik:
- Narration ile görsel uyuşmalı.
- Art arda çok benzer panel olmamalı.
- Düşük çözünürlük hissi olmamalı.
- Slideshow hissi minimum olmalı.
- Yanlış karakter/aksiyon gösterilmemeli.

SADECE JSON:
{{
  "visual_story_match": 0,
  "visual_diversity": 0,
  "image_quality": 0,
  "crop_quality": 0,
  "narrative_flow": 0,
  "professional_feel": 0,
  "rewatch_value": 0,
  "overall": 0,
  "weak_scenes": [3,7],
  "problems": ["..."]
}}
"""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=sheet.read_bytes(),
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.08,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name="Final Visual Director",
    )

    return parse_json_response(
        response.text
        or "{}"
    )


def run_final_visual_gate(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
    global_candidates: list[ImageCandidate],
    enable_reconstruction: bool,
) -> list[Scene]:
    """Final Director geçene kadar zayıf sahneleri yeniden üretir."""
    print()
    print("=" * 78)
    print("6/13 - FINAL VISUAL FLOW REPAIR LOOP")
    print("=" * 78)

    cycle = 1

    asset_directory = (
        ASSET_DIR
        / event[
            "id"
        ]
    )

    reconstruction_counter = {
        "count": sum(
            1
            for scene in scenes
            if scene.visual_source
            == "ai_reconstruction"
        ),
    }

    while checkpoint_cycle_allowed(
        cycle
    ):
        review = review_full_visual_flow(
            client,
            scenes,
        )

        score = float(
            review.get(
                "overall",
                0,
            )
            or 0
        )

        problems = "; ".join(
            clean(
                item
            )
            for item in review.get(
                "problems",
                [],
            )
        )

        passed = CHECKPOINTS.record(
            name="final_visual_flow",
            score=score,
            threshold=FINAL_DIRECTOR_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        if passed:
            return scenes

        weak_numbers: set[
            int
        ] = set()

        for raw in review.get(
            "weak_scenes",
            [],
        ):
            try:
                weak_numbers.add(
                    int(
                        raw
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        if not weak_numbers:
            weakest = min(
                scenes,
                key=lambda scene: (
                    scene.visual_match_score
                    + scene.image_quality_score
                    + scene.composition_score
                ),
            )

            weak_numbers.add(
                weakest.scene_number
            )

        for scene_number in sorted(
            weak_numbers
        ):
            scene_data = next(
                item
                for item in storyboard[
                    "scenes"
                ]
                if int(
                    item[
                        "scene_number"
                    ]
                )
                == scene_number
            )

            used_ids = {
                scene.selected_candidate_id
                for scene in scenes
                if scene.scene_number
                != scene_number
                and scene.selected_candidate_id
                != "AI"
            }

            replacement = resolve_scene_until_pass(
                client=client,
                event=event,
                scene_data=scene_data,
                global_candidates=global_candidates,
                used_candidate_ids=used_ids,
                asset_directory=asset_directory,
                reconstruction_counter=reconstruction_counter,
                enable_reconstruction=enable_reconstruction,
                force_new_visual=True,
            )

            scenes[
                scene_number - 1
            ] = replacement

        cycle += 1

    raise ComicFactoryError(
        "Final visual flow repair loop güvenlik limiti aşıldı."
    )


def choose_motion_direction(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[int, dict[str, Any]]:
    """AI Cinematic Director motion planı üretir."""
    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\n"
            f"ROLE: {scene.story_role}\n"
            f"NARRATION: {scene.narration}\n"
            f"VISUAL: {scene.visual_description}"
        )
        for scene in scenes
    )

    prompt = f"""
Premium vertical comic-video Cinematic Director'sın.

SCENES:
{plan}

Her scene için motion ve outgoing transition seç.

motion:
{", ".join(ALLOWED_MOTIONS)}

transition:
{", ".join(ALLOWED_TRANSITIONS)}

Kurallar:
- Aynı motion art arda kullanma.
- hook hızlı ama temiz.
- context daha sakin.
- escalation daha dinamik.
- twist/payoff gerektiğinde impact.
- cta sakinleşsin.
- Gelecek scene görseli kendi narration'ından önce ASLA görünmemeli.
- transition yalnız mevcut scene'in sonunda kendi görüntüsü üzerinde çalışmalı.

SADECE JSON:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "motion": "slow_push",
      "transition": "soft_black",
      "transition_duration": 0.12
    }}
  ]
}}
"""

    payload = ask_gemini_json(
        client,
        prompt,
        temperature=0.20,
        operation_name="Cinematic Director",
    )

    result: dict[
        int,
        dict[str, Any]
    ] = {}

    for item in payload.get(
        "scenes",
        [],
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            scene_number = int(
                item.get(
                    "scene_number",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        result[
            scene_number
        ] = item

    return result


def apply_motion_plan(
    scenes: list[Scene],
    direction: dict[int, dict[str, Any]],
) -> None:
    """Motion planını güvenli kurallarla uygular."""
    previous_motion = ""

    for index, scene in enumerate(
        scenes
    ):
        item = direction.get(
            scene.scene_number,
            {},
        )

        motion = clean(
            item.get(
                "motion"
            )
        )

        if motion not in ALLOWED_MOTIONS:
            motion = ALLOWED_MOTIONS[
                index
                % len(
                    ALLOWED_MOTIONS
                )
            ]

        if motion == previous_motion:
            motion = ALLOWED_MOTIONS[
                (
                    ALLOWED_MOTIONS.index(
                        motion
                    )
                    + 1
                )
                % len(
                    ALLOWED_MOTIONS
                )
            ]

        transition = clean(
            item.get(
                "transition"
            )
        )

        if transition not in ALLOWED_TRANSITIONS:
            transition = "cut"

        try:
            duration = float(
                item.get(
                    "transition_duration",
                    0.12,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            duration = 0.12

        if transition == "cut":
            duration = 0.0

        else:
            duration = max(
                0.08,
                min(
                    0.20,
                    duration,
                ),
            )

        scene.motion = motion
        scene.transition = transition
        scene.transition_duration = duration

        previous_motion = motion


def motion_filter(
    motion: str,
) -> str:
    """Sinematik kamera hareketi filtresi oluşturur."""
    motions = {
        "slow_push": (
            "zoompan="
            "z='min(zoom+0.00013,1.04)':"
            "x='iw/2-iw/zoom/2':"
            "y='ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
        "slow_pull": (
            "zoompan="
            "z='if(eq(on,0),1.04,max(1.0,zoom-0.00013))':"
            "x='iw/2-iw/zoom/2':"
            "y='ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
        "pan_left": (
            "zoompan="
            "z='1.035':"
            "x='max(0,(iw-iw/zoom)*(1-on/240))':"
            "y='ih/2-iw*0+ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
        "pan_right": (
            "zoompan="
            "z='1.035':"
            "x='min(iw-iw/zoom,(iw-iw/zoom)*(on/240))':"
            "y='ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
        "vertical_scan": (
            "zoompan="
            "z='1.03':"
            "x='iw/2-iw/zoom/2':"
            "y='min(ih-ih/zoom,(ih-ih/zoom)*(on/240))':"
            "d=1:s=1080x1920:fps=30"
        ),
        "impact": (
            "zoompan="
            "z='min(zoom+0.00062,1.08)':"
            "x='iw/2-iw/zoom/2':"
            "y='ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
        "hold": (
            "zoompan="
            "z='1.008':"
            "x='iw/2-iw/zoom/2':"
            "y='ih/2-ih/zoom/2':"
            "d=1:s=1080x1920:fps=30"
        ),
    }

    return motions.get(
        motion,
        motions[
            "slow_push"
        ],
    )


def transition_filter(
    transition: str,
    duration: float,
    scene_duration: float,
) -> str:
    """Gelecek scene'i erken göstermeyen transition filtresi üretir."""
    if (
        transition == "cut"
        or duration <= 0.0
    ):
        return ""

    duration = min(
        duration,
        max(
            0.05,
            scene_duration
            / 4.0,
        ),
    )

    start = max(
        0.0,
        scene_duration
        - duration,
    )

    if transition in {
        "soft_black",
        "dip_black",
    }:
        return (
            f",fade=t=out:"
            f"st={start:.3f}:"
            f"d={duration:.3f}:"
            "color=black"
        )

    if transition == "flash_white":
        return (
            f",fade=t=out:"
            f"st={start:.3f}:"
            f"d={duration:.3f}:"
            "color=white"
        )

    return ""


def evaluate_motion_plan(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Motion planını anlam ve tekrar açısından değerlendirir."""
    prompt = f"""
Premium comic video Motion QC.

PLAN:
{json.dumps(
    [
        {
            "scene": scene.scene_number,
            "role": scene.story_role,
            "narration": scene.narration,
            "motion": scene.motion,
            "transition": scene.transition,
        }
        for scene in scenes
    ],
    ensure_ascii=False,
    indent=2,
)}

0-100:
- story_fit
- variety
- restraint
- reveal_emphasis
- sync_safety
- overall

sync_safety:
Gelecek scene görselinin narration başlamadan görünmesine
neden olabilecek geçiş olmamalı.

SADECE JSON:
{{
  "overall": 0,
  "problems": ["..."]
}}
"""

    return ask_gemini_json(
        client,
        prompt,
        temperature=0.05,
        operation_name="Motion QC",
    )


def build_quality_motion(
    client: genai.Client,
    scenes: list[Scene],
) -> None:
    """Motion checkpoint geçene kadar planı yeniden üretir."""
    print()
    print("=" * 78)
    print("7/13 - MOTION REPAIR LOOP")
    print("=" * 78)

    cycle = 1

    while checkpoint_cycle_allowed(
        cycle
    ):
        direction = choose_motion_direction(
            client,
            scenes,
        )

        apply_motion_plan(
            scenes,
            direction,
        )

        review = evaluate_motion_plan(
            client,
            scenes,
        )

        score = float(
            review.get(
                "overall",
                0,
            )
            or 0
        )

        problems = "; ".join(
            clean(
                item
            )
            for item in review.get(
                "problems",
                [],
            )
        )

        passed = CHECKPOINTS.record(
            name="motion_quality",
            score=score,
            threshold=MOTION_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        if passed:
            return

        cycle += 1

    raise ComicFactoryError(
        "Motion repair loop güvenlik limiti aşıldı."
    )


def get_subtitle_font(
    size: int,
) -> ImageFont.ImageFont:
    """Subtitle ölçümü için mevcut fontu bulur."""
    paths = (
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\arialbd.ttf"
        ),
    )

    for path in paths:
        if path.exists():
            return ImageFont.truetype(
                str(
                    path
                ),
                size=size,
            )

    return ImageFont.load_default()


def subtitle_text_width(
    text: str,
    font_size: int,
) -> int:
    """Subtitle metninin pixel genişliğini ölçer."""
    font = get_subtitle_font(
        font_size
    )

    image = Image.new(
        "RGB",
        (
            10,
            10,
        ),
        "black",
    )

    draw = ImageDraw.Draw(
        image
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
        stroke_width=5,
    )

    return (
        bbox[
            2
        ]
        - bbox[
            0
        ]
    )


def build_subtitle_groups(
    words: list[dict[str, Any]],
) -> list[
    dict[str, Any]
]:
    """Taşmayacak şekilde 1-3 kelimelik subtitle grupları oluşturur."""
    groups: list[
        dict[str, Any]
    ] = []

    index = 0

    while index < len(
        words
    ):
        best_group: dict[str, Any] | None = None

        for count in range(
            min(
                SUBTITLE_MAX_WORDS,
                len(
                    words
                )
                - index,
            ),
            0,
            -1,
        ):
            indexes = list(
                range(
                    index,
                    index + count,
                )
            )

            text = " ".join(
                clean(
                    words[
                        word_index
                    ][
                        "word"
                    ]
                )
                for word_index in indexes
            ).upper()

            chosen_size = None

            for font_size in range(
                SUBTITLE_BASE_FONT_SIZE,
                SUBTITLE_MIN_FONT_SIZE - 1,
                -2,
            ):
                width = subtitle_text_width(
                    text,
                    font_size,
                )

                # Aktif kelimenin büyümesi için ekstra alan.
                safety_width = round(
                    width
                    * 1.10
                )

                if safety_width <= SUBTITLE_AVAILABLE_WIDTH:
                    chosen_size = font_size
                    break

            if chosen_size is not None:
                best_group = {
                    "indexes": indexes,
                    "font_size": chosen_size,
                    "text": text,
                }

                break

        if best_group is None:
            word = clean(
                words[
                    index
                ][
                    "word"
                ]
            ).upper()

            font_size = SUBTITLE_MIN_FONT_SIZE

            width = subtitle_text_width(
                word,
                font_size,
            )

            if width > SUBTITLE_AVAILABLE_WIDTH:
                raise ComicFactoryError(
                    "Tek bir subtitle kelimesi safe-zone'a "
                    f"sığmıyor: {word}"
                )

            best_group = {
                "indexes": [
                    index
                ],
                "font_size": font_size,
                "text": word,
            }

        groups.append(
            best_group
        )

        index = (
            best_group[
                "indexes"
            ][
                -1
            ]
            + 1
        )

    return groups


def build_group_map(
    groups: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Her kelimenin ait olduğu subtitle grubunu döndürür."""
    result: dict[
        int,
        dict[str, Any]
    ] = {}

    for group in groups:
        for index in group[
            "indexes"
        ]:
            result[
                index
            ] = group

    return result


def ass_time(
    seconds: float,
) -> str:
    """Saniyeyi ASS timestamp'e dönüştürür."""
    centiseconds = round(
        max(
            0.0,
            seconds,
        )
        * 100
    )

    hours, remainder = divmod(
        centiseconds,
        360000,
    )

    minutes, remainder = divmod(
        remainder,
        6000,
    )

    seconds_value, cs = divmod(
        remainder,
        100,
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{seconds_value:02d}."
        f"{cs:02d}"
    )


def ass_escape(
    text: str,
) -> str:
    """ASS özel karakterlerini temizler."""
    return (
        clean(
            text
        )
        .replace(
            "\\",
            r"\\"
        )
        .replace(
            "{",
            "("
        )
        .replace(
            "}",
            ")"
        )
    )


def emphasis_words(
    scenes: list[Scene],
) -> set[str]:
    """Story emphasis kelimelerini normalize eder."""
    result: set[
        str
    ] = set()

    for scene in scenes:
        for word in scene.emphasis_words:
            normalized = normalize_alignment_word(
                word
            )

            if normalized:
                result.add(
                    normalized
                )

    return result


def subtitle_line(
    words: list[dict[str, Any]],
    active_index: int,
    group_map: dict[int, dict[str, Any]],
    emphasis: set[str],
) -> str:
    """Aktif kelimeyi sarı yapan subtitle metni oluşturur."""
    group = group_map[
        active_index
    ]

    size = int(
        group[
            "font_size"
        ]
    )

    parts: list[
        str
    ] = []

    for index in group[
        "indexes"
    ]:
        raw = ass_escape(
            words[
                index
            ][
                "word"
            ]
        )

        display = raw.upper()

        normalized = normalize_alignment_word(
            raw
        )

        if index == active_index:
            scale = (
                111
                if normalized in emphasis
                else 108
            )

            parts.append(
                (
                    r"{"
                    rf"\fs{size}"
                    r"\1c&H0030D7FF&"
                    rf"\fscx{scale}"
                    rf"\fscy{scale}"
                    r"\bord5\shad2\b1"
                    r"}"
                    + display
                    + r"{\r}"
                )
            )

        else:
            parts.append(
                (
                    r"{"
                    rf"\fs{size}"
                    r"\fscx100\fscy100"
                    r"}"
                    + display
                    + r"{\r}"
                )
            )

    return " ".join(
        parts
    )


def create_subtitles_until_pass(
    aligned_words: list[dict[str, Any]],
    scenes: list[Scene],
    narration: str,
    offset: float,
) -> Path:
    """Subtitle layout deterministic checkpoint geçene kadar düzenler."""
    print()
    print("=" * 78)
    print("8/13 - SUBTITLE QUALITY GATE")
    print("=" * 78)

    expected_tokens = narration_tokens(
        narration
    )

    actual_tokens = [
        clean(
            item[
                "word"
            ]
        )
        for item in aligned_words
    ]

    if expected_tokens != actual_tokens:
        raise ComicFactoryError(
            "Subtitle text narration ile birebir aynı değil."
        )

    cycle = 1

    while checkpoint_cycle_allowed(
        cycle
    ):
        groups = build_subtitle_groups(
            aligned_words
        )

        overflow = False

        for group in groups:
            width = subtitle_text_width(
                group[
                    "text"
                ],
                int(
                    group[
                        "font_size"
                    ]
                ),
            )

            width = round(
                width
                * 1.10
            )

            if width > SUBTITLE_AVAILABLE_WIDTH:
                overflow = True
                break

        sync_monotonic = all(
            float(
                aligned_words[
                    index
                ][
                    "start"
                ]
            )
            >= (
                float(
                    aligned_words[
                        index - 1
                    ][
                        "start"
                    ]
                )
                if index > 0
                else 0.0
            )
            for index in range(
                len(
                    aligned_words
                )
            )
        )

        passed = (
            not overflow
            and sync_monotonic
            and expected_tokens
            == actual_tokens
        )

        score = (
            100.0
            if passed
            else 0.0
        )

        CHECKPOINTS.record(
            name="subtitle_technical",
            score=score,
            threshold=100.0,
            cycle=cycle,
            details=(
                f"overflow={overflow} | "
                f"sync_monotonic={sync_monotonic} | "
                "text_exact="
                f"{expected_tokens == actual_tokens}"
            ),
            passed=passed,
        )

        if not passed:
            cycle += 1
            continue

        group_map = build_group_map(
            groups
        )

        emphasis = emphasis_words(
            scenes
        )

        path = (
            WORK_DIR
            / "precise.ass"
        )

        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            (
                "Format: Name,Fontname,Fontsize,"
                "PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,"
                "Italic,Underline,StrikeOut,"
                "ScaleX,ScaleY,Spacing,Angle,"
                "BorderStyle,Outline,Shadow,"
                "Alignment,MarginL,MarginR,"
                "MarginV,Encoding"
            ),
            (
                "Style: Main,DejaVu Sans,"
                f"{SUBTITLE_BASE_FONT_SIZE},"
                "&H00FFFFFF,&H00FFFFFF,"
                "&H00121212,&H00000000,"
                "-1,0,0,0,100,100,0,0,"
                "1,5,2,2,"
                f"{SUBTITLE_MARGIN_LEFT},"
                f"{SUBTITLE_MARGIN_RIGHT},"
                f"{SUBTITLE_MARGIN_BOTTOM},1"
            ),
            "",
            "[Events]",
            (
                "Format: Layer,Start,End,Style,"
                "Name,MarginL,MarginR,MarginV,"
                "Effect,Text"
            ),
        ]

        for index, item in enumerate(
            aligned_words
        ):
            start = (
                float(
                    item[
                        "start"
                    ]
                )
                + offset
            )

            end = (
                float(
                    item[
                        "end"
                    ]
                )
                + offset
            )

            if end <= 0:
                continue

            start = max(
                0.0,
                start,
            )

            end = max(
                start + 0.05,
                end,
            )

            lines.append(
                "Dialogue: 0,"
                f"{ass_time(start)},"
                f"{ass_time(end)},"
                "Main,,0,0,0,,"
                f"{subtitle_line(
                    aligned_words,
                    index,
                    group_map,
                    emphasis,
                )}"
            )

        path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        return path

    raise ComicFactoryError(
        "Subtitle repair loop güvenlik limiti aşıldı."
    )


def render_scene_segment(
    ffmpeg: str,
    scene: Scene,
    frame_path: Path,
    output_path: Path,
) -> None:
    """Tek scene'i kendi kesin audio aralığı kadar render eder."""
    duration = (
        scene.audio_end
        - scene.audio_start
    )

    if duration <= 0:
        raise ComicFactoryError(
            f"Scene {scene.scene_number} süresi geçersiz."
        )

    filter_text = (
        "scale=1080:1920,"
        + motion_filter(
            scene.motion
        )
        + transition_filter(
            scene.transition,
            scene.transition_duration,
            duration,
        )
        + ",format=yuv420p"
    )

    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(
                FPS
            ),
            "-i",
            str(
                frame_path
            ),
            "-t",
            f"{duration:.4f}",
            "-vf",
            filter_text,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(
                output_path
            ),
        ],
        (
            f"Scene {scene.scene_number} "
            "render edilemedi."
        ),
    )


def attach_timeline(
    scenes: list[Scene],
    timeline: list[
        tuple[
            float,
            float,
        ]
    ],
) -> None:
    """Audio timeline'ı scene nesnelerine bağlar."""
    if len(
        scenes
    ) != len(
        timeline
    ):
        raise ComicFactoryError(
            "Scene ve timeline sayısı uyuşmuyor."
        )

    for scene, (
        start,
        end,
    ) in zip(
        scenes,
        timeline,
        strict=True,
    ):
        scene.audio_start = start
        scene.audio_end = end


def render_video(
    scenes: list[Scene],
    audio_file: Path,
    ass_file: Path,
) -> Path:
    """Exact scene timeline ile final videoyu render eder."""
    print()
    print("=" * 78)
    print("9/13 - EXACT-TIMELINE RENDER")
    print("=" * 78)

    ffmpeg = ffmpeg_path()

    shutil.rmtree(
        WORK_DIR
        / "render",
        ignore_errors=True,
    )

    render_directory = (
        WORK_DIR
        / "render"
    )

    render_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments: list[
        Path
    ] = []

    for scene in scenes:
        frame_path = (
            render_directory
            / (
                f"frame_"
                f"{scene.scene_number:02d}.jpg"
            )
        )

        segment_path = (
            render_directory
            / (
                f"segment_"
                f"{scene.scene_number:02d}.mp4"
            )
        )

        frame = compose_vertical(
            Path(
                scene.visual_file
            )
        )

        frame.save(
            frame_path,
            "JPEG",
            quality=96,
        )

        render_scene_segment(
            ffmpeg,
            scene,
            frame_path,
            segment_path,
        )

        segments.append(
            segment_path
        )

    concat_file = (
        render_directory
        / "concat.txt"
    )

    concat_file.write_text(
        "\n".join(
            (
                "file '"
                + segment.resolve().as_posix()
                + "'"
            )
            for segment in segments
        ),
        encoding="utf-8",
    )

    silent_video = (
        render_directory
        / "silent.mp4"
    )

    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(
                concat_file
            ),
            "-c",
            "copy",
            str(
                silent_video
            ),
        ],
        "Scene segmentleri birleştirilemedi.",
    )

    timestamp = datetime.now(
        TZ
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive = (
        VIDEO_DIR
        / (
            f"comic_factory_v4_"
            f"{timestamp}.mp4"
        )
    )

    subtitle_filter = (
        "ass="
        + ass_file.resolve()
        .as_posix()
        .replace(
            ":",
            r"\:",
        )
    )

    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(
                silent_video
            ),
            "-i",
            str(
                audio_file
            ),
            "-vf",
            subtitle_filter,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(
                archive
            ),
        ],
        "Final V4 video render edilemedi.",
    )

    return archive


def technical_video_check(
    video: Path,
    audio: Path,
) -> dict[str, Any]:
    """Final render için deterministic teknik QC yapar."""
    ffmpeg = ffmpeg_path()

    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                video
            ),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stderr = process.stderr

    resolution_match = re.search(
        r"Video:.*?(\d{3,5})x(\d{3,5})",
        stderr,
    )

    video_duration = media_duration(
        ffmpeg,
        video,
    )

    audio_duration = media_duration(
        ffmpeg,
        audio,
    )

    width = 0
    height = 0

    if resolution_match:
        width = int(
            resolution_match.group(
                1
            )
        )

        height = int(
            resolution_match.group(
                2
            )
        )

    has_audio = (
        "Audio:"
        in stderr
    )

    duration_difference = abs(
        video_duration
        - audio_duration
    )

    resolution_ok = (
        width == WIDTH
        and height == HEIGHT
    )

    duration_ok = (
        duration_difference
        <= 0.15
    )

    passed = (
        resolution_ok
        and has_audio
        and duration_ok
    )

    return {
        "passed": passed,
        "resolution_ok": resolution_ok,
        "width": width,
        "height": height,
        "has_audio": has_audio,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "duration_difference": duration_difference,
    }


def run_render_until_pass(
    scenes: list[Scene],
    audio_file: Path,
    ass_file: Path,
) -> Path:
    """Technical QC geçene kadar videoyu yeniden render eder."""
    print()
    print("=" * 78)
    print("10/13 - TECHNICAL RENDER REPAIR LOOP")
    print("=" * 78)

    cycle = 1

    while checkpoint_cycle_allowed(
        cycle
    ):
        archive = render_video(
            scenes,
            audio_file,
            ass_file,
        )

        metrics = technical_video_check(
            archive,
            audio_file,
        )

        score = (
            100.0
            if metrics[
                "passed"
            ]
            else 0.0
        )

        passed = CHECKPOINTS.record(
            name="technical_video_qc",
            score=score,
            threshold=100.0,
            cycle=cycle,
            details=(
                f"resolution="
                f"{metrics['width']}x"
                f"{metrics['height']} | "
                f"audio={metrics['has_audio']} | "
                f"duration_delta="
                f"{metrics['duration_difference']:.3f}s"
            ),
            passed=bool(
                metrics[
                    "passed"
                ]
            ),
        )

        if passed:
            shutil.copy2(
                archive,
                LATEST_VIDEO_FILE,
            )

            return archive

        archive.unlink(
            missing_ok=True
        )

        cycle += 1

    raise ComicFactoryError(
        "Technical render repair loop güvenlik limiti aşıldı."
    )


def save_script_metadata(
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
) -> None:
    """Uploader'ların kullandığı latest.json dosyasını oluşturur."""
    hashtags = [
        clean(
            tag
        )
        for tag in storyboard.get(
            "hashtags",
            [],
        )
        if clean(
            tag
        )
    ]

    if not any(
        tag.casefold()
        == "#shorts"
        for tag in hashtags
    ):
        hashtags.append(
            "#Shorts"
        )

    source_lines = []

    seen_urls: set[
        str
    ] = set()

    for source in event.get(
        "sources",
        [],
    ):
        if not isinstance(
            source,
            dict,
        ):
            continue

        url = clean(
            source.get(
                "url"
            )
        )

        name = clean(
            source.get(
                "name"
            )
        )

        if (
            not url
            or url in seen_urls
        ):
            continue

        seen_urls.add(
            url
        )

        source_lines.append(
            f"- {name or 'Source'}: {url}"
        )

    description = clean(
        storyboard.get(
            "description"
        )
    )

    full_description = (
        description
        + "\n\n"
        + " ".join(
            hashtags
        )
    )

    if source_lines:
        full_description += (
            "\n\nKaynaklar:\n"
            + "\n".join(
                source_lines
            )
        )

    script = {
        "title": clean(
            storyboard.get(
                "title"
            )
        ),
        "description": description,
        "full_description": full_description,
        "hashtags": hashtags,
        "narration": clean(
            storyboard.get(
                "narration"
            )
        ),
        "scene_narrations": [
            scene.narration
            for scene in scenes
        ],
    }

    save_json(
        LATEST_SCRIPT_FILE,
        {
            "event_id": event[
                "id"
            ],
            "generated_at": datetime.now(
                TZ
            ).isoformat(),
            "script": script,
        },
    )


def save_visual_manifest(
    event: dict[str, Any],
    scenes: list[Scene],
) -> None:
    """Final V4 sahne kararlarını kaydeder."""
    directory = (
        ASSET_DIR
        / event[
            "id"
        ]
    )

    save_json(
        directory
        / "visual_manifest.json",
        {
            "event": {
                "event_title": event.get(
                    "event_title"
                ),
                "series": event.get(
                    "series"
                ),
                "issue": event.get(
                    "issue"
                ),
            },
            "scenes": [
                asdict(
                    scene
                )
                for scene in scenes
            ],
        },
    )


def mark_used(
    event: dict[str, Any],
) -> None:
    """Başarılı event'i kullanılmış olarak kaydeder."""
    payload = load_json(
        USED_EVENTS_FILE,
        {
            "events": [],
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        payload = {
            "events": [],
        }

    events = payload.setdefault(
        "events",
        [],
    )

    key = clean(
        event.get(
            "event_key"
        )
    )

    if not any(
        isinstance(
            item,
            dict,
        )
        and clean(
            item.get(
                "event_key"
            )
        )
        == key
        for item in events
    ):
        events.append(
            {
                "event_key": key,
                "event_title": event[
                    "event_title"
                ],
                "series": event[
                    "series"
                ],
                "issue": event[
                    "issue"
                ],
                "completed_at": datetime.now(
                    TZ
                ).isoformat(),
            }
        )

    save_json(
        USED_EVENTS_FILE,
        payload,
    )


def upload_youtube() -> None:
    """Mevcut YouTube uploader'ı çalıştırır."""
    uploader = (
        ROOT
        / "youtube_uploader.py"
    )

    if not uploader.exists():
        raise ComicFactoryError(
            "youtube_uploader.py bulunamadı."
        )

    result = subprocess.run(
        [
            sys.executable,
            str(
                uploader
            ),
        ],
        cwd=str(
            ROOT
        ),
    )

    if result.returncode != 0:
        raise ComicFactoryError(
            "YouTube upload başarısız."
        )


def system_check(
    upload: bool,
) -> None:
    """Gerekli API ve runtime bağımlılıklarını kontrol eder."""
    problems: list[
        str
    ] = []

    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ):
        if not os.getenv(
            name,
            "",
        ).strip():
            problems.append(
                f"{name} yok."
            )

    try:
        ffmpeg_path()

    except Exception as error:
        problems.append(
            f"FFmpeg: {error}"
        )

    if (
        upload
        and not (
            ROOT
            / "youtube_uploader.py"
        ).exists()
    ):
        problems.append(
            "youtube_uploader.py yok."
        )

    if problems:
        raise ComicFactoryError(
            "\n".join(
                problems
            )
        )

    print(
        "✓ Gemini hazır"
    )

    print(
        f"✓ Gacrux voice: "
        f"{GEMINI_TTS_VOICE} (LOCKED)"
    )

    print(
        "✓ Groq hazır"
    )

    print(
        "✓ FFmpeg hazır"
    )

    print(
        "✓ Quality repair loops hazır"
    )


def main() -> None:
    """Comic Factory V4 ana kalite kontrollü üretim hattı."""
    args = parse_args()

    ensure_dirs()

    if args.check:
        system_check(
            args.upload
        )
        return

    client = gemini_client()

    enable_reconstruction = (
        ENABLE_AI_RECONSTRUCTION
        and not args.disable_ai_reconstruction
    )

    try:
        print()
        print("=" * 78)
        print("COMIC FACTORY V4")
        print("REPAIR UNTIL PASS")
        print("=" * 78)
        print()

        print(
            f"Voice: "
            f"{GEMINI_TTS_VOICE} (LOCKED)"
        )

        print(
            "AI reconstruction: "
            + (
                "ON"
                if enable_reconstruction
                else "OFF"
            )
        )

        print(
            "Quality repair cycles: "
            + (
                "UNLIMITED"
                if QUALITY_GATE_MAX_CYCLES == 0
                else str(
                    QUALITY_GATE_MAX_CYCLES
                )
            )
        )

        if args.reuse_active:
            event = load_active_event()

        else:
            event = activate_event(
                research_events(
                    client,
                    args.count,
                )[
                    0
                ]
            )

        storyboard = build_quality_storyboard(
            client,
            event,
        )

        narration = clean(
            storyboard[
                "narration"
            ]
        )

        audio_file, whisper_words, aligned_words = (
            build_quality_audio(
                client,
                narration,
            )
        )

        ffmpeg = ffmpeg_path()

        duration = media_duration(
            ffmpeg,
            audio_file,
        )

        timeline = build_scene_timeline(
            storyboard,
            aligned_words,
            duration,
        )

        print()
        print("=" * 78)
        print("4/13 - EXACT AUDIO SCENE TIMELINE")
        print("=" * 78)

        for index, (
            start,
            end,
        ) in enumerate(
            timeline,
            start=1,
        ):
            print(
                f"✓ Scene {index:02d}: "
                f"{start:.3f}s → {end:.3f}s"
            )

        scenes, global_candidates = (
            build_quality_visuals(
                client,
                event,
                storyboard,
                max(
                    30,
                    args.max_images,
                ),
                enable_reconstruction,
            )
        )

        scenes = run_final_visual_gate(
            client,
            event,
            storyboard,
            scenes,
            global_candidates,
            enable_reconstruction,
        )

        attach_timeline(
            scenes,
            timeline,
        )

        build_quality_motion(
            client,
            scenes,
        )

        ass_file = create_subtitles_until_pass(
            aligned_words,
            scenes,
            narration,
            args.subtitle_offset,
        )

        save_script_metadata(
            event,
            storyboard,
            scenes,
        )

        save_visual_manifest(
            event,
            scenes,
        )

        archive = run_render_until_pass(
            scenes,
            audio_file,
            ass_file,
        )

        print()
        print("=" * 78)
        print("11/13 - CHECKPOINT SUMMARY")
        print("=" * 78)

        CHECKPOINTS.save()

        failed_checkpoints = [
            result
            for result in CHECKPOINTS.results
            if not result.passed
        ]

        passed_checkpoints = [
            result
            for result in CHECKPOINTS.results
            if result.passed
        ]

        print(
            f"✓ Passed checkpoint events: "
            f"{len(passed_checkpoints)}"
        )

        print(
            f"↻ Repaired failures: "
            f"{len(failed_checkpoints)}"
        )

        CHECKPOINTS.persist_lessons()

        print()
        print("=" * 78)
        print("12/13 - QUALITY LESSONS SAVED")
        print("=" * 78)

        print(
            f"✓ {LESSONS_FILE}"
        )

        mark_used(
            event
        )

        print()
        print("=" * 78)
        print("13/13 - COMIC FACTORY V4 BAŞARILI")
        print("=" * 78)
        print()

        print(
            f"Konu: "
            f"{clean(event['event_title'])}"
        )

        print(
            f"Voice: "
            f"{GEMINI_TTS_VOICE} (LOCKED)"
        )

        print(
            f"Video: "
            f"{LATEST_VIDEO_FILE}"
        )

        print(
            f"Arşiv: "
            f"{archive.relative_to(ROOT)}"
        )

        print(
            "Story quality: PASS"
        )

        print(
            "Audio alignment: PASS"
        )

        print(
            "Scene visual match: PASS"
        )

        print(
            "Image quality: PASS"
        )

        print(
            "Composition: PASS"
        )

        print(
            "Motion quality: PASS"
        )

        print(
            "Subtitle technical QC: PASS"
        )

        print(
            "Final visual flow: PASS"
        )

        print(
            "Technical video QC: PASS"
        )

        if args.upload:
            upload_youtube()

    except KeyboardInterrupt:
        CHECKPOINTS.persist_lessons()

        raise SystemExit(
            1
        )

    except Exception as error:
        CHECKPOINTS.persist_lessons()

        print()
        print("=" * 78)
        print("COMIC FACTORY V4 DURDU")
        print("=" * 78)
        print()

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
