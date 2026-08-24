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
REJECTED_EVENTS_FILE = DIRECTOR_DIR / "rejected_events.json"

TZ = ZoneInfo("Europe/Istanbul")

WIDTH = 1080
HEIGHT = 1920
FPS = 30

SCENE_COUNT = 14

STORY_THRESHOLD = 94.0
AUDIO_ALIGNMENT_THRESHOLD = 95.0
VISUAL_MATCH_THRESHOLD = 95.0
IMAGE_QUALITY_THRESHOLD = 92.0
COMPOSITION_THRESHOLD = 94.0
MOTION_THRESHOLD = 92.0
FINAL_VISUAL_THRESHOLD = 94.0

MAX_EVENT_CANDIDATES_PER_RESEARCH = 8
MAX_EVENT_SWITCHES_PER_RUN = 12

STORY_REPAIR_CYCLES_PER_EVENT = 5
AUDIO_REPAIR_CYCLES = 5
VISUAL_REPAIR_CYCLES = 7
AI_RECONSTRUCTION_REPAIR_CYCLES = 4
FINAL_VISUAL_REPAIR_CYCLES = 5
MOTION_REPAIR_CYCLES = 4
RENDER_REPAIR_CYCLES = 3

MAX_GLOBAL_IMAGES = 40
MAX_SCENE_IMAGES = 20
MAX_VISION_IMAGES = 28

ENABLE_AI_RECONSTRUCTION = (
    os.getenv("ENABLE_AI_RECONSTRUCTION", "1").strip() == "1"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.1-flash-tts-preview",
)

# Thor V2 reference voice. Locked deliberately.
GEMINI_TTS_VOICE = "Gacrux"

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
SUBTITLE_MIN_FONT_SIZE = 38
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

load_dotenv(ROOT / ".env")


class ComicFactoryError(RuntimeError):
    """Comic Factory V4 pipeline hatası."""


class EventRejectedError(ComicFactoryError):
    """Mevcut event kalite standardına ulaşamadığında oluşur."""


@dataclass
class CheckpointResult:
    """Tek kalite kontrol sonucunu temsil eder."""

    event_id: str
    name: str
    passed: bool
    score: float
    threshold: float
    cycle: int
    details: str
    created_at: str


@dataclass
class ImageCandidate:
    """İndirilen gerçek comic görsel adayı."""

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
    """Visual Director tarafından sıralanan panel adayı."""

    candidate_id: str
    relevance_score: int
    crop_box: list[float]
    reason: str


@dataclass
class Scene:
    """Final videoda kullanılacak sahne."""

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
    """Checkpoint geçmişini ve öğrenilen kalite derslerini saklar."""

    def __init__(self) -> None:
        self.results: list[CheckpointResult] = []
        self.current_event_id = ""

    def set_event(self, event_id: str) -> None:
        self.current_event_id = event_id

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
            event_id=self.current_event_id,
            name=name,
            passed=bool(passed),
            score=round(float(score), 2),
            threshold=round(float(threshold), 2),
            cycle=cycle,
            details=clean(details),
            created_at=datetime.now(TZ).isoformat(),
        )

        self.results.append(result)

        icon = "✓" if result.passed else "✗"

        print(
            f"{icon} CHECKPOINT {result.name}: "
            f"{result.score:.2f}/100 "
            f"(min {result.threshold:.2f}) "
            f"| cycle={result.cycle}"
        )

        if result.details:
            print(f"  {result.details}")

        self.save()

        return result.passed

    def save(self) -> None:
        save_json(
            CHECKPOINT_FILE,
            {
                "updated_at": datetime.now(TZ).isoformat(),
                "results": [
                    asdict(result)
                    for result in self.results
                ],
            },
        )

    def persist_lessons(self) -> None:
        payload = load_json(
            LESSONS_FILE,
            {"lessons": []},
        )

        if not isinstance(payload, dict):
            payload = {"lessons": []}

        lessons = payload.setdefault(
            "lessons",
            [],
        )

        known = {
            (
                clean(item.get("checkpoint")),
                clean(item.get("details")),
            )
            for item in lessons
            if isinstance(item, dict)
        }

        for result in self.results:
            if result.passed:
                continue

            key = (
                result.name,
                result.details,
            )

            if key in known:
                continue

            lessons.append(
                {
                    "event_id": result.event_id,
                    "checkpoint": result.name,
                    "score": result.score,
                    "threshold": result.threshold,
                    "details": result.details,
                    "recorded_at": result.created_at,
                }
            )

            known.add(key)

        payload["lessons"] = lessons[-250:]
        payload["updated_at"] = datetime.now(TZ).isoformat()

        save_json(
            LESSONS_FILE,
            payload,
        )


CHECKPOINTS = CheckpointManager()


def parse_args() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""
    parser = argparse.ArgumentParser(
        description=(
            "Comic Factory V4 - "
            "quality-gated automatic comic video engine."
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
        default=MAX_EVENT_CANDIDATES_PER_RESEARCH,
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


def clean(value: Any) -> str:
    """Metni tek satırlık normalize edilmiş biçime getirir."""
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def slug(value: str) -> str:
    """Güvenli dosya adı üretir."""
    value = clean(value).lower()

    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
    }

    for source, target in replacements.items():
        value = value.replace(
            source,
            target,
        )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")[:80] or "event"


def ensure_dirs() -> None:
    """Pipeline klasörlerini oluşturur."""
    for directory in (
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
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_json(
    path: Path,
    default: Any = None,
) -> Any:
    """JSON dosyasını güvenli şekilde okur."""
    if not path.exists():
        return default

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ComicFactoryError(
            f"JSON okunamadı: {path}"
        ) from error


def save_json(
    path: Path,
    payload: Any,
) -> None:
    """JSON dosyasını atomik biçimde kaydeder."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def require_env(name: str) -> str:
    """Zorunlu environment variable değerini döndürür."""
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
    """Önceki kalite hatalarını yeni promptlara bağlar."""
    payload = load_json(
        LESSONS_FILE,
        {"lessons": []},
    )

    if not isinstance(payload, dict):
        return "Henüz kalite dersi yok."

    lessons = payload.get(
        "lessons",
        [],
    )

    if not isinstance(lessons, list):
        return "Henüz kalite dersi yok."

    recent = lessons[-30:]

    if not recent:
        return "Henüz kalite dersi yok."

    return "\n".join(
        (
            f"- {clean(item.get('checkpoint'))}: "
            f"{clean(item.get('details'))}"
        )
        for item in recent
        if isinstance(item, dict)
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
    """Model JSON cevabını ayrıştırır."""
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
        payload = json.loads(text)

    except json.JSONDecodeError as error:
        raise ComicFactoryError(
            "Gemini geçerli JSON döndürmedi."
        ) from error

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list):
        if not payload:
            return {"items": []}

        first = payload[0]

        if isinstance(first, dict):
            if any(
                key in first
                for key in (
                    "event_title",
                    "publisher",
                    "series",
                )
            ):
                return {"events": payload}

            if "scene_number" in first:
                return {"scenes": payload}

        return {"items": payload}

    raise ComicFactoryError(
        "Beklenmeyen Gemini JSON yapısı."
    )


def is_retryable_gemini_error(
    error: Exception,
) -> bool:
    """Geçici Gemini API hatalarını ayırt eder."""
    message = str(error).casefold()

    retryable_terms = (
        "503",
        "429",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "service unavailable",
    )

    return any(
        term in message
        for term in retryable_terms
    )


def gemini_with_retry(
    operation: Any,
    *,
    operation_name: str,
    max_attempts: int = 6,
) -> Any:
    """Geçici API problemlerinde exponential backoff uygular."""
    delays = (
        10,
        20,
        40,
        60,
        90,
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
                    f"{attempt}/{max_attempts}. API denemesi"
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

            delay = delays[
                min(
                    attempt - 1,
                    len(delays) - 1,
                )
            ]

            delay += random.uniform(
                0.0,
                3.0,
            )

            print(
                f"! {operation_name}: geçici servis hatası."
            )

            print(
                f"  {type(error).__name__}: {error}"
            )

            print(
                f"  {delay:.1f}s sonra tekrar denenecek."
            )

            time.sleep(delay)

    raise ComicFactoryError(
        f"{operation_name} API hizmeti kullanılamıyor: "
        f"{last_error}"
    )


def ask_gemini_json(
    client: genai.Client,
    prompt: str,
    *,
    temperature: float,
    operation_name: str,
) -> dict[str, Any]:
    """Gemini'den JSON yanıt alır."""

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
            f"{operation_name} boş cevap verdi."
        )

    return parse_json_response(
        response.text
    )


def search_web(
    queries: list[str],
    each: int,
) -> list[dict[str, str]]:
    """Ücretsiz web araştırması yapar."""
    results_out: list[dict[str, str]] = []
    seen: set[str] = set()

    for query in queries:
        try:
            results = DDGS(
                timeout=15
            ).text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=each,
            )

        except Exception as error:
            print(
                f"! Search skipped: {query}: {error}"
            )
            continue

        for item in results or []:
            url = clean(
                item.get("href")
                or item.get("url")
            )

            if (
                not url.startswith("http")
                or url in seen
            ):
                continue

            seen.add(url)

            results_out.append(
                {
                    "title": clean(
                        item.get("title")
                    ),
                    "url": url,
                    "body": clean(
                        item.get("body")
                    ),
                }
            )

    return results_out


def event_key(
    event: dict[str, Any],
) -> str:
    """Event benzersiz anahtarını oluşturur."""
    return "|".join(
        clean(
            event.get(key)
        ).casefold()
        for key in (
            "publisher",
            "series",
            "issue",
            "event_title",
        )
    )


def used_event_keys() -> set[str]:
    """Daha önce başarıyla üretilmiş event anahtarlarını döndürür."""
    payload = load_json(
        USED_EVENTS_FILE,
        {"events": []},
    )

    if not isinstance(payload, dict):
        return set()

    output: set[str] = set()

    for item in payload.get(
        "events",
        [],
    ):
        if not isinstance(item, dict):
            continue

        key = clean(
            item.get("event_key")
        )

        if key:
            output.add(key)

    return output


def rejected_event_keys() -> set[str]:
    """Bu run veya önceki runlarda kalite nedeniyle reddedilmiş eventleri döndürür."""
    payload = load_json(
        REJECTED_EVENTS_FILE,
        {"events": []},
    )

    if not isinstance(payload, dict):
        return set()

    return {
        clean(item.get("event_key"))
        for item in payload.get("events", [])
        if isinstance(item, dict)
        and clean(item.get("event_key"))
    }


def record_rejected_event(
    event: dict[str, Any],
    reason: str,
) -> None:
    """Kaliteye ulaşamayan eventi kaydeder."""
    payload = load_json(
        REJECTED_EVENTS_FILE,
        {"events": []},
    )

    if not isinstance(payload, dict):
        payload = {"events": []}

    events = payload.setdefault(
        "events",
        [],
    )

    key = event_key(event)

    events.append(
        {
            "event_key": key,
            "event_title": clean(
                event.get("event_title")
            ),
            "series": clean(
                event.get("series")
            ),
            "issue": clean(
                event.get("issue")
            ),
            "reason": clean(reason),
            "rejected_at": datetime.now(
                TZ
            ).isoformat(),
        }
    )

    payload["events"] = events[-100:]

    save_json(
        REJECTED_EVENTS_FILE,
        payload,
    )


def research_events(
    client: genai.Client,
    count: int,
    run_rejected: set[str],
) -> list[dict[str, Any]]:
    """Shorts için güçlü ve görsel olarak zengin yeni comic olayları seçer."""
    print()
    print("=" * 78)
    print("EVENT RESEARCH")
    print("=" * 78)

    web_results = search_web(
        [
            "Marvel comics shocking moments specific issue review",
            "DC comics shocking moment specific issue review",
            "Marvel specific issue huge battle comic review panels",
            "DC specific issue transformation comic review panels",
            "comic books best shocking reveals specific issue",
            "Image Comics shocking issue review panels",
            "Dark Horse comic shocking issue review",
            "site:marvel.com comics preview issue",
            "site:dc.com comics preview issue",
        ],
        each=8,
    )

    if not web_results:
        raise ComicFactoryError(
            "Event research için web sonucu bulunamadı."
        )

    evidence = "\n\n".join(
        (
            f"TITLE: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"SNIPPET: {item['body']}"
        )
        for item in web_results[:55]
    )

    forbidden = (
        used_event_keys()
        | rejected_event_keys()
        | run_rejected
    )

    forbidden_text = (
        "\n".join(
            f"- {key}"
            for key in sorted(forbidden)
        )
        or "Yok."
    )

    prompt = f"""
Premium comic Shorts research director'sın.

WEB EVIDENCE:
{evidence}

KULLANILMAYACAK EVENTLER:
{forbidden_text}

ÖNCEKİ KALİTE DERSLERİ:
{load_quality_lessons()}

TAM {max(5, count)} event adayı üret.

Aday seçerken en önemli kriter:
Bu event 14 ayrı, anlamlı ve kaliteli görsel sahneye bölünebilmeli.

Her event:
- tek spesifik comic olayı olsun
- series / issue / yıl net olsun
- 45-60 saniyede hook → escalation → payoff taşısın
- gerçek comic paneli bulma ihtimali yüksek olsun
- sadece tek görselle anlatılabilecek basit bir olay olmasın
- en az 10-14 farklı görsel moment sunsun

quality_score:
hook 25
story/payoff 25
visual variety 25
source confidence 15
Shorts retention 10

92+ yalnız gerçekten güçlü adaylara ver.

URL uydurma.
Kaynak URL yalnız WEB EVIDENCE içinden gelsin.

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
      "visual_moment_count": 14,
      "quality_score": 95,
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
        temperature=0.30,
        operation_name="Event Research Director",
    )

    events = payload.get(
        "events",
        [],
    )

    if not isinstance(events, list):
        return []

    eligible: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        key = event_key(event)

        if key in forbidden:
            continue

        score = float(
            event.get(
                "quality_score",
                0,
            )
            or 0
        )

        visual_count = int(
            event.get(
                "visual_moment_count",
                0,
            )
            or 0
        )

        if score < 88:
            continue

        if visual_count < 10:
            continue

        eligible.append(event)

    eligible.sort(
        key=lambda event: (
            float(
                event.get(
                    "quality_score",
                    0,
                )
                or 0
            ),
            int(
                event.get(
                    "visual_moment_count",
                    0,
                )
                or 0
            ),
        ),
        reverse=True,
    )

    save_json(
        RESEARCH_DIR
        / (
            "research_v4_"
            + datetime.now(TZ).strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        ),
        {
            "events": events,
            "eligible": eligible,
        },
    )

    return eligible


def activate_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    """Eventi aktif pipeline eventine dönüştürür."""
    active = dict(event)

    active["id"] = slug(
        f"{clean(event.get('series'))}_"
        f"{clean(event.get('issue'))}_"
        f"{clean(event.get('event_title'))}"
    )

    active["event_key"] = event_key(
        active
    )

    active["activated_at"] = datetime.now(
        TZ
    ).isoformat()

    save_json(
        ACTIVE_EVENT_FILE,
        active,
    )

    CHECKPOINTS.set_event(
        active["id"]
    )

    print()
    print(
        "✓ EVENT: "
        + clean(
            active.get("event_title")
        )
    )
    print(
        f"  {clean(active.get('series'))} "
        f"{clean(active.get('issue'))}"
    )

    return active


def load_active_event() -> dict[str, Any]:
    """Aktif eventi diskten yükler."""
    event = load_json(
        ACTIVE_EVENT_FILE
    )

    if (
        not isinstance(event, dict)
        or not clean(
            event.get("id")
        )
    ):
        raise ComicFactoryError(
            "active_event.json bulunamadı."
        )

    CHECKPOINTS.set_event(
        clean(event["id"])
    )

    return event


def generate_storyboard(
    client: genai.Client,
    event: dict[str, Any],
    feedback: str,
) -> dict[str, Any]:
    """Tek story repair turu üretir."""
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
        if isinstance(item, dict)
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

CHARACTERS:
{", ".join(event.get("characters", []))}

HOOK:
{clean(event.get("hook"))}

SUMMARY:
{clean(event.get("event_summary"))}

FEAT:
{clean(event.get("power_feat"))}

SOURCES:
{sources}

QUALITY LESSONS:
{load_quality_lessons()}

PREVIOUS REPAIR FEEDBACK:
{feedback or "İlk story üretimi."}

KRİTİK:
TAM {SCENE_COUNT} SAHNE.
Ne 13 ne 15.

scene_number:
1,2,3,4,5,6,7,8,9,10,11,12,13,14

Story:
- 120-150 Türkçe kelime
- ilk iki sahne çok güçlü hook
- her sahne yalnız bir ana görsel olayı anlatsın
- her sahne yeni bilgi veya escalation getirsin
- son dört sahne payoff
- Wikipedia özeti gibi olmasın
- bilgi uydurma
- narration ile visual_description birebir aynı olayı anlatsın
- o narration okunurken başka olayın paneli gerekmesin
- 14 sahnenin görsel fikirleri farklı olsun
- özel isimler orijinal yazılsın
- Thor -> Tor gibi fonetik yazım yapma

story_role:
hook
context
escalation
twist
payoff
cta

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
        temperature=0.42,
        operation_name="Story Director",
    )

    scenes = storyboard.get(
        "scenes",
        [],
    )

    if not isinstance(scenes, list):
        storyboard["_structure_valid"] = False
        storyboard["_structure_error"] = (
            "scenes alanı liste değil."
        )
        storyboard["_scene_count"] = 0
        return storyboard

    numbers: list[int] = []

    for scene in scenes:
        if not isinstance(scene, dict):
            continue

        try:
            numbers.append(
                int(
                    scene.get(
                        "scene_number",
                        0,
                    )
                )
            )

        except (TypeError, ValueError):
            numbers.append(0)

    expected = list(
        range(
            1,
            SCENE_COUNT + 1,
        )
    )

    structure_valid = (
        len(scenes) == SCENE_COUNT
        and numbers == expected
    )

    storyboard["_structure_valid"] = structure_valid
    storyboard["_scene_count"] = len(scenes)

    if not structure_valid:
        storyboard["_structure_error"] = (
            f"TAM {SCENE_COUNT} sahne gerekli. "
            f"Gelen={len(scenes)}. "
            f"Scene numbers={numbers}."
        )

        return storyboard

    valid_roles = {
        "hook",
        "context",
        "escalation",
        "twist",
        "payoff",
        "cta",
    }

    narration_parts: list[str] = []

    for scene in scenes:
        narration = clean(
            scene.get("narration")
        )

        visual = clean(
            scene.get(
                "visual_description"
            )
        )

        role = clean(
            scene.get(
                "story_role"
            )
        )

        if (
            not narration
            or not visual
            or role not in valid_roles
        ):
            storyboard["_structure_valid"] = False
            storyboard["_structure_error"] = (
                "Bir sahnede narration, visual_description "
                "veya geçerli story_role eksik."
            )

            return storyboard

        narration_parts.append(
            narration
        )

    storyboard["narration"] = " ".join(
        narration_parts
    )

    return storyboard


def score_storyboard(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> dict[str, Any]:
    """Storyboard kalitesini sert eşiklerle değerlendirir."""
    prompt = f"""
Premium Shorts Supervising Story Director'sın.

EVENT:
{clean(event.get("event_title"))}

STORYBOARD:
{json.dumps(
    storyboard.get("scenes", []),
    ensure_ascii=False,
    indent=2,
)}

0-100 puanla:
hook
clarity
escalation
payoff
factual_discipline
visual_storytelling
narration_visual_match
retention
overall

ÇOK KATI PUANLAMA:

94+ overall:
yalnız gerçekten yayınlanabilir premium story.

narration_visual_match:
Bir narration sırasında hangi comic panelinin
gösterileceği açık ve tek anlamlı olmalı.

Bir sahne iki ayrı büyük olayı anlatıyorsa puan kır.

Aynı bilgi veya aynı görsel fikir tekrar ediyorsa puan kır.

Story yalnız bilgi listesi gibi ilerliyorsa ciddi puan kır.

Factual karışıklık varsa ciddi puan kır.

Son dört sahnede gerçek payoff yoksa 94+ verme.

SADECE JSON:
{{
  "hook": 0,
  "clarity": 0,
  "escalation": 0,
  "payoff": 0,
  "factual_discipline": 0,
  "visual_storytelling": 0,
  "narration_visual_match": 0,
  "retention": 0,
  "overall": 0,
  "problems": ["..."],
  "revision_instruction": "..."
}}
"""

    result = ask_gemini_json(
        client,
        prompt,
        temperature=0.08,
        operation_name="Story Quality Check",
    )

    for key in (
        "hook",
        "clarity",
        "escalation",
        "payoff",
        "factual_discipline",
        "visual_storytelling",
        "narration_visual_match",
        "retention",
        "overall",
    ):
        try:
            score = float(
                result.get(
                    key,
                    0,
                )
                or 0
            )

        except (TypeError, ValueError):
            score = 0.0

        result[key] = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

    return result


def build_quality_storyboard(
    client: genai.Client,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Story yeterli değilse eventi reddedip üst pipeline'a geri gönderir."""
    print()
    print("=" * 78)
    print("STORY QUALITY GATE")
    print("=" * 78)

    feedback = ""
    best_score = 0.0
    best_problems = ""

    for cycle in range(
        1,
        STORY_REPAIR_CYCLES_PER_EVENT + 1,
    ):
        print(
            f"\n→ Story repair cycle {cycle}/"
            f"{STORY_REPAIR_CYCLES_PER_EVENT}"
        )

        storyboard = generate_storyboard(
            client,
            event,
            feedback,
        )

        if not bool(
            storyboard.get(
                "_structure_valid",
                False,
            )
        ):
            error = clean(
                storyboard.get(
                    "_structure_error"
                )
            )

            CHECKPOINTS.record(
                name="story_structure",
                score=0.0,
                threshold=100.0,
                cycle=cycle,
                details=error,
                passed=False,
            )

            feedback = (
                f"Önceki cevap yapısal olarak hatalıydı. "
                f"{error} "
                f"TAM {SCENE_COUNT} sahne üret. "
                "Scene sayılarını düzeltirken story kalitesini düşürme."
            )

            continue

        CHECKPOINTS.record(
            name="story_structure",
            score=100.0,
            threshold=100.0,
            cycle=cycle,
            details=(
                f"{SCENE_COUNT}/{SCENE_COUNT} sahne."
            ),
            passed=True,
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
            clean(item)
            for item in review.get(
                "problems",
                [],
            )
            if clean(item)
        )

        if score > best_score:
            best_score = score
            best_problems = problems

        passed = CHECKPOINTS.record(
            name="story_quality",
            score=score,
            threshold=STORY_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        if passed:
            storyboard.pop(
                "_structure_valid",
                None,
            )

            storyboard.pop(
                "_structure_error",
                None,
            )

            storyboard.pop(
                "_scene_count",
                None,
            )

            save_json(
                SCRIPT_DIR
                / "storyboard_v4.json",
                storyboard,
            )

            return storyboard

        feedback = (
            f"Önceki overall={score:.1f}/100. "
            f"Minimum={STORY_THRESHOLD:.1f}. "
            f"Problems: {problems}. "
            f"Repair instruction: "
            f"{clean(review.get('revision_instruction'))}. "
            f"TAM {SCENE_COUNT} sahneyi koru. "
            "İyi sahneleri bozma. "
            "Düşük puanlı story, factual ve "
            "narration-visual bölümlerini düzelt."
        )

    raise EventRejectedError(
        "Story bu event için yeterince güçlü değil. "
        f"Best={best_score:.1f}/100. "
        f"{best_problems}"
    )


def create_voice_direction(
    client: genai.Client,
    narration: str,
) -> dict[str, Any]:
    """Kilitli Gacrux voice delivery talimatı üretir."""
    prompt = f"""
Türkçe premium YouTube Shorts Voice Director'sın.

TRANSCRIPT:
{narration}

KELİMELERİ DEĞİŞTİRME.

Kilitli voice karakteri:
- doğal erkek storyteller
- arkadaşına inanılmaz comic hikayesi anlatır gibi
- yapay TikTok sesi değil
- haber spikeri değil
- ağır belgesel değil
- enerjik ama bağırmayan
- İngilizce özel isimleri doğal İngilizce telaffuz et
- sonra doğal Türkçeye dön
- cümle sonlarını aynı melodiyle bitirme
- reveal öncesi kısa doğal pause
- doğal nefes ve konuşma ritmi
- yaklaşık 1.0x tempo

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
        temperature=0.18,
        operation_name="Locked Voice Director",
    )


def write_pcm_wave(
    path: Path,
    pcm: bytes,
) -> None:
    """24kHz mono PCM sesini WAV dosyasına yazar."""
    with wave.open(
        str(path),
        "wb",
    ) as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(pcm)


def generate_gacrux_voice(
    client: genai.Client,
    narration: str,
    cycle: int,
) -> Path:
    """Kilitli Gacrux sesi üretir."""
    direction = create_voice_direction(
        client,
        narration,
    )

    prompt = f"""
Read ONLY the transcript inside <TRANSCRIPT>.

VOICE:
{clean(direction.get("style_instruction"))}

PRONUNCIATION:
{clean(direction.get("pronunciation_instruction"))}

PACE:
{clean(direction.get("pace_instruction"))}

Critical:
The surrounding language is Turkish.
English proper nouns must sound naturally English.
After the proper noun, return naturally to Turkish.

Do not:
add words
remove words
paraphrase
read instructions
sound synthetic
sound like an announcer

<TRANSCRIPT>
{narration}
</TRANSCRIPT>
"""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
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
            "Gemini Gacrux audio döndürmedi."
        ) from error

    if not pcm:
        raise ComicFactoryError(
            "Gemini Gacrux boş audio döndürdü."
        )

    output = (
        AUDIO_DIR
        / f"gacrux_cycle_{cycle:02d}.wav"
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
    """SDK nesnesi veya dict alanını okur."""
    if isinstance(value, dict):
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
    client = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with audio_path.open(
        "rb"
    ) as audio:
        transcription = (
            client.audio.transcriptions.create(
                file=audio,
                model=GROQ_MODEL,
                language="tr",
                response_format="verbose_json",
                timestamp_granularities=["word"],
                prompt=narration[:700],
                temperature=0,
            )
        )

    words: list[dict[str, Any]] = []

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
                    "start": float(start),
                    "end": float(end),
                }
            )

    if len(words) < 20:
        raise ComicFactoryError(
            "Groq yeterli word timestamp üretmedi."
        )

    return words


def normalize_alignment_word(
    value: str,
) -> str:
    """Alignment için kelimeyi normalize eder."""
    return re.sub(
        r"[^\wçğıöşü'-]",
        "",
        clean(value).casefold(),
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
        if clean(token)
    ]


def word_similarity(
    first: str,
    second: str,
) -> float:
    """Kelime benzerlik skorunu hesaplar."""
    first = normalize_alignment_word(first)
    second = normalize_alignment_word(second)

    if not first or not second:
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
    """Gerçek narration kelimelerini ses timestamp'lerine hizalar."""
    script_words = narration_tokens(
        narration
    )

    if not script_words:
        raise ComicFactoryError(
            "Narration boş."
        )

    aligned: list[dict[str, Any]] = []

    whisper_index = 0
    previous_end = 0.0

    strong_matches = 0
    similarity_total = 0.0

    for script_index, script_word in enumerate(
        script_words
    ):
        best_index: int | None = None
        best_score = 0.0

        search_end = min(
            len(whisper_words),
            whisper_index + 9,
        )

        for candidate_index in range(
            whisper_index,
            search_end,
        ):
            candidate_word = clean(
                whisper_words[
                    candidate_index
                ].get(
                    "word"
                )
            )

            score = word_similarity(
                script_word,
                candidate_word,
            )

            score -= (
                candidate_index
                - whisper_index
            ) * 0.018

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
                match["start"]
            )

            end = float(
                match["end"]
            )

            whisper_index = min(
                len(whisper_words),
                best_index + 1,
            )

            similarity_total += max(
                0.0,
                best_score,
            )

            if best_score >= 0.72:
                strong_matches += 1

        elif whisper_index < len(
            whisper_words
        ):
            reference = whisper_words[
                whisper_index
            ]

            start = max(
                previous_end,
                float(reference["start"]),
            )

            duration = max(
                0.10,
                min(
                    0.42,
                    float(reference["end"])
                    - float(reference["start"]),
                ),
            )

            end = start + duration

        else:
            last_audio_end = float(
                whisper_words[-1]["end"]
            )

            remaining = max(
                1,
                len(script_words)
                - script_index,
            )

            duration = max(
                0.10,
                (
                    last_audio_end
                    - previous_end
                )
                / remaining,
            )

            start = previous_end
            end = start + duration

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
        strong_matches
        / max(
            1,
            len(script_words),
        )
        * 100.0
    )

    similarity = (
        similarity_total
        / max(
            1,
            len(script_words),
        )
        * 100.0
    )

    score = (
        coverage * 0.72
        + similarity * 0.28
    )

    return (
        aligned,
        {
            "coverage": coverage,
            "similarity": similarity,
            "score": score,
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
    """Kilitli Gacrux sesi alignment geçene kadar tekrar üretir."""
    print()
    print("=" * 78)
    print("AUDIO ALIGNMENT GATE")
    print("=" * 78)

    best_score = 0.0

    for cycle in range(
        1,
        AUDIO_REPAIR_CYCLES + 1,
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
            metrics["score"]
        )

        best_score = max(
            best_score,
            score,
        )

        passed = CHECKPOINTS.record(
            name="audio_alignment",
            score=score,
            threshold=AUDIO_ALIGNMENT_THRESHOLD,
            cycle=cycle,
            details=(
                f"coverage={metrics['coverage']:.2f}% | "
                f"similarity={metrics['similarity']:.2f}%"
            ),
        )

        if not passed:
            continue

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

    raise EventRejectedError(
        "Bu eventin narration/audio kombinasyonu "
        "istenen alignment seviyesine ulaşamadı. "
        f"Best={best_score:.1f}."
    )


def ffmpeg_path() -> str:
    """FFmpeg executable yolunu döndürür."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(
    command: list[str],
    message: str,
) -> None:
    """FFmpeg komutunu çalıştırır."""
    process = subprocess.run(
        command,
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
            + process.stderr[-4000:]
        )


def media_duration(
    ffmpeg: str,
    path: Path,
) -> float:
    """Media duration değerini okur."""
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
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
            f"Media duration okunamadı: {path}"
        )

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + float(match.group(3))
    )


def build_scene_timeline(
    storyboard: dict[str, Any],
    aligned_words: list[dict[str, Any]],
    audio_duration_seconds: float,
) -> list[tuple[float, float]]:
    """Her sahneyi narration'daki gerçek audio kelime sınırlarına bağlar."""
    scenes = storyboard["scenes"]

    counts = [
        len(
            narration_tokens(
                clean(
                    scene.get("narration")
                )
            )
        )
        for scene in scenes
    ]

    expected_words = sum(counts)

    if abs(
        expected_words
        - len(aligned_words)
    ) > 2:
        raise EventRejectedError(
            "Scene narration kelimeleri ile audio alignment "
            "kelimeleri uyuşmadı."
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
        if count <= 0:
            raise EventRejectedError(
                "Boş scene narration bulundu."
            )

        start_index = min(
            cursor,
            len(aligned_words) - 1,
        )

        end_index = min(
            cursor + count - 1,
            len(aligned_words) - 1,
        )

        if scene_index == 0:
            start = 0.0

        else:
            previous_end = float(
                aligned_words[
                    start_index - 1
                ]["end"]
            )

            current_start = float(
                aligned_words[
                    start_index
                ]["start"]
            )

            start = (
                previous_end
                + current_start
            ) / 2.0

        if scene_index == len(scenes) - 1:
            end = audio_duration_seconds

        else:
            current_end = float(
                aligned_words[
                    end_index
                ]["end"]
            )

            next_index = min(
                end_index + 1,
                len(aligned_words) - 1,
            )

            next_start = float(
                aligned_words[
                    next_index
                ]["start"]
            )

            end = (
                current_end
                + next_start
            ) / 2.0

        end = max(
            start + 0.20,
            end,
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
    each: int,
) -> list[dict[str, str]]:
    """Comic image aramalarını normalize eder."""
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for query in queries:
        try:
            results = DDGS(
                timeout=15
            ).images(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=each,
            )

        except Exception as error:
            print(
                f"! Image search skipped: {error}"
            )
            continue

        for item in results or []:
            image_url = clean(
                item.get("image")
            )

            if (
                not image_url.startswith("http")
                or image_url in seen
            ):
                continue

            seen.add(image_url)

            output.append(
                {
                    "image_url": image_url,
                    "source_page": clean(
                        item.get("url")
                    ),
                    "title": clean(
                        item.get("title")
                    ),
                }
            )

    return output


def global_image_search(
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    """Event için geniş gerçek comic görsel havuzu arar."""
    series = clean(
        event.get("series")
    )

    issue = clean(
        event.get("issue")
    )

    title = clean(
        event.get("event_title")
    )

    queries = [
        f'"{series}" "{issue}" comic panels',
        f'"{series}" "{issue}" comic pages',
        f'"{series}" "{issue}" preview images',
        f'"{series}" "{issue}" review panels',
        f'"{title}" comic panels',
        f'"{title}" comic page',
    ]

    for scene in storyboard["scenes"]:
        visual = clean(
            scene.get(
                "visual_description"
            )
        )

        queries.append(
            f'"{series}" "{issue}" '
            f'{visual[:90]}'
        )

    return search_image_queries(
        queries,
        each=12,
    )


def average_hash(
    image: Image.Image,
) -> str:
    """64-bit average hash oluşturur."""
    tiny = image.convert(
        "L"
    ).resize(
        (8, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = list(
        tiny.getdata()
    )

    average = sum(pixels) / len(pixels)

    bits = "".join(
        "1"
        if pixel >= average
        else "0"
        for pixel in pixels
    )

    return f"{int(bits, 2):016x}"


def hash_distance(
    first: str,
    second: str,
) -> int:
    """İki perceptual hash arasındaki mesafeyi hesaplar."""
    return bin(
        int(first, 16)
        ^ int(second, 16)
    ).count("1")


def source_image_quality(
    image: Image.Image,
) -> float:
    """Kaynak görsel teknik kalite skorunu hesaplar."""
    width, height = image.size

    megapixels = (
        width
        * height
        / 1_000_000
    )

    minimum_side = min(
        width,
        height,
    )

    entropy = image.convert(
        "L"
    ).entropy()

    resolution_score = min(
        50.0,
        megapixels
        / 1.4
        * 50.0,
    )

    dimension_score = min(
        30.0,
        minimum_side
        / 900.0
        * 30.0,
    )

    entropy_score = min(
        20.0,
        max(
            0.0,
            (
                entropy - 3.0
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
        / event["id"]
        / directory_name
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = list(
        known_hashes or []
    )

    output: list[ImageCandidate] = []

    for item in results:
        if len(output) >= max_candidates:
            break

        try:
            response = session.get(
                item["image_url"],
                headers={
                    "Referer": item[
                        "source_page"
                    ],
                },
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

        except requests.RequestException:
            continue

        if len(response.content) < 18_000:
            continue

        try:
            with Image.open(
                io.BytesIO(
                    response.content
                )
            ) as opened:
                image = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

        except Exception:
            continue

        width, height = image.size

        if (
            width < 500
            or height < 500
            or width * height < 450_000
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

        hashes.append(phash)

        candidate_id = (
            f"c{start_index + len(output):04d}"
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

        output.append(
            ImageCandidate(
                candidate_id=candidate_id,
                local_file=str(local_file),
                source_page=item[
                    "source_page"
                ],
                image_url=item[
                    "image_url"
                ],
                title=item["title"],
                width=width,
                height=height,
                quality_score=round(
                    source_image_quality(
                        image
                    ),
                    2,
                ),
                perceptual_hash=phash,
            )
        )

    return output


def thumbnail_bytes(
    path: Path,
) -> bytes:
    """Gemini Vision için düşük maliyetli thumbnail üretir."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

    image.thumbnail(
        (640, 640),
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
    """Crop koordinatlarını normalize eder."""
    if (
        not isinstance(crop_box, list)
        or len(crop_box) != 4
    ):
        return [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

    try:
        left, top, right, bottom = [
            float(value)
            for value in crop_box
        ]

    except (
        TypeError,
        ValueError,
    ):
        return [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

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
    """Bir sahne için gerçek panelleri relevance'a göre sıralar."""
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.quality_score
        ),
        reverse=True,
    )[:MAX_VISION_IMAGES]

    if not candidates:
        return []

    prompt = f"""
Premium comic Visual Director'sın.

EVENT:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("series"))}
{clean(event.get("issue"))}

NARRATION:
{clean(scene_data.get("narration"))}

O SIRADA EKRANDA GÖRÜLMESİ GEREKEN:
{clean(scene_data.get("visual_description"))}

Aday görselleri tek tek incele.

En iyi 6 adayı sırala.

relevance_score:
100 = tam anlatılan olay
95 = aynı spesifik aksiyon açıkça görülüyor
85 = doğru karakterler ama yanlış/eksik an
70 = yalnız konu benziyor
50 = zayıf
0 = alakasız

95+ kolay verme.

Crop:
Doğru comic panelini seç.
Ana karakter/aksiyon crop içinde kalmalı.

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

    contents: list[Any] = [prompt]

    for candidate in candidates:
        contents.append(
            (
                f"CANDIDATE {candidate.candidate_id}\n"
                f"TITLE: {candidate.title}\n"
                f"SIZE: "
                f"{candidate.width}x{candidate.height}\n"
                f"QUALITY: "
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
                temperature=0.04,
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
        response.text or "{}"
    )

    raw_items = payload.get(
        "ranked_visuals",
        payload.get(
            "items",
            [],
        ),
    )

    ranked: list[RankedVisual] = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        candidate_id = clean(
            item.get("candidate_id")
        )

        if not candidate_id:
            continue

        ranked.append(
            RankedVisual(
                candidate_id=candidate_id,
                relevance_score=int(
                    item.get(
                        "relevance_score",
                        0,
                    )
                    or 0
                ),
                crop_box=normalize_crop_box(
                    item.get("crop_box")
                ),
                reason=clean(
                    item.get("reason")
                ),
            )
        )

    return ranked


def restore_crop(
    candidate: ImageCandidate,
    crop_box: list[float],
    output: Path,
) -> tuple[Path, float]:
    """Comic panelini değiştirmeden crop/restoration uygular."""
    with Image.open(
        candidate.local_file
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

    left, top, right, bottom = crop_box

    x1 = round(
        left * image.width
    )

    y1 = round(
        top * image.height
    )

    x2 = round(
        right * image.width
    )

    y2 = round(
        bottom * image.height
    )

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    original_width = max(
        1,
        x2 - x1,
    )

    original_height = max(
        1,
        y2 - y1,
    )

    megapixels = (
        original_width
        * original_height
        / 1_000_000
    )

    minimum_side = min(
        original_width,
        original_height,
    )

    entropy = crop.convert(
        "L"
    ).entropy()

    technical_quality = (
        min(
            50.0,
            megapixels
            / 1.20
            * 50.0,
        )
        + min(
            30.0,
            minimum_side
            / 800.0
            * 30.0,
        )
        + min(
            20.0,
            max(
                0.0,
                (
                    entropy - 3.0
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

    maximum_side = max(
        crop.size
    )

    if maximum_side < 2600:
        scale = min(
            2.4,
            2600
            / max(
                1,
                maximum_side,
            ),
        )

        crop = crop.resize(
            (
                round(
                    crop.width * scale
                ),
                round(
                    crop.height * scale
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    crop = ImageEnhance.Contrast(
        crop
    ).enhance(1.04)

    crop = ImageEnhance.Color(
        crop
    ).enhance(1.015)

    crop = crop.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
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
    """Final sahne panelini narration, kalite ve composition açısından inceler."""
    prompt = f"""
Premium comic Shorts Scene QC.

NARRATION:
{clean(scene_data.get("narration"))}

EXPECTED VISUAL:
{clean(scene_data.get("visual_description"))}

Final görseli değerlendir.

visual_match 0-100:
Narration'daki spesifik olay gerçekten ekranda mı?

image_quality 0-100:
Netlik, çözünürlük hissi, bozulma, artefact.

composition 0-100:
Ana olay okunuyor mu?
Karakter/aksiyon crop dışında mı?
9:16 videoda kullanışlı mı?

SOURCE TECHNICAL QUALITY:
{technical_quality:.1f}

95+ visual_match yalnız tam olay görünüyorsa ver.

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
                temperature=0.03,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name=(
            "Scene Visual QC "
            f"{scene_data['scene_number']}"
        ),
    )

    return parse_json_response(
        response.text or "{}"
    )


def scene_specific_search(
    event: dict[str, Any],
    scene_data: dict[str, Any],
    feedback: str,
) -> list[dict[str, str]]:
    """Başarısız sahne için daha spesifik panel araştırması yapar."""
    series = clean(
        event.get("series")
    )

    issue = clean(
        event.get("issue")
    )

    narration = clean(
        scene_data.get("narration")
    )

    visual = clean(
        scene_data.get(
            "visual_description"
        )
    )

    queries = [
        f'"{series}" "{issue}" {visual[:100]}',
        f'"{series}" "{issue}" {narration[:95]} panel',
        f'"{series}" "{issue}" comic page preview',
        f'"{series}" "{issue}" review scan',
    ]

    if feedback:
        queries.append(
            f'"{series}" "{issue}" '
            f'{clean(feedback)[:90]}'
        )

    return search_image_queries(
        queries,
        each=10,
    )


def generate_reconstruction(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    references: list[Path],
    output: Path,
    feedback: str,
) -> Path:
    """Gerçek panel bulunamadığında referanslı AI reconstruction üretir."""
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

PREVIOUS QC FEEDBACK:
{feedback or "First reconstruction."}

Use references only for:
character appearance
costume
era
setting
color language

Create the exact missing event.

Requirements:
premium comic artwork
high detail
cinematic depth
correct anatomy
clear action
9:16 composition
no text
no speech bubble
no watermark
no logo
no issue number
no UI

Correct every problem from PREVIOUS QC FEEDBACK.
Do not copy the reference composition exactly.
"""

    inputs: list[dict[str, str]] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for reference in references[:3]:
        mime_type = (
            "image/png"
            if reference.suffix.lower()
            == ".png"
            else "image/jpeg"
        )

        inputs.append(
            {
                "type": "image",
                "data": base64.b64encode(
                    reference.read_bytes()
                ).decode("ascii"),
                "mime_type": mime_type,
            }
        )

    def request() -> Any:
        return client.interactions.create(
            model=GEMINI_IMAGE_MODEL,
            input=inputs,
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


def make_scene(
    *,
    scene_data: dict[str, Any],
    rankings: list[RankedVisual],
    selected_candidate_id: str,
    relevance_score: int,
    crop_box: list[float],
    visual_source: str,
    visual_file: Path,
    visual_match: float,
    image_quality: float,
    composition: float,
) -> Scene:
    """Final Scene nesnesi oluşturur."""
    return Scene(
        scene_number=int(
            scene_data["scene_number"]
        ),
        narration=clean(
            scene_data.get("narration")
        ),
        visual_description=clean(
            scene_data.get(
                "visual_description"
            )
        ),
        story_role=clean(
            scene_data.get("story_role")
        ),
        emphasis_words=[
            clean(word)
            for word in scene_data.get(
                "emphasis_words",
                [],
            )
            if clean(word)
        ],
        ranked_visuals=rankings,
        selected_candidate_id=(
            selected_candidate_id
        ),
        relevance_score=(
            relevance_score
        ),
        crop_box=crop_box,
        visual_source=visual_source,
        visual_file=str(
            visual_file
        ),
        visual_match_score=(
            visual_match
        ),
        image_quality_score=(
            image_quality
        ),
        composition_score=(
            composition
        ),
    )


def resolve_scene_until_pass(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    global_candidates: list[ImageCandidate],
    used_ids: set[str],
    asset_directory: Path,
) -> Scene:
    """Sahne visual checkpointleri geçene kadar farklı çözüm dener."""
    number = int(
        scene_data["scene_number"]
    )

    candidates = list(
        global_candidates
    )

    candidate_map = {
        item.candidate_id: item
        for item in candidates
    }

    known_hashes = [
        item.perceptual_hash
        for item in candidates
    ]

    rankings = rank_scene_candidates(
        client,
        event,
        scene_data,
        candidates,
    )

    attempted: set[str] = set()
    feedback = ""

    for cycle in range(
        1,
        VISUAL_REPAIR_CYCLES + 1,
    ):
        print(
            f"\n→ Scene {number:02d} "
            f"real visual cycle {cycle}/"
            f"{VISUAL_REPAIR_CYCLES}"
        )

        selection: RankedVisual | None = None

        for option in rankings:
            if option.candidate_id in attempted:
                continue

            if (
                option.candidate_id in used_ids
                and option.relevance_score < 98
            ):
                continue

            if option.candidate_id not in candidate_map:
                continue

            selection = option
            break

        if selection is not None:
            attempted.add(
                selection.candidate_id
            )

            candidate = candidate_map[
                selection.candidate_id
            ]

            output = (
                asset_directory
                / (
                    f"scene_{number:02d}_"
                    f"{selection.candidate_id}_"
                    f"{cycle:02d}.png"
                )
            )

            visual, technical_quality = (
                restore_crop(
                    candidate,
                    selection.crop_box,
                    output,
                )
            )

            evaluation = evaluate_visual(
                client,
                scene_data,
                visual,
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
                technical_quality * 0.45
                + ai_quality * 0.55
            )

            composition = float(
                evaluation.get(
                    "composition",
                    0,
                )
                or 0
            )

            problems = "; ".join(
                clean(item)
                for item in evaluation.get(
                    "problems",
                    [],
                )
                if clean(item)
            )

            match_pass = CHECKPOINTS.record(
                name=(
                    f"scene_{number:02d}_"
                    "visual_match"
                ),
                score=visual_match,
                threshold=VISUAL_MATCH_THRESHOLD,
                cycle=cycle,
                details=problems,
            )

            quality_pass = CHECKPOINTS.record(
                name=(
                    f"scene_{number:02d}_"
                    "image_quality"
                ),
                score=image_quality,
                threshold=IMAGE_QUALITY_THRESHOLD,
                cycle=cycle,
                details=(
                    f"source={technical_quality:.1f}, "
                    f"vision={ai_quality:.1f}"
                ),
            )

            composition_pass = (
                CHECKPOINTS.record(
                    name=(
                        f"scene_{number:02d}_"
                        "composition"
                    ),
                    score=composition,
                    threshold=COMPOSITION_THRESHOLD,
                    cycle=cycle,
                    details=problems,
                )
            )

            if (
                match_pass
                and quality_pass
                and composition_pass
            ):
                used_ids.add(
                    selection.candidate_id
                )

                return make_scene(
                    scene_data=scene_data,
                    rankings=rankings,
                    selected_candidate_id=(
                        selection.candidate_id
                    ),
                    relevance_score=(
                        selection.relevance_score
                    ),
                    crop_box=(
                        selection.crop_box
                    ),
                    visual_source="real_comic",
                    visual_file=visual,
                    visual_match=visual_match,
                    image_quality=image_quality,
                    composition=composition,
                )

            feedback = clean(
                evaluation.get(
                    "repair_instruction"
                )
            ) or problems

        search_results = scene_specific_search(
            event,
            scene_data,
            feedback,
        )

        supplemental = download_candidates(
            event,
            search_results,
            max_candidates=MAX_SCENE_IMAGES,
            directory_name=(
                f"scene_{number:02d}_"
                f"search_{cycle:02d}"
            ),
            start_index=(
                1000
                + number * 100
                + cycle * 20
            ),
            known_hashes=known_hashes,
        )

        for candidate in supplemental:
            known_hashes.append(
                candidate.perceptual_hash
            )

            candidates.append(
                candidate
            )

            candidate_map[
                candidate.candidate_id
            ] = candidate

        if supplemental:
            new_rankings = (
                rank_scene_candidates(
                    client,
                    event,
                    scene_data,
                    supplemental
                    + global_candidates[:8],
                )
            )

            rankings = (
                new_rankings
                + rankings
            )

    if not ENABLE_AI_RECONSTRUCTION:
        raise EventRejectedError(
            f"Scene {number} için yeterli gerçek görsel bulunamadı."
        )

    references: list[Path] = []

    for option in rankings:
        candidate = candidate_map.get(
            option.candidate_id
        )

        if candidate is None:
            continue

        references.append(
            Path(
                candidate.local_file
            )
        )

        if len(references) >= 3:
            break

    if not references:
        references = [
            Path(
                candidate.local_file
            )
            for candidate in global_candidates[:3]
        ]

    feedback = feedback or (
        "Gerçek comic panelleri exact narration ile "
        "yeterince eşleşmedi."
    )

    for cycle in range(
        1,
        AI_RECONSTRUCTION_REPAIR_CYCLES + 1,
    ):
        print(
            f"🎨 Scene {number:02d} "
            f"AI reconstruction cycle {cycle}"
        )

        generated = (
            asset_directory
            / (
                f"scene_{number:02d}_"
                f"ai_{cycle:02d}.jpg"
            )
        )

        generate_reconstruction(
            client,
            event,
            scene_data,
            references,
            generated,
            feedback,
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
            clean(item)
            for item in evaluation.get(
                "problems",
                [],
            )
            if clean(item)
        )

        match_pass = CHECKPOINTS.record(
            name=(
                f"scene_{number:02d}_"
                "ai_visual_match"
            ),
            score=visual_match,
            threshold=VISUAL_MATCH_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        quality_pass = CHECKPOINTS.record(
            name=(
                f"scene_{number:02d}_"
                "ai_image_quality"
            ),
            score=image_quality,
            threshold=IMAGE_QUALITY_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        composition_pass = (
            CHECKPOINTS.record(
                name=(
                    f"scene_{number:02d}_"
                    "ai_composition"
                ),
                score=composition,
                threshold=COMPOSITION_THRESHOLD,
                cycle=cycle,
                details=problems,
            )
        )

        if (
            match_pass
            and quality_pass
            and composition_pass
        ):
            return make_scene(
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
                visual_source=(
                    "ai_reconstruction"
                ),
                visual_file=generated,
                visual_match=visual_match,
                image_quality=image_quality,
                composition=composition,
            )

        feedback = clean(
            evaluation.get(
                "repair_instruction"
            )
        ) or problems

    raise EventRejectedError(
        f"Scene {number} gerçek panel + AI reconstruction "
        "ile kalite checkpointlerini geçemedi."
    )


def build_quality_visuals(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    max_images: int,
) -> tuple[
    list[Scene],
    list[ImageCandidate],
]:
    """14 sahnenin tamamı için kalite kontrollü görseller üretir."""
    print()
    print("=" * 78)
    print("VISUAL QUALITY ENGINE")
    print("=" * 78)

    results = global_image_search(
        event,
        storyboard,
    )

    candidate_root = (
        CANDIDATE_DIR
        / event["id"]
    )

    shutil.rmtree(
        candidate_root,
        ignore_errors=True,
    )

    global_candidates = download_candidates(
        event,
        results,
        max_candidates=max(
            25,
            max_images,
        ),
        directory_name="global",
        start_index=1,
    )

    if len(global_candidates) < 10:
        raise EventRejectedError(
            "Bu event için yeterli kaliteli comic "
            "görsel havuzu bulunamadı."
        )

    asset_directory = (
        ASSET_DIR
        / event["id"]
    )

    shutil.rmtree(
        asset_directory,
        ignore_errors=True,
    )

    asset_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    used_ids: set[str] = set()

    scenes: list[Scene] = []

    for scene_data in storyboard[
        "scenes"
    ]:
        scene = resolve_scene_until_pass(
            client,
            event,
            scene_data,
            global_candidates,
            used_ids,
            asset_directory,
        )

        scenes.append(scene)

    return (
        scenes,
        global_candidates,
    )


def compose_vertical(
    visual_file: Path,
) -> Image.Image:
    """Paneli temiz 9:16 video kompozisyonuna dönüştürür."""
    with Image.open(
        visual_file
    ) as opened:
        source = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

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
    ).enhance(0.28)

    background = ImageEnhance.Color(
        background
    ).enhance(0.78)

    foreground = source.copy()

    ratio = (
        source.width
        / max(
            1,
            source.height,
        )
    )

    if 0.48 <= ratio <= 0.70:
        foreground.thumbnail(
            (
                WIDTH - 10,
                HEIGHT - 90,
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
    """Final Visual Director için contact sheet üretir."""
    columns = 4
    cell_width = 270
    cell_height = 480

    rows = math.ceil(
        len(scenes)
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
            index % columns
        ) * cell_width

        y = (
            index // columns
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
    """Bütün videonun görsel akışını kontrol eder."""
    contact_sheet = build_contact_sheet(
        scenes
    )

    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\n"
            f"NARRATION: {scene.narration}\n"
            f"EXPECTED: {scene.visual_description}\n"
            f"MATCH: {scene.visual_match_score:.1f}\n"
            f"IMAGE QUALITY: {scene.image_quality_score:.1f}\n"
            f"COMPOSITION: {scene.composition_score:.1f}\n"
            f"SOURCE: {scene.visual_source}"
        )
        for scene in scenes
    )

    prompt = f"""
Premium comic Shorts Final Visual Director'sın.

PLAN:
{plan}

Contact sheet'i incele.

0-100:
visual_story_match
visual_diversity
image_quality
crop_quality
narrative_flow
professional_feel
rewatch_value
overall

94+ overall yalnız gerçekten premium video için ver.

Kontrol et:
- Sesin anlatacağı olayın görseli o sahnede mi?
- Çok benzer paneller var mı?
- Düşük kaliteli/upscale çamur görsel var mı?
- Crop aksiyonu kesiyor mu?
- Slideshow hissi çok mu güçlü?
- AI reconstruction varsa gerçek comic estetiğine uyuyor mu?

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
                    data=contact_sheet.read_bytes(),
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.05,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name="Final Visual Director",
    )

    return parse_json_response(
        response.text or "{}"
    )


def run_final_visual_gate(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
    global_candidates: list[ImageCandidate],
) -> list[Scene]:
    """Final görsel akışını zayıf sahneleri yeniden üreterek düzeltir."""
    asset_directory = (
        ASSET_DIR
        / event["id"]
    )

    for cycle in range(
        1,
        FINAL_VISUAL_REPAIR_CYCLES + 1,
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
            clean(item)
            for item in review.get(
                "problems",
                [],
            )
            if clean(item)
        )

        passed = CHECKPOINTS.record(
            name="final_visual_flow",
            score=score,
            threshold=FINAL_VISUAL_THRESHOLD,
            cycle=cycle,
            details=problems,
        )

        if passed:
            return scenes

        weak_numbers: set[int] = set()

        for raw in review.get(
            "weak_scenes",
            [],
        ):
            try:
                weak_numbers.add(
                    int(raw)
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

        for number in sorted(
            weak_numbers
        ):
            scene_data = next(
                scene
                for scene in storyboard[
                    "scenes"
                ]
                if int(
                    scene["scene_number"]
                )
                == number
            )

            used_ids = {
                scene.selected_candidate_id
                for scene in scenes
                if scene.scene_number != number
                and scene.selected_candidate_id
                != "AI"
            }

            replacement = (
                resolve_scene_until_pass(
                    client,
                    event,
                    scene_data,
                    global_candidates,
                    used_ids,
                    asset_directory,
                )
            )

            scenes[
                number - 1
            ] = replacement

    raise EventRejectedError(
        "Final görsel akış bu event için "
        "premium kalite seviyesine ulaşamadı."
    )


def choose_motion_plan(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[int, dict[str, Any]]:
    """Sahne anlamına uygun kamera hareketleri belirler."""
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
Premium 9:16 comic Cinematic Director'sın.

SCENES:
{plan}

motion:
{", ".join(ALLOWED_MOTIONS)}

transition:
{", ".join(ALLOWED_TRANSITIONS)}

Kurallar:
- aynı motion arka arkaya gelmesin
- hareketler hafif ve premium olsun
- comic paneli gereksiz zoom ile bozma
- context sakin
- escalation dinamik
- twist/payoff impact kullanılabilir
- cta sakin
- sonraki scene görselini narration başlamadan gösterme
- transition yalnız mevcut sahnenin sonunda fade olarak çalışsın

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
        temperature=0.16,
        operation_name="Cinematic Director",
    )

    output: dict[
        int,
        dict[str, Any]
    ] = {}

    for item in payload.get(
        "scenes",
        [],
    ):
        if not isinstance(item, dict):
            continue

        try:
            number = int(
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

        output[number] = item

    return output


def apply_motion_plan(
    scenes: list[Scene],
    plan: dict[int, dict[str, Any]],
) -> None:
    """Motion planını güvenli değerlerle scene'lere uygular."""
    previous_motion = ""

    for index, scene in enumerate(
        scenes
    ):
        item = plan.get(
            scene.scene_number,
            {},
        )

        motion = clean(
            item.get("motion")
        )

        if motion not in ALLOWED_MOTIONS:
            motion = ALLOWED_MOTIONS[
                index
                % len(ALLOWED_MOTIONS)
            ]

        if motion == previous_motion:
            motion = ALLOWED_MOTIONS[
                (
                    ALLOWED_MOTIONS.index(
                        motion
                    )
                    + 1
                )
                % len(ALLOWED_MOTIONS)
            ]

        transition = clean(
            item.get("transition")
        )

        if transition not in ALLOWED_TRANSITIONS:
            transition = "cut"

        try:
            transition_duration = float(
                item.get(
                    "transition_duration",
                    0.12,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            transition_duration = 0.12

        if transition == "cut":
            transition_duration = 0.0

        else:
            transition_duration = max(
                0.08,
                min(
                    0.18,
                    transition_duration,
                ),
            )

        scene.motion = motion
        scene.transition = transition
        scene.transition_duration = (
            transition_duration
        )

        previous_motion = motion


def evaluate_motion_plan(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Motion planını kalite açısından değerlendirir."""
    payload = [
        {
            "scene": scene.scene_number,
            "role": scene.story_role,
            "narration": scene.narration,
            "motion": scene.motion,
            "transition": scene.transition,
        }
        for scene in scenes
    ]

    prompt = f"""
Premium vertical comic video Motion QC.

PLAN:
{json.dumps(
    payload,
    ensure_ascii=False,
    indent=2,
)}

0-100:
story_fit
variety
restraint
reveal_emphasis
sync_safety
overall

sync_safety çok önemli:
Bir sonraki sahnenin görüntüsü narration başlamadan
görünmemeli.

Aşırı effect varsa puan kır.

SADECE JSON:
{{
  "overall": 0,
  "problems": ["..."]
}}
"""

    return ask_gemini_json(
        client,
        prompt,
        temperature=0.04,
        operation_name="Motion QC",
    )


def build_quality_motion(
    client: genai.Client,
    scenes: list[Scene],
) -> None:
    """Motion planını checkpoint geçene kadar yeniden seçer."""
    for cycle in range(
        1,
        MOTION_REPAIR_CYCLES + 1,
    ):
        plan = choose_motion_plan(
            client,
            scenes,
        )

        apply_motion_plan(
            scenes,
            plan,
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
            clean(item)
            for item in review.get(
                "problems",
                [],
            )
            if clean(item)
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

    raise EventRejectedError(
        "Motion plan kalite checkpointini geçemedi."
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
    """Exact audio timeline değerlerini sahnelere bağlar."""
    if len(scenes) != len(timeline):
        raise EventRejectedError(
            "Scene count ve timeline uyuşmuyor."
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


def get_subtitle_font(
    size: int,
) -> ImageFont.ImageFont:
    """Subtitle pixel ölçümü için font döndürür."""
    candidate_paths = (
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf"
        ),
        Path(
            r"C:\Windows\Fonts\arialbd.ttf"
        ),
    )

    for path in candidate_paths:
        if path.exists():
            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


def text_width(
    text: str,
    font_size: int,
) -> int:
    """Subtitle pixel genişliğini ölçer."""
    font = get_subtitle_font(
        font_size
    )

    image = Image.new(
        "RGB",
        (10, 10),
        "black",
    )

    draw = ImageDraw.Draw(
        image
    )

    box = draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=5,
    )

    return box[2] - box[0]


def build_subtitle_groups(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """1-3 kelimelik safe-zone garantili subtitle grupları kurar."""
    groups: list[
        dict[str, Any]
    ] = []

    index = 0

    while index < len(words):
        selected: dict[
            str,
            Any
        ] | None = None

        maximum_count = min(
            SUBTITLE_MAX_WORDS,
            len(words) - index,
        )

        for count in range(
            maximum_count,
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
                    ]["word"]
                )
                for word_index in indexes
            ).upper()

            for font_size in range(
                SUBTITLE_BASE_FONT_SIZE,
                SUBTITLE_MIN_FONT_SIZE - 1,
                -2,
            ):
                width = round(
                    text_width(
                        text,
                        font_size,
                    )
                    * 1.12
                )

                if (
                    width
                    <= SUBTITLE_AVAILABLE_WIDTH
                ):
                    selected = {
                        "indexes": indexes,
                        "text": text,
                        "font_size": font_size,
                    }
                    break

            if selected is not None:
                break

        if selected is None:
            raise EventRejectedError(
                "Subtitle safe-zone içine sığdırılamadı."
            )

        groups.append(selected)

        index = (
            selected[
                "indexes"
            ][-1]
            + 1
        )

    return groups


def build_group_map(
    groups: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Kelime indexini subtitle grubuna bağlar."""
    output: dict[
        int,
        dict[str, Any]
    ] = {}

    for group in groups:
        for index in group[
            "indexes"
        ]:
            output[index] = group

    return output


def ass_time(
    seconds: float,
) -> str:
    """Saniyeyi ASS timestamp formatına dönüştürür."""
    total_cs = round(
        max(
            0.0,
            seconds,
        )
        * 100
    )

    hours, remainder = divmod(
        total_cs,
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
        clean(text)
        .replace(
            "\\",
            r"\\",
        )
        .replace(
            "{",
            "(",
        )
        .replace(
            "}",
            ")",
        )
    )


def all_emphasis_words(
    scenes: list[Scene],
) -> set[str]:
    """Tüm vurgulu kelimeleri toplar."""
    output: set[str] = set()

    for scene in scenes:
        for word in scene.emphasis_words:
            normalized = (
                normalize_alignment_word(
                    word
                )
            )

            if normalized:
                output.add(normalized)

    return output


def subtitle_text(
    words: list[dict[str, Any]],
    active_index: int,
    group_map: dict[int, dict[str, Any]],
    emphasis: set[str],
) -> str:
    """Aktif kelime vurgulu ASS subtitle metni üretir."""
    group = group_map[
        active_index
    ]

    font_size = int(
        group["font_size"]
    )

    parts: list[str] = []

    for index in group["indexes"]:
        raw = ass_escape(
            words[index]["word"]
        )

        display = raw.upper()

        normalized = (
            normalize_alignment_word(raw)
        )

        if index == active_index:
            scale = (
                110
                if normalized in emphasis
                else 107
            )

            parts.append(
                (
                    r"{"
                    rf"\fs{font_size}"
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
                    rf"\fs{font_size}"
                    r"\fscx100\fscy100"
                    r"}"
                    + display
                    + r"{\r}"
                )
            )

    return " ".join(parts)


def create_subtitles(
    aligned_words: list[dict[str, Any]],
    scenes: list[Scene],
    narration: str,
    offset: float,
) -> Path:
    """Narration metnine kilitli safe-zone subtitle dosyası üretir."""
    expected = narration_tokens(
        narration
    )

    actual = [
        clean(word["word"])
        for word in aligned_words
    ]

    if expected != actual:
        raise EventRejectedError(
            "Subtitle text narration ile birebir aynı değil."
        )

    groups = build_subtitle_groups(
        aligned_words
    )

    for group in groups:
        width = round(
            text_width(
                group["text"],
                int(
                    group["font_size"]
                ),
            )
            * 1.12
        )

        if width > SUBTITLE_AVAILABLE_WIDTH:
            raise EventRejectedError(
                "Subtitle safe-zone overflow tespit edildi."
            )

    CHECKPOINTS.record(
        name="subtitle_safe_zone",
        score=100.0,
        threshold=100.0,
        cycle=1,
        details=(
            "text_exact=True, overflow=False"
        ),
        passed=True,
    )

    group_map = build_group_map(
        groups
    )

    emphasis = all_emphasis_words(
        scenes
    )

    output = (
        WORK_DIR
        / "precise.ass"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    for index, word in enumerate(
        aligned_words
    ):
        start = (
            float(word["start"])
            + offset
        )

        end = (
            float(word["end"])
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
            + subtitle_text(
                aligned_words,
                index,
                group_map,
                emphasis,
            )
        )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output


def motion_filter(
    motion: str,
) -> str:
    """Seçilen sinematik panel hareketini döndürür."""
    filters = {
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
            "y='ih/2-ih/zoom/2':"
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
            "z='min(zoom+0.00060,1.075)':"
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

    return filters.get(
        motion,
        filters["slow_push"],
    )


def transition_filter(
    transition: str,
    duration: float,
    scene_duration: float,
) -> str:
    """Sahne sonunda yalnız mevcut paneli fade ederek sync'i korur."""
    if (
        transition == "cut"
        or duration <= 0
    ):
        return ""

    duration = min(
        duration,
        scene_duration / 4.0,
    )

    start = max(
        0.0,
        scene_duration - duration,
    )

    color = (
        "white"
        if transition
        == "flash_white"
        else "black"
    )

    return (
        f",fade=t=out:"
        f"st={start:.3f}:"
        f"d={duration:.3f}:"
        f"color={color}"
    )


def render_scene(
    ffmpeg: str,
    scene: Scene,
    frame_file: Path,
    output_file: Path,
) -> None:
    """Exact narration aralığı kadar tek scene render eder."""
    duration = (
        scene.audio_end
        - scene.audio_start
    )

    if duration <= 0:
        raise ComicFactoryError(
            f"Scene {scene.scene_number} "
            "duration geçersiz."
        )

    filters = (
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
            str(FPS),
            "-i",
            str(frame_file),
            "-t",
            f"{duration:.4f}",
            "-vf",
            filters,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(output_file),
        ],
        (
            f"Scene {scene.scene_number} "
            "render başarısız."
        ),
    )


def render_video(
    scenes: list[Scene],
    audio_file: Path,
    subtitle_file: Path,
) -> Path:
    """Final V4 videosunu exact timeline ile oluşturur."""
    ffmpeg = ffmpeg_path()

    render_directory = (
        WORK_DIR
        / "render"
    )

    shutil.rmtree(
        render_directory,
        ignore_errors=True,
    )

    render_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments: list[Path] = []

    for scene in scenes:
        frame_file = (
            render_directory
            / (
                f"frame_"
                f"{scene.scene_number:02d}.jpg"
            )
        )

        segment_file = (
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
            frame_file,
            "JPEG",
            quality=96,
        )

        render_scene(
            ffmpeg,
            scene,
            frame_file,
            segment_file,
        )

        segments.append(
            segment_file
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
            str(concat_file),
            "-c",
            "copy",
            str(silent_video),
        ],
        "Scene concat başarısız.",
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
        + subtitle_file.resolve()
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
            str(silent_video),
            "-i",
            str(audio_file),
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
            str(archive),
        ],
        "Final render başarısız.",
    )

    return archive


def technical_video_check(
    video: Path,
    audio: Path,
) -> dict[str, Any]:
    """Final dosyanın çözünürlük, audio ve duration kontrolünü yapar."""
    ffmpeg = ffmpeg_path()

    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video),
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

    width = 0
    height = 0

    if resolution_match:
        width = int(
            resolution_match.group(1)
        )

        height = int(
            resolution_match.group(2)
        )

    video_duration = media_duration(
        ffmpeg,
        video,
    )

    audio_duration = media_duration(
        ffmpeg,
        audio,
    )

    duration_delta = abs(
        video_duration
        - audio_duration
    )

    has_audio = "Audio:" in stderr

    passed = (
        width == WIDTH
        and height == HEIGHT
        and has_audio
        and duration_delta <= 0.20
    )

    return {
        "passed": passed,
        "width": width,
        "height": height,
        "has_audio": has_audio,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "duration_delta": duration_delta,
    }


def render_until_pass(
    scenes: list[Scene],
    audio_file: Path,
    subtitle_file: Path,
) -> Path:
    """Teknik QC geçene kadar final videoyu yeniden render eder."""
    for cycle in range(
        1,
        RENDER_REPAIR_CYCLES + 1,
    ):
        archive = render_video(
            scenes,
            audio_file,
            subtitle_file,
        )

        metrics = technical_video_check(
            archive,
            audio_file,
        )

        passed = CHECKPOINTS.record(
            name="technical_video_qc",
            score=(
                100.0
                if metrics["passed"]
                else 0.0
            ),
            threshold=100.0,
            cycle=cycle,
            details=(
                f"resolution="
                f"{metrics['width']}x"
                f"{metrics['height']} | "
                f"audio={metrics['has_audio']} | "
                f"duration_delta="
                f"{metrics['duration_delta']:.3f}s"
            ),
            passed=bool(
                metrics["passed"]
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

    raise EventRejectedError(
        "Final render teknik QC geçemedi."
    )


def save_script_metadata(
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
) -> None:
    """Mevcut upload scriptleri için latest.json üretir."""
    hashtags = [
        clean(tag)
        for tag in storyboard.get(
            "hashtags",
            [],
        )
        if clean(tag)
    ]

    if not any(
        tag.casefold() == "#shorts"
        for tag in hashtags
    ):
        hashtags.append(
            "#Shorts"
        )

    source_lines: list[str] = []
    seen_urls: set[str] = set()

    for source in event.get(
        "sources",
        [],
    ):
        if not isinstance(source, dict):
            continue

        url = clean(
            source.get("url")
        )

        if (
            not url
            or url in seen_urls
        ):
            continue

        seen_urls.add(url)

        source_lines.append(
            f"- {clean(source.get('name')) or 'Source'}: "
            f"{url}"
        )

    description = clean(
        storyboard.get("description")
    )

    full_description = (
        description
        + "\n\n"
        + " ".join(hashtags)
    )

    if source_lines:
        full_description += (
            "\n\nKaynaklar:\n"
            + "\n".join(source_lines)
        )

    save_json(
        LATEST_SCRIPT_FILE,
        {
            "event_id": event["id"],
            "generated_at": datetime.now(
                TZ
            ).isoformat(),
            "script": {
                "title": clean(
                    storyboard.get(
                        "title"
                    )
                ),
                "description": description,
                "full_description": (
                    full_description
                ),
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
            },
        },
    )


def save_visual_manifest(
    event: dict[str, Any],
    scenes: list[Scene],
) -> None:
    """Final scene planını kalite analizi için kaydeder."""
    directory = (
        ASSET_DIR
        / event["id"]
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
                asdict(scene)
                for scene in scenes
            ],
        },
    )


def mark_used(
    event: dict[str, Any],
) -> None:
    """Yalnız başarıyla video üretilmiş eventi kullanılmış olarak işaretler."""
    payload = load_json(
        USED_EVENTS_FILE,
        {"events": []},
    )

    if not isinstance(payload, dict):
        payload = {"events": []}

    events = payload.setdefault(
        "events",
        [],
    )

    key = event_key(event)

    if not any(
        isinstance(item, dict)
        and clean(
            item.get("event_key")
        )
        == key
        for item in events
    ):
        events.append(
            {
                "event_key": key,
                "event_title": clean(
                    event.get(
                        "event_title"
                    )
                ),
                "series": clean(
                    event.get("series")
                ),
                "issue": clean(
                    event.get("issue")
                ),
                "completed_at": datetime.now(
                    TZ
                ).isoformat(),
            }
        )

    save_json(
        USED_EVENTS_FILE,
        payload,
    )


def run_python_script(
    filename: str,
) -> None:
    """Repo içindeki uploader scriptini çalıştırır."""
    script = ROOT / filename

    if not script.exists():
        print(
            f"! {filename} bulunamadı, atlandı."
        )
        return

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=str(ROOT),
    )

    if result.returncode != 0:
        raise ComicFactoryError(
            f"{filename} başarısız."
        )


def upload_outputs() -> None:
    """Başarılı videoyu mevcut platform uploaderlarına gönderir."""
    run_python_script(
        "youtube_uploader.py"
    )

    run_python_script(
        "instagram_uploader.py"
    )


def reset_event_workspace(
    event_id: str,
) -> None:
    """Reddedilen eventten kalan geçici dosyaları temizler."""
    shutil.rmtree(
        CANDIDATE_DIR / event_id,
        ignore_errors=True,
    )

    shutil.rmtree(
        ASSET_DIR / event_id,
        ignore_errors=True,
    )

    shutil.rmtree(
        WORK_DIR,
        ignore_errors=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def build_single_event_video(
    client: genai.Client,
    event: dict[str, Any],
    *,
    max_images: int,
    subtitle_offset: float,
) -> Path:
    """Tek eventi bütün kalite checkpointlerinden geçirerek video üretir."""
    event = activate_event(event)

    storyboard = build_quality_storyboard(
        client,
        event,
    )

    narration = clean(
        storyboard["narration"]
    )

    audio_file, _, aligned_words = (
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

    for index, (
        start,
        end,
    ) in enumerate(
        timeline,
        start=1,
    ):
        print(
            f"✓ Timeline Scene {index:02d}: "
            f"{start:.3f} → {end:.3f}"
        )

    scenes, global_candidates = (
        build_quality_visuals(
            client,
            event,
            storyboard,
            max_images,
        )
    )

    scenes = run_final_visual_gate(
        client,
        event,
        storyboard,
        scenes,
        global_candidates,
    )

    attach_timeline(
        scenes,
        timeline,
    )

    build_quality_motion(
        client,
        scenes,
    )

    subtitle_file = create_subtitles(
        aligned_words,
        scenes,
        narration,
        subtitle_offset,
    )

    archive = render_until_pass(
        scenes,
        audio_file,
        subtitle_file,
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

    mark_used(event)

    return archive


def system_check(
    upload: bool,
) -> None:
    """Pipeline'ın temel bağımlılıklarını kontrol eder."""
    problems: list[str] = []

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

    if upload:
        if not (
            ROOT
            / "youtube_uploader.py"
        ).exists():
            problems.append(
                "youtube_uploader.py yok."
            )

    if problems:
        raise ComicFactoryError(
            "\n".join(problems)
        )

    print("✓ Gemini hazır")
    print("✓ Groq hazır")
    print("✓ FFmpeg hazır")
    print(
        f"✓ Voice: "
        f"{GEMINI_TTS_VOICE} LOCKED"
    )


def automatic_video_loop(
    client: genai.Client,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    Path,
]:
    """Bir run içinde kaliteli video oluşana kadar farklı eventler dener."""
    run_rejected: set[str] = set()

    event_queue: list[
        dict[str, Any]
    ] = []

    switch_count = 0

    if args.reuse_active:
        active = load_active_event()

        event_queue.append(
            active
        )

    while switch_count < MAX_EVENT_SWITCHES_PER_RUN:
        if not event_queue:
            print()
            print("=" * 78)
            print("NEW EVENT BATCH")
            print("=" * 78)

            event_queue = research_events(
                client,
                max(
                    args.count,
                    MAX_EVENT_CANDIDATES_PER_RESEARCH,
                ),
                run_rejected,
            )

            if not event_queue:
                print(
                    "! Uygun event bulunamadı. "
                    "Yeni araştırma turu başlatılıyor."
                )

                time.sleep(5)
                continue

        event = event_queue.pop(0)

        key = event_key(event)

        if key in run_rejected:
            continue

        switch_count += 1

        print()
        print("#" * 78)
        print(
            f"VIDEO ATTEMPT "
            f"{switch_count}/"
            f"{MAX_EVENT_SWITCHES_PER_RUN}"
        )
        print("#" * 78)
        print(
            clean(
                event.get(
                    "event_title"
                )
            )
        )

        try:
            archive = build_single_event_video(
                client,
                event,
                max_images=max(
                    30,
                    args.max_images,
                ),
                subtitle_offset=(
                    args.subtitle_offset
                ),
            )

            return (
                event,
                archive,
            )

        except EventRejectedError as error:
            reason = clean(error)

            print()
            print(
                "=" * 78
            )
            print(
                "EVENT QUALITY REJECTED"
            )
            print(
                "=" * 78
            )
            print(reason)
            print(
                "→ Başka story/event seçilecek."
            )

            run_rejected.add(key)

            record_rejected_event(
                event,
                reason,
            )

            event_id = slug(
                f"{clean(event.get('series'))}_"
                f"{clean(event.get('issue'))}_"
                f"{clean(event.get('event_title'))}"
            )

            reset_event_workspace(
                event_id
            )

            continue

    raise ComicFactoryError(
        "Bu run içinde maksimum event değişim "
        "sayısına ulaşıldı. "
        "Dış kaynaklar veya API cevapları nedeniyle "
        "yayınlanabilir video üretilemedi."
    )


def main() -> None:
    """Comic Factory V4 production entry point."""
    args = parse_args()

    ensure_dirs()

    if args.check:
        system_check(
            args.upload
        )
        return

    client = gemini_client()

    try:
        print()
        print("=" * 78)
        print("COMIC FACTORY V4")
        print("ONE RUN → ONE QUALITY VIDEO")
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
                if ENABLE_AI_RECONSTRUCTION
                and not args.disable_ai_reconstruction
                else "OFF"
            )
        )

        global ENABLE_AI_RECONSTRUCTION

        if args.disable_ai_reconstruction:
            ENABLE_AI_RECONSTRUCTION = False

        event, archive = automatic_video_loop(
            client,
            args,
        )

        CHECKPOINTS.persist_lessons()
        CHECKPOINTS.save()

        print()
        print("=" * 78)
        print("COMIC FACTORY V4 SUCCESS")
        print("=" * 78)
        print()

        print(
            "Event: "
            + clean(
                event.get(
                    "event_title"
                )
            )
        )

        print(
            "Comic: "
            + clean(
                event.get(
                    "series"
                )
            )
            + " "
            + clean(
                event.get(
                    "issue"
                )
            )
        )

        print(
            f"Voice: "
            f"{GEMINI_TTS_VOICE} (LOCKED)"
        )

        print(
            f"Latest video: "
            f"{LATEST_VIDEO_FILE}"
        )

        print(
            f"Archive: {archive}"
        )

        print()
        print("✓ Story PASS")
        print("✓ Audio alignment PASS")
        print("✓ Scene timing PASS")
        print("✓ Visual match PASS")
        print("✓ Image quality PASS")
        print("✓ Composition PASS")
        print("✓ Final visual flow PASS")
        print("✓ Motion PASS")
        print("✓ Subtitle safe-zone PASS")
        print("✓ Technical render PASS")

        if args.upload:
            print()
            print(
                "=" * 78
            )
            print(
                "PLATFORM UPLOAD"
            )
            print(
                "=" * 78
            )

            upload_outputs()

    except KeyboardInterrupt:
        CHECKPOINTS.persist_lessons()
        CHECKPOINTS.save()

        raise SystemExit(1)

    except Exception as error:
        CHECKPOINTS.persist_lessons()
        CHECKPOINTS.save()

        print()
        print("=" * 78)
        print("COMIC FACTORY V4 DURDU")
        print("=" * 78)
        print()
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
