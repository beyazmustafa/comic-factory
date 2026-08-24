# comic_factory_v3.py
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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

EVENT_DIR = DATA / "events"
RESEARCH_DIR = DATA / "research"
SCRIPT_DIR = DATA / "scripts"
AUDIO_DIR = DATA / "audio"
VIDEO_DIR = DATA / "videos"
CANDIDATE_DIR = DATA / "comic_candidates"
WORK_DIR = DATA / "video_work"
ASSET_DIR = ROOT / "assets" / "comic_pages"

ACTIVE_EVENT_FILE = EVENT_DIR / "active_event.json"
USED_EVENTS_FILE = EVENT_DIR / "used_events.json"
LATEST_SCRIPT_FILE = SCRIPT_DIR / "latest.json"
LATEST_AUDIO_FILE = AUDIO_DIR / "latest.wav"
LATEST_WORDS_FILE = AUDIO_DIR / "latest_word_timestamps.json"
LATEST_VIDEO_FILE = VIDEO_DIR / "latest.mp4"

TZ = ZoneInfo("Europe/Istanbul")

WIDTH = 1080
HEIGHT = 1920
FPS = 30

SCENE_COUNT = 14
MAX_VISION_IMAGES = 28

MAX_AI_RECONSTRUCTIONS = int(
    os.getenv(
        "MAX_AI_RECONSTRUCTIONS",
        "4",
    )
)

ENABLE_AI_RECONSTRUCTION = (
    os.getenv(
        "ENABLE_AI_RECONSTRUCTION",
        "1",
    ).strip()
    == "1"
)

MIN_REAL_RELEVANCE = 80
MIN_ALT_RELEVANCE = 72

SUBTITLE_MAX_WORDS = 3
SUBTITLE_MAX_CHARACTERS = 18
SUBTITLE_MARGIN_LEFT = 125
SUBTITLE_MARGIN_RIGHT = 125
SUBTITLE_MARGIN_BOTTOM = 390
SUBTITLE_BASE_FONT_SIZE = 56

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.1-flash-tts-preview",
)

# Voice is intentionally locked.
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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
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
    "fade",
    "fadeblack",
    "fadewhite",
    "wipeleft",
    "wiperight",
    "slideleft",
    "slideright",
    "dissolve",
)

load_dotenv(
    ROOT / ".env"
)


class ComicFactoryError(RuntimeError):
    """Comic Factory üretimi başarısız olduğunda oluşur."""


@dataclass
class ImageCandidate:
    """Web'den bulunan gerçek comic görseli."""

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
    """Bir sahne için Visual Director tarafından puanlanan görsel."""

    candidate_id: str
    relevance_score: int
    crop_box: list[float]
    reason: str


@dataclass
class Scene:
    """Render edilecek tek video sahnesi."""

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
    motion: str = "slow_push"
    transition: str = "fade"
    transition_duration: float = 0.24


def parse_args() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""
    parser = argparse.ArgumentParser(
        description=(
            "Comic Factory V3 - "
            "AI Director production engine."
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
        default=36,
    )

    parser.add_argument(
        "--subtitle-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--disable-ai-reconstruction",
        action="store_true",
        help="AI image reconstruction fallback'ını kapatır.",
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
        clean(value)
        .lower()
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return (
        value.strip("_")[:80]
        or "event"
    )


def load_json(
    path: Path,
    default: Any = None,
) -> Any:
    """JSON dosyasını okur."""
    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def save_json(
    path: Path,
    payload: Any,
) -> None:
    """JSON dosyasını atomik kaydeder."""
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


def gemini_client() -> genai.Client:
    """Gemini API istemcisi oluşturur."""
    return genai.Client(
        api_key=require_env(
            "GEMINI_API_KEY"
        )
    )


def parse_json_response(
    text: str,
) -> dict[str, Any]:
    """Gemini JSON cevabını güvenli ayrıştırır."""
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
                "Gemini beklenmeyen JSON listesi döndürdü."
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
        "Gemini beklenmeyen JSON yapısı döndürdü."
    )


def is_retryable_gemini_error(
    error: Exception,
) -> bool:
    """Geçici Gemini servis hatalarını ayırt eder."""
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
    """Gemini işlemini exponential backoff ile tekrar dener."""
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
                    )
                    - 1,
                )
            ]

            wait_seconds += random.uniform(
                0.0,
                3.0,
            )

            print(
                f"! {operation_name} geçici hata: "
                f"{error}\n"
                f"  {wait_seconds:.1f} saniye "
                "sonra tekrar denenecek."
            )

            time.sleep(
                wait_seconds
            )

    raise ComicFactoryError(
        f"{operation_name} "
        f"{max_attempts} denemeden sonra başarısız: "
        f"{last_error}"
    )


def ask_gemini_json(
    client: genai.Client,
    prompt: str,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Gemini'den retry destekli JSON cevap alır."""

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
        operation_name="Gemini JSON",
    )

    if not response.text:
        raise ComicFactoryError(
            "Gemini boş yanıt döndürdü."
        )

    return parse_json_response(
        response.text
    )


def search_web(
    queries: list[str],
    each: int = 8,
) -> list[dict[str, str]]:
    """DDGS ile ücretsiz metin araması yapar."""
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
                f"! Arama başarısız: "
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
    """Olay için benzersiz anahtar oluşturur."""
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
    """Daha önce kullanılan olay anahtarlarını döndürür."""
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
    """Görsel ve hikâye potansiyeli yüksek comic olaylarını seçer."""
    count = max(
        3,
        min(
            count,
            10,
        ),
    )

    print()
    print(
        "=" * 78
    )
    print(
        "1/11 - EVENT RESEARCH"
    )
    print(
        "=" * 78
    )
    print()

    results = search_web(
        [
            "Marvel comics shocking moments specific issue review",
            "DC comics shocking moments specific issue review",
            "comic book craziest feats specific issue panels",
            "comic book transformation specific issue review Marvel DC",
            "Image Comics shocking moments specific issue review",
            "Dark Horse comics shocking moments specific issue review",
            "site:marvel.com comics preview issue",
            "site:dc.com comics preview issue",
        ],
        each=8,
    )

    if not results:
        raise ComicFactoryError(
            "Web araması sonuç vermedi."
        )

    evidence = "\n\n".join(
        (
            f"TITLE: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"SNIPPET: {item['body']}"
        )
        for item in results[
            :45
        ]
    )

    used = (
        "\n".join(
            f"- {key}"
            for key in sorted(
                used_event_keys()
            )
        )
        or "Yok."
    )

    prompt = f"""
Aşağıdaki web sonuçlarından TAM {count} adet premium Shorts'a uygun
tek ve spesifik comic-book olayı seç.

WEB:
{evidence}

DAHA ÖNCE KULLANILANLAR:
{used}

Her adayı şu kalite mantığıyla değerlendir:
- hook /25
- şaşırtıcılık /20
- hikâye-payoff /20
- görsel potansiyel /15
- kaynak güvenilirliği /10
- Shorts uygunluğu /10

Kurallar:
- Marvel, DC, Image ve Dark Horse öncelikli.
- Series, issue ve yıl doğru olmalı.
- URL veya issue uydurma.
- Kaynak URL'leri yalnızca WEB bölümünden gelsin.
- Olay 45-60 saniyede net başlangıç, escalation ve payoff taşısın.
- Birden çok farklı panel/aksiyon anı bulunabilecek kadar görsel olsun.
- Özel isimler orijinal kalsın.
- Metin alanları Türkçe.
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
      "quality_score": 92,
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
            "Gemini events listesi döndürmedi."
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
        >= 82
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
            "82+ kalite puanlı yeni comic olayı bulunamadı."
        )

    timestamp = datetime.now(
        TZ
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    save_json(
        RESEARCH_DIR
        / f"research_{timestamp}.json",
        {
            "events": events,
        },
    )

    for index, event in enumerate(
        eligible[
            :5
        ],
        start=1,
    ):
        print(
            f"{index}. "
            f"{clean(event.get('event_title'))} "
            f"({event.get('quality_score')}/100)"
        )

    return eligible


def activate_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    """Aktif olayı kaydeder."""
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
        f"✓ Seçilen olay: "
        f"{clean(active.get('event_title'))}"
    )

    return active


def load_active_event() -> dict[str, Any]:
    """Aktif olayı yükler."""
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


def build_storyboard(
    client: genai.Client,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Story Director ile 14 sahnelik anlatı oluşturur."""
    print()
    print(
        "=" * 78
    )
    print(
        "2/11 - STORY DIRECTOR"
    )
    print(
        "=" * 78
    )
    print()

    source_text = "\n".join(
        (
            f"- {clean(source.get('name'))}: "
            f"{clean(source.get('supports'))} "
            f"({clean(source.get('url'))})"
        )
        for source in event.get(
            "sources",
            [],
        )
        if isinstance(
            source,
            dict,
        )
    )

    prompt = f"""
Premium Türkçe comic Shorts Story Director'sın.

OLAY:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("publisher"))} / {clean(event.get("series"))}
Issue: {clean(event.get("issue"))}
Year: {event.get("publication_year")}

KARAKTERLER:
{", ".join(event.get("characters", []))}

HOOK:
{clean(event.get("hook"))}

ÖZET:
{clean(event.get("event_summary"))}

GÜÇ / FEAT:
{clean(event.get("power_feat"))}

KAYNAK NOTLARI:
{source_text}

TAM {SCENE_COUNT} sahnelik 45-60 saniyelik video yaz.

Kurallar:
- Toplam yaklaşık 120-150 Türkçe kelime.
- Wikipedia özeti gibi olmasın.
- İlk 2 sahne hook.
- Sonraki sahnelerde sürekli yeni bilgi ve escalation.
- Son 4 sahne payoff.
- Bilgi uydurma.
- Her sahnenin visual_description alanı, gerçek comic panelinde aranabilecek
  somut bir anı tarif etsin.
- Aynı görsel fikri iki sahnede tekrar etmesin.
- Özel isimler orijinal yazılsın.
- Thor -> Tor gibi Türkçe fonetik yazım YAPMA.
- emphasis_words 0-3 kelime.
- story_role yalnızca:
  hook, context, escalation, twist, payoff, cta

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
        temperature=0.55,
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
            f"Story Director tam "
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

    save_json(
        SCRIPT_DIR
        / "storyboard.json",
        storyboard,
    )

    return storyboard


def image_search(
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    """Gerçek comic sayfası ve panel görsellerini geniş arar."""
    print()
    print(
        "=" * 78
    )
    print(
        "3/11 - VISUAL RESEARCH"
    )
    print(
        "=" * 78
    )
    print()

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

    title = clean(
        event.get(
            "event_title"
        )
    )

    queries = [
        f'"{series}" "{issue}" comic panels',
        f'"{series}" "{issue}" comic page',
        f'"{series}" "{issue}" preview images',
        f'"{series}" "{issue}" review panels',
        f'"{title}" comic panels',
        f'"{title}" comic page',
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
                f'{visual[:80]}'
            )

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
                max_results=12,
            )

        except Exception as error:
            print(
                f"! Görsel araması atlandı: "
                f"{error}"
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

    if not output:
        raise ComicFactoryError(
            "Comic görseli bulunamadı."
        )

    return output


def average_hash(
    image: Image.Image,
) -> str:
    """64-bit average hash üretir."""
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

    average = (
        sum(
            pixels
        )
        / len(
            pixels
        )
    )

    bits = "".join(
        "1"
        if value >= average
        else "0"
        for value in pixels
    )

    return f"{int(bits, 2):016x}"


def hash_distance(
    first: str,
    second: str,
) -> int:
    """İki perceptual hash arasındaki Hamming mesafesini döndürür."""
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


def download_candidates(
    event: dict[str, Any],
    results: list[dict[str, str]],
    max_candidates: int,
) -> list[ImageCandidate]:
    """Yüksek çözünürlüklü, benzersiz görselleri indirir."""
    print()
    print(
        "=" * 78
    )
    print(
        "4/11 - DOWNLOAD + DEDUPLICATE"
    )
    print(
        "=" * 78
    )
    print()

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
    )

    shutil.rmtree(
        directory,
        ignore_errors=True,
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates: list[
        ImageCandidate
    ] = []

    hashes: list[
        str
    ] = []

    for result in results:
        if len(
            candidates
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
            f"c{len(candidates) + 1:02d}"
        )

        local_file = (
            directory
            / f"{candidate_id}.jpg"
        )

        image.save(
            local_file,
            "JPEG",
            quality=95,
            optimize=True,
        )

        megapixels = (
            width
            * height
            / 1_000_000
        )

        quality_score = min(
            100.0,
            45.0
            + megapixels
            * 13.0
            + min(
                width,
                height,
            )
            / 80.0,
        )

        candidates.append(
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
                    quality_score,
                    2,
                ),
                perceptual_hash=phash,
            )
        )

        print(
            f"✓ {candidate_id}: "
            f"{width}x{height}"
        )

    if len(
        candidates
    ) < 8:
        raise ComicFactoryError(
            "Yeterli farklı kaliteli comic "
            "görseli bulunamadı. "
            f"Bulunan: {len(candidates)}"
        )

    return candidates


def thumbnail_bytes(
    path: Path,
) -> bytes:
    """Gemini Vision için küçük JPEG oluşturur."""
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
        quality=80,
    )

    return buffer.getvalue()


def normalize_crop_box(
    crop_box: Any,
) -> list[float]:
    """Normalize panel crop koordinatlarını doğrular."""
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


def rank_visuals(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    candidates: list[ImageCandidate],
) -> list[dict[str, Any]]:
    """Visual Director her sahne için en iyi 3 paneli sıralar."""
    print()
    print(
        "=" * 78
    )
    print(
        "5/11 - VISUAL DIRECTOR"
    )
    print(
        "=" * 78
    )
    print()

    vision_candidates = candidates[
        :MAX_VISION_IMAGES
    ]

    scenes_text = "\n\n".join(
        (
            f"SCENE {scene['scene_number']}\n"
            f"NARRATION: {scene['narration']}\n"
            f"VISUAL NEEDED: "
            f"{scene['visual_description']}"
        )
        for scene in storyboard[
            "scenes"
        ]
    )

    prompt = f"""
Premium comic-video Visual Director'sın.

EVENT:
{clean(event.get("event_title"))}
{clean(event.get("series"))} {clean(event.get("issue"))}

SAHNELER:
{scenes_text}

Aşağıdaki gerçek comic görsellerini incele.

Her sahne için EN İYİ 3 adayı sırala.

Kurallar:
- Yalnızca karakter aynı diye yüksek puan verme.
- Anlatılan spesifik olay gerçekten görselde bulunmalı.
- Gerçek aksiyon paneli kapaktan üstündür.
- Bir comic sayfasının içindeki doğru paneli crop et.
- crop_box normalize [left, top, right, bottom], 0-1.
- Tüm sayfa gerekliyse [0,0,1,1].
- Karakter yüzü/ana aksiyon crop dışında kalmasın.
- Aynı görseli gereksiz tekrar etme.
- relevance_score 0-100.

SADECE JSON:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "ranked_visuals": [
        {{
          "candidate_id": "c01",
          "relevance_score": 94,
          "crop_box": [0.05,0.1,0.95,0.9],
          "reason": "..."
        }},
        {{
          "candidate_id": "c02",
          "relevance_score": 85,
          "crop_box": [0,0,1,1],
          "reason": "..."
        }},
        {{
          "candidate_id": "c03",
          "relevance_score": 72,
          "crop_box": [0,0,1,1],
          "reason": "..."
        }}
      ]
    }}
  ]
}}

Tam {SCENE_COUNT} scene döndür.
"""

    contents: list[
        Any
    ] = [
        prompt
    ]

    for candidate in vision_candidates:
        contents.append(
            (
                f"CANDIDATE "
                f"{candidate.candidate_id}\n"
                f"TITLE: "
                f"{candidate.title}\n"
                f"SIZE: "
                f"{candidate.width}x"
                f"{candidate.height}\n"
                f"QUALITY: "
                f"{candidate.quality_score}"
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
                temperature=0.12,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name="Visual Director",
    )

    payload = parse_json_response(
        response.text
        or "{}"
    )

    scenes = payload.get(
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
            f"Visual Director tam "
            f"{SCENE_COUNT} sahne döndürmedi."
        )

    return scenes


def parse_ranked_visuals(
    ranking_payload: list[dict[str, Any]],
) -> dict[int, list[RankedVisual]]:
    """Gemini görsel sıralamasını tipli yapıya dönüştürür."""
    output: dict[
        int,
        list[RankedVisual],
    ] = {}

    for scene_data in ranking_payload:
        try:
            scene_number = int(
                scene_data.get(
                    "scene_number",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        ranked: list[
            RankedVisual
        ] = []

        for item in scene_data.get(
            "ranked_visuals",
            [],
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            candidate_id = clean(
                item.get(
                    "candidate_id"
                )
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
                        item.get(
                            "crop_box"
                        )
                    ),
                    reason=clean(
                        item.get(
                            "reason"
                        )
                    ),
                )
            )

        if ranked:
            output[
                scene_number
            ] = ranked

    return output


def select_unique_visuals(
    rankings: dict[int, list[RankedVisual]],
) -> dict[int, RankedVisual]:
    """Görsel tekrarını azaltarak her sahne için seçim yapar."""
    selected: dict[
        int,
        RankedVisual,
    ] = {}

    usage: dict[
        str,
        int,
    ] = {}

    for scene_number in range(
        1,
        SCENE_COUNT + 1,
    ):
        options = rankings.get(
            scene_number,
            [],
        )

        if not options:
            raise ComicFactoryError(
                f"Sahne {scene_number} için "
                "görsel sıralaması yok."
            )

        choice = options[
            0
        ]

        for option in options:
            used_count = usage.get(
                option.candidate_id,
                0,
            )

            if used_count == 0:
                choice = option
                break

            if (
                used_count == 1
                and option.relevance_score
                >= 92
            ):
                choice = option
                break

        selected[
            scene_number
        ] = choice

        usage[
            choice.candidate_id
        ] = (
            usage.get(
                choice.candidate_id,
                0,
            )
            + 1
        )

    return selected


def restore_crop(
    candidate: ImageCandidate,
    crop_box: list[float],
    event_directory: Path,
    scene_number: int,
) -> Path:
    """Gerçek comic panelini değiştirmeden crop/restore eder."""
    output = (
        event_directory
        / (
            f"scene_{scene_number:02d}_"
            f"{candidate.candidate_id}.png"
        )
    )

    with Image.open(
        candidate.local_file
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    left, top, right, bottom = crop_box

    pixel_box = (
        round(
            left
            * image.width
        ),
        round(
            top
            * image.height
        ),
        round(
            right
            * image.width
        ),
        round(
            bottom
            * image.height
        ),
    )

    image = image.crop(
        pixel_box
    )

    max_side = max(
        image.size
    )

    if max_side < 2400:
        scale = min(
            2.25,
            2400
            / max(
                1,
                max_side,
            ),
        )

        image = image.resize(
            (
                round(
                    image.width
                    * scale
                ),
                round(
                    image.height
                    * scale
                ),
            ),
            Image.Resampling.LANCZOS,
        )

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        1.045
    )

    image = ImageEnhance.Color(
        image
    ).enhance(
        1.015
    )

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.25,
            percent=105,
            threshold=3,
        )
    )

    image.save(
        output,
        "PNG",
        optimize=True,
    )

    return output


def generate_reconstruction(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    reference_paths: list[Path],
    output: Path,
) -> Path:
    """Gerçek panel yoksa referanslardan AI reconstruction üretir."""
    prompt = f"""
Create a premium 9:16 American comic-book illustration.

EVENT:
{clean(event.get("event_title"))}

COMIC CONTEXT:
{clean(event.get("series"))} {clean(event.get("issue"))}

EXACT MOMENT:
{clean(scene_data.get("visual_description"))}

NARRATION:
{clean(scene_data.get("narration"))}

Use the supplied comic images only as references for:
- character appearance
- costume continuity
- era-specific color language
- setting and atmosphere

Create the exact missing moment.
Do not copy a source composition verbatim.

Requirements:
- professional premium comic artwork
- cinematic composition
- clear focal point
- strong depth and readable anatomy
- no text
- no speech bubbles
- no logo
- no watermark
- no issue number
- no UI
""".strip()

    interaction_input: list[
        dict[str, str]
    ] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for path in reference_paths[
        :3
    ]:
        mime_type = (
            "image/png"
            if path.suffix.lower()
            == ".png"
            else "image/jpeg"
        )

        interaction_input.append(
            {
                "type": "image",
                "data": base64.b64encode(
                    path.read_bytes()
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


def build_scenes(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    candidates: list[ImageCandidate],
    ranking_payload: list[dict[str, Any]],
    enable_reconstruction: bool,
) -> list[Scene]:
    """Gerçek panel öncelikli final sahne görsellerini oluşturur."""
    print()
    print(
        "=" * 78
    )
    print(
        "6/11 - BUILD VISUAL PLAN"
    )
    print(
        "=" * 78
    )
    print()

    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    rankings = parse_ranked_visuals(
        ranking_payload
    )

    selected = select_unique_visuals(
        rankings
    )

    event_directory = (
        ASSET_DIR
        / event[
            "id"
        ]
    )

    shutil.rmtree(
        event_directory,
        ignore_errors=True,
    )

    event_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenes: list[
        Scene
    ] = []

    reconstruction_count = 0

    for scene_data in storyboard[
        "scenes"
    ]:
        number = int(
            scene_data[
                "scene_number"
            ]
        )

        choice = selected[
            number
        ]

        candidate = candidate_map.get(
            choice.candidate_id
        )

        if candidate is None:
            raise ComicFactoryError(
                "Candidate bulunamadı: "
                f"{choice.candidate_id}"
            )

        visual_file = restore_crop(
            candidate,
            choice.crop_box,
            event_directory,
            number,
        )

        visual_source = (
            "real_comic"
        )

        if (
            enable_reconstruction
            and choice.relevance_score
            < MIN_REAL_RELEVANCE
            and reconstruction_count
            < MAX_AI_RECONSTRUCTIONS
        ):
            references: list[
                Path
            ] = []

            for option in rankings.get(
                number,
                [],
            )[
                :3
            ]:
                reference = candidate_map.get(
                    option.candidate_id
                )

                if reference is not None:
                    references.append(
                        Path(
                            reference.local_file
                        )
                    )

            try:
                generated = (
                    event_directory
                    / f"scene_{number:02d}_ai.jpg"
                )

                visual_file = generate_reconstruction(
                    client,
                    event,
                    scene_data,
                    references,
                    generated,
                )

                visual_source = (
                    "ai_reconstruction"
                )

                reconstruction_count += 1

                print(
                    f"🎨 Sahne {number}: "
                    "AI reconstruction"
                )

            except Exception as error:
                print(
                    f"! Sahne {number} reconstruction "
                    "başarısız, gerçek panel kullanılıyor: "
                    f"{error}"
                )

        scenes.append(
            Scene(
                scene_number=number,
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
                ranked_visuals=rankings.get(
                    number,
                    [],
                ),
                selected_candidate_id=(
                    choice.candidate_id
                ),
                relevance_score=(
                    choice.relevance_score
                ),
                crop_box=(
                    choice.crop_box
                ),
                visual_source=visual_source,
                visual_file=str(
                    visual_file
                ),
            )
        )

    return scenes


def compose_vertical(
    visual_file: Path,
) -> Image.Image:
    """Comic panelini temiz 9:16 kompozisyona dönüştürür."""
    with Image.open(
        visual_file
    ) as opened:
        source = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
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
            radius=48
        )
    )

    background = ImageEnhance.Brightness(
        background
    ).enhance(
        0.30
    )

    background = ImageEnhance.Color(
        background
    ).enhance(
        0.78
    )

    source_ratio = (
        source.width
        / max(
            1,
            source.height,
        )
    )

    vertical_like = (
        0.48
        <= source_ratio
        <= 0.76
    )

    max_height = (
        1700
        if vertical_like
        else 1540
    )

    foreground = source.copy()

    foreground.thumbnail(
        (
            WIDTH - 26,
            max_height,
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
    ) // 2 - 40

    y = max(
        55,
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
    """AI Director için sahneleri tek görselde toplar."""
    columns = 4
    cell_width = 270
    cell_height = 460

    rows = math.ceil(
        len(
            scenes
        )
        / columns
    )

    sheet = Image.new(
        "RGB",
        (
            columns
            * cell_width,
            rows
            * cell_height,
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
        quality=92,
    )

    return output


def supervising_director(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Görsel kalite kontrolü yapar."""
    print()
    print(
        "=" * 78
    )
    print(
        "7/11 - SUPERVISING DIRECTOR"
    )
    print(
        "=" * 78
    )
    print()

    sheet = build_contact_sheet(
        scenes
    )

    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\n"
            f"NARRATION: {scene.narration}\n"
            f"VISUAL: {scene.visual_description}\n"
            f"RELEVANCE: {scene.relevance_score}\n"
            f"SOURCE: {scene.visual_source}"
        )
        for scene in scenes
    )

    prompt = f"""
Premium comic Shorts supervising director'sın.

14 sahnelik contact sheet ve planı değerlendir:

{plan}

0-10 puanla:
- visual_relevance
- visual_diversity
- crop_quality
- story_visual_match
- cinematic_potential
- overall

weak_scenes:
yanlış görsel, kötü crop, tekrar, kapak kullanımı veya anlatımla
uyuşmayan sahnelerin numaraları.

SADECE JSON:
{{
  "visual_relevance": 0,
  "visual_diversity": 0,
  "crop_quality": 0,
  "story_visual_match": 0,
  "cinematic_potential": 0,
  "overall": 0,
  "weak_scenes": [2, 8],
  "lessons": ["..."]
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
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

    response = gemini_with_retry(
        request,
        operation_name="Supervising Director",
    )

    critique = parse_json_response(
        response.text
        or "{}"
    )

    save_json(
        WORK_DIR
        / "director_critique.json",
        critique,
    )

    return critique


def revise_weak_scenes(
    scenes: list[Scene],
    candidates: list[ImageCandidate],
    critique: dict[str, Any],
    event: dict[str, Any],
) -> list[Scene]:
    """Zayıf sahnelerde alternatif gerçek paneli kullanır."""
    weak: set[
        int
    ] = set()

    for value in critique.get(
        "weak_scenes",
        [],
    ):
        try:
            weak.add(
                int(
                    value
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    used = {
        scene.selected_candidate_id
        for scene in scenes
        if scene.visual_source
        != "ai_reconstruction"
    }

    event_directory = (
        ASSET_DIR
        / event[
            "id"
        ]
    )

    for scene in scenes:
        if scene.scene_number not in weak:
            continue

        for option in scene.ranked_visuals[
            1:
        ]:
            if option.relevance_score < MIN_ALT_RELEVANCE:
                continue

            if option.candidate_id in used:
                continue

            candidate = candidate_map.get(
                option.candidate_id
            )

            if candidate is None:
                continue

            new_file = restore_crop(
                candidate,
                option.crop_box,
                event_directory,
                scene.scene_number,
            )

            used.discard(
                scene.selected_candidate_id
            )

            used.add(
                option.candidate_id
            )

            scene.selected_candidate_id = (
                option.candidate_id
            )

            scene.relevance_score = (
                option.relevance_score
            )

            scene.crop_box = (
                option.crop_box
            )

            scene.visual_source = (
                "real_comic_alternative"
            )

            scene.visual_file = str(
                new_file
            )

            break

    return scenes


def cinematic_director(
    client: genai.Client,
    scenes: list[Scene],
) -> list[Scene]:
    """Her sahneye hareket ve sinematik geçiş seçer."""
    print()
    print(
        "=" * 78
    )
    print(
        "8/11 - CINEMATIC DIRECTOR"
    )
    print(
        "=" * 78
    )
    print()

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
Premium 9:16 comic-video motion director'sın.

SAHNELER:
{plan}

Her sahne için motion ve bir SONRAKİ sahneye transition seç.

motion yalnızca:
{", ".join(ALLOWED_MOTIONS)}

transition yalnızca:
{", ".join(ALLOWED_TRANSITIONS)}

Kurallar:
- Aynı motion art arda gelmesin.
- Aynı transition art arda gelmesin.
- hook: hızlı ama temiz.
- context: sakin push/pan/dissolve.
- escalation: yönlü pan veya daha dinamik geçiş.
- twist/reveal: impact + kısa white/black transition olabilir.
- payoff: en güçlü ama aşırı olmayan hareket.
- cta: sakinleş.
- transition_duration 0.16 ile 0.34 saniye.
- fadewhite yalnız büyük reveal'da.
- Her sahneye transition ver; son sahneninki renderda kullanılmayacak.

SADECE JSON:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "motion": "slow_push",
      "transition": "dissolve",
      "transition_duration": 0.24
    }}
  ]
}}
"""

    payload = ask_gemini_json(
        client,
        prompt,
        temperature=0.25,
    )

    direction = {
        int(
            item.get(
                "scene_number",
                0,
            )
        ): item
        for item in payload.get(
            "scenes",
            [],
        )
        if isinstance(
            item,
            dict,
        )
    }

    previous_motion = ""
    previous_transition = ""

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

        transition = clean(
            item.get(
                "transition"
            )
        )

        if motion not in ALLOWED_MOTIONS:
            motion = ALLOWED_MOTIONS[
                index
                % len(
                    ALLOWED_MOTIONS
                )
            ]

        if transition not in ALLOWED_TRANSITIONS:
            transition = ALLOWED_TRANSITIONS[
                index
                % len(
                    ALLOWED_TRANSITIONS
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

        if transition == previous_transition:
            transition = ALLOWED_TRANSITIONS[
                (
                    ALLOWED_TRANSITIONS.index(
                        transition
                    )
                    + 1
                )
                % len(
                    ALLOWED_TRANSITIONS
                )
            ]

        try:
            transition_duration = float(
                item.get(
                    "transition_duration",
                    0.24,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            transition_duration = 0.24

        scene.motion = motion

        scene.transition = transition

        scene.transition_duration = max(
            0.16,
            min(
                0.34,
                transition_duration,
            ),
        )

        previous_motion = motion
        previous_transition = transition

    return scenes


def save_script_metadata(
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
) -> dict[str, Any]:
    """YouTube/Instagram yükleyicileri için latest.json üretir."""
    unique_sources: list[
        str
    ] = []

    source_lines: list[
        str
    ] = []

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
            url
            and url not in unique_sources
        ):
            unique_sources.append(
                url
            )

            source_lines.append(
                f"- {name or 'Source'}: "
                f"{url}"
            )

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

    if "#shorts" not in [
        tag.casefold()
        for tag in hashtags
    ]:
        hashtags.append(
            "#Shorts"
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
        "full_description": full_description,
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

    return script


def create_voice_direction(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Kilitli Gacrux sesinin doğal okuma talimatını üretir."""
    narration = " ".join(
        scene.narration
        for scene in scenes
    )

    prompt = f"""
Sen Türkçe premium YouTube Shorts Voice Director'sın.

METİN:
{narration}

Anlamı ve kelimeleri değiştirme.

Kurallar:
- İngilizce özel isimleri Türkçe fonetik yazıma çevirme.
- İngilizce özel isimleri doğal İngilizce telaffuz et,
  sonra akıcı biçimde Türkçeye dön.
- Doğal İstanbul Türkçesi.
- Haber spikeri gibi değil.
- Belgesel gibi ağır değil.
- Robot/TikTok sesi gibi değil.
- 25-35 yaş doğal erkek YouTube storyteller hissi.
- Arkadaşına inanılmaz bir comic olayını anlatıyormuş gibi.
- Enerjik ama bağırmayan.
- Büyük reveal öncesi kısa doğal pause.
- Cümle sonlarını sürekli aynı melodide bitirme.
- Yaklaşık 1.0x doğal tempo.

SADECE JSON:
{{
  "exact_narration": "...",
  "style_instruction": "...",
  "pronunciation_instruction": "...",
  "pace_instruction": "..."
}}

exact_narration orijinal metindeki kelimeleri korusun.
"""

    return ask_gemini_json(
        client,
        prompt,
        temperature=0.25,
    )


def write_pcm_wave(
    path: Path,
    pcm: bytes,
) -> None:
    """24 kHz mono PCM sesini WAV olarak kaydeder."""
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


def generate_voice(
    client: genai.Client,
    scenes: list[Scene],
) -> tuple[Path, str]:
    """Kilitli Gemini Gacrux sesiyle narration üretir."""
    print()
    print(
        "=" * 78
    )
    print(
        "9/11 - LOCKED GACRUX VOICE"
    )
    print(
        "=" * 78
    )
    print()

    voice_direction = create_voice_direction(
        client,
        scenes,
    )

    narration = clean(
        voice_direction.get(
            "exact_narration"
        )
    )

    if not narration:
        narration = " ".join(
            scene.narration
            for scene in scenes
        )

    prompt = f"""
Read ONLY the transcript enclosed in <TRANSCRIPT> tags.

VOICE DIRECTION:
{clean(voice_direction.get("style_instruction"))}

PRONUNCIATION:
{clean(voice_direction.get("pronunciation_instruction"))}

PACE:
{clean(voice_direction.get("pace_instruction"))}

Critical pronunciation rule:
The surrounding language is Turkish.
Whenever an English proper noun appears, pronounce that proper noun
naturally in English, then return smoothly to Turkish pronunciation.

Do not read stage directions.
Do not read XML tags.
Do not add or remove words.
Do not sound like an announcer.
Do not sound synthetic.
Use natural breaths and subtle conversational rhythm.

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
        operation_name="Gemini Gacrux TTS",
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
            "Gemini TTS audio döndürmedi."
        ) from error

    if not pcm:
        raise ComicFactoryError(
            "Gemini TTS boş audio döndürdü."
        )

    timestamp = datetime.now(
        TZ
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive = (
        AUDIO_DIR
        / f"comic_gacrux_{timestamp}.wav"
    )

    write_pcm_wave(
        archive,
        pcm,
    )

    shutil.copy2(
        archive,
        LATEST_AUDIO_FILE,
    )

    print(
        f"✓ Gemini TTS voice: "
        f"{GEMINI_TTS_VOICE} (LOCKED)"
    )

    return (
        LATEST_AUDIO_FILE,
        narration,
    )


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


def align_words(
    audio_file: Path,
    narration: str,
) -> list[dict[str, Any]]:
    """Groq Whisper ile gerçek kelime zamanlarını çıkarır."""
    print()
    print(
        "=" * 78
    )
    print(
        "10/11 - WORD ALIGNMENT"
    )
    print(
        "=" * 78
    )
    print()

    client = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with audio_file.open(
        "rb"
    ) as audio:
        transcription = (
            client.audio.transcriptions.create(
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
            "Groq word timestamps alınamadı."
        )

    save_json(
        LATEST_WORDS_FILE,
        {
            "provider": "groq",
            "model": GROQ_MODEL,
            "words": words,
        },
    )

    return words


def normalize_alignment_word(
    value: str,
) -> str:
    """Kelimeyi yalnızca zaman eşleştirmesi için normalize eder."""
    value = clean(
        value
    ).casefold()

    return re.sub(
        r"[^\wçğıöşü'-]",
        "",
        value,
        flags=re.UNICODE,
    )


def narration_tokens(
    narration: str,
) -> list[str]:
    """Gerçek narration kelimelerini çıkarır."""
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
    """İki kelimenin yaklaşık benzerlik oranını döndürür."""
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


def align_narration_to_timestamps(
    narration: str,
    whisper_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Altyazı metnini narration'a, zamanları Whisper'a kilitler."""
    script_words = narration_tokens(
        narration
    )

    if not script_words:
        raise ComicFactoryError(
            "Narration kelimeleri bulunamadı."
        )

    if not whisper_words:
        raise ComicFactoryError(
            "Whisper timestamp bulunamadı."
        )

    aligned: list[
        dict[str, Any]
    ] = []

    whisper_index = 0
    previous_end = 0.0

    for script_index, script_word in enumerate(
        script_words
    ):
        best_index: int | None = None
        best_score = 0.0

        search_end = min(
            len(
                whisper_words
            ),
            whisper_index + 7,
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
            ) * 0.025

            if score > best_score:
                best_score = score
                best_index = candidate_index

        if (
            best_index is not None
            and best_score >= 0.48
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

        elif whisper_index < len(
            whisper_words
        ):
            reference = whisper_words[
                whisper_index
            ]

            reference_start = float(
                reference[
                    "start"
                ]
            )

            reference_end = float(
                reference[
                    "end"
                ]
            )

            start = max(
                previous_end,
                reference_start,
            )

            estimated_duration = max(
                0.10,
                min(
                    0.42,
                    reference_end
                    - reference_start,
                ),
            )

            end = (
                start
                + estimated_duration
            )

        else:
            remaining_words = max(
                1,
                len(
                    script_words
                )
                - script_index,
            )

            last_audio_end = float(
                whisper_words[
                    -1
                ][
                    "end"
                ]
            )

            remaining_time = max(
                0.12,
                last_audio_end
                - previous_end,
            )

            estimated_duration = (
                remaining_time
                / remaining_words
            )

            start = previous_end

            end = (
                start
                + estimated_duration
            )

        start = max(
            previous_end,
            start,
        )

        end = max(
            start + 0.06,
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

    audio_end = float(
        whisper_words[
            -1
        ][
            "end"
        ]
    )

    if (
        aligned
        and aligned[
            -1
        ][
            "end"
        ]
        > audio_end + 0.5
    ):
        scale = (
            audio_end
            / aligned[
                -1
            ][
                "end"
            ]
        )

        for item in aligned:
            item[
                "start"
            ] *= scale

            item[
                "end"
            ] *= scale

    save_json(
        AUDIO_DIR
        / "aligned_subtitle_words.json",
        aligned,
    )

    return aligned


def create_subtitle_groups(
    words: list[dict[str, Any]],
) -> list[list[int]]:
    """Ekran genişliğine göre 1-3 kelimelik gruplar oluşturur."""
    groups: list[
        list[int]
    ] = []

    current: list[
        int
    ] = []

    current_length = 0

    for index, item in enumerate(
        words
    ):
        word = clean(
            item[
                "word"
            ]
        )

        added_length = len(
            word
        ) + (
            1
            if current
            else 0
        )

        should_break = (
            bool(
                current
            )
            and (
                len(
                    current
                )
                >= SUBTITLE_MAX_WORDS
                or current_length
                + added_length
                > SUBTITLE_MAX_CHARACTERS
            )
        )

        if should_break:
            groups.append(
                current
            )

            current = []
            current_length = 0

        current.append(
            index
        )

        current_length += len(
            word
        ) + (
            1
            if len(
                current
            )
            > 1
            else 0
        )

        if re.search(
            r"[.!?…,:;]$",
            word,
        ):
            groups.append(
                current
            )

            current = []
            current_length = 0

    if current:
        groups.append(
            current
        )

    return groups


def create_word_to_group_map(
    groups: list[list[int]],
) -> dict[int, list[int]]:
    """Kelime indexinden subtitle grubuna harita oluşturur."""
    result: dict[
        int,
        list[int]
    ] = {}

    for group in groups:
        for index in group:
            result[
                index
            ] = group

    return result


def subtitle_scale_for_group(
    words: list[dict[str, Any]],
    group: list[int],
) -> int:
    """Uzun subtitle grubunda yatay ölçeği azaltır."""
    text = " ".join(
        clean(
            words[
                index
            ][
                "word"
            ]
        )
        for index in group
    )

    length = len(
        text
    )

    if length <= 12:
        return 100

    if length <= 16:
        return 96

    if length <= 20:
        return 91

    return 86


def all_emphasis_words(
    scenes: list[Scene],
) -> set[str]:
    """Story Director emphasis kelimelerini toplar."""
    output: set[
        str
    ] = set()

    for scene in scenes:
        for word in scene.emphasis_words:
            normalized = normalize_alignment_word(
                word
            )

            if normalized:
                output.add(
                    normalized
                )

    return output


def ass_time(
    seconds: float,
) -> str:
    """Saniyeyi ASS timestamp'e çevirir."""
    centiseconds = round(
        max(
            seconds,
            0.0,
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

    secs, cs = divmod(
        remainder,
        100,
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
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


def subtitle_chunk(
    words: list[dict[str, Any]],
    active_index: int,
    group_map: dict[int, list[int]],
    emphasis: set[str],
) -> str:
    """Safe-zone sınırlarını aşmayan dinamik subtitle üretir."""
    group = group_map.get(
        active_index,
        [
            active_index
        ],
    )

    group_scale = subtitle_scale_for_group(
        words,
        group,
    )

    output: list[
        str
    ] = []

    for index in group:
        raw_word = ass_escape(
            words[
                index
            ][
                "word"
            ]
        )

        normalized = normalize_alignment_word(
            raw_word
        )

        display_word = raw_word.upper()

        if index == active_index:
            active_scale = min(
                112,
                group_scale + 8,
            )

            if normalized in emphasis:
                output.append(
                    (
                        r"{"
                        rf"\1c&H0030D7FF&"
                        rf"\fscx{active_scale}"
                        rf"\fscy{active_scale}"
                        r"\bord5\shad2\b1"
                        r"}"
                        + display_word
                        + r"{\r}"
                    )
                )

            else:
                output.append(
                    (
                        r"{"
                        rf"\1c&H0030D7FF&"
                        rf"\fscx{active_scale}"
                        rf"\fscy{active_scale}"
                        r"\b1"
                        r"}"
                        + display_word
                        + r"{\r}"
                    )
                )

        else:
            output.append(
                (
                    r"{"
                    rf"\fscx{group_scale}"
                    rf"\fscy{group_scale}"
                    r"}"
                    + display_word
                    + r"{\r}"
                )
            )

    return " ".join(
        output
    )


def create_subtitles(
    whisper_words: list[dict[str, Any]],
    scenes: list[Scene],
    narration: str,
    offset: float,
) -> Path:
    """Kilitli Thor V2 subtitle sistemini üretir."""
    words = align_narration_to_timestamps(
        narration,
        whisper_words,
    )

    groups = create_subtitle_groups(
        words
    )

    group_map = create_word_to_group_map(
        groups
    )

    emphasis = all_emphasis_words(
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
            "Format: Name,Fontname,Fontsize,PrimaryColour,"
            "SecondaryColour,OutlineColour,BackColour,Bold,"
            "Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,"
            "Angle,BorderStyle,Outline,Shadow,Alignment,"
            "MarginL,MarginR,MarginV,Encoding"
        ),
        (
            f"Style: Main,DejaVu Sans,"
            f"{SUBTITLE_BASE_FONT_SIZE},"
            "&H00FFFFFF,&H00FFFFFF,"
            "&H00121212,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,5,2,2,"
            f"{SUBTITLE_MARGIN_LEFT},"
            f"{SUBTITLE_MARGIN_RIGHT},"
            f"{SUBTITLE_MARGIN_BOTTOM},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer,Start,End,Style,Name,"
            "MarginL,MarginR,MarginV,Effect,Text"
        ),
    ]

    for index, word in enumerate(
        words
    ):
        start = float(
            word[
                "start"
            ]
        ) + offset

        end = float(
            word[
                "end"
            ]
        ) + offset

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
            f"{subtitle_chunk(words, index, group_map, emphasis)}"
        )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return path


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


def audio_duration(
    ffmpeg: str,
    audio_file: Path,
) -> float:
    """Ses süresini ölçer."""
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                audio_file
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
            "Ses süresi okunamadı."
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


def scene_durations(
    scenes: list[Scene],
    aligned_words: list[dict[str, Any]],
    total_duration: float,
) -> list[float]:
    """Sahne sürelerini gerçek narration zamanlarına göre böler."""
    counts = [
        max(
            1,
            len(
                re.findall(
                    r"\w+",
                    scene.narration,
                    flags=re.UNICODE,
                )
            ),
        )
        for scene in scenes
    ]

    total_words = sum(
        counts
    )

    boundaries = [
        0.0
    ]

    cumulative = 0

    for count in counts[
        :-1
    ]:
        cumulative += count

        ratio = (
            cumulative
            / total_words
        )

        index = min(
            len(
                aligned_words
            )
            - 1,
            max(
                0,
                round(
                    ratio
                    * len(
                        aligned_words
                    )
                )
                - 1,
            ),
        )

        boundaries.append(
            float(
                aligned_words[
                    index
                ][
                    "end"
                ]
            )
        )

    boundaries.append(
        total_duration
    )

    durations = [
        max(
            0.7,
            boundaries[
                index + 1
            ]
            - boundaries[
                index
            ],
        )
        for index in range(
            len(
                boundaries
            )
            - 1
        )
    ]

    durations[
        -1
    ] += (
        total_duration
        - sum(
            durations
        )
    )

    return durations


def motion_filter(
    motion: str,
) -> str:
    """Sahneye uygun kamera hareketini üretir."""
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
            "z='min(zoom+0.00065,1.085)':"
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

    return (
        "scale=1080:1920,"
        + motions.get(
            motion,
            motions[
                "slow_push"
            ],
        )
        + ",format=yuv420p"
    )


def render_motion_segment(
    ffmpeg: str,
    frame: Path,
    output: Path,
    duration: float,
    motion: str,
) -> None:
    """Tek sahnenin hareketli video segmentini oluşturur."""
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
                frame
            ),
            "-t",
            f"{duration:.3f}",
            "-vf",
            motion_filter(
                motion
            ),
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
                output
            ),
        ],
        "Video sahnesi render edilemedi.",
    )


def build_xfade_filter(
    scenes: list[Scene],
    segment_durations: list[float],
) -> tuple[str, str]:
    """Segmentler için zincirlenmiş sinematik xfade filtresi üretir."""
    if len(
        scenes
    ) == 1:
        return (
            "[0:v]null[vout]",
            "[vout]",
        )

    filters: list[
        str
    ] = []

    current_label = (
        "[0:v]"
    )

    cumulative_duration = (
        segment_durations[
            0
        ]
    )

    for index in range(
        1,
        len(
            scenes
        ),
    ):
        previous_scene = scenes[
            index - 1
        ]

        transition_duration = (
            previous_scene.transition_duration
        )

        offset = max(
            0.0,
            cumulative_duration
            - transition_duration,
        )

        output_label = (
            f"[vx{index}]"
        )

        filters.append(
            f"{current_label}"
            f"[{index}:v]"
            f"xfade="
            f"transition={previous_scene.transition}:"
            f"duration={transition_duration:.3f}:"
            f"offset={offset:.3f}"
            f"{output_label}"
        )

        cumulative_duration = (
            cumulative_duration
            + segment_durations[
                index
            ]
            - transition_duration
        )

        current_label = output_label

    return (
        ";".join(
            filters
        ),
        current_label,
    )


def render_video(
    scenes: list[Scene],
    audio_file: Path,
    whisper_words: list[dict[str, Any]],
    narration: str,
    subtitle_offset: float,
) -> Path:
    """Sinematik geçişli final videoyu render eder."""
    print()
    print(
        "=" * 78
    )
    print(
        "11/11 - CINEMATIC FINAL RENDER"
    )
    print(
        "=" * 78
    )
    print()

    ffmpeg = ffmpeg_path()

    total_duration = audio_duration(
        ffmpeg,
        audio_file,
    )

    aligned_words = align_narration_to_timestamps(
        narration,
        whisper_words,
    )

    base_durations = scene_durations(
        scenes,
        aligned_words,
        total_duration,
    )

    shutil.rmtree(
        WORK_DIR,
        ignore_errors=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segment_files: list[
        Path
    ] = []

    rendered_durations: list[
        float
    ] = []

    for index, (
        scene,
        base_duration,
    ) in enumerate(
        zip(
            scenes,
            base_durations,
            strict=True,
        )
    ):
        transition_extra = (
            scene.transition_duration
            if index
            < len(
                scenes
            )
            - 1
            else 0.0
        )

        render_duration = (
            base_duration
            + transition_extra
        )

        frame_file = (
            WORK_DIR
            / f"frame_{index + 1:02d}.jpg"
        )

        segment_file = (
            WORK_DIR
            / f"segment_{index + 1:02d}.mp4"
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

        render_motion_segment(
            ffmpeg,
            frame_file,
            segment_file,
            render_duration,
            scene.motion,
        )

        segment_files.append(
            segment_file
        )

        rendered_durations.append(
            render_duration
        )

        print(
            f"✓ Sahne {scene.scene_number}: "
            f"{base_duration:.2f}s | "
            f"{scene.motion} | "
            f"{scene.transition}"
        )

    filter_complex, output_label = build_xfade_filter(
        scenes,
        rendered_durations,
    )

    silent_video = (
        WORK_DIR
        / "cinematic_silent.mp4"
    )

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    for segment in segment_files:
        command.extend(
            [
                "-i",
                str(
                    segment
                ),
            ]
        )

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            output_label,
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
                silent_video
            ),
        ]
    )

    run_ffmpeg(
        command,
        "Sinematik sahne geçişleri oluşturulamadı.",
    )

    ass_file = create_subtitles(
        whisper_words,
        scenes,
        narration,
        subtitle_offset,
    )

    timestamp = datetime.now(
        TZ
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive = (
        VIDEO_DIR
        / f"comic_factory_{timestamp}.mp4"
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
        "Final video render edilemedi.",
    )

    shutil.copy2(
        archive,
        LATEST_VIDEO_FILE,
    )

    if not LATEST_VIDEO_FILE.exists():
        raise ComicFactoryError(
            "latest.mp4 oluşmadı."
        )

    return archive


def save_scene_manifest(
    event: dict[str, Any],
    scenes: list[Scene],
    critique: dict[str, Any],
) -> None:
    """AI Director kararlarını sonraki geliştirmeler için kaydeder."""
    save_json(
        ASSET_DIR
        / event[
            "id"
        ]
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
            "director_critique": critique,
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
    """Olayı kullanılmış olarak işaretler."""
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
        and item.get(
            "event_key"
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
    """Mevcut YouTube yükleyicisini çalıştırır."""
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
            "YouTube yükleme başarısız oldu."
        )


def system_check(
    upload: bool,
) -> None:
    """Production bağımlılıklarını kontrol eder."""
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
        f"✓ Voice: "
        f"{GEMINI_TTS_VOICE} (LOCKED)"
    )

    print(
        "✓ Groq hazır"
    )

    print(
        "✓ FFmpeg hazır"
    )


def main() -> None:
    """Comic Factory V3 ana üretim hattı."""
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

        storyboard = build_storyboard(
            client,
            event,
        )

        image_results = image_search(
            event,
            storyboard,
        )

        candidates = download_candidates(
            event,
            image_results,
            max(
                14,
                args.max_images,
            ),
        )

        ranking_payload = rank_visuals(
            client,
            event,
            storyboard,
            candidates,
        )

        scenes = build_scenes(
            client,
            event,
            storyboard,
            candidates,
            ranking_payload,
            enable_reconstruction,
        )

        critique = supervising_director(
            client,
            scenes,
        )

        if critique.get(
            "weak_scenes"
        ):
            scenes = revise_weak_scenes(
                scenes,
                candidates,
                critique,
                event,
            )

        scenes = cinematic_director(
            client,
            scenes,
        )

        save_script_metadata(
            event,
            storyboard,
            scenes,
        )

        audio_file, narration = generate_voice(
            client,
            scenes,
        )

        words = align_words(
            audio_file,
            narration,
        )

        archive = render_video(
            scenes,
            audio_file,
            words,
            narration,
            args.subtitle_offset,
        )

        save_scene_manifest(
            event,
            scenes,
            critique,
        )

        mark_used(
            event
        )

        print()
        print(
            "=" * 78
        )
        print(
            "COMIC FACTORY V3 BAŞARILI"
        )
        print(
            "=" * 78
        )

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
            "AI reconstruction: "
            + (
                "AÇIK"
                if enable_reconstruction
                else "KAPALI"
            )
        )

        if args.upload:
            upload_youtube()

    except KeyboardInterrupt:
        raise SystemExit(
            1
        )

    except Exception as error:
        print()
        print(
            "=" * 78
        )
        print(
            "COMIC FACTORY V3 DURDU"
        )
        print(
            "=" * 78
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
