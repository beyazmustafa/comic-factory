# prototype_thor_galactus_v2.py
from __future__ import annotations

import argparse
import asyncio
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
from urllib.parse import quote, urlsplit, urlunsplit

import edge_tts
import imageio_ffmpeg
import requests
from ddgs import DDGS
from google import genai
from google.genai import types
from groq import Groq
from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PROTOTYPE_ROOT = DATA_DIR / "prototype" / "thor_galactus_v2"

RAW_DIR = PROTOTYPE_ROOT / "raw"
RESTORED_DIR = PROTOTYPE_ROOT / "restored"
GENERATED_DIR = PROTOTYPE_ROOT / "generated"
FRAME_DIR = PROTOTYPE_ROOT / "frames"
SEGMENT_DIR = PROTOTYPE_ROOT / "segments"
AUDIO_DIR = PROTOTYPE_ROOT / "audio"

STORYBOARD_FILE = PROTOTYPE_ROOT / "storyboard.json"
VISUAL_PLAN_FILE = PROTOTYPE_ROOT / "visual_plan.json"
CRITIQUE_FILE = PROTOTYPE_ROOT / "director_critique.json"
LESSONS_FILE = PROTOTYPE_ROOT / "lessons.json"
WORDS_FILE = PROTOTYPE_ROOT / "words.json"
ALIGNED_WORDS_FILE = PROTOTYPE_ROOT / "aligned_subtitle_words.json"
SUBTITLE_FILE = PROTOTYPE_ROOT / "subtitles.ass"
CONTACT_SHEET_FILE = PROTOTYPE_ROOT / "contact_sheet.jpg"
MANIFEST_FILE = PROTOTYPE_ROOT / "manifest.json"
VOICE_DIRECTION_FILE = PROTOTYPE_ROOT / "voice_direction.json"

OUTPUT_VIDEO = DATA_DIR / "videos" / "latest.mp4"
LATEST_SCRIPT_FILE = DATA_DIR / "scripts" / "latest.json"
ACTIVE_EVENT_FILE = DATA_DIR / "events" / "active_event.json"
USED_EVENTS_FILE = DATA_DIR / "events" / "used_events.json"
FACTORY_MODULE_FILE = ROOT / "comic_factory_event_source.py"

WIDTH = 1080
HEIGHT = 1920
FPS = 30

SCENE_COUNT = 14
MAX_DOWNLOAD_IMAGES = 36
MAX_VISION_IMAGES = 28

MIN_IMAGE_WIDTH = 500
MIN_IMAGE_HEIGHT = 500
MIN_IMAGE_PIXELS = 400_000

MIN_REAL_RELEVANCE = 80
MIN_RELEVANCE_FOR_REUSE = 90

SUBTITLE_MAX_WORDS = 3
SUBTITLE_MAX_CHARACTERS = 18
SUBTITLE_MARGIN_LEFT = 125
SUBTITLE_MARGIN_RIGHT = 125
SUBTITLE_MARGIN_BOTTOM = 390
SUBTITLE_BASE_FONT_SIZE = 56

REQUEST_TIMEOUT = 25

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.1-flash-tts-preview",
)

# Bu ses artık sabit. Değiştirmiyoruz.
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

EDGE_FALLBACK_VOICE = os.getenv(
    "EDGE_FALLBACK_VOICE",
    "tr-TR-AhmetNeural",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0 Safari/537.36"
)

EVENT_CONTEXT = """
Comic: Thor #6 (2020)
Writer: Donny Cates
Artist: Nic Klein

Odak:
Thor, Galactus ve Black Winter çatışmasının finali.

Özellikle:
Thor'un Galactus'un kozmik gücüyle ilişkisi,
Galactus'a karşı dönmesi,
Galactus'un ölümü
ve Black Winter finali.

Video:
Türkçe YouTube Shorts / Instagram Reels.

Amaç:
Slideshow değil, premium comic-video hissi.

Kurallar:
- Bilgi uydurma.
- Gerçek comic panellerini ana kaynak olarak kullan.
- Gerçek panel bulunamazsa ancak son çare AI reconstruction.
- Aynı görseli gereksiz tekrar etme.
- Video üzerinde issue adı, kaynak adı veya sahne numarası gösterme.
""".strip()



def load_factory_event() -> dict[str, Any]:
    """Load the event selected by the main Comic Factory."""
    if not ACTIVE_EVENT_FILE.exists():
        raise PrototypeError("active_event.json bulunamadı.")

    payload = json.loads(ACTIVE_EVENT_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PrototypeError("active_event.json geçersiz.")

    return payload


def factory_event_context(event: dict[str, Any]) -> str:
    """Build the dynamic event context consumed by the V2 directors."""
    characters = ", ".join(event.get("characters", []))
    beats = "\n".join(
        f"- {clean(item)}"
        for item in event.get("visual_beats", [])
        if clean(item)
    )

    return f"""
Comic event: {clean(event.get("event_title"))}
Publisher: {clean(event.get("publisher"))}
Series: {clean(event.get("series"))}
Issue: {clean(event.get("issue"))}
Year: {clean(event.get("publication_year"))}
Characters: {characters}

Hook:
{clean(event.get("hook"))}

Summary:
{clean(event.get("event_summary"))}

Power / key feat:
{clean(event.get("power_feat"))}

Known visual beats:
{beats}

Video:
Türkçe YouTube Shorts / Instagram Reels.

Amaç:
Slideshow değil, premium comic-video hissi.

Kurallar:
- Bilgi uydurma.
- Gerçek comic panellerini ana kaynak olarak kullan.
- Gerçek panel bulunamazsa ancak son çare AI reconstruction.
- Aynı görseli gereksiz tekrar etme.
- Video üzerinde issue adı, kaynak adı veya sahne numarası gösterme.
""".strip()


def dynamic_research_event(event: dict[str, Any]) -> list[dict[str, str]]:
    """Research the selected factory event instead of Thor #6."""
    series = clean(event.get("series"))
    issue = clean(event.get("issue"))
    title = clean(event.get("event_title"))
    characters = " ".join(event.get("characters", [])[:3])

    queries = [
        f'"{series}" "{issue}" {title} review',
        f'"{series}" "{issue}" {characters} comic',
        f'"{title}" comic panels review',
        f'"{series}" "{issue}" preview',
        f'"{series}" "{issue}" ending explained',
    ]

    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for source in event.get("sources", []):
        if not isinstance(source, dict):
            continue
        url = clean(source.get("url"))
        if url.startswith("http") and url not in seen:
            seen.add(url)
            output.append(
                {
                    "title": clean(source.get("name")),
                    "url": url,
                    "body": clean(source.get("supports")),
                }
            )

    for query in queries:
        print(f"Web: {query}")
        try:
            results = DDGS(timeout=12).text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=8,
            )
        except Exception as error:
            print(f"! Arama atlandı: {error}")
            continue

        for item in results or []:
            url = clean(item.get("href") or item.get("url"))
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            output.append(
                {
                    "title": clean(item.get("title")),
                    "url": url,
                    "body": clean(item.get("body")),
                }
            )

    if not output:
        raise PrototypeError("Seçilen event için araştırma sonucu bulunamadı.")

    return output


def dynamic_image_search(
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    """Search real comic imagery for the selected event."""
    series = clean(event.get("series"))
    issue = clean(event.get("issue"))
    title = clean(event.get("event_title"))
    characters = " ".join(event.get("characters", [])[:3])

    queries = [
        f'"{series}" "{issue}" comic panels',
        f'"{series}" "{issue}" preview images',
        f'"{title}" comic panels',
        f'"{series}" "{issue}" {characters}',
    ]

    for scene in storyboard["scenes"]:
        visual = clean(scene.get("visual_description"))
        if visual:
            queries.append(f'"{series}" "{issue}" {visual[:90]}')

    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for query in queries:
        try:
            results = DDGS(timeout=15).images(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=12,
            )
        except Exception as error:
            print(f"! Image search atlandı: {error}")
            continue

        for item in results or []:
            image_url = clean(item.get("image"))
            if not image_url.startswith("http") or image_url in seen:
                continue
            seen.add(image_url)
            output.append(
                {
                    "image_url": image_url,
                    "source_page": clean(item.get("url")),
                    "title": clean(item.get("title")),
                }
            )

    if not output:
        raise PrototypeError("Comic görseli bulunamadı.")

    return output


def build_dynamic_storyboard(
    client: genai.Client,
    event: dict[str, Any],
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Use the V2 Story Director for the event selected by Comic Factory."""
    evidence = "\n\n".join(
        f"TITLE: {item['title']}\nURL: {item['url']}\nSNIPPET: {item['body']}"
        for item in sources[:35]
    )
    context = factory_event_context(event)

    prompt = f"""
{context}

ARAŞTIRMA:
{evidence}

Sen Story Director'sın.

Premium bir 45-60 saniyelik Türkçe comic video yaz.
TAM {SCENE_COUNT} sahne oluştur.

Önemli:
- 120-150 kelime.
- Wikipedia özeti gibi olmasın.
- Her sahne yeni bilgi veya gerilim taşısın.
- İlk 2 sahne çok güçlü hook.
- Ortada escalation.
- Son 4 sahne gerçek payoff.
- Finalde kısa doğal soru olabilir.
- İngilizce özel isimleri orijinal yaz; Türkçe fonetik yazma.
- visual_description anlatılan anı gerçek panelde arayacak kadar somut olsun.
- narration ile visual_description aynı anı anlatmalı.
- Aynı görsel fikrini tekrar etme.
- Videoda kaynak adı, issue etiketi veya sahne başlığı gösterme.

motion:
slow_push, slow_pull, pan_left, pan_right, vertical_scan, impact, hold

Her sahne için emphasis_words 0-3 kelime.

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
      "motion": "slow_push",
      "emphasis_words": ["..."]
    }}
  ]
}}
"""

    payload = ask_json(client, prompt, temperature=0.55)
    scenes = payload.get("scenes", [])

    if not isinstance(scenes, list) or len(scenes) != SCENE_COUNT:
        raise PrototypeError(f"Storyboard tam {SCENE_COUNT} sahne değil.")

    payload["sources"] = sources
    save_json(STORYBOARD_FILE, payload)
    return payload


def save_factory_metadata(
    event: dict[str, Any],
    storyboard: dict[str, Any],
) -> None:
    """Write metadata in the format expected by the existing uploaders."""
    narration = " ".join(
        clean(scene.get("narration"))
        for scene in storyboard.get("scenes", [])
    )
    hashtags = storyboard.get("hashtags", ["#comics", "#shorts", "#reels"])
    if not isinstance(hashtags, list):
        hashtags = ["#comics", "#shorts", "#reels"]

    description = clean(storyboard.get("description"))
    source_lines = []
    for source in event.get("sources", []):
        if isinstance(source, dict) and clean(source.get("url")):
            source_lines.append(
                f"{clean(source.get('name'))}: {clean(source.get('url'))}"
            )

    full_description = description
    if source_lines:
        full_description += "\n\nKaynaklar:\n" + "\n".join(source_lines)

    payload = {
        "event_id": clean(event.get("id")),
        "generated_at": datetime.now().isoformat(),
        "script": {
            "title": clean(storyboard.get("title"))
            or clean(event.get("event_title")),
            "narration": narration,
            "description": description,
            "full_description": full_description,
            "hashtags": hashtags,
            "scene_narrations": [
                clean(scene.get("narration"))
                for scene in storyboard.get("scenes", [])
            ],
            "scene_captions": ["" for _ in storyboard.get("scenes", [])],
        },
    }
    save_json(LATEST_SCRIPT_FILE, payload)


def mark_factory_event_used(event: dict[str, Any]) -> None:
    """Mark the event only after a successful V2 render."""
    payload: dict[str, Any] = {"events": []}
    if USED_EVENTS_FILE.exists():
        try:
            loaded = json.loads(USED_EVENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            pass

    events = payload.setdefault("events", [])
    key = clean(event.get("event_key"))
    if not any(
        isinstance(item, dict) and clean(item.get("event_key")) == key
        for item in events
    ):
        events.append(
            {
                "event_key": key,
                "event_title": clean(event.get("event_title")),
                "series": clean(event.get("series")),
                "issue": clean(event.get("issue")),
                "completed_at": datetime.now().isoformat(),
            }
        )
    save_json(USED_EVENTS_FILE, payload)

class PrototypeError(RuntimeError):
    """Prototype üretimi başarısız olduğunda oluşur."""


@dataclass
class Candidate:
    """Gerçek comic görsel adayı."""

    candidate_id: str
    path: str
    source_url: str
    source_page: str
    title: str
    width: int
    height: int
    quality_score: float
    perceptual_hash: str


@dataclass
class RankedVisual:
    """Bir sahne için Gemini tarafından sıralanan görsel."""

    candidate_id: str
    relevance_score: int
    crop_box: list[float]
    reason: str


@dataclass
class Scene:
    """Final video sahnesi."""

    scene_number: int
    narration: str
    visual_description: str
    story_role: str
    motion: str
    emphasis_words: list[str]
    ranked_visuals: list[RankedVisual]
    selected_candidate_id: str
    relevance_score: int
    crop_box: list[float]
    visual_source: str
    visual_file: str


def parse_args() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""
    parser = argparse.ArgumentParser(
        description="Thor #6 AI Director V2 prototype."
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Tüm çalışma klasörlerini yeniden oluştur.",
    )

    parser.add_argument(
        "--enable-ai-reconstruction",
        action="store_true",
        help=(
            "Düşük relevance sahnelerinde Gemini Image "
            "reconstruction kullanımına izin ver."
        ),
    )

    parser.add_argument(
        "--edge-voice",
        action="store_true",
        help="Gemini TTS yerine Edge TTS kullan.",
    )

    return parser.parse_args()


def clean(value: Any) -> str:
    """Metni normalize eder."""
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def safe_http_url(value: Any) -> str:
    """Unicode içeren URL'yi güvenli ASCII HTTP URL'sine dönüştürür."""
    url = clean(value)

    if not url.startswith(("http://", "https://")):
        return ""

    try:
        parts = urlsplit(url)

        hostname = (
            parts.hostname.encode("idna").decode("ascii")
            if parts.hostname
            else ""
        )

        if not hostname:
            return ""

        netloc = hostname

        if parts.port:
            netloc += f":{parts.port}"

        if parts.username:
            credentials = quote(parts.username, safe="")

            if parts.password:
                credentials += ":" + quote(
                    parts.password,
                    safe="",
                )

            netloc = credentials + "@" + netloc

        return urlunsplit(
            (
                parts.scheme,
                netloc,
                quote(
                    parts.path,
                    safe="/:@-._~!$&'()*+,;=",
                ),
                quote(
                    parts.query,
                    safe="=&?/:@-._~!$'()*+,;",
                ),
                quote(parts.fragment, safe=""),
            )
        )

    except Exception:
        return ""


def require_env(name: str) -> str:
    """Zorunlu environment variable döndürür."""
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise PrototypeError(
            f"{name} bulunamadı."
        )

    return value


def ensure_directories() -> None:
    """Çalışma klasörlerini oluşturur."""
    for directory in (
        PROTOTYPE_ROOT,
        RAW_DIR,
        RESTORED_DIR,
        GENERATED_DIR,
        FRAME_DIR,
        SEGMENT_DIR,
        AUDIO_DIR,
        OUTPUT_VIDEO.parent,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def reset_directories() -> None:
    """Prototype çalışma alanını temizler."""
    shutil.rmtree(
        PROTOTYPE_ROOT,
        ignore_errors=True,
    )

    OUTPUT_VIDEO.unlink(
        missing_ok=True,
    )

    ensure_directories()


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


def parse_json_response(
    text: str,
) -> dict[str, Any]:
    """Gemini JSON cevabını güvenli ayrıştırır."""
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
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
        raise PrototypeError(
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
        return {
            "items": payload,
        }

    raise PrototypeError(
        "Beklenmeyen Gemini JSON yapısı."
    )


def create_gemini_client() -> genai.Client:
    """Gemini API istemcisi oluşturur."""
    return genai.Client(
        api_key=require_env(
            "GEMINI_API_KEY"
        )
    )


def is_retryable_gemini_error(
    error: Exception,
) -> bool:
    """Geçici Gemini servis hatalarını ayırt eder."""
    message = str(
        error
    ).casefold()

    retryable_terms = (
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
        for term in retryable_terms
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

            base_delay = delays[
                min(
                    attempt - 1,
                    len(delays) - 1,
                )
            ]

            jitter = random.uniform(
                0.0,
                3.0,
            )

            wait_seconds = (
                base_delay
                + jitter
            )

            print()
            print(
                f"! {operation_name} geçici hata verdi:"
            )
            print(
                f"  {type(error).__name__}: {error}"
            )
            print(
                f"  {wait_seconds:.1f} saniye "
                "sonra tekrar denenecek."
            )
            print()

            time.sleep(
                wait_seconds
            )

    raise PrototypeError(
        f"{operation_name} "
        f"{max_attempts} denemeden sonra başarısız oldu: "
        f"{last_error}"
    )


def ask_json(
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
        raise PrototypeError(
            "Gemini boş cevap döndürdü."
        )

    return parse_json_response(
        response.text
    )


def research_event() -> list[dict[str, str]]:
    """Thor #6 hakkında ücretsiz web araştırması yapar."""
    queries = [
        '"Thor #6" 2020 Galactus Black Winter review',
        '"Thor 6" Donny Cates Nic Klein Galactus Black Winter',
        '"Thor #6" Galactus death Black Winter',
        '"Thor #6" Marvel preview Galactus',
        '"Thor 2020 #6" Black Winter ending',
        '"Thor #6" God of Thunder Galactus review',
    ]

    output: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for query in queries:
        print(
            f"Web: {query}"
        )

        try:
            results = DDGS(
                timeout=12
            ).text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=8,
            )

        except Exception as error:
            print(
                f"! Arama atlandı: {error}"
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

            seen.add(
                url
            )

            output.append(
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

    if not output:
        raise PrototypeError(
            "Thor araştırması sonuç vermedi."
        )

    return output


def build_storyboard(
    client: genai.Client,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Story Director ile 14 sahnelik anlatı oluşturur."""
    evidence = "\n\n".join(
        (
            f"TITLE: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"SNIPPET: {item['body']}"
        )
        for item in sources[
            :35
        ]
    )

    prompt = f"""
{EVENT_CONTEXT}

ARAŞTIRMA:
{evidence}

Sen Story Director'sın.

Premium bir 45-60 saniyelik Türkçe comic video yaz.

TAM {SCENE_COUNT} sahne oluştur.

Önemli:
- 120-150 kelime.
- Wikipedia özeti gibi olmasın.
- Her sahne yeni bilgi veya gerilim taşısın.
- İlk 2 sahne çok güçlü hook.
- Ortada escalation.
- Son 4 sahne gerçek payoff.
- Son cümlede kısa doğal soru olabilir.
- "Thor", "Galactus", "Black Winter", "Marvel"
  gibi İngilizce özel isimleri orijinal yaz.
- Bu isimlerin Türkçe fonetik yazımını YAPMA.
- Görsel açıklaması çok somut olsun.
- Aynı görsel fikrini tekrar etme.
- visual_description tek bir comic panelde aranabilecek kadar net olsun.
- Videoda kaynak adı, issue etiketi veya başlık gösterilmeyecek.

motion seçenekleri:
slow_push
slow_pull
pan_left
pan_right
vertical_scan
impact
hold

impact yalnızca gerçek büyük reveal anlarında.

Her sahne için emphasis_words:
altyazıda özellikle vurgulanması gereken 0-3 kelime.

SADECE JSON:

{{
  "title": "...",
  "description": "...",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "...",
      "visual_description": "...",
      "story_role": "hook",
      "motion": "slow_push",
      "emphasis_words": ["Thor"]
    }}
  ]
}}
"""

    payload = ask_json(
        client,
        prompt,
        temperature=0.55,
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
        raise PrototypeError(
            f"Storyboard tam {SCENE_COUNT} sahne değil."
        )

    payload[
        "sources"
    ] = sources

    save_json(
        STORYBOARD_FILE,
        payload,
    )

    return payload


def image_search(
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    """Gerçek comic görsellerini arar."""
    queries = [
        '"Thor #6" comic panels Galactus',
        '"Thor #6" Black Winter comic page',
        '"Thor 6" 2020 Nic Klein panels',
        '"Thor #6" Galactus death panels',
        '"Thor #6" Black Winter ending comic',
        '"Thor #6" Marvel preview images',
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
                f'"Thor #6" {visual[:80]}'
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
                not image_url.startswith("http")
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
        raise PrototypeError(
            "Comic görseli bulunamadı."
        )

    return output


def average_hash(
    image: Image.Image,
) -> str:
    """Basit perceptual hash üretir."""
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
        if pixel >= average
        else "0"
        for pixel in pixels
    )

    return hex(
        int(
            bits,
            2,
        )
    )


def download_candidates(
    results: list[dict[str, str]],
) -> list[Candidate]:
    """Kaliteli ve benzersiz comic görsellerini indirir."""
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    candidates: list[
        Candidate
    ] = []

    hashes: set[str] = set()

    for item in results:
        if len(
            candidates
        ) >= MAX_DOWNLOAD_IMAGES:
            break

        try:
            image_url = safe_http_url(
                item.get(
                    "image_url"
                )
            )

            source_page = safe_http_url(
                item.get(
                    "source_page"
                )
            )

            if not image_url:
                continue

            request_headers = {
                "User-Agent": USER_AGENT,
            }

            if source_page:
                request_headers[
                    "Referer"
                ] = source_page

            response = session.get(
                image_url,
                headers=request_headers,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

        except (
            requests.RequestException,
            UnicodeEncodeError,
            UnicodeError,
            ValueError,
        ):
            continue

        if len(
            response.content
        ) < 15_000:
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
            width < MIN_IMAGE_WIDTH
            or height < MIN_IMAGE_HEIGHT
            or width
            * height
            < MIN_IMAGE_PIXELS
        ):
            continue

        phash = average_hash(
            image
        )

        if phash in hashes:
            continue

        hashes.add(
            phash
        )

        candidate_id = (
            f"real_{len(candidates) + 1:02d}"
        )

        file_path = (
            RAW_DIR
            / f"{candidate_id}.jpg"
        )

        image.save(
            file_path,
            "JPEG",
            quality=95,
            optimize=True,
        )

        megapixels = (
            width
            * height
            / 1_000_000
        )

        quality = min(
            100.0,
            40.0
            + megapixels
            * 14.0
            + min(
                width,
                height,
            )
            / 80.0,
        )

        candidate = Candidate(
            candidate_id=candidate_id,
            path=str(
                file_path
            ),
            source_url=image_url,
            source_page=source_page,
            title=item[
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

        candidates.append(
            candidate
        )

        print(
            f"✓ {candidate_id}: "
            f"{width}x{height}"
        )

    if len(
        candidates
    ) < 8:
        raise PrototypeError(
            "Yeterli farklı comic görseli bulunamadı. "
            f"Bulunan: {len(candidates)}"
        )

    return candidates


def thumbnail_bytes(
    path: Path,
) -> bytes:
    """Vision için küçük JPEG üretir."""
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


def vision_rank_scenes(
    client: genai.Client,
    storyboard: dict[str, Any],
    candidates: list[Candidate],
) -> list[dict[str, Any]]:
    """Visual Director her sahne için ilk 3 görseli seçer."""
    candidates = candidates[
        :MAX_VISION_IMAGES
    ]

    scenes_text = "\n\n".join(
        (
            f"SCENE {scene['scene_number']}\n"
            f"NARRATION: {scene['narration']}\n"
            f"VISUAL NEEDED: {scene['visual_description']}"
        )
        for scene in storyboard[
            "scenes"
        ]
    )

    prompt = f"""
Sen premium comic video Visual Director'sın.

EVENT:
Thor #6 (2020), Thor / Galactus / Black Winter.

SAHNELER:
{scenes_text}

Aşağıda gerçek comic görselleri var.

Her sahne için EN İYİ 3 adayı sırala.

Kurallar:
- Sadece karakter aynı diye yüksek puan verme.
- Anlatılan olay gerçekten görüntüde bulunmalı.
- Kapak, gerçek olay panelinden düşük değerlidir.
- Aynı resmi her sahneye vermekten kaçın.
- Bir comic sayfasının içindeki doğru paneli crop etmek gerekiyorsa
  crop_box döndür.
- crop_box normalize 0-1 koordinatıdır:
  [left, top, right, bottom]
- Eğer tüm sayfa kullanılmalıysa:
  [0, 0, 1, 1]
- Ana karakter veya aksiyon crop dışında kalmasın.
- Crop gereksiz dar olmasın.

JSON:

{{
  "scenes": [
    {{
      "scene_number": 1,
      "ranked_visuals": [
        {{
          "candidate_id": "real_01",
          "relevance_score": 94,
          "crop_box": [0.05, 0.10, 0.95, 0.88],
          "reason": "..."
        }},
        {{
          "candidate_id": "real_02",
          "relevance_score": 87,
          "crop_box": [0, 0, 1, 1],
          "reason": "..."
        }},
        {{
          "candidate_id": "real_03",
          "relevance_score": 75,
          "crop_box": [0, 0, 1, 1],
          "reason": "..."
        }}
      ]
    }}
  ]
}}

Tam {SCENE_COUNT} scene döndür.
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
                f"QUALITY: {candidate.quality_score}"
            )
        )

        contents.append(
            types.Part.from_bytes(
                data=thumbnail_bytes(
                    Path(
                        candidate.path
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

    if len(
        scenes
    ) != SCENE_COUNT:
        raise PrototypeError(
            "Visual Director sahne sayısı hatalı."
        )

    return scenes


def normalize_crop_box(
    crop_box: Any,
) -> list[float]:
    """Crop koordinatlarını güvenli normalize eder."""
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
            0.9,
            left,
        ),
    )

    top = max(
        0.0,
        min(
            0.9,
            top,
        ),
    )

    right = max(
        left + 0.1,
        min(
            1.0,
            right,
        ),
    )

    bottom = max(
        top + 0.1,
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


def choose_unique_visuals(
    ranking_payload: list[dict[str, Any]],
) -> dict[int, RankedVisual]:
    """Aynı görsel tekrarını azaltarak final seçim yapar."""
    chosen: dict[
        int,
        RankedVisual
    ] = {}

    usage: dict[
        str,
        int
    ] = {}

    for scene_data in ranking_payload:
        scene_number = int(
            scene_data.get(
                "scene_number",
                0,
            )
        )

        ranked_raw = scene_data.get(
            "ranked_visuals",
            [],
        )

        ranked: list[
            RankedVisual
        ] = []

        for item in ranked_raw:
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

        if not ranked:
            raise PrototypeError(
                f"Sahne {scene_number} için görsel yok."
            )

        selected = ranked[
            0
        ]

        for option in ranked:
            used_count = usage.get(
                option.candidate_id,
                0,
            )

            if used_count == 0:
                selected = option
                break

            if (
                used_count == 1
                and option.relevance_score
                >= MIN_RELEVANCE_FOR_REUSE
            ):
                selected = option
                break

        chosen[
            scene_number
        ] = selected

        usage[
            selected.candidate_id
        ] = (
            usage.get(
                selected.candidate_id,
                0,
            )
            + 1
        )

    return chosen


def crop_candidate(
    candidate: Candidate,
    crop_box: list[float],
    scene_number: int,
) -> Path:
    """Seçilen panel bölgesini crop edip restore eder."""
    output = (
        RESTORED_DIR
        / (
            f"scene_{scene_number:02d}_"
            f"{candidate.candidate_id}.png"
        )
    )

    with Image.open(
        candidate.path
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

    largest_side = max(
        image.size
    )

    if largest_side < 2200:
        scale = min(
            2.2,
            2200
            / max(
                1,
                largest_side,
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

    image = image.filter(
        ImageFilter.MedianFilter(
            size=3
        )
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        1.05
    )

    image = ImageEnhance.Color(
        image
    ).enhance(
        1.02
    )

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.3,
            percent=110,
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
    scene_data: dict[str, Any],
    reference_paths: list[Path],
) -> Path:
    """Eksik sahneyi referans comic görsellerinden oluşturur."""
    scene_number = int(
        scene_data[
            "scene_number"
        ]
    )

    output = (
        GENERATED_DIR
        / f"scene_{scene_number:02d}.jpg"
    )

    prompt = f"""
Create a premium 9:16 American comic-book illustration.

This is an artistic reconstruction for a video about Thor #6 (2020).

EXACT MOMENT:
{clean(scene_data["visual_description"])}

NARRATION:
{clean(scene_data["narration"])}

Use supplied references for:
- Thor's appearance
- Galactus's appearance
- costume continuity
- comic-era color language
- cosmic atmosphere

Do NOT copy a source panel composition exactly.

Requirements:
- exceptionally detailed professional comic artwork
- cinematic composition
- clear focal subject
- dynamic lighting
- dramatic cosmic scale
- correct readable anatomy
- no text
- no speech bubble
- no logo
- no watermark
- no issue number
- no UI
- no fake caption
""".strip()

    interaction_input: list[
        dict[str, str]
    ] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for reference_path in reference_paths[
        :3
    ]:
        mime_type = (
            "image/png"
            if reference_path.suffix.lower()
            == ".png"
            else "image/jpeg"
        )

        interaction_input.append(
            {
                "type": "image",
                "data": base64.b64encode(
                    reference_path.read_bytes()
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
            f"AI Reconstruction Scene {scene_number}"
        ),
    )

    if interaction.output_image is None:
        raise PrototypeError(
            "Gemini Image görsel döndürmedi."
        )

    output.write_bytes(
        base64.b64decode(
            interaction.output_image.data
        )
    )

    return output


def build_scene_objects(
    client: genai.Client,
    storyboard: dict[str, Any],
    candidates: list[Candidate],
    rankings: list[dict[str, Any]],
    enable_reconstruction: bool,
) -> list[Scene]:
    """Visual Director planını Scene nesnelerine çevirir."""
    candidate_map = {
        item.candidate_id: item
        for item in candidates
    }

    chosen = choose_unique_visuals(
        rankings
    )

    ranking_map = {
        int(
            item[
                "scene_number"
            ]
        ): item
        for item in rankings
    }

    scenes: list[
        Scene
    ] = []

    for storyboard_scene in storyboard[
        "scenes"
    ]:
        number = int(
            storyboard_scene[
                "scene_number"
            ]
        )

        selected = chosen[
            number
        ]

        candidate = candidate_map.get(
            selected.candidate_id
        )

        if candidate is None:
            raise PrototypeError(
                f"Candidate bulunamadı: "
                f"{selected.candidate_id}"
            )

        visual_file = crop_candidate(
            candidate,
            selected.crop_box,
            number,
        )

        visual_source = (
            "real_comic"
        )

        if (
            selected.relevance_score
            < MIN_REAL_RELEVANCE
            and enable_reconstruction
        ):
            ranking = ranking_map[
                number
            ].get(
                "ranked_visuals",
                [],
            )

            references: list[
                Path
            ] = []

            for item in ranking[
                :3
            ]:
                reference_candidate = (
                    candidate_map.get(
                        clean(
                            item.get(
                                "candidate_id"
                            )
                        )
                    )
                )

                if reference_candidate is None:
                    continue

                references.append(
                    Path(
                        reference_candidate.path
                    )
                )

            try:
                visual_file = generate_reconstruction(
                    client,
                    storyboard_scene,
                    references,
                )

                visual_source = (
                    "ai_reconstruction"
                )

                print(
                    f"🎨 Scene {number}: "
                    "AI reconstruction"
                )

            except Exception as error:
                print(
                    f"! Scene {number} reconstruction "
                    f"başarısız: {error}"
                )

        ranked_visuals: list[
            RankedVisual
        ] = []

        for raw_visual in ranking_map[
            number
        ].get(
            "ranked_visuals",
            [],
        ):
            ranked_visuals.append(
                RankedVisual(
                    candidate_id=clean(
                        raw_visual.get(
                            "candidate_id"
                        )
                    ),
                    relevance_score=int(
                        raw_visual.get(
                            "relevance_score",
                            0,
                        )
                        or 0
                    ),
                    crop_box=normalize_crop_box(
                        raw_visual.get(
                            "crop_box"
                        )
                    ),
                    reason=clean(
                        raw_visual.get(
                            "reason"
                        )
                    ),
                )
            )

        scenes.append(
            Scene(
                scene_number=number,
                narration=clean(
                    storyboard_scene[
                        "narration"
                    ]
                ),
                visual_description=clean(
                    storyboard_scene[
                        "visual_description"
                    ]
                ),
                story_role=clean(
                    storyboard_scene.get(
                        "story_role"
                    )
                ),
                motion=clean(
                    storyboard_scene.get(
                        "motion"
                    )
                )
                or "slow_push",
                emphasis_words=[
                    clean(
                        word
                    )
                    for word in storyboard_scene.get(
                        "emphasis_words",
                        [],
                    )
                    if clean(
                        word
                    )
                ],
                ranked_visuals=ranked_visuals,
                selected_candidate_id=(
                    selected.candidate_id
                ),
                relevance_score=(
                    selected.relevance_score
                ),
                crop_box=(
                    selected.crop_box
                ),
                visual_source=visual_source,
                visual_file=str(
                    visual_file
                ),
            )
        )

    return scenes


def unique_visual_ratio(
    scenes: list[Scene],
) -> float:
    """Final sahnelerdeki benzersiz görsel oranını ölçer."""
    if not scenes:
        return 0.0

    unique = {
        scene.visual_file
        for scene in scenes
    }

    return (
        len(
            unique
        )
        / len(
            scenes
        )
    )


def compose_vertical(
    visual_file: Path,
) -> Image.Image:
    """Comic görselini 9:16 kompozisyona dönüştürür."""
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

    foreground = source.copy()

    foreground.thumbnail(
        (
            WIDTH - 26,
            1540,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = background.copy()

    x = (
        WIDTH
        - foreground.width
    ) // 2

    y = (
        105
        + (
            1510
            - foreground.height
        )
        // 2
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
    """AI Director için sahne contact sheet oluşturur."""
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

        column = (
            index
            % columns
        )

        row = (
            index
            // columns
        )

        x = (
            column
            * cell_width
        )

        y = (
            row
            * cell_height
        )

        sheet.paste(
            frame,
            (
                x,
                y,
            ),
        )

    sheet.save(
        CONTACT_SHEET_FILE,
        "JPEG",
        quality=92,
    )

    return CONTACT_SHEET_FILE


def critique_visuals(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """AI Director contact sheet'i değerlendirir."""
    contact_sheet = build_contact_sheet(
        scenes
    )

    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\n"
            f"NARRATION: {scene.narration}\n"
            f"VISUAL: {scene.visual_description}\n"
            f"RELEVANCE: {scene.relevance_score}\n"
            f"SOURCE: {scene.visual_source}\n"
            f"MOTION: {scene.motion}"
        )
        for scene in scenes
    )

    prompt = f"""
You are the supervising director of a premium vertical comic video.

Review the contact sheet and scene plan.

{plan}

Evaluate 0-10:
- visual_relevance
- visual_diversity
- crop_quality
- cinematic_flow
- repetition
- story_visual_match
- overall

Return weak scene numbers.

A scene is weak if:
- wrong visual
- crop misses main action
- nearly identical to surrounding scene
- cover used when better action should exist
- visual does not support narration

JSON:
{{
  "visual_relevance": 0,
  "visual_diversity": 0,
  "crop_quality": 0,
  "cinematic_flow": 0,
  "repetition": 0,
  "story_visual_match": 0,
  "overall": 0,
  "weak_scenes": [2, 6],
  "lessons": ["..."]
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

    critique[
        "unique_visual_ratio"
    ] = round(
        unique_visual_ratio(
            scenes
        ),
        3,
    )

    save_json(
        CRITIQUE_FILE,
        critique,
    )

    save_json(
        LESSONS_FILE,
        {
            "created_at": datetime.now().isoformat(),
            "lessons": critique.get(
                "lessons",
                [],
            ),
        },
    )

    return critique


def improve_weak_scenes(
    scenes: list[Scene],
    candidates: list[Candidate],
    critique: dict[str, Any],
) -> list[Scene]:
    """Zayıf sahnelerde alternatif gerçek panel kullanır."""
    weak: set[int] = set()

    for number in critique.get(
        "weak_scenes",
        [],
    ):
        try:
            weak.add(
                int(
                    number
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    candidate_map = {
        item.candidate_id: item
        for item in candidates
    }

    currently_used = {
        scene.selected_candidate_id
        for scene in scenes
    }

    for scene in scenes:
        if scene.scene_number not in weak:
            continue

        for alternative in scene.ranked_visuals[
            1:
        ]:
            if alternative.relevance_score < 72:
                continue

            if alternative.candidate_id in currently_used:
                continue

            candidate = candidate_map.get(
                alternative.candidate_id
            )

            if candidate is None:
                continue

            new_file = crop_candidate(
                candidate,
                alternative.crop_box,
                scene.scene_number,
            )

            currently_used.discard(
                scene.selected_candidate_id
            )

            currently_used.add(
                alternative.candidate_id
            )

            scene.selected_candidate_id = (
                alternative.candidate_id
            )

            scene.relevance_score = (
                alternative.relevance_score
            )

            scene.crop_box = (
                alternative.crop_box
            )

            scene.visual_source = (
                "real_comic_alternative"
            )

            scene.visual_file = str(
                new_file
            )

            break

    return scenes


def create_voice_direction(
    client: genai.Client,
    scenes: list[Scene],
) -> dict[str, Any]:
    """Voice Director anlatım ve telaffuz talimatı oluşturur."""
    narration = " ".join(
        scene.narration
        for scene in scenes
    )

    prompt = f"""
Sen Türkçe bir premium YouTube Shorts Voice Director'sın.

METİN:
{narration}

Bu metnin ANLAMINI DEĞİŞTİRME.

Ama anlatıcının doğal duyulması için bir voice direction üret.

Özel isimler:
Thor
Galactus
Black Winter
Marvel
Silver Surfer
Donny Cates
Nic Klein

KRİTİK:
- İngilizce özel isimleri Türkçe fonetik yazıma çevirme.
- "Thor" -> "Tor" gibi değiştirme.
- İngilizce özel isimleri doğal İngilizce telaffuz et.
- Çevresindeki Türkçe cümleler doğal İstanbul Türkçesi olsun.
- Haber spikeri gibi konuşmasın.
- Belgesel sesi gibi ağır olmasın.
- Yapay TikTok robot sesi gibi olmasın.
- 25-35 yaşlarında doğal erkek YouTube storyteller hissi.
- Arkadaşına inanılmaz bir comic olayını anlatıyormuş gibi.
- Enerjik ama bağırmayan.
- Büyük reveal öncesinde kısa pause.
- Cümle sonlarını sürekli aynı melodiyle bitirme.
- İngilizce isimlere geçince aksanı doğal İngilizce telaffuza geçir,
  sonra Türkçeye geri dön.
- Yaklaşık 1.0x doğal konuşma temposu.

JSON:
{{
  "exact_narration": "...",
  "style_instruction": "...",
  "pronunciation_instruction": "...",
  "pace_instruction": "..."
}}

exact_narration orijinal narration ile aynı kelimeleri korusun.
"""

    return ask_json(
        client,
        prompt,
        temperature=0.25,
    )


def write_pcm_wave(
    path: Path,
    pcm: bytes,
) -> None:
    """Gemini PCM sesini WAV olarak kaydeder."""
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


def generate_gemini_voice(
    client: genai.Client,
    voice_direction: dict[str, Any],
) -> Path:
    """Gemini Gacrux ile doğal narration üretir."""
    output = (
        AUDIO_DIR
        / "narration_gemini.wav"
    )

    narration = clean(
        voice_direction.get(
            "exact_narration"
        )
    )

    if not narration:
        raise PrototypeError(
            "Voice Director narration üretmedi."
        )

    # Bu prompt ve voice sabit tutuluyor.
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
Whenever an English proper noun appears, especially
Thor, Galactus, Black Winter, Marvel, Silver Surfer,
pronounce that proper noun naturally in English,
then return smoothly to Turkish pronunciation.

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
        operation_name="Gemini TTS",
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
        raise PrototypeError(
            "Gemini TTS audio döndürmedi."
        ) from error

    if not pcm:
        raise PrototypeError(
            "Gemini TTS boş audio döndürdü."
        )

    write_pcm_wave(
        output,
        pcm,
    )

    latest_audio = DATA_DIR / "audio" / "latest.wav"
    latest_audio.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, latest_audio)
    return output


def generate_edge_fallback(
    narration: str,
) -> Path:
    """Gemini TTS kullanılamazsa Edge fallback üretir."""
    output = (
        AUDIO_DIR
        / "narration_edge.mp3"
    )

    async def create() -> None:
        communicate = edge_tts.Communicate(
            narration,
            EDGE_FALLBACK_VOICE,
            rate="+3%",
            pitch="+0Hz",
        )

        await communicate.save(
            str(
                output
            )
        )

    asyncio.run(
        create()
    )

    return output


def generate_voice(
    client: genai.Client,
    scenes: list[Scene],
    force_edge: bool,
) -> tuple[Path, str]:
    """Voice Director + TTS pipeline."""
    voice_direction = create_voice_direction(
        client,
        scenes,
    )

    save_json(
        VOICE_DIRECTION_FILE,
        voice_direction,
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

    if force_edge:
        return (
            generate_edge_fallback(
                narration
            ),
            narration,
        )

    try:
        audio = generate_gemini_voice(
            client,
            voice_direction,
        )

        print(
            f"✓ Gemini TTS voice: {GEMINI_TTS_VOICE}"
        )

        return (
            audio,
            narration,
        )

    except Exception as error:
        print(
            f"! Gemini TTS başarısız: {error}"
        )

        print(
            "→ Edge TTS fallback kullanılıyor."
        )

        return (
            generate_edge_fallback(
                narration
            ),
            narration,
        )


def groq_word_timestamps(
    audio_path: Path,
    narration: str,
) -> list[dict[str, Any]]:
    """Groq Whisper ile kelime zamanlarını çıkarır."""
    groq = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with audio_path.open(
        "rb"
    ) as audio_file:
        transcription = (
            groq.audio.transcriptions.create(
                file=audio_file,
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

    raw_words = getattr(
        transcription,
        "words",
        [],
    )

    words: list[
        dict[str, Any]
    ] = []

    for raw in raw_words or []:
        if isinstance(
            raw,
            dict,
        ):
            word = clean(
                raw.get(
                    "word"
                )
            )

            start = raw.get(
                "start"
            )

            end = raw.get(
                "end"
            )

        else:
            word = clean(
                getattr(
                    raw,
                    "word",
                    "",
                )
            )

            start = getattr(
                raw,
                "start",
                None,
            )

            end = getattr(
                raw,
                "end",
                None,
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
        raise PrototypeError(
            "Groq word timestamp yeterli değil."
        )

    save_json(
        WORDS_FILE,
        words,
    )

    return words


def normalize_alignment_word(
    value: str,
) -> str:
    """Kelimeyi yalnızca timestamp eşleştirmesi için normalize eder."""
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
    """Gerçek Voice Director narration kelimelerini çıkarır."""
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
    """İki kelimenin yaklaşık eşleşme oranını döndürür."""
    first_normalized = normalize_alignment_word(
        first
    )

    second_normalized = normalize_alignment_word(
        second
    )

    if (
        not first_normalized
        or not second_normalized
    ):
        return 0.0

    if first_normalized == second_normalized:
        return 1.0

    return SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
    ).ratio()


def align_narration_to_timestamps(
    narration: str,
    whisper_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ekranda her zaman gerçek narration metnini kullanır.

    Groq yalnızca kelime zaman referansı sağlar.
    """
    script_words = narration_tokens(
        narration
    )

    if not script_words:
        raise PrototypeError(
            "Narration kelimeleri bulunamadı."
        )

    if not whisper_words:
        raise PrototypeError(
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

        else:
            if whisper_index < len(
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
        ALIGNED_WORDS_FILE,
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
        )

        if current:
            added_length += 1

        should_break = (
            current
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
        )

        if len(
            current
        ) > 1:
            current_length += 1

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
    """Uzun gruplarda altyazıyı yatay küçültür."""
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
) -> None:
    """Gerçek narration metninden güvenli altyazı oluşturur."""
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
            "-1,0,0,0,100,100,0,0,"
            "1,5,2,2,"
            f"{SUBTITLE_MARGIN_LEFT},"
            f"{SUBTITLE_MARGIN_RIGHT},"
            f"{SUBTITLE_MARGIN_BOTTOM},"
            "1"
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
        )

        end = max(
            start + 0.05,
            float(
                word[
                    "end"
                ]
            ),
        )

        text = subtitle_chunk(
            words,
            index,
            group_map,
            emphasis,
        )

        lines.append(
            "Dialogue: 0,"
            f"{ass_time(start)},"
            f"{ass_time(end)},"
            "Main,,0,0,0,,"
            f"{text}"
        )

    SUBTITLE_FILE.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def get_ffmpeg() -> str:
    """FFmpeg executable yolunu döndürür."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(
    command: list[str],
    error_message: str,
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
        raise PrototypeError(
            error_message
            + "\n"
            + process.stderr[
                -3500:
            ]
        )


def get_audio_duration(
    ffmpeg: str,
    path: Path,
) -> float:
    """Audio süresini ölçer."""
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                path
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
        raise PrototypeError(
            "Audio duration okunamadı."
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
    words: list[dict[str, Any]],
    total_duration: float,
) -> list[float]:
    """Sahne sürelerini narration dağılımına göre çıkarır."""
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
                words
            )
            - 1,
            max(
                0,
                round(
                    ratio
                    * len(
                        words
                    )
                )
                - 1,
            ),
        )

        boundaries.append(
            float(
                words[
                    index
                ][
                    "end"
                ]
            )
        )

    boundaries.append(
        total_duration
    )

    durations = []

    for index in range(
        len(
            boundaries
        )
        - 1
    ):
        durations.append(
            max(
                0.65,
                boundaries[
                    index + 1
                ]
                - boundaries[
                    index
                ],
            )
        )

    difference = (
        total_duration
        - sum(
            durations
        )
    )

    durations[
        -1
    ] += difference

    return durations


def motion_filter(
    motion: str,
) -> str:
    """Sahneye uygun kamera hareketi filtresi üretir."""
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


def render_video(
    scenes: list[Scene],
    audio_file: Path,
    words: list[dict[str, Any]],
    narration: str,
) -> None:
    """V2 videosunu render eder."""
    ffmpeg = get_ffmpeg()

    total_duration = get_audio_duration(
        ffmpeg,
        audio_file,
    )

    durations = scene_durations(
        scenes,
        words,
        total_duration,
    )

    shutil.rmtree(
        FRAME_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        SEGMENT_DIR,
        ignore_errors=True,
    )

    FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SEGMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segment_files: list[
        Path
    ] = []

    for scene, duration in zip(
        scenes,
        durations,
        strict=True,
    ):
        frame_file = (
            FRAME_DIR
            / f"scene_{scene.scene_number:02d}.jpg"
        )

        segment_file = (
            SEGMENT_DIR
            / f"scene_{scene.scene_number:02d}.mp4"
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
                    frame_file
                ),
                "-t",
                f"{duration:.3f}",
                "-vf",
                motion_filter(
                    scene.motion
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
                    segment_file
                ),
            ],
            (
                f"Scene {scene.scene_number} "
                "render başarısız."
            ),
        )

        segment_files.append(
            segment_file
        )

        print(
            f"✓ Scene {scene.scene_number}: "
            f"{duration:.2f}s | "
            f"{scene.motion} | "
            f"{scene.visual_source}"
        )

    concat_file = (
        PROTOTYPE_ROOT
        / "concat.txt"
    )

    concat_file.write_text(
        "\n".join(
            "file '"
            + file.resolve().as_posix()
            + "'"
            for file in segment_files
        ),
        encoding="utf-8",
    )

    silent_video = (
        PROTOTYPE_ROOT
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
        "Segment concat başarısız.",
    )

    create_subtitles(
        words,
        scenes,
        narration,
    )

    OUTPUT_VIDEO.unlink(
        missing_ok=True
    )

    subtitle_filter = (
        "ass="
        + SUBTITLE_FILE.resolve()
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
                OUTPUT_VIDEO
            ),
        ],
        "Final V2 render başarısız.",
    )


def save_visual_plan(
    scenes: list[Scene],
) -> None:
    """Visual planı debug için kaydeder."""
    save_json(
        VISUAL_PLAN_FILE,
        [
            asdict(
                scene
            )
            for scene in scenes
        ],
    )


def run_v2_pipeline() -> int:
    """Thor V2 rendering pipeline."""
    arguments = parse_args()

    ensure_directories()

    if arguments.rebuild:
        reset_directories()

    print()
    print("=" * 78)
    print("COMIC FACTORY - THOR V2 ENGINE")
    print("=" * 78)
    print()

    print(
        "Voice Director     : ON"
    )

    print(
        "Voice              : Gacrux LOCKED"
    )

    print(
        "Visual Director    : ON"
    )

    print(
        "Subtitle Director  : ON"
    )

    print(
        "AI Reconstruction  : "
        + (
            "ON"
            if arguments.enable_ai_reconstruction
            else "OFF"
        )
    )

    print()

    client = create_gemini_client()
    event = load_factory_event()

    global EVENT_CONTEXT
    EVENT_CONTEXT = factory_event_context(event)

    print(f"Event              : {clean(event.get('event_title'))}")
    print(f"Comic              : {clean(event.get('series'))} {clean(event.get('issue'))}")
    print()

    print("[1/10] Deep research")
    sources = dynamic_research_event(event)

    print(
        "[2/10] Story Director"
    )

    storyboard = build_dynamic_storyboard(client, event, sources)

    print(
        "[3/10] Comic visual search"
    )

    image_results = dynamic_image_search(event, storyboard)

    print(
        "[4/10] Download + deduplicate"
    )

    candidates = download_candidates(
        image_results
    )

    print(
        f"✓ {len(candidates)} "
        "unique visual candidates"
    )

    print(
        "[5/10] Visual Director ranking + panel crop"
    )

    rankings = vision_rank_scenes(
        client,
        storyboard,
        candidates,
    )

    print(
        "[6/10] Build visual plan"
    )

    scenes = build_scene_objects(
        client,
        storyboard,
        candidates,
        rankings,
        arguments.enable_ai_reconstruction,
    )

    print(
        "Unique visual ratio: "
        f"{unique_visual_ratio(scenes):.0%}"
    )

    print(
        "[7/10] Supervising Director critique"
    )

    critique = critique_visuals(
        client,
        scenes,
    )

    print(
        "Director overall: "
        f"{critique.get('overall', '?')}/10"
    )

    if critique.get(
        "weak_scenes"
    ):
        print(
            "→ Weak scenes are being revised."
        )

        scenes = improve_weak_scenes(
            scenes,
            candidates,
            critique,
        )

    save_visual_plan(
        scenes
    )

    print(
        "[8/10] Voice Director + natural TTS"
    )

    audio_file, narration = generate_voice(
        client,
        scenes,
        arguments.edge_voice,
    )

    print(
        "[9/10] Groq word alignment"
    )

    words = groq_word_timestamps(
        audio_file,
        narration,
    )

    print(
        "[10/10] Subtitle Director + final render"
    )

    render_video(
        scenes,
        audio_file,
        words,
        narration,
    )

    save_factory_metadata(event, storyboard)
    mark_factory_event_used(event)

    save_json(
        MANIFEST_FILE,
        {
            "created_at": datetime.now().isoformat(),
            "version": "v2-subtitle-sync",
            "event": (
                "Thor #6 (2020) - "
                "Galactus / Black Winter"
            ),
            "gemini_model": GEMINI_MODEL,
            "tts_model": (
                "edge"
                if arguments.edge_voice
                else GEMINI_TTS_MODEL
            ),
            "tts_voice": (
                EDGE_FALLBACK_VOICE
                if arguments.edge_voice
                else GEMINI_TTS_VOICE
            ),
            "voice_locked": (
                not arguments.edge_voice
            ),
            "ai_reconstruction": (
                arguments.enable_ai_reconstruction
            ),
            "unique_visual_ratio": (
                unique_visual_ratio(
                    scenes
                )
            ),
            "director_critique": critique,
            "scene_count": len(
                scenes
            ),
            "subtitle_sync_source": (
                "voice_director_narration"
            ),
            "subtitle_timing_source": (
                "groq_whisper"
            ),
            "output": str(
                OUTPUT_VIDEO
            ),
            "published": False,
            "scenes": [
                asdict(
                    scene
                )
                for scene in scenes
            ],
        },
    )

    print()
    print("=" * 78)
    print("THOR V2 PROTOTYPE READY")
    print("=" * 78)
    print()

    print(
        f"Video: {OUTPUT_VIDEO}"
    )

    print(
        f"Voice: {GEMINI_TTS_VOICE} (LOCKED)"
    )

    print(
        "YouTube: NOT UPLOADED"
    )

    print(
        "Instagram: NOT UPLOADED"
    )

    return 0


def select_factory_event(
    *,
    reuse_active: bool,
    count: int,
) -> dict[str, Any]:
    """Orijinal Comic Factory araştırma motoruyla event seçer."""
    import importlib.util

    module_path = (
        ROOT
        / "comic_factory_event_source.py"
    )

    if not module_path.exists():
        raise PrototypeError(
            "comic_factory_event_source.py bulunamadı. "
            "Bu dosya comic_factory.py ile aynı klasörde olmalı."
        )

    module_name = (
        "comic_factory_event_source"
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise PrototypeError(
            "Event source modülü yüklenemedi."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    try:
        spec.loader.exec_module(
            module
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    if reuse_active:
        return module.load_active_event()

    client = module.gemini_client()

    events = module.research_events(
        client,
        count,
    )

    if not events:
        raise PrototypeError(
            "Comic Factory event bulamadı."
        )

    return module.activate_event(
        events[0]
    )


def main() -> int:
    """Comic Factory event selector + proven Thor V2 production engine."""
    parser = argparse.ArgumentParser(
        description="Comic Factory powered by the Thor V2 video engine."
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reuse-active", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--enable-ai-reconstruction", action="store_true")
    parser.add_argument("--edge-voice", action="store_true")
    arguments = parser.parse_args()

    if arguments.check:
        require_env("GEMINI_API_KEY")
        require_env("GROQ_API_KEY")
        print("✓ Gemini hazır")
        print("✓ Groq hazır")
        print("✓ FFmpeg hazır")
        print(f"✓ Voice: {GEMINI_TTS_VOICE} LOCKED")
        return 0

    event = select_factory_event(
        reuse_active=arguments.reuse_active,
        count=arguments.count,
    )
    save_json(ACTIVE_EVENT_FILE, event)

    forwarded = [sys.argv[0]]
    if arguments.rebuild:
        forwarded.append("--rebuild")
    if arguments.enable_ai_reconstruction:
        forwarded.append("--enable-ai-reconstruction")
    if arguments.edge_voice:
        forwarded.append("--edge-voice")

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        result = run_v2_pipeline()
    finally:
        sys.argv = original_argv

    if arguments.upload:
        uploader = ROOT / "youtube_uploader.py"
        if not uploader.exists():
            raise PrototypeError("youtube_uploader.py bulunamadı.")
        process = subprocess.run(
            [sys.executable, str(uploader)],
            cwd=str(ROOT),
        )
        if process.returncode != 0:
            raise PrototypeError("YouTube upload başarısız.")

    print()
    print("=" * 78)
    print("COMIC FACTORY + THOR V2 ENGINE BAŞARILI")
    print("=" * 78)
    print(f"Video: {OUTPUT_VIDEO}")
    print(f"Metadata: {LATEST_SCRIPT_FILE}")
    return result or 0


if __name__ == "__main__":
    raise SystemExit(main())
