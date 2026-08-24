# comic_factory.py
from __future__ import annotations

import argparse
import asyncio
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import edge_tts
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
CANDIDATE_DIR = DATA / "comic_candidates"
WORK_DIR = DATA / "video_work"
ASSET_DIR = ROOT / "assets" / "comic_pages"

ACTIVE_EVENT_FILE = EVENT_DIR / "active_event.json"
USED_EVENTS_FILE = EVENT_DIR / "used_events.json"
LATEST_SCRIPT_FILE = SCRIPT_DIR / "latest.json"
LATEST_AUDIO_FILE = AUDIO_DIR / "latest.mp3"
LATEST_WORDS_FILE = AUDIO_DIR / "latest_word_timestamps.json"
LATEST_VIDEO_FILE = VIDEO_DIR / "latest.mp4"

TZ = ZoneInfo("Europe/Istanbul")

WIDTH = 1080
HEIGHT = 1920
FPS = 30
SCENE_COUNT = 8
SUBTITLE_CHUNK = 4

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite",
)

GROQ_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo",
)

TTS_VOICE = "tr-TR-AhmetNeural"
TTS_RATE = "+12%"
TTS_PITCH = "+3Hz"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
)

REQUEST_TIMEOUT = 30

BAD_IMAGE_TERMS = {
    "logo",
    "avatar",
    "author",
    "gravatar",
    "sprite",
    "favicon",
    "tracking",
    "pixel",
    "advert",
    "advertisement",
    "social-icon",
    "emoji",
    "placeholder",
}

load_dotenv(
    ROOT / ".env"
)


class ComicFactoryError(RuntimeError):
    """Comic Factory üretimi başarısız olduğunda oluşur."""


@dataclass
class ImageCandidate:
    """Comic görsel adayı."""

    candidate_id: str
    source_name: str
    source_page: str
    image_url: str
    alt_text: str
    local_file: str
    width: int
    height: int
    score: float


class PageImageParser(HTMLParser):
    """HTML içindeki görsel adaylarını toplar."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        data = {
            key.lower(): value or ""
            for key, value in attrs
        }

        if tag.lower() == "meta":
            property_name = (
                data.get("property")
                or data.get("name")
                or ""
            ).lower()

            if property_name in {
                "og:image",
                "og:image:url",
                "twitter:image",
            }:
                value = data.get(
                    "content",
                    "",
                ).strip()

                if value:
                    self.images.append(
                        {
                            "url": value,
                            "alt": "",
                        }
                    )

            return

        if tag.lower() != "img":
            return

        alt_text = (
            data.get("alt")
            or data.get("title")
            or ""
        ).strip()

        urls = [
            data.get("src", ""),
            data.get("data-src", ""),
            data.get("data-lazy-src", ""),
            data.get("data-original", ""),
        ]

        for key in (
            "srcset",
            "data-srcset",
        ):
            srcset = data.get(
                key,
                "",
            )

            if srcset:
                urls.extend(
                    part.strip().split(" ")[0]
                    for part in srcset.split(",")
                    if part.strip()
                )

        for value in urls:
            value = value.strip()

            if value:
                self.images.append(
                    {
                        "url": value,
                        "alt": alt_text,
                    }
                )


def parse_args() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""

    parser = argparse.ArgumentParser()

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
        default=18,
    )

    parser.add_argument(
        "--subtitle-offset",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def ensure_dirs() -> None:
    """Gerekli klasörleri oluşturur."""

    directories = (
        EVENT_DIR,
        RESEARCH_DIR,
        SCRIPT_DIR,
        AUDIO_DIR,
        VIDEO_DIR,
        CANDIDATE_DIR,
        ASSET_DIR,
    )

    for path in directories:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def clean(value: Any) -> str:
    """Metni temizler."""

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
    """JSON dosyasını kaydeder."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def require_env(
    name: str,
) -> str:
    """Zorunlu ortam değişkenini döndürür."""

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
    """Gemini istemcisini oluşturur."""

    return genai.Client(
        api_key=require_env(
            "GEMINI_API_KEY"
        )
    )


def parse_json_response(
    text: str,
) -> dict[str, Any]:
    """Gemini JSON cevabını güvenli biçimde ayrıştırır."""

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

        first_item = payload[
            0
        ]

        if not isinstance(
            first_item,
            dict,
        ):
            raise ComicFactoryError(
                "Gemini beklenmeyen JSON listesi döndürdü."
            )

        if (
            "event_title"
            in first_item
            or "publisher"
            in first_item
            or "series"
            in first_item
        ):
            return {
                "events": payload,
            }

        if (
            "scene_number"
            in first_item
            or "candidate_id"
            in first_item
        ):
            return {
                "assignments": payload,
            }

    raise ComicFactoryError(
        "Gemini beklenmeyen JSON yapısı döndürdü."
    )
def ask_gemini_json(
    client: genai.Client,
    prompt: str,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Gemini'den JSON cevap alır."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
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
    """DDGS ile ücretsiz web araması yapar."""

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

            found.append(
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
    """Daha önce kullanılan olayları döndürür."""

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
    """DDGS sonuçlarından Gemini ile comic olayı seçer."""

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
        "1/7 - ÜCRETSİZ WEB + GEMINI COMIC ARAŞTIRMASI"
    )
    print(
        "=" * 78
    )
    print()

    results = search_web(
        [
            "Marvel comics craziest feats specific issue review",
            "DC comics craziest feats specific issue review",
            "comic book shocking moments specific issue panels",
            "comic book impossible feats issue review Marvel DC",
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
            :40
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
Aşağıdaki web arama sonuçlarına dayanarak TAM {count} adet
şaşırtıcı ve güçlü comic-book event adayı seç.

WEB:
{evidence}

DAHA ÖNCE KULLANILANLAR:
{used}

Kurallar:

- Tek ve spesifik olay/feat/dönüşüm seç.
- Marvel, DC, Image ve Dark Horse öncelikli.
- Series, issue, yıl ve olay doğru olmalı.
- URL veya issue uydurma.
- Kaynak URL'lerini yalnızca WEB bölümünden kullan.
- 45-60 saniyelik Shorts'a uygun olsun.
- Görsel olarak güçlü olsun.
- TAM 8 visual_beats olsun.
- shorts_score 0-100.
- Metin alanları Türkçe olsun.
- Comic ve karakter özel isimleri orijinal kalsın.

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
      "shorts_score": 95,
      "sources": [
        {{
          "name": "...",
          "url": "https://...",
          "supports": "..."
        }}
      ],
      "visual_beats": [
        "...",
        "...",
        "...",
        "...",
        "...",
        "...",
        "...",
        "..."
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

    already_used = used_event_keys()

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
        not in already_used
        and isinstance(
            event.get(
                "visual_beats"
            ),
            list,
        )
        and len(
            event[
                "visual_beats"
            ]
        )
        == SCENE_COUNT
    ]

    eligible.sort(
        key=lambda item: float(
            item.get(
                "shorts_score",
                0,
            )
        ),
        reverse=True,
    )

    if not eligible:
        raise ComicFactoryError(
            "Yeni kullanılabilir comic olayı bulunamadı."
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
            f"({event.get('shorts_score')}/100)"
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


def discover_visual_pages(
    event: dict[str, Any],
) -> list[dict[str, str]]:
    """Kamuya açık görsel kaynak sayfalarını bulur."""

    print()
    print(
        "=" * 78
    )
    print(
        "2/7 - KAMUYA AÇIK COMIC GÖRSEL KAYNAKLARI"
    )
    print(
        "=" * 78
    )
    print()

    queries = [
        (
            f'"{clean(event.get("series"))}" '
            f'"{clean(event.get("issue"))}" preview'
        ),
        (
            f'"{clean(event.get("series"))}" '
            f'"{clean(event.get("issue"))}" '
            "review panels"
        ),
        (
            f'{clean(event.get("event_title"))} '
            "comic review panels"
        ),
    ]

    searched = search_web(
        queries,
        each=8,
    )

    pages: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

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

        if (
            url.startswith(
                "http"
            )
            and url not in seen
        ):
            seen.add(
                url
            )

            pages.append(
                {
                    "name": (
                        clean(
                            source.get(
                                "name"
                            )
                        )
                        or "Research source"
                    ),
                    "url": url,
                    "reason": clean(
                        source.get(
                            "supports"
                        )
                    ),
                }
            )

    for item in searched:
        url = item[
            "url"
        ]

        if url not in seen:
            seen.add(
                url
            )

            pages.append(
                {
                    "name": (
                        item[
                            "title"
                        ]
                        or "Web source"
                    ),
                    "url": url,
                    "reason": item[
                        "body"
                    ],
                }
            )

    if not pages:
        raise ComicFactoryError(
            "Görsel kaynak sayfası bulunamadı."
        )

    print(
        f"✓ {min(len(pages), 10)} "
        "kaynak sayfası bulundu."
    )

    return pages[
        :10
    ]


def http_session() -> requests.Session:
    """HTTP oturumu oluşturur."""

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": (
                "en-US,en;q=0.9,tr;q=0.8"
            ),
        }
    )

    return session


def scrape_page_images(
    session: requests.Session,
    source: dict[str, str],
) -> list[dict[str, str]]:
    """Web sayfasından görsel URL'leri çıkarır."""

    try:
        response = session.get(
            source[
                "url"
            ],
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print(
            f"! Sayfa okunamadı: "
            f"{source['url']} ({error})"
        )
        return []

    parser = PageImageParser()

    try:
        parser.feed(
            response.text
        )

    except Exception:
        return []

    output: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for item in parser.images:
        url = html.unescape(
            clean(
                item.get(
                    "url"
                )
            )
        )

        if url.startswith(
            "//"
        ):
            url = (
                "https:"
                + url
            )

        url = urljoin(
            source[
                "url"
            ],
            url,
        )

        alt = clean(
            item.get(
                "alt"
            )
        )

        combined = (
            url
            + " "
            + alt
        ).casefold()

        extension = Path(
            urlparse(
                url
            ).path
        ).suffix.casefold()

        if (
            not url.startswith(
                "http"
            )
            or url in seen
            or any(
                term in combined
                for term in BAD_IMAGE_TERMS
            )
            or extension in {
                ".svg",
                ".gif",
                ".ico",
            }
        ):
            continue

        seen.add(
            url
        )

        output.append(
            {
                "source_name": source[
                    "name"
                ],
                "source_page": source[
                    "url"
                ],
                "image_url": url,
                "alt_text": alt,
            }
        )

    return output


def candidate_score(
    item: dict[str, str],
    event: dict[str, Any],
) -> float:
    """Görsel adayına kaba uygunluk puanı verir."""

    text = (
        item[
            "image_url"
        ]
        + " "
        + item[
            "alt_text"
        ]
    ).casefold()

    score = 0.0

    terms = [
        clean(
            event.get(
                "series"
            )
        ),
        clean(
            event.get(
                "issue"
            )
        ),
        *[
            clean(
                name
            )
            for name in event.get(
                "characters",
                [],
            )
        ],
    ]

    for term in terms:
        if (
            len(
                term
            )
            >= 3
            and term.casefold()
            in text
        ):
            score += 3.0

    if any(
        term in text
        for term in (
            "comic",
            "panel",
            "preview",
            "issue",
            "page",
        )
    ):
        score += 1.5

    return score


def average_hash(
    image: Image.Image,
) -> str:
    """Benzer görselleri tespit eder."""

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

    return hex(
        int(
            bits,
            2,
        )
    )


def download_candidate(
    session: requests.Session,
    item: dict[str, str],
    output: Path,
) -> tuple[int, int, str] | None:
    """Görseli indirir."""

    try:
        response = session.get(
            item[
                "image_url"
            ],
            headers={
                "Referer": item[
                    "source_page"
                ],
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException:
        return None

    if len(
        response.content
    ) < 12_000:
        return None

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
        return None

    width, height = image.size

    if (
        max(
            width,
            height,
        )
        < 500
        or width
        * height
        < 350_000
    ):
        return None

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output,
        "JPEG",
        quality=92,
        optimize=True,
    )

    return (
        width,
        height,
        average_hash(
            image
        ),
    )


def collect_candidates(
    event: dict[str, Any],
    sources: list[dict[str, str]],
    max_candidates: int,
) -> list[ImageCandidate]:
    """Comic görsellerini toplar."""

    print()
    print(
        "=" * 78
    )
    print(
        "3/7 - COMIC GÖRSELLERİ TOPLANIYOR"
    )
    print(
        "=" * 78
    )
    print()

    session = http_session()

    raw: list[
        dict[str, str]
    ] = []

    seen_urls: set[str] = set()

    for source in sources:
        print(
            f"Kaynak: "
            f"{source['name']}"
        )

        for item in scrape_page_images(
            session,
            source,
        ):
            if item[
                "image_url"
            ] not in seen_urls:
                seen_urls.add(
                    item[
                        "image_url"
                    ]
                )

                raw.append(
                    item
                )

    raw.sort(
        key=lambda item: candidate_score(
            item,
            event,
        ),
        reverse=True,
    )

    event_directory = (
        CANDIDATE_DIR
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

    accepted: list[
        ImageCandidate
    ] = []

    hashes: set[str] = set()

    for item in raw:
        if len(
            accepted
        ) >= max_candidates:
            break

        candidate_id = (
            f"c{len(accepted) + 1:02d}"
        )

        output = (
            event_directory
            / f"{candidate_id}.jpg"
        )

        downloaded = download_candidate(
            session,
            item,
            output,
        )

        if downloaded is None:
            continue

        width, height, image_hash = downloaded

        if image_hash in hashes:
            output.unlink(
                missing_ok=True
            )
            continue

        hashes.add(
            image_hash
        )

        score = (
            candidate_score(
                item,
                event,
            )
            + min(
                4.0,
                width
                * height
                / 1_000_000,
            )
        )

        accepted.append(
            ImageCandidate(
                candidate_id,
                item[
                    "source_name"
                ],
                item[
                    "source_page"
                ],
                item[
                    "image_url"
                ],
                item[
                    "alt_text"
                ],
                str(
                    output
                ),
                width,
                height,
                score,
            )
        )

        print(
            f"✓ {candidate_id}: "
            f"{width}x{height}"
        )

    if len(
        accepted
    ) < 4:
        raise ComicFactoryError(
            "Yeterli kaliteli comic görseli "
            f"bulunamadı. Bulunan: {len(accepted)}"
        )

    accepted.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return accepted


def assign_images(
    client: genai.Client,
    event: dict[str, Any],
    candidates: list[ImageCandidate],
) -> list[dict[str, Any]]:
    """Gemini Vision ile panelleri seçer."""

    print()
    print(
        "=" * 78
    )
    print(
        "4/7 - GEMINI PANELLERİ GÖREREK SEÇİYOR"
    )
    print(
        "=" * 78
    )
    print()

    beats = [
        clean(
            item
        )
        for item in event.get(
            "visual_beats",
            [],
        )
    ][
        :SCENE_COUNT
    ]

    while len(
        beats
    ) < SCENE_COUNT:
        beats.append(
            f"Comic olayının "
            f"{len(beats) + 1}. önemli anı"
        )

    prompt = (
        f"EVENT: "
        f"{clean(event.get('event_title'))}\n\n"
        + "VISUAL BEATS:\n"
        + "\n".join(
            f"{index}. {beat}"
            for index, beat in enumerate(
                beats,
                start=1,
            )
        )
        + (
            "\n\nHer sahneye en uygun "
            "candidate_id seç. "
            "Aynı görseli zorunlu olmadıkça "
            "tekrar kullanma. "
            "Kapak yerine olay panelini tercih et. "
            "SADECE JSON döndür: "
            '{"assignments":['
            '{"scene_number":1,'
            '"candidate_id":"c01",'
            '"confidence":90,'
            '"reason":"..."}'
            "]}. "
            "TAM 8 assignment olmalı."
        )
    )

    contents: list[Any] = [
        prompt
    ]

    for candidate in candidates:
        contents.append(
            (
                f"CANDIDATE "
                f"{candidate.candidate_id}\n"
                f"Source: "
                f"{candidate.source_name}\n"
                f"Alt: "
                f"{candidate.alt_text}\n"
                f"Size: "
                f"{candidate.width}x"
                f"{candidate.height}"
            )
        )

        contents.append(
            types.Part.from_bytes(
                data=Path(
                    candidate.local_file
                ).read_bytes(),
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    payload = parse_json_response(
        response.text
        or "{}"
    )

    assignments = payload.get(
        "assignments",
        [],
    )

    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    selected: list[
        dict[str, Any]
    ] = []

    for scene in range(
        1,
        SCENE_COUNT + 1,
    ):
        assignment = next(
            (
                item
                for item in assignments
                if isinstance(
                    item,
                    dict,
                )
                and int(
                    item.get(
                        "scene_number",
                        0,
                    )
                    or 0
                )
                == scene
                and clean(
                    item.get(
                        "candidate_id"
                    )
                )
                in candidate_map
            ),
            None,
        )

        if assignment is None:
            fallback = candidates[
                (
                    scene
                    - 1
                )
                % len(
                    candidates
                )
            ]

            assignment = {
                "candidate_id": (
                    fallback.candidate_id
                ),
                "confidence": 30,
                "reason": "fallback",
            }

        candidate = candidate_map[
            clean(
                assignment[
                    "candidate_id"
                ]
            )
        ]

        selected.append(
            {
                "scene_number": scene,
                "visual_beat": beats[
                    scene
                    - 1
                ],
                "candidate": candidate,
                "confidence": int(
                    assignment.get(
                        "confidence",
                        50,
                    )
                    or 50
                ),
                "reason": clean(
                    assignment.get(
                        "reason"
                    )
                ),
            }
        )

        print(
            f"✓ Sahne {scene}: "
            f"{candidate.candidate_id}"
        )

    return selected


def build_assets(
    event: dict[str, Any],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Final görselleri asset klasörüne kopyalar."""

    directory = (
        ASSET_DIR
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

    assets: list[
        dict[str, Any]
    ] = []

    for item in selected:
        candidate: ImageCandidate = item[
            "candidate"
        ]

        number = int(
            item[
                "scene_number"
            ]
        )

        output = (
            directory
            / f"{number:02d}_comic.jpg"
        )

        shutil.copy2(
            candidate.local_file,
            output,
        )

        assets.append(
            {
                "scene_number": number,
                "local_file": output.relative_to(
                    ROOT
                ).as_posix(),
                "visual_beat": item[
                    "visual_beat"
                ],
                "source_name": (
                    candidate.source_name
                ),
                "source_page": (
                    candidate.source_page
                ),
                "image_url": (
                    candidate.image_url
                ),
            }
        )

    save_json(
        directory
        / "downloaded_sources.json",
        {
            "event_title": event[
                "event_title"
            ],
            "series": event[
                "series"
            ],
            "issue": event[
                "issue"
            ],
            "assets": assets,
        },
    )

    return assets


def generate_script(
    client: genai.Client,
    event: dict[str, Any],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Gemini ile Türkçe Shorts senaryosu üretir."""

    print()
    print(
        "=" * 78
    )
    print(
        "5/7 - GEMINI TÜRKÇE SHORTS SENARYOSU"
    )
    print(
        "=" * 78
    )
    print()

    visual_plan = "\n".join(
        (
            f"{item['scene_number']}. "
            f"{item['visual_beat']}"
        )
        for item in assets
    )

    prompt = f"""
Türkçe yüksek-retention comic-book Shorts senaryosu yaz.

OLAY:
{clean(event.get("event_title"))}

COMIC:
{clean(event.get("publisher"))}
{clean(event.get("series"))}

ISSUE:
{clean(event.get("issue"))}
({event.get("publication_year")})

KARAKTERLER:
{", ".join(event.get("characters", []))}

HOOK:
{clean(event.get("hook"))}

ÖZET:
{clean(event.get("event_summary"))}

GÜÇ:
{clean(event.get("power_feat"))}

GÖRSEL AKIŞ:
{visual_plan}

Kurallar:

- 45-60 saniye.
- Yaklaşık 120-145 Türkçe kelime.
- İlk cümle güçlü hook.
- Her cümlede olay ilerlesin.
- Bilgi uydurma.
- TAM 8 scene_narrations.
- TAM 8 scene_captions.
- Caption en fazla 4 kelime.
- Finalde kısa soru.
- Özel isimler orijinal.

SADECE JSON:

{{
  "title":"...",
  "narration":"...",
  "description":"...",
  "hashtags":["#comics"],
  "scene_captions":[
    "...","...","...","...",
    "...","...","...","..."
  ],
  "scene_narrations":[
    "...","...","...","...",
    "...","...","...","..."
  ]
}}
"""

    script = ask_gemini_json(
        client,
        prompt,
        temperature=0.6,
    )

    if len(
        script.get(
            "scene_captions",
            [],
        )
    ) != SCENE_COUNT:
        raise ComicFactoryError(
            "scene_captions tam 8 değil."
        )

    if len(
        script.get(
            "scene_narrations",
            [],
        )
    ) != SCENE_COUNT:
        raise ComicFactoryError(
            "scene_narrations tam 8 değil."
        )

    unique_sources: list[
        tuple[str, str]
    ] = []

    for asset in assets:
        pair = (
            clean(
                asset[
                    "source_name"
                ]
            ),
            clean(
                asset[
                    "source_page"
                ]
            ),
        )

        if pair not in unique_sources:
            unique_sources.append(
                pair
            )

    hashtags = " ".join(
        clean(
            tag
        )
        for tag in script.get(
            "hashtags",
            [],
        )
    )

    sources = "\n".join(
        f"- {name}: {url}"
        for name, url in unique_sources
    )

    script[
        "full_description"
    ] = (
        f"{clean(script.get('description'))}"
        f"\n\n{hashtags}"
        f"\n\nKaynaklar:\n"
        f"{sources}"
    )

    payload = {
        "event_id": event[
            "id"
        ],
        "generated_at": datetime.now(
            TZ
        ).isoformat(),
        "script": script,
    }

    save_json(
        LATEST_SCRIPT_FILE,
        payload,
    )

    print(
        f"✓ Başlık: "
        f"{script['title']}"
    )

    return script


def generate_voice(
    narration: str,
) -> None:
    """Edge TTS ile ücretsiz Türkçe erkek ses üretir."""

    print()
    print(
        "=" * 78
    )
    print(
        "6/7 - EDGE TTS TÜRKÇE ERKEK SES"
    )
    print(
        "=" * 78
    )
    print()

    timestamp = datetime.now(
        TZ
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive = (
        AUDIO_DIR
        / f"comic_ahmet_{timestamp}.mp3"
    )

    async def create() -> None:
        speech = edge_tts.Communicate(
            narration,
            TTS_VOICE,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
        )

        await speech.save(
            str(
                archive
            )
        )

    asyncio.run(
        create()
    )

    if (
        not archive.exists()
        or archive.stat().st_size
        < 10_000
    ):
        raise ComicFactoryError(
            "Edge TTS ses üretmedi."
        )

    shutil.copy2(
        archive,
        LATEST_AUDIO_FILE,
    )

    print(
        f"✓ {TTS_VOICE} hazır."
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
    narration: str,
) -> list[dict[str, Any]]:
    """Groq Whisper ile kelime zamanlarını çıkarır."""

    print(
        "Groq Whisper kelime "
        "zamanlaması çıkarılıyor..."
    )

    client = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with LATEST_AUDIO_FILE.open(
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
    ) < 10:
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

    print(
        f"✓ {len(words)} "
        "kelime timestamp hazır."
    )

    return words


def get_font(
    size: int,
    comic: bool = False,
) -> ImageFont.ImageFont:
    """Windows/Linux fontu yükler."""

    paths = (
        [
            Path(
                r"C:\Windows\Fonts\impact.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\ariblk.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/"
                "dejavu/"
                "DejaVuSansCondensed-Bold.ttf"
            ),
        ]
        if comic
        else [
            Path(
                r"C:\Windows\Fonts\arialbd.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/"
                "dejavu/"
                "DejaVuSans-Bold.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/"
                "dejavu/"
                "DejaVuSans.ttf"
            ),
        ]
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


def compose_frame(
    image: Image.Image,
) -> Image.Image:
    """Comic görselini kırpmadan blur arka plana yerleştirir."""

    background = ImageOps.fit(
        image,
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
            38
        )
    )

    background = ImageEnhance.Brightness(
        background
    ).enhance(
        0.42
    )

    background = ImageEnhance.Color(
        background
    ).enhance(
        0.85
    )

    canvas = background.convert(
        "RGBA"
    )

    max_width = (
        WIDTH
        - 64
    )

    max_height = (
        HEIGHT
        - 205
        - 315
    )

    foreground = image.copy()

    foreground.thumbnail(
        (
            max_width,
            max_height,
        ),
        Image.Resampling.LANCZOS,
    )

    foreground_width, foreground_height = (
        foreground.size
    )

    x = (
        WIDTH
        - foreground_width
    ) // 2

    y = (
        205
        + (
            max_height
            - foreground_height
        )
        // 2
    )

    shadow = Image.new(
        "RGBA",
        (
            WIDTH,
            HEIGHT,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    shadow_draw = ImageDraw.Draw(
        shadow
    )

    shadow_draw.rounded_rectangle(
        (
            x - 12,
            y + 14,
            x + foreground_width + 12,
            y + foreground_height + 38,
        ),
        radius=28,
        fill=(
            0,
            0,
            0,
            145,
        ),
    )

    shadow = shadow.filter(
        ImageFilter.GaussianBlur(
            22
        )
    )

    canvas = Image.alpha_composite(
        canvas,
        shadow,
    )

    canvas.alpha_composite(
        foreground.convert(
            "RGBA"
        ),
        (
            x,
            y,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.rectangle(
        (
            x - 2,
            y - 2,
            x + foreground_width + 1,
            y + foreground_height + 1,
        ),
        outline=(
            245,
            245,
            245,
            210,
        ),
        width=3,
    )

    return canvas.convert(
        "RGB"
    )


def create_scene_frame(
    event: dict[str, Any],
    asset: dict[str, Any],
    caption: str,
    output: Path,
) -> None:
    """Final sahne karesi oluşturur."""

    source = (
        ROOT
        / asset[
            "local_file"
        ]
    )

    with Image.open(
        source
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    frame = compose_frame(
        image
    )

    frame = ImageEnhance.Contrast(
        frame
    ).enhance(
        1.06
    )

    frame = ImageEnhance.Color(
        frame
    ).enhance(
        1.04
    )

    canvas = frame.convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        (
            WIDTH,
            HEIGHT,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    overlay_draw = ImageDraw.Draw(
        overlay
    )

    overlay_draw.rectangle(
        (
            0,
            0,
            WIDTH,
            210,
        ),
        fill=(
            0,
            0,
            0,
            90,
        ),
    )

    overlay_draw.rectangle(
        (
            0,
            1730,
            WIDTH,
            HEIGHT,
        ),
        fill=(
            0,
            0,
            0,
            120,
        ),
    )

    canvas = Image.alpha_composite(
        canvas,
        overlay,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    issue_font = get_font(
        25
    )

    caption_font = get_font(
        34,
        comic=True,
    )

    source_font = get_font(
        18
    )

    issue = (
        f"{clean(event.get('series'))} "
        f"{clean(event.get('issue'))}"
    )

    if event.get(
        "publication_year"
    ):
        issue += (
            f" • "
            f"{event['publication_year']}"
        )

    draw.rounded_rectangle(
        (
            24,
            28,
            520,
            82,
        ),
        radius=16,
        fill=(
            5,
            5,
            8,
            175,
        ),
    )

    draw.text(
        (
            40,
            41,
        ),
        issue[
            :35
        ].upper(),
        font=issue_font,
        fill="white",
    )

    short_caption = " ".join(
        clean(
            caption
        ).split()[
            :4
        ]
    ).upper()

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        short_caption,
        font=caption_font,
    )

    caption_width = (
        bbox[
            2
        ]
        - bbox[
            0
        ]
    )

    caption_x = max(
        35,
        WIDTH
        - caption_width
        - 38,
    )

    draw.rounded_rectangle(
        (
            caption_x - 17,
            103,
            WIDTH - 22,
            166,
        ),
        radius=17,
        fill=(
            5,
            5,
            8,
            155,
        ),
    )

    draw.text(
        (
            caption_x,
            116,
        ),
        short_caption,
        font=caption_font,
        fill=(
            255,
            225,
            65,
            255,
        ),
        stroke_width=1,
        stroke_fill=(
            0,
            0,
            0,
            230,
        ),
    )

    draw.text(
        (
            26,
            1880,
        ),
        clean(
            asset.get(
                "source_name"
            )
        )[
            :55
        ],
        font=source_font,
        fill=(
            225,
            225,
            225,
            175,
        ),
        stroke_width=1,
        stroke_fill=(
            0,
            0,
            0,
            180,
        ),
    )

    canvas.convert(
        "RGB"
    ).save(
        output,
        "PNG",
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
                -3000:
            ]
        )


def audio_duration(
    ffmpeg: str,
) -> float:
    """Ses süresini ölçer."""

    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                LATEST_AUDIO_FILE
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
            match[
                1
            ]
        )
        * 3600
        + int(
            match[
                2
            ]
        )
        * 60
        + float(
            match[
                3
            ]
        )
    )


def scene_durations(
    narrations: list[str],
    words: list[dict[str, Any]],
    duration: float,
) -> list[float]:
    """Sahne sürelerini konuşma akışına göre böler."""

    counts = [
        max(
            1,
            len(
                re.findall(
                    r"\w+",
                    text,
                )
            ),
        )
        for text in narrations
    ]

    total = sum(
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
            / total
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
        duration
    )

    result = [
        max(
            0.3,
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

    result[
        -1
    ] += (
        duration
        - sum(
            result
        )
    )

    return result


def render_segment(
    ffmpeg: str,
    frame: Path,
    output: Path,
    duration: float,
    index: int,
) -> None:
    """Comic görseline hafif zoom uygular."""

    speed = (
        0.00010
        if index
        % 2
        else 0.00008
    )

    filter_text = (
        "scale=1080:1920,"
        "zoompan="
        f"z='min(zoom+{speed},1.025)':"
        "x='iw/2-(iw/zoom/2)':"
        "y='ih/2-(ih/zoom/2)':"
        "d=1:"
        "s=1080x1920:"
        "fps=30,"
        "format=yuv420p"
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
                frame
            ),
            "-t",
            f"{duration:.3f}",
            "-vf",
            filter_text,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(
                output
            ),
        ],
        "Video sahnesi render edilemedi.",
    )


def ass_time(
    seconds: float,
) -> str:
    """Saniyeyi ASS zamanına çevirir."""

    centiseconds = round(
        max(
            0.0,
            seconds,
        )
        * 100
    )

    hours, centiseconds = divmod(
        centiseconds,
        360000,
    )

    minutes, centiseconds = divmod(
        centiseconds,
        6000,
    )

    secs, cs = divmod(
        centiseconds,
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


def subtitle_line(
    words: list[dict[str, Any]],
    active: int,
) -> str:
    """Aktif kelimeyi sarı yapar."""

    start = (
        active
        // SUBTITLE_CHUNK
        * SUBTITLE_CHUNK
    )

    end = min(
        len(
            words
        ),
        start
        + SUBTITLE_CHUNK,
    )

    parts: list[str] = []

    for index in range(
        start,
        end,
    ):
        word = ass_escape(
            words[
                index
            ][
                "word"
            ]
        )

        if index == active:
            parts.append(
                r"{\1c&H0000E6FF&"
                r"\b1\fscx112\fscy112}"
                + word
                + r"{\r}"
            )

        else:
            parts.append(
                word
            )

    return " ".join(
        parts
    )


def create_ass(
    words: list[dict[str, Any]],
    offset: float,
) -> Path:
    """ASS altyazı dosyasını oluşturur."""

    path = (
        WORK_DIR
        / "precise.ass"
    )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
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
            "Style: Main,DejaVu Sans,52,"
            "&H00FFFFFF,&H00FFFFFF,"
            "&H00101010,&H78000000,"
            "-1,0,0,0,100,100,0,0,"
            "1,4,1,2,85,85,145,1"
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
        words
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
            0,
            start,
        )

        end = max(
            start
            + 0.03,
            end,
        )

        lines.append(
            "Dialogue: 0,"
            f"{ass_time(start)},"
            f"{ass_time(end)},"
            "Main,,0,0,0,,"
            f"{subtitle_line(words, index)}"
        )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return path


def render_video(
    event: dict[str, Any],
    assets: list[dict[str, Any]],
    script: dict[str, Any],
    words: list[dict[str, Any]],
    subtitle_offset: float,
) -> Path:
    """Final videoyu render eder."""

    print()
    print(
        "=" * 78
    )
    print(
        "7/7 - FINAL VIDEO RENDER"
    )
    print(
        "=" * 78
    )
    print()

    ffmpeg = ffmpeg_path()

    duration = audio_duration(
        ffmpeg
    )

    shutil.rmtree(
        WORK_DIR,
        ignore_errors=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    durations = scene_durations(
        script[
            "scene_narrations"
        ],
        words,
        duration,
    )

    segments: list[
        Path
    ] = []

    for index, (
        asset,
        caption,
        scene_duration,
    ) in enumerate(
        zip(
            assets,
            script[
                "scene_captions"
            ],
            durations,
            strict=True,
        ),
        start=1,
    ):
        frame = (
            WORK_DIR
            / f"frame_{index:02d}.png"
        )

        segment = (
            WORK_DIR
            / f"segment_{index:02d}.mp4"
        )

        create_scene_frame(
            event,
            asset,
            clean(
                caption
            ),
            frame,
        )

        render_segment(
            ffmpeg,
            frame,
            segment,
            scene_duration,
            index,
        )

        segments.append(
            segment
        )

        print(
            f"✓ Sahne {index}: "
            f"{scene_duration:.2f} sn"
        )

    concat = (
        WORK_DIR
        / "concat.txt"
    )

    concat.write_text(
        "\n".join(
            (
                "file '"
                + item.resolve().as_posix()
                + "'"
            )
            for item in segments
        ),
        encoding="utf-8",
    )

    silent = (
        WORK_DIR
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
                concat
            ),
            "-c",
            "copy",
            str(
                silent
            ),
        ],
        "Sahneler birleştirilemedi.",
    )

    with_audio = (
        WORK_DIR
        / "with_audio.mp4"
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
                silent
            ),
            "-i",
            str(
                LATEST_AUDIO_FILE
            ),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(
                with_audio
            ),
        ],
        "Ses videoya eklenemedi.",
    )

    ass_file = create_ass(
        words,
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

    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(
                with_audio.resolve()
            ),
            "-vf",
            f"ass={ass_file.name}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(
                archive.resolve()
            ),
        ],
        "Final video render edilemedi.",
        cwd=WORK_DIR,
    )

    shutil.copy2(
        archive,
        LATEST_VIDEO_FILE,
    )

    shutil.rmtree(
        WORK_DIR,
        ignore_errors=True,
    )

    if not LATEST_VIDEO_FILE.exists():
        raise ComicFactoryError(
            "latest.mp4 oluşmadı."
        )

    print(
        f"✓ Video hazır: "
        f"{LATEST_VIDEO_FILE}"
    )

    return archive


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
    """Sistem kontrolü yapar."""

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
        "✓ Groq hazır"
    )
    print(
        "✓ Edge TTS hazır"
    )
    print(
        "✓ FFmpeg hazır"
    )


def main() -> None:
    """Comic Factory ana üretim hattı."""

    args = parse_args()

    ensure_dirs()

    if args.check:
        system_check(
            args.upload
        )
        return

    client = gemini_client()

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

        sources = discover_visual_pages(
            event
        )

        candidates = collect_candidates(
            event,
            sources,
            max(
                8,
                args.max_images,
            ),
        )

        selected = assign_images(
            client,
            event,
            candidates,
        )

        assets = build_assets(
            event,
            selected,
        )

        script = generate_script(
            client,
            event,
            assets,
        )

        narration = clean(
            script[
                "narration"
            ]
        )

        generate_voice(
            narration
        )

        words = align_words(
            narration
        )

        archive = render_video(
            event,
            assets,
            script,
            words,
            args.subtitle_offset,
        )

        mark_used(
            event
        )

        print()
        print(
            "=" * 78
        )
        print(
            "COMIC FACTORY BAŞARILI"
        )
        print(
            "=" * 78
        )

        print(
            f"Konu: "
            f"{clean(event['event_title'])}"
        )

        print(
            f"Video: "
            f"{LATEST_VIDEO_FILE}"
        )

        print(
            f"Arşiv: "
            f"{archive.relative_to(ROOT)}"
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
            "COMIC FACTORY DURDU"
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
