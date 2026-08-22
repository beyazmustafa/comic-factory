# comic_factory.py
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import imageio_ffmpeg
import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

EVENT_DIRECTORY = DATA_DIRECTORY / "events"
RESEARCH_DIRECTORY = DATA_DIRECTORY / "research"
SCRIPT_DIRECTORY = DATA_DIRECTORY / "scripts"
AUDIO_DIRECTORY = DATA_DIRECTORY / "audio"
VIDEO_DIRECTORY = DATA_DIRECTORY / "videos"

ACTIVE_EVENT_FILE = EVENT_DIRECTORY / "active_event.json"
USED_EVENTS_FILE = EVENT_DIRECTORY / "used_events.json"

LATEST_SCRIPT_FILE = SCRIPT_DIRECTORY / "latest.json"
LATEST_AUDIO_FILE = AUDIO_DIRECTORY / "latest.mp3"
LATEST_AUDIO_META_FILE = AUDIO_DIRECTORY / "latest_meta.json"
LATEST_WORDS_FILE = AUDIO_DIRECTORY / "latest_word_timestamps.json"
LATEST_VIDEO_FILE = VIDEO_DIRECTORY / "latest.mp4"

ASSET_ROOT = PROJECT_DIRECTORY / "assets" / "comic_pages"
CANDIDATE_ROOT = DATA_DIRECTORY / "comic_candidates"
WORK_DIRECTORY = DATA_DIRECTORY / "video_work"

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

DEFAULT_RESEARCH_COUNT = 6
DEFAULT_SCENE_COUNT = 8
DEFAULT_MAX_IMAGE_CANDIDATES = 18

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "ash"
TTS_SPEED = 1.08
ALIGNMENT_MODEL = "whisper-1"

SUBTITLE_WORDS_PER_CHUNK = 4

REQUEST_TIMEOUT = 30
DOWNLOAD_DELAY_SECONDS = 0.45

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0 Safari/537.36"
)

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
    "banner-ad",
    "social-icon",
    "emoji",
    "placeholder",
}

load_dotenv(PROJECT_DIRECTORY / ".env")


class ComicFactoryError(RuntimeError):
    """Comic Factory işlemi başarısız olduğunda oluşur."""


@dataclass
class ImageCandidate:
    """Bir web sayfasında bulunan kullanılabilir görsel."""

    candidate_id: str
    source_name: str
    source_page: str
    image_url: str
    alt_text: str
    local_file: str
    width: int
    height: int
    preliminary_score: float


class PageImageParser(HTMLParser):
    """HTML içinden comic görsel adaylarını çıkarır."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {
            key.lower(): value or ""
            for key, value in attrs
        }

        if tag.lower() == "meta":
            property_name = (
                attributes.get("property")
                or attributes.get("name")
                or ""
            ).lower()

            if property_name in {
                "og:image",
                "og:image:url",
                "twitter:image",
                "twitter:image:src",
            }:
                content = attributes.get(
                    "content",
                    "",
                ).strip()

                if content:
                    self.images.append(
                        {
                            "url": content,
                            "alt": "",
                        }
                    )

            return

        if tag.lower() != "img":
            return

        alt_text = (
            attributes.get("alt")
            or attributes.get("title")
            or ""
        ).strip()

        possible_urls = [
            attributes.get("src", ""),
            attributes.get("data-src", ""),
            attributes.get("data-lazy-src", ""),
            attributes.get("data-original", ""),
            attributes.get("data-image", ""),
        ]

        for srcset_key in (
            "srcset",
            "data-srcset",
        ):
            srcset = attributes.get(
                srcset_key,
                "",
            )

            if srcset:
                urls = []

                for item in srcset.split(","):
                    url = item.strip().split(" ")[0]

                    if url:
                        urls.append(url)

                possible_urls.extend(
                    reversed(urls)
                )

        for image_url in possible_urls:
            image_url = image_url.strip()

            if image_url:
                self.images.append(
                    {
                        "url": image_url,
                        "alt": alt_text,
                    }
                )


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""

    parser = argparse.ArgumentParser(
        description=(
            "Comic-book olaylarını araştırıp görsellerini "
            "toplayarak otomatik Shorts üretir."
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="API kullanmadan proje hazırlığını kontrol eder.",
    )

    parser.add_argument(
        "--reuse-active",
        action="store_true",
        help=(
            "Yeni konu araştırmak yerine "
            "active_event.json olayını kullanır."
        ),
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Video tamamlanınca youtube_uploader.py "
            "ile YouTube'a yollar."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_RESEARCH_COUNT,
        help="Araştırılacak comic-event aday sayısı.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGE_CANDIDATES,
        help="AI seçiminde değerlendirilecek maksimum görsel.",
    )

    parser.add_argument(
        "--subtitle-offset",
        type=float,
        default=0.0,
        help="Kelime vurgusunu saniye olarak kaydırır.",
    )

    return parser.parse_args()


def ensure_directories() -> None:
    """Gerekli proje klasörlerini oluşturur."""

    directories = (
        EVENT_DIRECTORY,
        RESEARCH_DIRECTORY,
        SCRIPT_DIRECTORY,
        AUDIO_DIRECTORY,
        VIDEO_DIRECTORY,
        ASSET_ROOT,
        CANDIDATE_ROOT,
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_json(
    file_path: Path,
    default: Any = None,
) -> Any:
    """JSON dosyasını yükler."""

    if not file_path.exists():
        return default

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ComicFactoryError(
            f"JSON okunamadı: {file_path}"
        ) from error


def save_json(
    file_path: Path,
    payload: Any,
) -> None:
    """JSON dosyasını kaydeder."""

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def clean_text(value: Any) -> str:
    """Metni normalize eder."""

    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def safe_slug(value: str) -> str:
    """Dosya sistemi için güvenli ID üretir."""

    value = (
        clean_text(value)
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

    return value.strip("_")[:80] or "comic_event"


def create_client() -> OpenAI:
    """OpenAI istemcisini oluşturur."""

    api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise ComicFactoryError(
            "OPENAI_API_KEY bulunamadı."
        )

    return OpenAI(
        api_key=api_key
    )


def get_model() -> str:
    """Ana model adını döndürür."""

    return (
        os.getenv(
            "OPENAI_MODEL",
            "",
        ).strip()
        or "gpt-5.6"
    )


def create_http_session() -> requests.Session:
    """Web sayfaları ve görseller için HTTP oturumu oluşturur."""

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
        }
    )

    return session


def run_check(
    upload_requested: bool,
) -> None:
    """API kullanmadan sistem kontrolü yapar."""

    print()
    print("=" * 78)
    print("COMIC FACTORY - SİSTEM KONTROLÜ")
    print("=" * 78)
    print()

    problems: list[str] = []

    if not os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip():
        problems.append(
            "OPENAI_API_KEY yok."
        )

    try:
        imageio_ffmpeg.get_ffmpeg_exe()

    except Exception as error:
        problems.append(
            f"FFmpeg kullanılamıyor: {error}"
        )

    if upload_requested:
        uploader = (
            PROJECT_DIRECTORY
            / "youtube_uploader.py"
        )

        if not uploader.exists():
            problems.append(
                "youtube_uploader.py bulunamadı."
            )

    if problems:
        print("HAZIR DEĞİL")
        print()

        for problem in problems:
            print(
                f"- {problem}"
            )

        raise SystemExit(1)

    print("✓ OpenAI API anahtarı hazır")
    print("✓ FFmpeg hazır")
    print("✓ Comic klasörleri hazır")
    print()
    print("✓ Sistem kontrolü başarılı")


def get_used_event_keys() -> set[str]:
    """Daha önce tamamlanmış comic olaylarını döndürür."""

    payload = load_json(
        USED_EVENTS_FILE,
        default={
            "events": [],
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        return set()

    events = payload.get(
        "events",
        [],
    )

    if not isinstance(
        events,
        list,
    ):
        return set()

    return {
        clean_text(
            event.get(
                "event_key",
                "",
            )
        )
        for event in events
        if isinstance(
            event,
            dict,
        )
        and clean_text(
            event.get(
                "event_key",
                "",
            )
        )
    }


def event_key(
    event: dict[str, Any],
) -> str:
    """Olay için tekrar kontrol anahtarı oluşturur."""

    return "|".join(
        [
            clean_text(
                event.get(
                    "publisher",
                    "",
                )
            ).casefold(),
            clean_text(
                event.get(
                    "series",
                    "",
                )
            ).casefold(),
            clean_text(
                event.get(
                    "issue",
                    "",
                )
            ).casefold(),
            clean_text(
                event.get(
                    "event_title",
                    "",
                )
            ).casefold(),
        ]
    )


def build_research_schema() -> dict[str, Any]:
    """Comic event araştırması JSON şeması."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_title": {
                            "type": "string",
                        },
                        "publisher": {
                            "type": "string",
                        },
                        "series": {
                            "type": "string",
                        },
                        "issue": {
                            "type": "string",
                        },
                        "publication_year": {
                            "type": [
                                "integer",
                                "null",
                            ],
                        },
                        "characters": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                        "hook": {
                            "type": "string",
                        },
                        "event_summary": {
                            "type": "string",
                        },
                        "power_feat": {
                            "type": "string",
                        },
                        "why_interesting": {
                            "type": "string",
                        },
                        "shorts_score": {
                            "type": "number",
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {
                                        "type": "string",
                                    },
                                    "url": {
                                        "type": "string",
                                    },
                                    "supports": {
                                        "type": "string",
                                    },
                                },
                                "required": [
                                    "name",
                                    "url",
                                    "supports",
                                ],
                            },
                        },
                        "visual_beats": {
                            "type": "array",
                            "minItems": DEFAULT_SCENE_COUNT,
                            "maxItems": DEFAULT_SCENE_COUNT,
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "event_title",
                        "publisher",
                        "series",
                        "issue",
                        "publication_year",
                        "characters",
                        "hook",
                        "event_summary",
                        "power_feat",
                        "why_interesting",
                        "shorts_score",
                        "sources",
                        "visual_beats",
                    ],
                },
            },
        },
        "required": [
            "events",
        ],
    }


def research_events(
    client: OpenAI,
    count: int,
) -> list[dict[str, Any]]:
    """Web araştırmasıyla yeni comic olayları bulur."""

    count = max(
        3,
        min(
            count,
            10,
        ),
    )

    used_keys = sorted(
        get_used_event_keys()
    )

    used_text = (
        "\n".join(
            f"- {key}"
            for key in used_keys
        )
        if used_keys
        else "Yok."
    )

    prompt = f"""
İnterneti araştır ve YouTube Shorts için tam olarak {count}
çok güçlü comic-book event adayı bul.

Öncelik:
Marvel, DC, Image, Dark Horse ve büyük Amerikan comic yayıncıları.

Aradığımız şey genel hikâye değil.
Spesifik tek bir olay, feat, dönüşüm, kozmik an
veya inanılmaz güç gösterisi.

İzleyici:
"Bu gerçekten çizgi romanda mı oldu?"
demeli.

Kriterler:
- şaşırtıcı
- güçlü
- görsel olarak iyi
- 45-60 saniyede anlatılabilir
- doğru issue tespit edilebilir
- web'de resmi preview, inceleme veya makale görselleri bulunabilir

Mümkün olduğunca:
- resmi publisher
- ciddi comic journalism
- comic database
- issue review
kullan.

Bilgi uydurma.
URL uydurma.
Issue uydurma.

Her olay için TAM 8 adet visual_beats oluştur.

shorts_score 0-100 arasında olsun.

Daha önce kullanılan olaylar:

{used_text}

Tüm anlatıcı metin alanlarını Türkçe yaz.
Comic ve karakter isimlerini orijinal bırak.
"""

    print()
    print("=" * 78)
    print("1/7 - YENİ COMIC OLAYLARI ARAŞTIRILIYOR")
    print("=" * 78)
    print()

    response = client.responses.create(
        model=get_model(),
        instructions=(
            "Titiz bir comic-book araştırmacısısın. "
            "Web aramasıyla doğrulanmış, görsel olarak güçlü "
            "ve Shorts formatına uygun spesifik olaylar bul."
        ),
        input=prompt,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium",
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "comic_factory_research",
                "strict": True,
                "schema": build_research_schema(),
            }
        },
        max_output_tokens=10000,
    )

    if not response.output_text:
        raise ComicFactoryError(
            "Comic araştırması boş sonuç döndürdü."
        )

    payload = json.loads(
        response.output_text
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
            "Araştırma sonucu events listesi içermiyor."
        )

    used_keys_set = get_used_event_keys()

    eligible_events = [
        event
        for event in events
        if isinstance(
            event,
            dict,
        )
        and event_key(
            event
        )
        not in used_keys_set
    ]

    eligible_events.sort(
        key=lambda event: float(
            event.get(
                "shorts_score",
                0,
            )
        ),
        reverse=True,
    )

    if not eligible_events:
        raise ComicFactoryError(
            "Kullanılmamış yeni comic olayı bulunamadı."
        )

    now = datetime.now(
        ISTANBUL_TIMEZONE
    )

    save_json(
        RESEARCH_DIRECTORY
        / (
            "research_"
            + now.strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        ),
        {
            "researched_at": now.isoformat(),
            "events": events,
        },
    )

    print(
        f"✓ {len(eligible_events)} kullanılabilir aday bulundu."
    )

    for index, event in enumerate(
        eligible_events[:5],
        start=1,
    ):
        print(
            f"{index}. "
            f"{clean_text(event.get('event_title'))} "
            f"({event.get('shorts_score', 0)}/100)"
        )

    return eligible_events


def activate_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    """Seçilen olayı aktif üretim konusuna dönüştürür."""

    active = dict(
        event
    )

    event_id = safe_slug(
        "_".join(
            [
                clean_text(
                    event.get(
                        "series",
                        "",
                    )
                ),
                clean_text(
                    event.get(
                        "issue",
                        "",
                    )
                ),
                clean_text(
                    event.get(
                        "event_title",
                        "",
                    )
                ),
            ]
        )
    )

    active[
        "id"
    ] = event_id

    active[
        "event_key"
    ] = event_key(
        event
    )

    active[
        "activated_at"
    ] = datetime.now(
        ISTANBUL_TIMEZONE
    ).isoformat()

    active[
        "status"
    ] = "active"

    save_json(
        ACTIVE_EVENT_FILE,
        active,
    )

    print()
    print(
        f"✓ Seçilen olay: "
        f"{clean_text(active.get('event_title'))}"
    )

    print(
        f"✓ Comic: "
        f"{clean_text(active.get('series'))} "
        f"{clean_text(active.get('issue'))}"
    )

    return active


def load_active_event() -> dict[str, Any]:
    """Mevcut aktif eventi yükler."""

    event = load_json(
        ACTIVE_EVENT_FILE
    )

    if not isinstance(
        event,
        dict,
    ):
        raise ComicFactoryError(
            "active_event.json bulunamadı."
        )

    if not clean_text(
        event.get(
            "id",
            "",
        )
    ):
        raise ComicFactoryError(
            "Aktif event geçersiz."
        )

    return event


def build_source_schema() -> dict[str, Any]:
    """Görsel kaynak keşfi için şema."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                        },
                        "url": {
                            "type": "string",
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "name",
                        "url",
                        "reason",
                    ],
                },
            },
        },
        "required": [
            "pages",
        ],
    }


def discover_visual_pages(
    client: OpenAI,
    event: dict[str, Any],
) -> list[dict[str, str]]:
    """Comic panellerini gösterme ihtimali yüksek sayfaları bulur."""

    existing_sources = event.get(
        "sources",
        [],
    )

    existing_text = json.dumps(
        existing_sources,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
Aşağıdaki comic event için internette görsel kaynakları ara.

EVENT:
{clean_text(event.get("event_title"))}

COMIC:
{clean_text(event.get("publisher"))}
{clean_text(event.get("series"))}
{clean_text(event.get("issue"))}
({event.get("publication_year")})

CHARACTERS:
{", ".join(event.get("characters", []))}

OLAY:
{clean_text(event.get("event_summary"))}

Mevcut araştırma kaynakları:
{existing_text}

Amaç:
Bu olayın comic panellerini / preview görsellerini /
issue sayfalarını gerçekten HTML içinde gösteren
4-8 adet kamuya açık web sayfası bul.

Özellikle:
- resmi preview
- resmi publisher issue page
- spoiler review
- comic issue review
- ciddi comic news article

Sadece ana sayfa değil,
olayın geçtiği spesifik article URL'sini ver.

Korsan arşiv veya rastgele image board önerme.

URL uydurma.
"""

    print()
    print("=" * 78)
    print("2/7 - COMIC GÖRSEL KAYNAKLARI BULUNUYOR")
    print("=" * 78)
    print()

    response = client.responses.create(
        model=get_model(),
        input=prompt,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium",
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "comic_visual_sources",
                "strict": True,
                "schema": build_source_schema(),
            }
        },
        max_output_tokens=4000,
    )

    if not response.output_text:
        raise ComicFactoryError(
            "Görsel kaynak araştırması boş döndü."
        )

    payload = json.loads(
        response.output_text
    )

    pages = payload.get(
        "pages",
        [],
    )

    if not isinstance(
        pages,
        list,
    ):
        pages = []

    result: list[
        dict[str, str]
    ] = []

    seen_urls: set[str] = set()

    for source in existing_sources:
        if not isinstance(
            source,
            dict,
        ):
            continue

        url = clean_text(
            source.get(
                "url",
                "",
            )
        )

        if (
            url.startswith(
                "http"
            )
            and url not in seen_urls
        ):
            result.append(
                {
                    "name": clean_text(
                        source.get(
                            "name",
                            "Research source",
                        )
                    ),
                    "url": url,
                    "reason": clean_text(
                        source.get(
                            "supports",
                            "",
                        )
                    ),
                }
            )

            seen_urls.add(
                url
            )

    for page in pages:
        if not isinstance(
            page,
            dict,
        ):
            continue

        url = clean_text(
            page.get(
                "url",
                "",
            )
        )

        if (
            not url.startswith(
                "http"
            )
            or url in seen_urls
        ):
            continue

        result.append(
            {
                "name": clean_text(
                    page.get(
                        "name",
                        "Web source",
                    )
                ),
                "url": url,
                "reason": clean_text(
                    page.get(
                        "reason",
                        "",
                    )
                ),
            }
        )

        seen_urls.add(
            url
        )

    if not result:
        raise ComicFactoryError(
            "Görsel kaynak sayfası bulunamadı."
        )

    print(
        f"✓ {len(result)} kaynak sayfası bulundu."
    )

    return result[:10]


def normalize_image_url(
    page_url: str,
    image_url: str,
) -> str:
    """HTML içindeki görsel URL'sini tam URL'ye dönüştürür."""

    image_url = html.unescape(
        clean_text(
            image_url
        )
    )

    if image_url.startswith(
        "//"
    ):
        image_url = (
            "https:"
            + image_url
        )

    return urljoin(
        page_url,
        image_url,
    )


def is_probably_bad_image_url(
    image_url: str,
    alt_text: str,
) -> bool:
    """Logo, avatar ve tracking görsellerini filtreler."""

    combined = (
        image_url
        + " "
        + alt_text
    ).casefold()

    if any(
        term in combined
        for term in BAD_IMAGE_TERMS
    ):
        return True

    parsed = urlparse(
        image_url
    )

    extension = Path(
        parsed.path
    ).suffix.casefold()

    if extension in {
        ".svg",
        ".gif",
        ".ico",
    }:
        return True

    return False


def scrape_page_images(
    session: requests.Session,
    source: dict[str, str],
) -> list[dict[str, str]]:
    """Tek web sayfasındaki görsel URL'lerini çıkarır."""

    url = source[
        "url"
    ]

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print(
            f"! Sayfa okunamadı: {url}"
        )
        print(
            f"  {error}"
        )

        return []

    parser = PageImageParser()

    try:
        parser.feed(
            response.text
        )

    except Exception:
        return []

    discovered: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for image in parser.images:
        image_url = normalize_image_url(
            url,
            image.get(
                "url",
                "",
            ),
        )

        alt_text = clean_text(
            image.get(
                "alt",
                "",
            )
        )

        if (
            not image_url.startswith(
                "http"
            )
            or image_url in seen
            or is_probably_bad_image_url(
                image_url,
                alt_text,
            )
        ):
            continue

        seen.add(
            image_url
        )

        discovered.append(
            {
                "source_name": source[
                    "name"
                ],
                "source_page": url,
                "image_url": image_url,
                "alt_text": alt_text,
            }
        )

    return discovered


def image_context_score(
    item: dict[str, str],
    event: dict[str, Any],
) -> float:
    """URL ve alt metinden kaba comic-ilgi puanı hesaplar."""

    text = (
        item.get(
            "image_url",
            ""
        )
        + " "
        + item.get(
            "alt_text",
            ""
        )
    ).casefold()

    score = 0.0

    terms = [
        clean_text(
            event.get(
                "series",
                "",
            )
        ),
        clean_text(
            event.get(
                "issue",
                "",
            )
        ),
        *[
            clean_text(
                character
            )
            for character in event.get(
                "characters",
                []
            )
        ],
    ]

    for term in terms:
        term = term.casefold()

        if (
            len(
                term
            )
            >= 3
            and term in text
        ):
            score += 3.0

    if any(
        keyword in text
        for keyword in [
            "comic",
            "panel",
            "preview",
            "issue",
            "page",
        ]
    ):
        score += 1.5

    if any(
        keyword in text
        for keyword in [
            "cover",
            "variant",
        ]
    ):
        score += 1.0

    return score


def perceptual_hash(
    image: Image.Image,
) -> str:
    """Benzer görselleri elemek için basit average hash üretir."""

    image = (
        image.convert(
            "L"
        )
        .resize(
            (
                8,
                8,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    pixels = list(
        image.getdata()
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


def download_candidate(
    session: requests.Session,
    raw_candidate: dict[str, str],
    output_file: Path,
) -> tuple[int, int, str] | None:
    """Web görselini indirip JPG'ye çevirir."""

    headers = {
        "Referer": raw_candidate[
            "source_page"
        ],
    }

    try:
        response = session.get(
            raw_candidate[
                "image_url"
            ],
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            time.sleep(
                3
            )

            response = session.get(
                raw_candidate[
                    "image_url"
                ],
                headers=headers,
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
    ):
        return None

    if (
        width
        * height
        < 350_000
    ):
        return None

    image_hash = perceptual_hash(
        image
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_file,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return (
        width,
        height,
        image_hash,
    )


def collect_candidates(
    event: dict[str, Any],
    sources: list[dict[str, str]],
    max_candidates: int,
) -> list[ImageCandidate]:
    """Kaynak sayfalarından gerçek görsel adaylarını toplar."""

    print()
    print("=" * 78)
    print("3/7 - COMIC GÖRSELLERİ OTOMATİK TOPLANIYOR")
    print("=" * 78)
    print()

    session = create_http_session()

    raw_candidates: list[
        dict[str, str]
    ] = []

    seen_urls: set[str] = set()

    for source in sources:
        print(
            f"Kaynak taranıyor: {source['name']}"
        )

        page_images = scrape_page_images(
            session,
            source,
        )

        for image in page_images:
            if image[
                "image_url"
            ] in seen_urls:
                continue

            seen_urls.add(
                image[
                    "image_url"
                ]
            )

            raw_candidates.append(
                image
            )

    if not raw_candidates:
        raise ComicFactoryError(
            "Kaynak sayfalardan hiçbir görsel URL'si çıkarılamadı."
        )

    raw_candidates.sort(
        key=lambda item: image_context_score(
            item,
            event,
        ),
        reverse=True,
    )

    event_id = clean_text(
        event[
            "id"
        ]
    )

    candidate_directory = (
        CANDIDATE_ROOT
        / event_id
    )

    if candidate_directory.exists():
        shutil.rmtree(
            candidate_directory,
            ignore_errors=True,
        )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted: list[
        ImageCandidate
    ] = []

    seen_hashes: set[str] = set()

    for raw_candidate in raw_candidates:
        if len(
            accepted
        ) >= max_candidates:
            break

        candidate_id = (
            f"c{len(accepted) + 1:02d}"
        )

        output_file = (
            candidate_directory
            / f"{candidate_id}.jpg"
        )

        result = download_candidate(
            session,
            raw_candidate,
            output_file,
        )

        if result is None:
            continue

        width, height, image_hash = result

        if image_hash in seen_hashes:
            output_file.unlink(
                missing_ok=True
            )

            continue

        seen_hashes.add(
            image_hash
        )

        preliminary_score = image_context_score(
            raw_candidate,
            event,
        )

        preliminary_score += min(
            4.0,
            (
                width
                * height
            )
            / 1_000_000,
        )

        accepted.append(
            ImageCandidate(
                candidate_id=candidate_id,
                source_name=raw_candidate[
                    "source_name"
                ],
                source_page=raw_candidate[
                    "source_page"
                ],
                image_url=raw_candidate[
                    "image_url"
                ],
                alt_text=raw_candidate[
                    "alt_text"
                ],
                local_file=str(
                    output_file
                ),
                width=width,
                height=height,
                preliminary_score=preliminary_score,
            )
        )

        print(
            f"✓ {candidate_id}: "
            f"{width}x{height} "
            f"({raw_candidate['source_name']})"
        )

        time.sleep(
            DOWNLOAD_DELAY_SECONDS
        )

    if len(
        accepted
    ) < 4:
        raise ComicFactoryError(
            "Yeterli kaliteli comic görseli bulunamadı. "
            f"Bulunan: {len(accepted)}"
        )

    accepted.sort(
        key=lambda candidate: candidate.preliminary_score,
        reverse=True,
    )

    print()
    print(
        f"✓ {len(accepted)} kaliteli görsel adayı hazır."
    )

    return accepted


def image_to_data_url(
    file_path: Path,
) -> str:
    """Yerel görseli düşük maliyetli vision thumbnail'a dönüştürür."""

    with Image.open(
        file_path
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
        format="JPEG",
        quality=72,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "ascii"
    )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def build_assignment_schema() -> dict[str, Any]:
    """Vision tabanlı sahne eşleştirmesi şeması."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": DEFAULT_SCENE_COUNT,
                "maxItems": DEFAULT_SCENE_COUNT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scene_number": {
                            "type": "integer",
                        },
                        "candidate_id": {
                            "type": "string",
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "scene_number",
                        "candidate_id",
                        "confidence",
                        "reason",
                    ],
                },
            },
        },
        "required": [
            "assignments",
        ],
    }


def assign_images_with_vision(
    client: OpenAI,
    event: dict[str, Any],
    candidates: list[ImageCandidate],
) -> list[dict[str, Any]]:
    """Görselleri gerçekten görerek 8 sahneye eşleştirir."""

    print()
    print("=" * 78)
    print("4/7 - AI COMIC PANELLERİNİ GÖREREK SEÇİYOR")
    print("=" * 78)
    print()

    visual_beats = event.get(
        "visual_beats",
        [],
    )

    if not isinstance(
        visual_beats,
        list,
    ):
        visual_beats = []

    visual_beats = [
        clean_text(
            beat
        )
        for beat in visual_beats
    ][:DEFAULT_SCENE_COUNT]

    while len(
        visual_beats
    ) < DEFAULT_SCENE_COUNT:
        visual_beats.append(
            f"Comic olayının {len(visual_beats) + 1}. önemli anı"
        )

    content: list[
        dict[str, Any]
    ] = []

    content.append(
        {
            "type": "input_text",
            "text": (
                "EVENT:\n"
                + clean_text(
                    event.get(
                        "event_title",
                        "",
                    )
                )
                + "\n\nVISUAL BEATS:\n"
                + "\n".join(
                    f"{index}. {beat}"
                    for index, beat in enumerate(
                        visual_beats,
                        start=1,
                    )
                )
                + (
                    "\n\nAşağıdaki comic görsellerini gerçekten incele. "
                    "Her sahne için en uygun candidate_id seç. "
                    "Aynı görseli zorunlu olmadıkça tekrar kullanma. "
                    "Kapak veya alakasız görsel yerine "
                    "olayı gerçekten anlatan panelleri tercih et."
                )
            ),
        }
    )

    for candidate in candidates:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"CANDIDATE {candidate.candidate_id}\n"
                    f"Source: {candidate.source_name}\n"
                    f"Alt: {candidate.alt_text}\n"
                    f"Size: {candidate.width}x{candidate.height}"
                ),
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    Path(
                        candidate.local_file
                    )
                ),
                "detail": "low",
            }
        )

    response = client.responses.create(
        model=get_model(),
        instructions=(
            "Sen comic-book video editörüsün. "
            "Verilen görselleri görsel olarak inceleyip "
            "anlatım sahnelerine en uygun şekilde eşleştir."
        ),
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "comic_image_assignment",
                "strict": True,
                "schema": build_assignment_schema(),
            }
        },
        max_output_tokens=3000,
    )

    if not response.output_text:
        raise ComicFactoryError(
            "Vision panel seçimi boş sonuç döndürdü."
        )

    payload = json.loads(
        response.output_text
    )

    assignments = payload.get(
        "assignments",
        [],
    )

    if not isinstance(
        assignments,
        list,
    ):
        raise ComicFactoryError(
            "Panel assignment listesi alınamadı."
        )

    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    selected: list[
        dict[str, Any]
    ] = []

    for scene_number in range(
        1,
        DEFAULT_SCENE_COUNT + 1,
    ):
        assignment = next(
            (
                item
                for item in assignments
                if isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "scene_number"
                )
                == scene_number
                and clean_text(
                    item.get(
                        "candidate_id",
                        "",
                    )
                )
                in candidate_map
            ),
            None,
        )

        if assignment is None:
            fallback = candidates[
                (
                    scene_number
                    - 1
                )
                % len(
                    candidates
                )
            ]

            assignment = {
                "scene_number": scene_number,
                "candidate_id": fallback.candidate_id,
                "confidence": 30,
                "reason": "Yerel fallback",
            }

        candidate = candidate_map[
            assignment[
                "candidate_id"
            ]
        ]

        selected.append(
            {
                "scene_number": scene_number,
                "visual_beat": visual_beats[
                    scene_number
                    - 1
                ],
                "candidate": candidate,
                "confidence": assignment[
                    "confidence"
                ],
                "reason": assignment[
                    "reason"
                ],
            }
        )

        print(
            f"✓ Sahne {scene_number}: "
            f"{candidate.candidate_id} "
            f"({assignment['confidence']}%)"
        )

    return selected


def build_assets(
    event: dict[str, Any],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Seçilmiş comic panellerini final asset klasörüne kopyalar."""

    event_directory = (
        ASSET_ROOT
        / clean_text(
            event[
                "id"
            ]
        )
    )

    if event_directory.exists():
        shutil.rmtree(
            event_directory,
            ignore_errors=True,
        )

    event_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    assets: list[
        dict[str, Any]
    ] = []

    for item in selected:
        scene_number = int(
            item[
                "scene_number"
            ]
        )

        candidate: ImageCandidate = item[
            "candidate"
        ]

        output_file = (
            event_directory
            / f"{scene_number:02d}_comic.jpg"
        )

        shutil.copy2(
            candidate.local_file,
            output_file,
        )

        assets.append(
            {
                "scene_number": scene_number,
                "filename": output_file.name,
                "local_file": str(
                    output_file.relative_to(
                        PROJECT_DIRECTORY
                    )
                ).replace(
                    "\\",
                    "/",
                ),
                "visual_beat": item[
                    "visual_beat"
                ],
                "source_name": candidate.source_name,
                "source_page": candidate.source_page,
                "image_url": candidate.image_url,
                "alt_text": candidate.alt_text,
                "selection_confidence": item[
                    "confidence"
                ],
                "selection_reason": item[
                    "reason"
                ],
            }
        )

    manifest = {
        "event_id": event[
            "id"
        ],
        "event_title": event[
            "event_title"
        ],
        "publisher": event[
            "publisher"
        ],
        "series": event[
            "series"
        ],
        "issue": event[
            "issue"
        ],
        "publication_year": event[
            "publication_year"
        ],
        "asset_count": len(
            assets
        ),
        "assets": assets,
    }

    save_json(
        event_directory
        / "downloaded_sources.json",
        manifest,
    )

    return assets


def build_script_schema() -> dict[str, Any]:
    """Shorts script structured output şeması."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {
                "type": "string",
            },
            "narration": {
                "type": "string",
            },
            "description": {
                "type": "string",
            },
            "hashtags": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "scene_captions": {
                "type": "array",
                "minItems": DEFAULT_SCENE_COUNT,
                "maxItems": DEFAULT_SCENE_COUNT,
                "items": {
                    "type": "string",
                },
            },
            "scene_narrations": {
                "type": "array",
                "minItems": DEFAULT_SCENE_COUNT,
                "maxItems": DEFAULT_SCENE_COUNT,
                "items": {
                    "type": "string",
                },
            },
        },
        "required": [
            "title",
            "narration",
            "description",
            "hashtags",
            "scene_captions",
            "scene_narrations",
        ],
    }


def generate_script(
    client: OpenAI,
    event: dict[str, Any],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aktif comic event için Türkçe Shorts senaryosu üretir."""

    print()
    print("=" * 78)
    print("5/7 - TÜRKÇE SHORTS SENARYOSU")
    print("=" * 78)
    print()

    visual_plan = "\n".join(
        (
            f"{asset['scene_number']}. "
            f"{asset['visual_beat']}"
        )
        for asset in assets
    )

    prompt = f"""
COMIC EVENT:

Başlık:
{clean_text(event.get("event_title"))}

Comic:
{clean_text(event.get("publisher"))}
{clean_text(event.get("series"))}
{clean_text(event.get("issue"))}
({event.get("publication_year")})

Karakterler:
{", ".join(event.get("characters", []))}

Hook:
{clean_text(event.get("hook"))}

Olay:
{clean_text(event.get("event_summary"))}

Güç olayı:
{clean_text(event.get("power_feat"))}

Neden ilginç:
{clean_text(event.get("why_interesting"))}

GÖRSEL AKIŞ:
{visual_plan}

Bundan Türkçe 45-60 saniyelik yüksek-retention
YouTube Shorts senaryosu oluştur.

Kurallar:

- İlk cümle izleyiciyi hemen yakalasın.
- İnanılmaz bir comic olayını arkadaşına anlatıyormuş gibi yaz.
- Gereksiz giriş yapma.
- Olay her cümlede ilerlesin.
- 120-145 kelime civarında tut.
- Doğrulanmayan bilgi ekleme.
- 8 scene_narrations üret.
- 8 kısa scene_captions üret.
- Caption en fazla 4 kelime olsun.
- Sahne numarası yazma.
- Finalde izleyiciye kısa soru sor.
- Comic/character isimlerini orijinal bırak.
"""

    response = client.responses.create(
        model=get_model(),
        instructions=(
            "Yüksek retention sağlayan Türkçe comic-book "
            "Shorts senaryoları yaz. "
            "Bilgi uydurma ve olay akışını güçlü tut."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "comic_factory_script",
                "strict": True,
                "schema": build_script_schema(),
            }
        },
        max_output_tokens=3500,
    )

    if not response.output_text:
        raise ComicFactoryError(
            "Senaryo üretilemedi."
        )

    script = json.loads(
        response.output_text
    )

    script[
        "titles"
    ] = [
        script[
            "title"
        ]
    ]

    sources: list[
        tuple[str, str]
    ] = []

    for asset in assets:
        pair = (
            clean_text(
                asset[
                    "source_name"
                ]
            ),
            clean_text(
                asset[
                    "source_page"
                ]
            ),
        )

        if pair not in sources:
            sources.append(
                pair
            )

    source_text = "\n".join(
        f"- {name}: {url}"
        for name, url in sources
    )

    hashtags = " ".join(
        clean_text(
            tag
        )
        for tag in script.get(
            "hashtags",
            []
        )
    )

    script[
        "full_description"
    ] = (
        clean_text(
            script[
                "description"
            ]
        )
        + "\n\n"
        + hashtags
        + "\n\nKaynaklar:\n"
        + source_text
    )

    payload = {
        "event_id": event[
            "id"
        ],
        "generated_at": datetime.now(
            ISTANBUL_TIMEZONE
        ).isoformat(),
        "script": script,
    }

    save_json(
        LATEST_SCRIPT_FILE,
        payload,
    )

    timestamp = datetime.now(
        ISTANBUL_TIMEZONE
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    save_json(
        SCRIPT_DIRECTORY
        / f"comic_script_{timestamp}.json",
        payload,
    )

    print(
        f"✓ Başlık: {script['title']}"
    )

    return script


def generate_voice(
    client: OpenAI,
    event: dict[str, Any],
    narration: str,
) -> None:
    """Yüksek enerjili Ash seslendirmeyi üretir."""

    print()
    print("=" * 78)
    print("6/7 - ASH YÜKSEK ENERJİLİ SES")
    print("=" * 78)
    print()

    characters = ", ".join(
        event.get(
            "characters",
            []
        )
    )

    instructions = f"""
Speak in natural Turkish as an excited adult male comic-book storyteller.

You have just discovered one of the craziest events in comic-book history
and you genuinely cannot wait to tell somebody.

Overall energy: 9/10.

Sound amazed, fascinated, entertained and confident.

Speak like a charismatic Turkish YouTube creator,
not like somebody reading written text.

Start with immediate energy.

Build tension quickly.
Use short dramatic pauses before major revelations.
Hit shocking reveals with much stronger emphasis.

Change pitch, speed and intensity naturally.

Do not become monotone after the hook.

Important character names in this story:
{characters}

Pronounce comic-book names confidently inside Turkish sentences.

Do not sound:
calm,
sleepy,
documentary-like,
like a newsreader,
like an audiobook,
corporate,
or robotic.

Do not scream constantly.

This is a high-retention YouTube Short about an unbelievable comic event.
Every sentence should make the viewer want to hear what happens next.
""".strip()

    timestamp = datetime.now(
        ISTANBUL_TIMEZONE
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive_file = (
        AUDIO_DIRECTORY
        / f"comic_ash_{timestamp}.mp3"
    )

    with (
        client.audio.speech
        .with_streaming_response
        .create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=narration,
            instructions=instructions,
            speed=TTS_SPEED,
        )
    ) as response:
        response.stream_to_file(
            archive_file
        )

    shutil.copy2(
        archive_file,
        LATEST_AUDIO_FILE,
    )

    save_json(
        LATEST_AUDIO_META_FILE,
        {
            "event_id": event[
                "id"
            ],
            "created_at": datetime.now(
                ISTANBUL_TIMEZONE
            ).isoformat(),
            "voice": TTS_VOICE,
            "model": TTS_MODEL,
            "speed": TTS_SPEED,
        },
    )

    print(
        "✓ Ash ses hazır."
    )


def get_object_value(
    value: Any,
    key: str,
    default: Any = None,
) -> Any:
    """SDK nesnesi veya dict içinden alan okur."""

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
    client: OpenAI,
    narration: str,
) -> list[dict[str, Any]]:
    """Ses dosyasından gerçek kelime zamanlarını çıkarır."""

    print()
    print(
        "Gerçek kelime zamanlaması çıkarılıyor..."
    )

    with LATEST_AUDIO_FILE.open(
        "rb"
    ) as audio_file:
        transcription = client.audio.transcriptions.create(
            model=ALIGNMENT_MODEL,
            file=audio_file,
            language="tr",
            response_format="verbose_json",
            timestamp_granularities=[
                "word"
            ],
            prompt=narration[
                :1000
            ],
            temperature=0,
        )

    raw_words = get_object_value(
        transcription,
        "words",
        [],
    )

    words: list[
        dict[str, Any]
    ] = []

    for raw_word in raw_words or []:
        word = clean_text(
            get_object_value(
                raw_word,
                "word",
                "",
            )
        )

        start = get_object_value(
            raw_word,
            "start",
        )

        end = get_object_value(
            raw_word,
            "end",
        )

        if (
            not word
            or start is None
            or end is None
        ):
            continue

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
            "Kelime timestamp verisi alınamadı."
        )

    save_json(
        LATEST_WORDS_FILE,
        {
            "word_count": len(
                words
            ),
            "words": words,
        },
    )

    print(
        f"✓ {len(words)} gerçek kelime zamanı hazır."
    )

    return words


def get_ffmpeg() -> str:
    """Bundled FFmpeg yolunu döndürür."""

    return imageio_ffmpeg.get_ffmpeg_exe()


def get_audio_duration(
    ffmpeg: str,
) -> float:
    """Ses dosyasının süresini bulur."""

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


def get_font(
    size: int,
    comic: bool = False,
) -> ImageFont.ImageFont:
    """Windows ve Linux üzerinde uygun font yükler."""

    if comic:
        candidates = [
            Path(
                r"C:\Windows\Fonts\impact.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\ariblk.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\arialbd.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSansCondensed-Bold.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf"
            ),
        ]

    else:
        candidates = [
            Path(
                r"C:\Windows\Fonts\arialbd.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\seguisb.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\arial.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans.ttf"
            ),
        ]

    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(
                str(
                    font_path
                ),
                size=size,
            )

    return ImageFont.load_default()


def resize_cover(
    image: Image.Image,
) -> Image.Image:
    """
    Comic sayfasını kırpmadan 9:16 videoya yerleştirir.

    Ana comic sayfası tamamen görünür.
    Arka plan aynı görselin bulanık versiyonudur.
    """

    background = ImageOps.fit(
        image,
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        method=Image.Resampling.LANCZOS,
        centering=(
            0.5,
            0.5,
        ),
    )

    background = background.filter(
        ImageFilter.GaussianBlur(
            radius=38
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

    side_padding = 32
    top_safe_area = 205
    bottom_safe_area = 315

    max_width = (
        VIDEO_WIDTH
        - side_padding
        * 2
    )

    max_height = (
        VIDEO_HEIGHT
        - top_safe_area
        - bottom_safe_area
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
        VIDEO_WIDTH
        - foreground_width
    ) // 2

    y = (
        top_safe_area
        + (
            max_height
            - foreground_height
        )
        // 2
    )

    shadow = Image.new(
        "RGBA",
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
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
            radius=22
        )
    )

    canvas = Image.alpha_composite(
        canvas,
        shadow,
    )

    panel = foreground.convert(
        "RGBA"
    )

    canvas.alpha_composite(
        panel,
        (
            x,
            y,
        ),
    )

    border_draw = ImageDraw.Draw(
        canvas
    )

    border_draw.rectangle(
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
    output_file: Path,
) -> None:
    """Video sahnesinin sabit temel karesini oluşturur."""

    source_file = (
        PROJECT_DIRECTORY
        / asset[
            "local_file"
        ]
    )

    with Image.open(
        source_file
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    frame = resize_cover(
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

    dark = Image.new(
        "RGBA",
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    dark_draw = ImageDraw.Draw(
        dark
    )

    dark_draw.rectangle(
        (
            0,
            0,
            VIDEO_WIDTH,
            210,
        ),
        fill=(
            0,
            0,
            0,
            90,
        ),
    )

    dark_draw.rectangle(
        (
            0,
            1730,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
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
        dark,
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

    issue_text = (
        f"{clean_text(event.get('series'))} "
        f"{clean_text(event.get('issue'))}"
    )

    year = event.get(
        "publication_year"
    )

    if year:
        issue_text += (
            f" • {year}"
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
        issue_text[
            :35
        ].upper(),
        font=issue_font,
        fill=(
            255,
            255,
            255,
            245,
        ),
    )

    short_caption = " ".join(
        clean_text(
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
        VIDEO_WIDTH
        - caption_width
        - 38,
    )

    draw.rounded_rectangle(
        (
            caption_x - 17,
            103,
            VIDEO_WIDTH - 22,
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

    source_name = clean_text(
        asset.get(
            "source_name",
            "",
        )
    )

    draw.text(
        (
            26,
            1880,
        ),
        source_name[
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
        output_file,
        format="PNG",
    )


def estimated_scene_durations(
    scene_narrations: list[str],
    words: list[dict[str, Any]],
    audio_duration: float,
) -> list[float]:
    """Sahne değişimlerini gerçek konuşma akışına yaklaştırır."""

    counts = [
        max(
            1,
            len(
                re.findall(
                    r"\w+",
                    scene,
                    flags=re.UNICODE,
                )
            ),
        )
        for scene in scene_narrations
    ]

    total_count = sum(
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
            / total_count
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
        audio_duration
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
                0.3,
                boundaries[
                    index + 1
                ]
                - boundaries[
                    index
                ],
            )
        )

    difference = (
        audio_duration
        - sum(
            durations
        )
    )

    durations[
        -1
    ] += difference

    return durations


def run_ffmpeg(
    command: list[str],
    error_message: str,
    cwd: Path | None = None,
) -> None:
    """FFmpeg komutunu güvenli biçimde çalıştırır."""

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
            error_message
            + "\n"
            + process.stderr[
                -4000:
            ]
        )


def render_segment(
    ffmpeg: str,
    frame_file: Path,
    output_file: Path,
    duration: float,
    index: int,
) -> None:
    """Tek comic görseline çok hafif hareket verir."""

    zoom_speed = (
        0.00010
        if index
        % 2
        else 0.00008
    )

    video_filter = (
        "scale=1080:1920,"
        "zoompan="
        f"z='min(zoom+{zoom_speed},1.025)':"
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
                VIDEO_FPS
            ),
            "-i",
            str(
                frame_file
            ),
            "-t",
            f"{duration:.3f}",
            "-vf",
            video_filter,
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
                output_file
            ),
        ],
        "Video sahnesi render edilemedi.",
    )


def seconds_to_ass_time(
    seconds: float,
) -> str:
    """Saniyeyi ASS zamanına çevirir."""

    seconds = max(
        0.0,
        seconds,
    )

    centiseconds = round(
        seconds
        * 100
    )

    hours = (
        centiseconds
        // 360000
    )

    centiseconds %= 360000

    minutes = (
        centiseconds
        // 6000
    )

    centiseconds %= 6000

    secs = (
        centiseconds
        // 100
    )

    cs = (
        centiseconds
        % 100
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
    """ASS altyazı özel karakterlerini temizler."""

    return (
        clean_text(
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


def subtitle_text(
    words: list[dict[str, Any]],
    active_index: int,
) -> str:
    """Aktif kelimeyi sarı gösteren 4 kelimelik parça üretir."""

    chunk_start = (
        active_index
        // SUBTITLE_WORDS_PER_CHUNK
    ) * SUBTITLE_WORDS_PER_CHUNK

    chunk_end = min(
        len(
            words
        ),
        chunk_start
        + SUBTITLE_WORDS_PER_CHUNK,
    )

    parts = []

    for index in range(
        chunk_start,
        chunk_end,
    ):
        word = ass_escape(
            words[
                index
            ][
                "word"
            ]
        )

        if index == active_index:
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
    """Gerçek word timestamp tabanlı subtitle oluşturur."""

    output_file = (
        WORK_DIRECTORY
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
            "Format: Name,Fontname,Fontsize,PrimaryColour,"
            "SecondaryColour,OutlineColour,BackColour,Bold,"
            "Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,"
            "Angle,BorderStyle,Outline,Shadow,Alignment,"
            "MarginL,MarginR,MarginV,Encoding"
        ),
        (
            "Style: Main,DejaVu Sans,52,"
            "&H00FFFFFF,&H00FFFFFF,&H00101010,&H78000000,"
            "-1,0,0,0,100,100,0,0,1,4,1,2,85,85,145,1"
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
        start = (
            float(
                word[
                    "start"
                ]
            )
            + offset
        )

        end = (
            float(
                word[
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
            f"{seconds_to_ass_time(start)},"
            f"{seconds_to_ass_time(end)},"
            "Main,,0,0,0,,"
            f"{subtitle_text(words, index)}"
        )

    output_file.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return output_file


def render_video(
    event: dict[str, Any],
    assets: list[dict[str, Any]],
    script: dict[str, Any],
    words: list[dict[str, Any]],
    subtitle_offset: float,
) -> Path:
    """Final Shorts videosunu oluşturur."""

    print()
    print("=" * 78)
    print("7/7 - FINAL COMIC SHORTS RENDER")
    print("=" * 78)
    print()

    ffmpeg = get_ffmpeg()

    audio_duration = get_audio_duration(
        ffmpeg
    )

    if WORK_DIRECTORY.exists():
        shutil.rmtree(
            WORK_DIRECTORY,
            ignore_errors=True,
        )

    WORK_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    captions = script[
        "scene_captions"
    ]

    narrations = script[
        "scene_narrations"
    ]

    durations = estimated_scene_durations(
        narrations,
        words,
        audio_duration,
    )

    segments: list[
        Path
    ] = []

    for index, (
        asset,
        caption,
        duration,
    ) in enumerate(
        zip(
            assets,
            captions,
            durations,
            strict=True,
        ),
        start=1,
    ):
        frame_file = (
            WORK_DIRECTORY
            / f"frame_{index:02d}.png"
        )

        segment_file = (
            WORK_DIRECTORY
            / f"segment_{index:02d}.mp4"
        )

        create_scene_frame(
            event,
            asset,
            clean_text(
                caption
            ),
            frame_file,
        )

        render_segment(
            ffmpeg,
            frame_file,
            segment_file,
            duration,
            index,
        )

        segments.append(
            segment_file
        )

        print(
            f"✓ Sahne {index}: "
            f"{duration:.2f} sn"
        )

    concat_file = (
        WORK_DIRECTORY
        / "concat.txt"
    )

    concat_file.write_text(
        "\n".join(
            "file '"
            + segment.resolve().as_posix()
            + "'"
            for segment in segments
        ),
        encoding="utf-8",
    )

    silent_file = (
        WORK_DIRECTORY
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
                silent_file
            ),
        ],
        "Segmentler birleştirilemedi.",
    )

    audio_video = (
        WORK_DIRECTORY
        / "audio_video.mp4"
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
                silent_file
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
                audio_video
            ),
        ],
        "Ses videoya eklenemedi.",
    )

    subtitle_file = create_ass(
        words,
        subtitle_offset,
    )

    timestamp = datetime.now(
        ISTANBUL_TIMEZONE
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    archive_file = (
        VIDEO_DIRECTORY
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
                audio_video.resolve()
            ),
            "-vf",
            f"ass={subtitle_file.name}",
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
                archive_file.resolve()
            ),
        ],
        "Final altyazı render edilemedi.",
        cwd=WORK_DIRECTORY,
    )

    shutil.copy2(
        archive_file,
        LATEST_VIDEO_FILE,
    )

    shutil.rmtree(
        WORK_DIRECTORY,
        ignore_errors=True,
    )

    print()
    print(
        f"✓ Video süresi: "
        f"{audio_duration:.1f} saniye"
    )

    print(
        f"✓ latest.mp4 hazır: "
        f"{LATEST_VIDEO_FILE}"
    )

    return archive_file


def mark_event_used(
    event: dict[str, Any],
) -> None:
    """Tamamlanmış eventi tekrar seçilmemek üzere kaydeder."""

    payload = load_json(
        USED_EVENTS_FILE,
        default={
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

    key = clean_text(
        event.get(
            "event_key",
            "",
        )
    )

    already_exists = any(
        isinstance(
            item,
            dict,
        )
        and item.get(
            "event_key"
        )
        == key
        for item in events
    )

    if not already_exists:
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
                    ISTANBUL_TIMEZONE
                ).isoformat(),
            }
        )

    save_json(
        USED_EVENTS_FILE,
        payload,
    )


def upload_to_youtube() -> None:
    """Mevcut YouTube slot uploader'ını çalıştırır."""

    uploader = (
        PROJECT_DIRECTORY
        / "youtube_uploader.py"
    )

    if not uploader.exists():
        raise ComicFactoryError(
            "youtube_uploader.py bulunamadı."
        )

    if not LATEST_VIDEO_FILE.exists():
        raise ComicFactoryError(
            "YouTube yüklemesinden önce latest.mp4 bulunamadı."
        )

    print()
    print("=" * 78)
    print("YOUTUBE PLANLI YÜKLEME")
    print("=" * 78)
    print()

    process = subprocess.run(
        [
            sys.executable,
            str(
                uploader
            ),
        ],
        cwd=str(
            PROJECT_DIRECTORY
        ),
    )

    if process.returncode != 0:
        raise ComicFactoryError(
            "YouTube yükleme aşaması başarısız oldu."
        )


def main() -> None:
    """Comic Factory üretim hattını çalıştırır."""

    arguments = parse_arguments()

    ensure_directories()

    if arguments.check:
        run_check(
            arguments.upload
        )
        return

    print()
    print("=" * 78)
    print("COMIC FACTORY - FULL AUTOMATION")
    print("=" * 78)
    print()

    print(
        "Araştırma -> Comic görselleri -> AI panel seçimi -> "
        "Senaryo -> Ash -> Kelime senkronu -> Video"
    )

    print()

    client = create_client()

    try:
        if arguments.reuse_active:
            event = load_active_event()

            print(
                "✓ Mevcut active event yeniden kullanılıyor:"
            )

            print(
                clean_text(
                    event.get(
                        "event_title"
                    )
                )
            )

        else:
            events = research_events(
                client,
                arguments.count,
            )

            event = activate_event(
                events[
                    0
                ]
            )

        sources = discover_visual_pages(
            client,
            event,
        )

        candidates = collect_candidates(
            event,
            sources,
            max(
                8,
                arguments.max_images,
            ),
        )

        selected = assign_images_with_vision(
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

        narration = clean_text(
            script[
                "narration"
            ]
        )

        generate_voice(
            client,
            event,
            narration,
        )

        words = align_words(
            client,
            narration,
        )

        archive_file = render_video(
            event,
            assets,
            script,
            words,
            arguments.subtitle_offset,
        )

        if not LATEST_VIDEO_FILE.exists():
            raise ComicFactoryError(
                "Render bitti ancak data/videos/latest.mp4 oluşmadı."
            )

        mark_event_used(
            event
        )

        print()
        print("=" * 78)
        print("COMIC FACTORY BAŞARIYLA TAMAMLANDI")
        print("=" * 78)
        print()

        print(
            "Konu:"
        )

        print(
            clean_text(
                event[
                    "event_title"
                ]
            )
        )

        print()

        print(
            "Comic:"
        )

        print(
            f"{clean_text(event['series'])} "
            f"{clean_text(event['issue'])}"
        )

        print()

        print(
            "Başlık:"
        )

        print(
            clean_text(
                script[
                    "title"
                ]
            )
        )

        print()

        print(
            "Video:"
        )

        print(
            r"data\videos\latest.mp4"
        )

        print()

        print(
            "Arşiv:"
        )

        print(
            archive_file.relative_to(
                PROJECT_DIRECTORY
            )
        )

        print()

        if arguments.upload:
            upload_to_youtube()

        else:
            print(
                "YouTube'a yükleme yapılmadı."
            )

            print(
                "Yüklemek için: "
                "python comic_factory.py --upload"
            )

    except KeyboardInterrupt:
        print()
        print(
            "İşlem kullanıcı tarafından durduruldu."
        )

        raise SystemExit(
            1
        )

    except Exception as error:
        print()
        print("=" * 78)
        print("COMIC FACTORY DURDU")
        print("=" * 78)
        print()

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()

        print(
            "Eksik video YouTube'a gönderilmedi."
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
