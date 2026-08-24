# prototype_thor_galactus.py

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import edge_tts
import imageio_ffmpeg
import requests
from ddgs import DDGS
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
PROTOTYPE_DIR = DATA / "prototype" / "thor_galactus"
RAW_DIR = PROTOTYPE_DIR / "raw"
RESTORED_DIR = PROTOTYPE_DIR / "restored"
GENERATED_DIR = PROTOTYPE_DIR / "generated"
FRAME_DIR = PROTOTYPE_DIR / "frames"
SEGMENT_DIR = PROTOTYPE_DIR / "segments"
AUDIO_DIR = PROTOTYPE_DIR / "audio"

OUTPUT_VIDEO = DATA / "prototype" / "thor_galactus_v1.mp4"
MANIFEST_FILE = PROTOTYPE_DIR / "manifest.json"
STORYBOARD_FILE = PROTOTYPE_DIR / "storyboard.json"
WORDS_FILE = PROTOTYPE_DIR / "words.json"
ASS_FILE = PROTOTYPE_DIR / "subtitles.ass"

WIDTH = 1080
HEIGHT = 1920
FPS = 30
SCENE_COUNT = 12

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL",
    "gemini-3.1-flash-image",
)

GROQ_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo",
)

EDGE_VOICE = os.getenv(
    "PROTOTYPE_TTS_VOICE",
    "tr-TR-AhmetNeural",
)

EDGE_RATE = "+8%"
EDGE_PITCH = "+2Hz"

ENABLE_AI_RECONSTRUCTION = (
    os.getenv(
        "ENABLE_AI_RECONSTRUCTION",
        "0",
    ).strip()
    == "1"
)

MIN_REAL_MATCH_SCORE = 82
MIN_UNIQUE_VISUAL_RATIO = 0.72
MAX_DOWNLOAD_IMAGES = 28
MAX_VISION_IMAGES = 20
REQUEST_TIMEOUT = 25

SUBTITLE_WORDS = 3
SUBTITLE_Y = 1480

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0 Safari/537.36"
)

EVENT_CONTEXT = """
Comic: Thor #6 (2020)
Writer: Donny Cates
Artist: Nic Klein

Prototype konusu:
Thor'un Galactus ile Black Winter çatışmasının finalinde,
Galactus'un kozmik enerjisini emmesi, Galactus'u öldürmesi
ve kalan enerjiyi Black Winter'a karşı kullanması.

Amaç:
45-60 saniyelik Türkçe yüksek kaliteli dikey comic video.

Önemli:
Kaynaklardan desteklenmeyen ayrıntı uydurma.
Gerçek comic görseli varsa onu tercih et.
""".strip()


class PrototypeError(RuntimeError):
    """Prototype üretimi başarısız olduğunda oluşur."""


@dataclass
class VisualCandidate:
    candidate_id: str
    path: str
    source_url: str
    source_page: str
    width: int
    height: int
    sha256: str
    quality_score: float


@dataclass
class ScenePlan:
    scene_number: int
    narration: str
    visual_description: str
    hook_role: str
    motion: str
    selected_candidate_id: str
    relevance_score: int
    visual_source: str
    visual_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thor #6 yüksek kalite Shorts prototipi."
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="İndirilmiş görselleri de sıfırdan oluşturur.",
    )

    parser.add_argument(
        "--enable-ai-reconstruction",
        action="store_true",
        help=(
            "Eksik sahnelerde ücretli Gemini image "
            "reconstruction kullanımına izin verir."
        ),
    )

    return parser.parse_args()


def clean(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
    )


def require_env(name: str) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise PrototypeError(
            f"Eksik secret/environment variable: {name}"
        )

    return value


def ensure_directories() -> None:
    for directory in (
        PROTOTYPE_DIR,
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


def reset_working_directories() -> None:
    for directory in (
        RAW_DIR,
        RESTORED_DIR,
        GENERATED_DIR,
        FRAME_DIR,
        SEGMENT_DIR,
        AUDIO_DIR,
    ):
        shutil.rmtree(
            directory,
            ignore_errors=True,
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_json(
    path: Path,
    payload: Any,
) -> None:
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
    except json.JSONDecodeError as exc:
        raise PrototypeError(
            "Gemini geçerli JSON döndürmedi."
        ) from exc

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
        "Gemini beklenmeyen JSON yapısı döndürdü."
    )


def gemini_client() -> genai.Client:
    return genai.Client(
        api_key=require_env(
            "GEMINI_API_KEY"
        )
    )


def ask_json(
    client: genai.Client,
    prompt: str,
    temperature: float = 0.4,
) -> dict[str, Any]:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise PrototypeError(
            "Gemini boş yanıt verdi."
        )

    return parse_json_response(
        response.text
    )


def web_search() -> list[dict[str, str]]:
    queries = [
        '"Thor #6" 2020 Galactus Black Winter preview',
        '"Thor 6" Galactus Black Winter panels',
        '"Thor #6" Donny Cates Nic Klein review',
        '"Thor 2020 #6" review Galactus',
        '"Thor #6" Galactus dies Black Winter comic',
        '"Thor #6" preview Marvel',
    ]

    results: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for query in queries:
        print(
            f"Aranıyor: {query}"
        )

        try:
            found = DDGS(
                timeout=12
            ).text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=8,
            )
        except Exception as exc:
            print(
                f"Arama atlandı: {exc}"
            )
            continue

        for item in found or []:
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

            results.append(
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

    if not results:
        raise PrototypeError(
            "Thor #6 için web araştırması sonuç vermedi."
        )

    return results


def build_storyboard(
    client: genai.Client,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    evidence = "\n\n".join(
        (
            f"TITLE: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"SNIPPET: {item['body']}"
        )
        for item in sources[:35]
    )

    prompt = f"""
{EVENT_CONTEXT}

ARAŞTIRMA SONUÇLARI:
{evidence}

Bu tek olay için kaliteli bir Shorts/Reels storyboard hazırla.

Tam 12 sahne oluştur.

Senaryo yapısı:
1-2: çok güçlü hook
3-4: Galactus ve tehdidin context'i
5-8: gerilimin ve Thor'un gücünün yükselişi
9-11: Galactus / Black Winter payoff
12: güçlü kapanış + kısa izleyici sorusu

Kurallar:
- Toplam narration yaklaşık 120-145 Türkçe kelime.
- Her sahnenin narration'ı kısa ve doğal olsun.
- visual_description gerçek bir comic panelde aranabilecek kadar somut olsun.
- Aynı fikir iki sahnede tekrar edilmesin.
- Sahne numarasını narration içinde söyleme.
- Kaynaklarda olmayan ayrıntı uydurma.
- Güçlü ama clickbait yalanı olmayan anlatım.
- Videoda comic adı, issue etiketi veya kaynak etiketi görünmeyecek.
- motion yalnızca:
  slow_push
  slow_pull
  pan_left
  pan_right
  vertical_scan
  impact
  hold
  seçeneklerinden biri olsun.
- impact sadece gerçekten büyük reveal sahnelerinde kullanılsın.

JSON:
{{
  "title": "...",
  "description": "...",
  "narration": "...",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "...",
      "visual_description": "...",
      "hook_role": "hook/context/escalation/twist/payoff/cta",
      "motion": "slow_push"
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
            "Storyboard tam 12 sahne üretmedi."
        )

    payload[
        "sources"
    ] = sources

    save_json(
        STORYBOARD_FILE,
        payload,
    )

    return payload


def search_images(
    storyboard: dict[str, Any],
) -> list[dict[str, str]]:
    scene_terms = []

    for scene in storyboard[
        "scenes"
    ]:
        scene_terms.append(
            clean(
                scene[
                    "visual_description"
                ]
            )
        )

    queries = [
        '"Thor #6" 2020 comic panels',
        '"Thor 6" Galactus Black Winter comic page',
        '"Thor #6" Nic Klein panels',
        '"Thor #6" Black Winter pages',
    ]

    for term in scene_terms[:6]:
        queries.append(
            f'"Thor #6" {term[:75]}'
        )

    results: list[
        dict[str, str]
    ] = []

    seen: set[str] = set()

    for query in queries:
        try:
            found = DDGS(
                timeout=15
            ).images(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=12,
            )
        except Exception as exc:
            print(
                f"Görsel araması atlandı: {exc}"
            )
            continue

        for item in found or []:
            image_url = clean(
                item.get("image")
            )

            if (
                not image_url.startswith("http")
                or image_url in seen
            ):
                continue

            seen.add(
                image_url
            )

            results.append(
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

    return results


def image_hash(
    image: Image.Image,
) -> str:
    tiny = (
        image.convert(
            "L"
        )
        .resize(
            (8, 8),
            Image.Resampling.LANCZOS,
        )
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


def file_sha256(
    content: bytes,
) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def download_visuals(
    image_results: list[dict[str, str]],
) -> list[VisualCandidate]:
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    accepted: list[
        VisualCandidate
    ] = []

    perceptual_hashes: set[str] = set()

    for result in image_results:
        if len(
            accepted
        ) >= MAX_DOWNLOAD_IMAGES:
            break

        try:
            response = session.get(
                result[
                    "image_url"
                ],
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Referer": result[
                        "source_page"
                    ]
                },
            )

            response.raise_for_status()

        except requests.RequestException:
            continue

        content = response.content

        if len(
            content
        ) < 20_000:
            continue

        try:
            with Image.open(
                io.BytesIO(
                    content
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
            < 500_000
        ):
            continue

        phash = image_hash(
            image
        )

        if phash in perceptual_hashes:
            continue

        perceptual_hashes.add(
            phash
        )

        candidate_id = (
            f"real_{len(accepted) + 1:02d}"
        )

        output = (
            RAW_DIR
            / f"{candidate_id}.jpg"
        )

        image.save(
            output,
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
            / 75.0,
        )

        accepted.append(
            VisualCandidate(
                candidate_id=candidate_id,
                path=str(
                    output
                ),
                source_url=result[
                    "image_url"
                ],
                source_page=result[
                    "source_page"
                ],
                width=width,
                height=height,
                sha256=file_sha256(
                    content
                ),
                quality_score=round(
                    quality_score,
                    2,
                ),
            )
        )

        print(
            f"✓ {candidate_id}: "
            f"{width}x{height}"
        )

    if len(
        accepted
    ) < 6:
        raise PrototypeError(
            "Prototip için yeterli farklı görsel bulunamadı. "
            f"Bulunan: {len(accepted)}"
        )

    return accepted


def make_vision_thumbnail(
    path: Path,
) -> bytes:
    with Image.open(
        path
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

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


def assign_visuals(
    client: genai.Client,
    storyboard: dict[str, Any],
    candidates: list[VisualCandidate],
) -> list[dict[str, Any]]:
    candidates = candidates[
        :MAX_VISION_IMAGES
    ]

    scene_text = "\n".join(
        (
            f"{scene['scene_number']}. "
            f"NARRATION: {scene['narration']}\n"
            f"NEEDED VISUAL: {scene['visual_description']}"
        )
        for scene in storyboard[
            "scenes"
        ]
    )

    prompt = f"""
Sen yüksek kaliteli bir comic video görsel editörüsün.

EVENT:
Thor #6 (2020), Thor / Galactus / Black Winter.

12 SAHNE:
{scene_text}

Aşağıda gerçek web görselleri var.

Her sahne için:
- Anlatılan olayı gerçekten gösteren en iyi candidate_id'yi seç.
- relevance_score 0-100 ver.
- Sadece karakter aynı diye yüksek puan verme.
- Gerçek aksiyon/an gerçekten eşleşmeli.
- Bir görseli mümkün olduğunca yalnızca bir sahnede kullan.
- Aynı candidate zorunlu değilse tekrar edilmesin.
- Kapak görseli olay panelinden daha düşük değerlidir.

JSON:
{{
  "assignments": [
    {{
      "scene_number": 1,
      "candidate_id": "real_01",
      "relevance_score": 92,
      "reason": "..."
    }}
  ]
}}

Tam 12 assignment döndür.
"""

    contents: list[Any] = [
        prompt
    ]

    for candidate in candidates:
        contents.append(
            (
                f"CANDIDATE "
                f"{candidate.candidate_id}\n"
                f"SIZE: "
                f"{candidate.width}x{candidate.height}\n"
                f"QUALITY: "
                f"{candidate.quality_score}"
            )
        )

        contents.append(
            types.Part.from_bytes(
                data=make_vision_thumbnail(
                    Path(
                        candidate.path
                    )
                ),
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.15,
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

    if len(
        assignments
    ) != SCENE_COUNT:
        raise PrototypeError(
            "Vision matcher tam 12 assignment döndürmedi."
        )

    return assignments


def restore_comic_image(
    candidate: VisualCandidate,
) -> Path:
    source = Path(
        candidate.path
    )

    output = (
        RESTORED_DIR
        / f"{candidate.candidate_id}.png"
    )

    with Image.open(
        source
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

    max_side = max(
        image.size
    )

    if max_side < 2200:
        scale = min(
            2.0,
            2200
            / max_side,
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
        1.055
    )

    image = ImageEnhance.Color(
        image
    ).enhance(
        1.025
    )

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.4,
            percent=115,
            threshold=3,
        )
    )

    image.save(
        output,
        "PNG",
        optimize=True,
    )

    return output


def image_mime_type(
    path: Path,
) -> str:
    suffix = path.suffix.lower()

    if suffix == ".png":
        return "image/png"

    return "image/jpeg"


def generate_reconstruction(
    client: genai.Client,
    scene: dict[str, Any],
    references: list[Path],
) -> Path:
    output = (
        GENERATED_DIR
        / (
            f"scene_"
            f"{int(scene['scene_number']):02d}.png"
        )
    )

    prompt = f"""
Create a premium vertical comic-book illustration for a short-form video.

This is an artistic reconstruction of a specific moment from:
Thor #6 (2020), Thor, Galactus and the Black Winter.

SCENE:
{clean(scene["visual_description"])}

NARRATION CONTEXT:
{clean(scene["narration"])}

Use the supplied comic images only as visual references for:
- costume continuity
- character appearance
- color language
- cosmic atmosphere

Create the exact action described above.

Requirements:
- premium modern American comic-book illustration
- dynamic composition
- cinematic lighting
- extremely detailed
- clear readable silhouettes
- strong depth
- no captions
- no speech bubbles
- no logos
- no issue numbers
- no watermark
- no UI
- no text
- suitable for 9:16 vertical video
- do not copy a reference panel composition verbatim
""".strip()

    inputs: list[
        dict[str, Any]
    ] = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for reference in references[
        :3
    ]:
        inputs.append(
            {
                "type": "image",
                "data": base64.b64encode(
                    reference.read_bytes()
                ).decode(
                    "ascii"
                ),
                "mime_type": image_mime_type(
                    reference
                ),
            }
        )

    interaction = (
        client.interactions.create(
            model=GEMINI_IMAGE_MODEL,
            input=inputs,
            response_format={
                "type": "image",
                "mime_type": "image/png",
                "aspect_ratio": "9:16",
                "image_size": "1K",
            },
        )
    )

    if (
        interaction.output_image
        is None
    ):
        raise PrototypeError(
            "AI reconstruction görsel döndürmedi."
        )

    output.write_bytes(
        base64.b64decode(
            interaction.output_image.data
        )
    )

    return output


def choose_scene_visuals(
    client: genai.Client,
    storyboard: dict[str, Any],
    candidates: list[VisualCandidate],
    assignments: list[dict[str, Any]],
    allow_reconstruction: bool,
) -> list[ScenePlan]:
    candidate_map = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    restored_cache: dict[
        str,
        Path
    ] = {}

    scene_plans: list[
        ScenePlan
    ] = []

    for scene in storyboard[
        "scenes"
    ]:
        number = int(
            scene[
                "scene_number"
            ]
        )

        assignment = next(
            (
                item
                for item in assignments
                if int(
                    item.get(
                        "scene_number",
                        0,
                    )
                )
                == number
            ),
            None,
        )

        if assignment is None:
            raise PrototypeError(
                f"Sahne {number} assignment bulunamadı."
            )

        candidate_id = clean(
            assignment.get(
                "candidate_id"
            )
        )

        relevance = int(
            assignment.get(
                "relevance_score",
                0,
            )
            or 0
        )

        candidate = candidate_map.get(
            candidate_id
        )

        if candidate is None:
            raise PrototypeError(
                f"Geçersiz candidate: {candidate_id}"
            )

        if (
            candidate_id
            not in restored_cache
        ):
            restored_cache[
                candidate_id
            ] = restore_comic_image(
                candidate
            )

        visual_source = (
            "real_comic"
        )

        visual_file = restored_cache[
            candidate_id
        ]

        if (
            relevance
            < MIN_REAL_MATCH_SCORE
            and allow_reconstruction
        ):
            ranked_refs = sorted(
                candidates,
                key=lambda item: (
                    item.candidate_id
                    != candidate_id,
                    -item.quality_score,
                ),
            )

            reference_files: list[
                Path
            ] = []

            for reference in ranked_refs[
                :3
            ]:
                if (
                    reference.candidate_id
                    not in restored_cache
                ):
                    restored_cache[
                        reference.candidate_id
                    ] = restore_comic_image(
                        reference
                    )

                reference_files.append(
                    restored_cache[
                        reference.candidate_id
                    ]
                )

            try:
                visual_file = generate_reconstruction(
                    client,
                    scene,
                    reference_files,
                )

                visual_source = (
                    "ai_reconstruction"
                )

                print(
                    f"🎨 Sahne {number}: "
                    "AI reconstruction"
                )

            except Exception as exc:
                print(
                    f"! Reconstruction başarısız, "
                    f"gerçek panel kullanılıyor: {exc}"
                )

        scene_plans.append(
            ScenePlan(
                scene_number=number,
                narration=clean(
                    scene[
                        "narration"
                    ]
                ),
                visual_description=clean(
                    scene[
                        "visual_description"
                    ]
                ),
                hook_role=clean(
                    scene.get(
                        "hook_role"
                    )
                ),
                motion=clean(
                    scene.get(
                        "motion"
                    )
                )
                or "slow_push",
                selected_candidate_id=candidate_id,
                relevance_score=relevance,
                visual_source=visual_source,
                visual_file=str(
                    visual_file
                ),
            )
        )

    return scene_plans


def unique_visual_ratio(
    plans: list[ScenePlan],
) -> float:
    unique = {
        plan.visual_file
        for plan in plans
    }

    return (
        len(
            unique
        )
        / len(
            plans
        )
    )


def build_contact_sheet(
    plans: list[ScenePlan],
) -> Path:
    thumb_width = 300
    thumb_height = 440
    columns = 4
    rows = math.ceil(
        len(
            plans
        )
        / columns
    )

    sheet = Image.new(
        "RGB",
        (
            thumb_width
            * columns,
            thumb_height
            * rows,
        ),
        "black",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for index, plan in enumerate(
        plans
    ):
        with Image.open(
            plan.visual_file
        ) as opened:
            image = ImageOps.contain(
                opened.convert(
                    "RGB"
                ),
                (
                    thumb_width,
                    thumb_height
                    - 40,
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
            * thumb_width
        )

        y = (
            row
            * thumb_height
        )

        image_x = (
            x
            + (
                thumb_width
                - image.width
            )
            // 2
        )

        sheet.paste(
            image,
            (
                image_x,
                y,
            ),
        )

        draw.text(
            (
                x + 8,
                y + thumb_height - 34,
            ),
            (
                f"{plan.scene_number} | "
                f"{plan.relevance_score}"
            ),
            fill="white",
        )

    output = (
        PROTOTYPE_DIR
        / "contact_sheet.jpg"
    )

    sheet.save(
        output,
        "JPEG",
        quality=92,
    )

    return output


def director_critique(
    client: genai.Client,
    plans: list[ScenePlan],
) -> dict[str, Any]:
    sheet = build_contact_sheet(
        plans
    )

    plan_text = "\n".join(
        (
            f"SCENE {plan.scene_number}: "
            f"{plan.narration}\n"
            f"VISUAL: {plan.visual_description}\n"
            f"MATCH: {plan.relevance_score}\n"
            f"SOURCE: {plan.visual_source}\n"
        )
        for plan in plans
    )

    prompt = f"""
You are the final video director for a premium comic Shorts video.

Review the 12-scene contact sheet and plan.

{plan_text}

Judge:
- visual relevance to narration
- visual diversity
- excessive repeated images
- story escalation
- whether visual order feels cinematic
- whether any scene is obviously weak

Return JSON:
{{
  "visual_relevance": 0,
  "visual_diversity": 0,
  "story_flow": 0,
  "overall": 0,
  "weak_scenes": [1, 4],
  "notes": ["..."]
}}

Scores are 0-10.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=sheet.read_bytes(),
                mime_type="image/jpeg",
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    critique = parse_json_response(
        response.text
        or "{}"
    )

    critique[
        "unique_visual_ratio"
    ] = round(
        unique_visual_ratio(
            plans
        ),
        3,
    )

    save_json(
        PROTOTYPE_DIR
        / "director_critique.json",
        critique,
    )

    return critique


def generate_audio(
    narration: str,
) -> Path:
    output = (
        AUDIO_DIR
        / "narration.mp3"
    )

    async def run() -> None:
        speech = edge_tts.Communicate(
            narration,
            EDGE_VOICE,
            rate=EDGE_RATE,
            pitch=EDGE_PITCH,
        )

        await speech.save(
            str(
                output
            )
        )

    asyncio.run(
        run()
    )

    if (
        not output.exists()
        or output.stat().st_size
        < 10_000
    ):
        raise PrototypeError(
            "Edge TTS ses üretemedi."
        )

    return output


def groq_word_timestamps(
    audio: Path,
    narration: str,
) -> list[dict[str, Any]]:
    client = Groq(
        api_key=require_env(
            "GROQ_API_KEY"
        )
    )

    with audio.open(
        "rb"
    ) as file:
        transcription = (
            client.audio.transcriptions.create(
                file=file,
                model=GROQ_MODEL,
                language="tr",
                response_format="verbose_json",
                timestamp_granularities=[
                    "word"
                ],
                prompt=narration[:700],
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

    for item in raw_words or []:
        if isinstance(
            item,
            dict,
        ):
            word = clean(
                item.get(
                    "word"
                )
            )
            start = item.get(
                "start"
            )
            end = item.get(
                "end"
            )
        else:
            word = clean(
                getattr(
                    item,
                    "word",
                    "",
                )
            )
            start = getattr(
                item,
                "start",
                None,
            )
            end = getattr(
                item,
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
            "Groq yeterli kelime timestamp üretmedi."
        )

    save_json(
        WORDS_FILE,
        words,
    )

    return words


def get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_audio_duration(
    ffmpeg: str,
    audio: Path,
) -> float:
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(
                audio
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
) -> ImageFont.ImageFont:
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

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(
                str(
                    candidate
                ),
                size=size,
            )

    return ImageFont.load_default()


def compose_vertical(
    image_path: Path,
) -> Image.Image:
    with Image.open(
        image_path
    ) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert(
            "RGB"
        )

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
            radius=44
        )
    )

    background = ImageEnhance.Brightness(
        background
    ).enhance(
        0.34
    )

    background = ImageEnhance.Color(
        background
    ).enhance(
        0.78
    )

    foreground = image.copy()

    foreground.thumbnail(
        (
            WIDTH - 36,
            1460,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = background.convert(
        "RGBA"
    )

    x = (
        WIDTH
        - foreground.width
    ) // 2

    y = (
        150
        + (
            1430
            - foreground.height
        )
        // 2
    )

    shadow = Image.new(
        "RGBA",
        canvas.size,
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

    shadow_draw.rectangle(
        (
            x - 14,
            y + 18,
            x + foreground.width + 14,
            y + foreground.height + 34,
        ),
        fill=(
            0,
            0,
            0,
            145,
        ),
    )

    shadow = shadow.filter(
        ImageFilter.GaussianBlur(
            radius=24
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

    return canvas.convert(
        "RGB"
    )


def run_ffmpeg(
    command: list[str],
    error_message: str,
    cwd: Path | None = None,
) -> None:
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
        raise PrototypeError(
            error_message
            + "\n"
            + process.stderr[-3500:]
        )


def motion_filter(
    motion: str,
) -> str:
    common = (
        "scale=1080:1920,"
    )

    filters = {
        "slow_push": (
            "zoompan="
            "z='min(zoom+0.00011,1.035)':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
        "slow_pull": (
            "zoompan="
            "z='if(eq(on,0),1.035,max(1.0,zoom-0.00011))':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
        "pan_left": (
            "zoompan="
            "z='1.025':"
            "x='max(0,(iw-iw/zoom)*(1-on/220))':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
        "pan_right": (
            "zoompan="
            "z='1.025':"
            "x='min(iw-iw/zoom,(iw-iw/zoom)*(on/220))':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
        "vertical_scan": (
            "zoompan="
            "z='1.025':"
            "x='iw/2-(iw/zoom/2)':"
            "y='min(ih-ih/zoom,(ih-ih/zoom)*(on/220))':"
            "d=1:s=1080x1920:fps=30"
        ),
        "impact": (
            "zoompan="
            "z='min(zoom+0.00055,1.075)':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
        "hold": (
            "zoompan="
            "z='1.008':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        ),
    }

    selected = filters.get(
        motion,
        filters[
            "slow_push"
        ],
    )

    return (
        common
        + selected
        + ",format=yuv420p"
    )


def estimated_scene_durations(
    scenes: list[ScenePlan],
    words: list[dict[str, Any]],
    audio_duration: float,
) -> list[float]:
    counts = [
        max(
            1,
            len(
                re.findall(
                    r"\w+",
                    scene.narration,
                    re.UNICODE,
                )
            ),
        )
        for scene in scenes
    ]

    total = sum(
        counts
    )

    boundaries = [
        0.0
    ]

    cumulative = 0

    for count in counts[:-1]:
        cumulative += count

        ratio = (
            cumulative
            / total
        )

        word_index = min(
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
                    word_index
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
        SCENE_COUNT
    ):
        durations.append(
            max(
                0.7,
                boundaries[
                    index + 1
                ]
                - boundaries[
                    index
                ],
            )
        )

    durations[
        -1
    ] += (
        audio_duration
        - sum(
            durations
        )
    )

    return durations


def ass_time(
    seconds: float,
) -> str:
    centiseconds = round(
        max(
            0.0,
            seconds,
        )
        * 100
    )

    hours, rest = divmod(
        centiseconds,
        360000,
    )

    minutes, rest = divmod(
        rest,
        6000,
    )

    seconds_value, cs = divmod(
        rest,
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
    active: int,
) -> str:
    start = (
        active
        // SUBTITLE_WORDS
        * SUBTITLE_WORDS
    )

    end = min(
        len(
            words
        ),
        start
        + SUBTITLE_WORDS,
    )

    parts = []

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
        ).upper()

        if index == active:
            parts.append(
                r"{\1c&H0034D7FF&"
                r"\fscx116\fscy116\b1}"
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
) -> None:
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
            "Style: Main,DejaVu Sans,60,"
            "&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,"
            "-1,0,0,0,100,100,1,0,1,5,2,2,"
            "90,90,310,1"
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
            start
            + 0.04,
            float(
                word[
                    "end"
                ]
            ),
        )

        lines.append(
            "Dialogue: 0,"
            f"{ass_time(start)},"
            f"{ass_time(end)},"
            "Main,,0,0,0,,"
            f"{subtitle_chunk(words, index)}"
        )

    ASS_FILE.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def render_video(
    scenes: list[ScenePlan],
    audio: Path,
    words: list[dict[str, Any]],
) -> None:
    ffmpeg = get_ffmpeg()

    duration = get_audio_duration(
        ffmpeg,
        audio,
    )

    durations = estimated_scene_durations(
        scenes,
        words,
        duration,
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

    segments = []

    for scene, scene_duration in zip(
        scenes,
        durations,
        strict=True,
    ):
        frame_path = (
            FRAME_DIR
            / (
                f"scene_"
                f"{scene.scene_number:02d}.jpg"
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

        segment_path = (
            SEGMENT_DIR
            / (
                f"scene_"
                f"{scene.scene_number:02d}.mp4"
            )
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
                f"{scene_duration:.3f}",
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
                    segment_path
                ),
            ],
            f"Sahne {scene.scene_number} render edilemedi.",
        )

        segments.append(
            segment_path
        )

    concat_file = (
        PROTOTYPE_DIR
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

    silent = (
        PROTOTYPE_DIR
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
                silent
            ),
        ],
        "Segmentler birleştirilemedi.",
    )

    create_ass(
        words
    )

    OUTPUT_VIDEO.unlink(
        missing_ok=True
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
                audio
            ),
            "-vf",
            (
                f"ass="
                f"{ASS_FILE.resolve().as_posix()}"
            ),
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
        "Final prototip render edilemedi.",
    )


def main() -> int:
    args = parse_args()

    ensure_directories()

    if args.rebuild:
        reset_working_directories()

    allow_reconstruction = (
        ENABLE_AI_RECONSTRUCTION
        or args.enable_ai_reconstruction
    )

    print()
    print("=" * 78)
    print("THOR #6 - HIGH QUALITY AI VIDEO PROTOTYPE")
    print("=" * 78)
    print()

    print(
        "AI reconstruction: "
        + (
            "AÇIK"
            if allow_reconstruction
            else "KAPALI"
        )
    )

    print()

    client = gemini_client()

    print("[1/9] Thor #6 araştırılıyor...")
    sources = web_search()

    print("[2/9] 12 sahnelik storyboard hazırlanıyor...")
    storyboard = build_storyboard(
        client,
        sources,
    )

    print("[3/9] Çok sayıda comic görseli aranıyor...")
    image_results = search_images(
        storyboard
    )

    print("[4/9] Kaliteli ve farklı görseller indiriliyor...")
    candidates = download_visuals(
        image_results
    )

    print(
        f"✓ {len(candidates)} farklı kaliteli görsel."
    )

    print("[5/9] Gemini sahne-panel eşleştirmesi yapıyor...")
    assignments = assign_visuals(
        client,
        storyboard,
        candidates,
    )

    print("[6/9] Görseller restore ediliyor...")
    plans = choose_scene_visuals(
        client,
        storyboard,
        candidates,
        assignments,
        allow_reconstruction,
    )

    ratio = unique_visual_ratio(
        plans
    )

    print(
        f"✓ Unique visual ratio: "
        f"{ratio:.0%}"
    )

    print("[7/9] AI Director kalite kontrolü...")
    critique = director_critique(
        client,
        plans,
    )

    print(
        "Director overall: "
        f"{critique.get('overall', '?')}/10"
    )

    print(
        "Visual diversity: "
        f"{critique.get('visual_diversity', '?')}/10"
    )

    if ratio < MIN_UNIQUE_VISUAL_RATIO:
        print(
            "! Görsel çeşitlilik hedefin altında. "
            "Manifestte işaretlendi."
        )

    narration = clean(
        storyboard[
            "narration"
        ]
    )

    if not narration:
        narration = " ".join(
            scene.narration
            for scene in plans
        )

    print("[8/9] Ses ve gerçek kelime zamanlaması...")
    audio = generate_audio(
        narration
    )

    words = groq_word_timestamps(
        audio,
        narration,
    )

    print("[9/9] Premium prototype render...")
    render_video(
        plans,
        audio,
        words,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "event": "Thor #6 (2020) - Galactus / Black Winter",
        "ai_reconstruction_enabled": allow_reconstruction,
        "unique_visual_ratio": ratio,
        "director_critique": critique,
        "scenes": [
            asdict(
                scene
            )
            for scene in plans
        ],
        "output": str(
            OUTPUT_VIDEO
        ),
        "published": False,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 78)
    print("PROTOTİP HAZIR")
    print("=" * 78)
    print()

    print(
        f"Video: {OUTPUT_VIDEO}"
    )

    print()
    print(
        "YouTube: YÜKLENMEDİ"
    )

    print(
        "Instagram: YÜKLENMEDİ"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
