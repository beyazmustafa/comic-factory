from __future__ import annotations
from . import core
from .config import Settings
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import requests
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq, GroqError
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVENT_DIR = DATA / "events"
RESEARCH_DIR = DATA / "research"
SCRIPT_DIR = DATA / "scripts"
AUDIO_DIR = DATA / "audio"
AUDIO_DIAGNOSTICS_DIR = AUDIO_DIR / "alignment_diagnostics"
VIDEO_DIR = DATA / "videos"
CANDIDATE_DIR = DATA / "comic_candidates"
WORK_DIR = DATA / "video_work"
DIRECTOR_DIR = DATA / "director"
ASSET_DIR = ROOT / "assets" / "comic_pages"
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
SETTINGS = Settings()
SCENE_COUNT = SETTINGS.scene_count
TOPIC = ""
PANEL_DIR = None
TARGET_SECONDS = SETTINGS.target_seconds
BRIEF = SETTINGS.brief
API_CALLS = 0
STARTED_AT = time.monotonic()
STORY_THRESHOLD = SETTINGS.story_threshold
AUDIO_ALIGNMENT_THRESHOLD = SETTINGS.alignment_threshold
VISUAL_MATCH_THRESHOLD = SETTINGS.visual_threshold
IMAGE_QUALITY_THRESHOLD = SETTINGS.image_threshold
COMPOSITION_THRESHOLD = SETTINGS.composition_threshold
FINAL_VISUAL_THRESHOLD = SETTINGS.final_visual_threshold
STORY_REPAIR_CYCLES_PER_EVENT = SETTINGS.repair_attempts
AUDIO_REPAIR_CYCLES = SETTINGS.repair_attempts
VISUAL_REPAIR_CYCLES = SETTINGS.repair_attempts
AI_RECONSTRUCTION_REPAIR_CYCLES = SETTINGS.repair_attempts
FINAL_VISUAL_REPAIR_CYCLES = SETTINGS.repair_attempts
MAX_GLOBAL_IMAGES = SETTINGS.max_images
MAX_SCENE_IMAGES = 20
MAX_VISION_IMAGES = 28
load_dotenv(ROOT / ".env")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
GEMINI_TTS_VOICE = "Gacrux"
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
GROQ_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
REQUEST_TIMEOUT = 25
ALLOWED_MOTIONS = (
    "slow_push",
    "slow_pull",
    "pan_left",
    "pan_right",
    "vertical_scan",
    "impact",
    "hold",
)
ALLOWED_TRANSITIONS = ("cut", "soft_black", "dip_black", "flash_white")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/139.0 Safari/537.36"


class ComicFactoryError(RuntimeError):
    """Comic Factory üretim hatası."""


class EventRejectedError(ComicFactoryError):
    """Mevcut event kalite standardına ulaşamadığında oluşur."""


class AudioAlignmentError(ComicFactoryError):
    """Ses sorunu konunun reddedilmesine veya değiştirilmesine yol açmaz."""


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
            f"{icon} CHECKPOINT {result.name}: {result.score:.2f}/100 (min {result.threshold:.2f}) | cycle={result.cycle}"
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
                "results": [asdict(result) for result in self.results],
            },
        )

    def persist_lessons(self) -> None:
        payload = load_json(LESSONS_FILE, {"lessons": []})
        if not isinstance(payload, dict):
            payload = {"lessons": []}
        lessons = payload.setdefault("lessons", [])
        known = {
            (clean(item.get("checkpoint")), clean(item.get("details")))
            for item in lessons
            if isinstance(item, dict)
        }
        for result in self.results:
            if result.passed:
                continue
            key = (result.name, result.details)
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
        save_json(LESSONS_FILE, payload)


CHECKPOINTS = CheckpointManager()


def clean(value: Any) -> str:
    """Metni normalize eder."""
    return re.sub("\\s+", " ", str(value or "").strip())


def slug(value: str) -> str:
    """Güvenli dosya adı üretir."""
    value = clean(value).lower()
    replacements = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub("[^a-z0-9]+", "_", value)
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
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    """JSON dosyasını okur."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ComicFactoryError(f"JSON okunamadı: {path}") from error


def save_json(path: Path, payload: Any) -> None:
    """JSON dosyasını atomik kaydeder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def require_env(name: str) -> str:
    """Zorunlu environment variable değerini döndürür."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ComicFactoryError(f"{name} bulunamadı.")
    return value


def load_quality_lessons() -> str:
    """Önceki kalite hatalarını yeni promptlara bağlar."""
    payload = load_json(LESSONS_FILE, {"lessons": []})
    if not isinstance(payload, dict):
        return "Henüz kalite dersi yok."
    lessons = payload.get("lessons", [])
    if not isinstance(lessons, list):
        return "Henüz kalite dersi yok."
    recent = lessons[-30:]
    if not recent:
        return "Henüz kalite dersi yok."
    return "\n".join(
        (
            f"- {clean(item.get('checkpoint'))}: {clean(item.get('details'))}"
            for item in recent
            if isinstance(item, dict)
        )
    )


def gemini_client():
    return genai.Client(
        api_key=require_env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=120000),
    )


def parse_json_response(text: str) -> dict[str, Any]:
    """Model JSON cevabını ayrıştırır."""
    text = re.sub("^```(?:json)?\\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub("\\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ComicFactoryError("Gemini geçerli JSON döndürmedi.") from error
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if not payload:
            return {"items": []}
        first = payload[0]
        if isinstance(first, dict):
            if any((key in first for key in ("event_title", "publisher", "series"))):
                return {"events": payload}
            if "scene_number" in first:
                return {"scenes": payload}
        return {"items": payload}
    raise ComicFactoryError("Beklenmeyen Gemini JSON yapısı.")


def is_retryable_gemini_error(error: Exception) -> bool:
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
    return any((term in message for term in retryable_terms))


def gemini_with_retry(operation, *, operation_name, max_attempts=3):
    global API_CALLS
    for attempt in range(max_attempts):
        check_budget()
        API_CALLS += 1
        try:
            return operation()
        except Exception as error:
            if not is_retryable_gemini_error(error) or attempt + 1 == max_attempts:
                raise ComicFactoryError(
                    f"{operation_name}: {type(error).__name__}: {error}"
                ) from error
            delay = min(30, 4 * 2**attempt)
            print(
                f"{operation_name}: geçici hata; {delay} saniye sonra yeniden denenecek."
            )
            time.sleep(delay)


def ask_gemini_json(
    client: genai.Client, prompt: str, *, temperature: float, operation_name: str
) -> dict[str, Any]:
    """Gemini'den JSON cevap alır."""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, response_mime_type="application/json"
            ),
        )

    response = gemini_with_retry(request, operation_name=operation_name)
    if not response.text:
        raise ComicFactoryError(f"{operation_name} boş cevap verdi.")
    return parse_json_response(response.text)


def search_web(queries: list[str], each: int) -> list[dict[str, str]]:
    """DDGS ile web araştırması yapar."""
    results_out: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            results = DDGS(timeout=15).text(
                query, region="us-en", safesearch="moderate", max_results=each
            )
        except Exception as error:
            print(f"! Search skipped: {query}: {error}")
            continue
        for item in results or []:
            url = clean(item.get("href") or item.get("url"))
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            results_out.append(
                {
                    "title": clean(item.get("title")),
                    "url": url,
                    "body": clean(item.get("body")),
                }
            )
    return results_out


def event_key(event: dict[str, Any]) -> str:
    """Event benzersiz anahtarını oluşturur."""
    return "|".join(
        (
            clean(event.get(key)).casefold()
            for key in ("publisher", "series", "issue", "event_title")
        )
    )


def used_event_keys() -> set[str]:
    """Daha önce başarıyla üretilmiş eventleri döndürür."""
    payload = load_json(USED_EVENTS_FILE, {"events": []})
    if not isinstance(payload, dict):
        return set()
    return {
        clean(item.get("event_key"))
        for item in payload.get("events", [])
        if isinstance(item, dict) and clean(item.get("event_key"))
    }


def rejected_event_keys() -> set[str]:
    """Kalite nedeniyle daha önce reddedilmiş eventleri döndürür."""
    payload = load_json(REJECTED_EVENTS_FILE, {"events": []})
    if not isinstance(payload, dict):
        return set()
    return {
        clean(item.get("event_key"))
        for item in payload.get("events", [])
        if isinstance(item, dict)
        and clean(item.get("event_key"))
        # Legacy audio failures were incorrectly cached as rejected stories.
        # Keep the history, but allow these specific cases to be researched again.
        and not clean(item.get("reason")).startswith(
            (
                "Geçersiz veya sırasız ses zamanları.",
                "Bu eventin narration/audio kombinasyonu istenen alignment seviyesine ulaşamadı.",
            )
        )
    }


def record_rejected_event(event: dict[str, Any], reason: str) -> None:
    """Kaliteye ulaşamayan eventi kaydeder."""
    payload = load_json(REJECTED_EVENTS_FILE, {"events": []})
    if not isinstance(payload, dict):
        payload = {"events": []}
    events = payload.setdefault("events", [])
    events.append(
        {
            "event_key": event_key(event),
            "event_title": clean(event.get("event_title")),
            "series": clean(event.get("series")),
            "issue": clean(event.get("issue")),
            "reason": clean(reason),
            "rejected_at": datetime.now(TZ).isoformat(),
        }
    )
    payload["events"] = events[-100:]
    save_json(REJECTED_EVENTS_FILE, payload)


def research_events(client, count, run_rejected):
    check_budget()
    if TOPIC:
        queries = [
            f"{TOPIC} comics issue review",
            f"{TOPIC} comic panels",
            f"{TOPIC} comic story explained",
        ]
    else:
        queries = [
            "Marvel comics surprising obscure events specific issue panels",
            "DC comics shocking moments specific issue review panels",
            "popular superheroes little known comic story issue",
        ]
    web_results = search_web(queries, each=8)
    if not web_results:
        return []
    forbidden = used_event_keys() | rejected_event_keys() | run_rejected
    if TOPIC:
        forbidden = run_rejected
    evidence = json.dumps(web_results[:24], ensure_ascii=False)
    prompt = f"""Türkçe çizgi roman videosu için kaynaklı konu seç.\nKullanıcının istediği konu: {TOPIC or "Otomatik: popüler kahramanlar ve az bilinen ilginç olaylar"}\nEditoryal istek: {BRIEF}\nHedef: yaklaşık {TARGET_SECONDS} saniye, {SCENE_COUNT} farklı sahne.\nYalnız aşağıdaki arama kanıtlarında desteklenen spesifik olayları seç.\nKaynak bağlantısı, sayı, tarih veya olay uydurma. Yeterli kanıt yoksa events boş olsun.\nKaynakta görünmeyen detayları kesin bilgi gibi yazma. Birden fazla evreni karıştırma.\nHer aday {SCENE_COUNT} sahneye bölünebilmeli ve gerçek panel bulunabilmeli.\nİstenen konu verildiyse o konudan ayrılma.\nEn fazla {count} aday. Kullanılmayacak anahtarlar: {json.dumps(sorted(forbidden))}\nKANIT: {evidence}\nJSON şeması:\n{{"events":[{{"event_title":"","publisher":"","series":"","issue":"",\n"publication_year":2000,"characters":[],"hook":"","event_summary":"",\n"power_feat":"","why_interesting":"","quality_score":90,\n"popularity_score":80,"niche_score":80,"visual_moment_count":{SCENE_COUNT},\n"sources":[{{"name":"","url":"arama sonuçlarındaki URL","supports":""}}]}}]}}\nPuanlar editoryal tahmindir; kanıt değildir."""
    payload = ask_gemini_json(
        client, prompt, temperature=0.25, operation_name="Konu araştırması"
    )
    allowed_urls = {item["url"].rstrip("/") for item in web_results}
    eligible, seen = ([], set())
    raw_events = payload.get("events", [])
    for event in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(event, dict) or not all(
            (
                clean(event.get(key))
                for key in ("event_title", "series", "issue", "event_summary")
            )
        ):
            continue
        sources = [
            item
            for item in event.get("sources", [])
            if isinstance(item, dict)
            and clean(item.get("url")).rstrip("/") in allowed_urls
        ]
        if not sources:
            continue
        event["sources"] = sources
        key = event_key(event)
        if key in forbidden or key in seen:
            continue
        try:
            quality = max(0, min(100, float(event.get("quality_score", 0))))
            popularity = max(0, min(100, float(event.get("popularity_score", 0))))
            niche = max(0, min(100, float(event.get("niche_score", 0))))
        except (ValueError, TypeError):
            continue
        if quality < 80:
            continue
        event["selection_score"] = quality * 0.6 + popularity * 0.2 + niche * 0.2
        eligible.append(event)
        seen.add(key)
    eligible.sort(key=lambda event: event["selection_score"], reverse=True)
    save_json(
        RESEARCH_DIR / f"research_{datetime.now(TZ):%Y%m%d_%H%M%S}.json",
        {"topic": TOPIC, "evidence": web_results, "eligible": eligible},
    )
    return eligible[:count]


def activate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Eventi aktif pipeline eventine dönüştürür."""
    active = dict(event)
    active["id"] = slug(
        f"{clean(event.get('series'))}_{clean(event.get('issue'))}_{clean(event.get('event_title'))}"
    )
    active["event_key"] = event_key(active)
    active["activated_at"] = datetime.now(TZ).isoformat()
    save_json(ACTIVE_EVENT_FILE, active)
    CHECKPOINTS.set_event(active["id"])
    print()
    print("✓ EVENT: " + clean(active.get("event_title")))
    print(f"  {clean(active.get('series'))} {clean(active.get('issue'))}")
    return active


def load_active_event() -> dict[str, Any]:
    """Aktif eventi diskten yükler."""
    event = load_json(ACTIVE_EVENT_FILE)
    if not isinstance(event, dict) or not clean(event.get("id")):
        raise ComicFactoryError("active_event.json bulunamadı.")
    CHECKPOINTS.set_event(clean(event["id"]))
    return event


def generate_storyboard(
    client: genai.Client, event: dict[str, Any], feedback: str
) -> dict[str, Any]:
    """Tek story repair turu üretir."""
    sources = "\n".join(
        (
            f"- {clean(item.get('name'))}: {clean(item.get('supports'))} ({clean(item.get('url'))})"
            for item in event.get("sources", [])
            if isinstance(item, dict)
        )
    )
    prompt = f"""\nPremium Türkçe comic Shorts Story Director'sın.\nKULLANICININ ANLATIM İSTEĞİ: {BRIEF}\n\nEVENT:\n{clean(event.get("event_title"))}\n\nCOMIC:\n{clean(event.get("publisher"))}\n{clean(event.get("series"))}\n{clean(event.get("issue"))}\n{event.get("publication_year")}\n\nCHARACTERS:\n{", ".join(event.get("characters", []))}\n\nHOOK:\n{clean(event.get("hook"))}\n\nSUMMARY:\n{clean(event.get("event_summary"))}\n\nFEAT:\n{clean(event.get("power_feat"))}\n\nSOURCES:\n{sources}\n\nQUALITY LESSONS:\n{load_quality_lessons()}\n\nPREVIOUS REPAIR FEEDBACK:\n{feedback or "İlk story üretimi."}\n\nKRİTİK:\nTAM {SCENE_COUNT} SAHNE.\nBelirtilen sahne sayısına tam uy.\n\nscene_number:\n{list(range(1, SCENE_COUNT + 1))}\n\nStory:\n- {int(TARGET_SECONDS * 1.9)}-{int(TARGET_SECONDS * 2.45)} Türkçe kelime; hedef yaklaşık {TARGET_SECONDS} saniye\n- ilk iki sahne çok güçlü hook\n- her sahne yalnız bir ana görsel olayı anlatsın\n- her sahne yeni bilgi veya escalation getirsin\n- son iki sahne payoff\n- Wikipedia özeti gibi olmasın\n- bilgi uydurma\n- narration ile visual_description birebir aynı olayı anlatsın\n- o narration okunurken başka olayın paneli gerekmesin\n- {SCENE_COUNT} sahnenin görsel fikirleri farklı olsun\n- özel isimler orijinal yazılsın\n- Thor -> Tor gibi fonetik yazım yapma\n\nstory_role:\nhook\ncontext\nescalation\ntwist\npayoff\ncta\n\nSADECE JSON:\n{{\n  "title": "...",\n  "description": "...",\n  "hashtags": ["#comics", "#shorts"],\n  "scenes": [\n    {{\n      "scene_number": 1,\n      "narration": "...",\n      "visual_description": "...",\n      "story_role": "hook",\n      "emphasis_words": ["..."]\n    }}\n  ]\n}}\n"""
    storyboard = ask_gemini_json(
        client, prompt, temperature=0.42, operation_name="Story Director"
    )
    scenes = storyboard.get("scenes", [])
    if not isinstance(scenes, list):
        storyboard["_structure_valid"] = False
        storyboard["_structure_error"] = "scenes alanı liste değil."
        storyboard["_scene_count"] = 0
        return storyboard
    numbers: list[int] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            numbers.append(0)
            continue
        try:
            numbers.append(int(scene.get("scene_number", 0)))
        except (TypeError, ValueError):
            numbers.append(0)
    expected = list(range(1, SCENE_COUNT + 1))
    structure_valid = len(scenes) == SCENE_COUNT and numbers == expected
    storyboard["_structure_valid"] = structure_valid
    storyboard["_scene_count"] = len(scenes)
    if not structure_valid:
        storyboard["_structure_error"] = (
            f"TAM {SCENE_COUNT} sahne gerekli. Gelen={len(scenes)}. Scene numbers={numbers}."
        )
        return storyboard
    valid_roles = {"hook", "context", "escalation", "twist", "payoff", "cta"}
    narration_parts: list[str] = []
    for scene in scenes:
        narration = clean(scene.get("narration"))
        visual = clean(scene.get("visual_description"))
        role = clean(scene.get("story_role"))
        if not narration or not visual or role not in valid_roles:
            storyboard["_structure_valid"] = False
            storyboard["_structure_error"] = (
                "Bir sahnede narration, visual_description veya geçerli story_role eksik."
            )
            return storyboard
        narration_parts.append(narration)
    storyboard["narration"] = " ".join(narration_parts)
    return storyboard


def score_storyboard(
    client: genai.Client, event: dict[str, Any], storyboard: dict[str, Any]
) -> dict[str, Any]:
    """Storyboard kalitesini sert eşiklerle değerlendirir."""
    prompt = f"""\nPremium Shorts Supervising Story Director'sın.\n\nEVENT:\n{clean(event.get("event_title"))}\n\nSTORYBOARD:\n{json.dumps(storyboard.get("scenes", []), ensure_ascii=False, indent=2)}\n\n0-100 puanla:\nhook\nclarity\nescalation\npayoff\nfactual_discipline\nvisual_storytelling\nnarration_visual_match\nretention\noverall\n\n{STORY_THRESHOLD}+ overall:\nyalnız gerçekten yayınlanabilir premium story.\n\nnarration_visual_match:\nBir narration sırasında hangi comic panelinin\ngösterileceği açık ve tek anlamlı olmalı.\n\nBir sahne iki ayrı büyük olayı anlatıyorsa puan kır.\nAynı bilgi veya aynı görsel fikir tekrar ediyorsa puan kır.\nStory yalnız bilgi listesi gibi ilerliyorsa ciddi puan kır.\nFactual karışıklık varsa ciddi puan kır.\nFinalde gerçek payoff yoksa yüksek puan verme.\n\nSADECE JSON:\n{{\n  "hook": 0,\n  "clarity": 0,\n  "escalation": 0,\n  "payoff": 0,\n  "factual_discipline": 0,\n  "visual_storytelling": 0,\n  "narration_visual_match": 0,\n  "retention": 0,\n  "overall": 0,\n  "problems": ["..."],\n  "revision_instruction": "..."\n}}\n"""
    result = ask_gemini_json(
        client, prompt, temperature=0.08, operation_name="Story Quality Check"
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
            score = float(result.get(key, 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        result[key] = max(0.0, min(100.0, score))
    return result


def build_quality_storyboard(
    client: genai.Client, event: dict[str, Any]
) -> dict[str, Any]:
    """Story yeterli değilse eventi reddedip üst pipeline'a döndürür."""
    print()
    print("=" * 78)
    print("STORY QUALITY GATE")
    print("=" * 78)
    feedback = ""
    best_score = 0.0
    best_problems = ""
    for cycle in range(1, STORY_REPAIR_CYCLES_PER_EVENT + 1):
        print(f"\n→ Story repair cycle {cycle}/{STORY_REPAIR_CYCLES_PER_EVENT}")
        storyboard = generate_storyboard(client, event, feedback)
        if not bool(storyboard.get("_structure_valid", False)):
            error = clean(storyboard.get("_structure_error"))
            CHECKPOINTS.record(
                name="story_structure",
                score=0.0,
                threshold=100.0,
                cycle=cycle,
                details=error,
                passed=False,
            )
            feedback = f"Önceki cevap yapısal olarak hatalıydı. {error} TAM {SCENE_COUNT} sahne üret. Scene sayılarını düzeltirken story kalitesini düşürme."
            continue
        CHECKPOINTS.record(
            name="story_structure",
            score=100.0,
            threshold=100.0,
            cycle=cycle,
            details=f"{SCENE_COUNT}/{SCENE_COUNT} sahne.",
            passed=True,
        )
        review = score_storyboard(client, event, storyboard)
        score = float(review.get("overall", 0) or 0)
        problems = "; ".join(
            (clean(item) for item in review.get("problems", []) if clean(item))
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
            storyboard.pop("_structure_valid", None)
            storyboard.pop("_structure_error", None)
            storyboard.pop("_scene_count", None)
            save_json(SCRIPT_DIR / "storyboard.json", storyboard)
            return storyboard
        feedback = f"Önceki overall={score:.1f}/100. Minimum={STORY_THRESHOLD:.1f}. Problems: {problems}. Repair instruction: {clean(review.get('revision_instruction'))}. TAM {SCENE_COUNT} sahneyi koru. İyi sahneleri bozma. Düşük puanlı story, factual ve narration-visual bölümlerini düzelt."
    raise EventRejectedError(
        f"Story bu event için yeterince güçlü değil. Best={best_score:.1f}/100. {best_problems}"
    )


def create_voice_direction(client: genai.Client, narration: str) -> dict[str, Any]:
    """Kilitli Gacrux voice delivery talimatı üretir."""
    prompt = f"""\nTürkçe premium YouTube Shorts Voice Director'sın.\n\nTRANSCRIPT:\n{narration}\n\nKELİMELERİ DEĞİŞTİRME.\n\nKilitli voice karakteri:\n- doğal erkek storyteller\n- arkadaşına inanılmaz comic hikayesi anlatır gibi\n- yapay TikTok sesi değil\n- haber spikeri değil\n- ağır belgesel değil\n- enerjik ama bağırmayan\n- İngilizce özel isimleri doğal İngilizce telaffuz et\n- sonra doğal Türkçeye dön\n- cümle sonlarını aynı melodiyle bitirme\n- reveal öncesi kısa doğal pause\n- doğal nefes ve konuşma ritmi\n- yaklaşık 1.0x tempo\n\nSADECE JSON:\n{{\n  "style_instruction": "...",\n  "pronunciation_instruction": "...",\n  "pace_instruction": "..."\n}}\n"""
    return ask_gemini_json(
        client, prompt, temperature=0.18, operation_name="Locked Voice Director"
    )


def write_pcm_wave(path: Path, pcm: bytes) -> None:
    """24kHz mono PCM sesini WAV dosyasına yazar."""
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(pcm)


def generate_gacrux_voice(client: genai.Client, narration: str, cycle: int) -> Path:
    """Kilitli Gacrux sesi üretir."""
    direction = create_voice_direction(client, narration)
    prompt = f"\nRead ONLY the transcript inside <TRANSCRIPT>.\n\nVOICE:\n{clean(direction.get('style_instruction'))}\n\nPRONUNCIATION:\n{clean(direction.get('pronunciation_instruction'))}\n\nPACE:\n{clean(direction.get('pace_instruction'))}\n\nCritical:\nThe surrounding language is Turkish.\nEnglish proper nouns must sound naturally English.\nAfter the proper noun, return naturally to Turkish.\n\nDo not:\nadd words\nremove words\nparaphrase\nread instructions\nsound synthetic\nsound like an announcer\n\n<TRANSCRIPT>\n{narration}\n</TRANSCRIPT>\n"

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=GEMINI_TTS_VOICE
                        )
                    )
                ),
            ),
        )

    response = gemini_with_retry(request, operation_name="Locked Gacrux TTS")
    pcm = None
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                pcm = inline.data
                break
        if pcm:
            break
    if not pcm:
        raise ComicFactoryError("Gemini Gacrux boş audio döndürdü.")
    output = AUDIO_DIR / f"gacrux_cycle_{cycle:02d}.wav"
    if isinstance(pcm, str):
        pcm = base64.b64decode(pcm)
    if pcm[:4] == b"RIFF":
        output.write_bytes(pcm)
    else:
        write_pcm_wave(output, pcm)
    return output


def object_value(value: Any, key: str, default: Any = None) -> Any:
    """SDK nesnesi veya dict alanını okur."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def transcribe_words(
    audio_path: Path,
    narration: str,
    *,
    model: str | None = None,
    diagnostics_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Preserve the unmodified ASR response before validating word times.

    Do not feed the expected script back as a prompt: this is a check of the
    actual recording, including words that TTS may have omitted or changed.
    """
    model = model or GROQ_MODEL
    with (
        Groq(api_key=require_env("GROQ_API_KEY"), timeout=120, max_retries=2) as client,
        audio_path.open("rb") as audio,
    ):
        transcription = client.audio.transcriptions.create(
            file=audio,
            model=model,
            language="tr",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0,
        )
    if diagnostics_path is not None:
        response = (
            transcription.model_dump(mode="json")
            if hasattr(transcription, "model_dump")
            else transcription
        )
        save_json(
            diagnostics_path,
            {
                "provider": "groq",
                "model": model,
                "expected_narration": narration,
                "response": response,
            },
        )
    words: list[dict[str, Any]] = []
    for item in object_value(transcription, "words", []) or []:
        word = clean(object_value(item, "word", ""))
        start = object_value(item, "start")
        end = object_value(item, "end")
        if word:
            words.append({"word": word, "start": start, "end": end})
    if not words:
        raise AudioAlignmentError("Groq kelime zamanları döndürmedi.")
    return words


def align_narration_to_audio(narration, whisper_words, audio_duration=None):
    try:
        return core.align_words(narration, whisper_words, audio_duration)
    except ValueError as error:
        raise AudioAlignmentError(str(error)) from error


def build_quality_audio(
    client: genai.Client, narration: str
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-transcribe the same audio before spending another TTS attempt."""
    print()
    print("=" * 78)
    print("AUDIO ALIGNMENT GATE")
    print("=" * 78)
    print("Ses işleme sürümü: 2026-09-12-audio-1")
    best_score = 0.0
    models = list(dict.fromkeys([GROQ_MODEL, "whisper-large-v3"]))
    diagnostics = AUDIO_DIAGNOSTICS_DIR / slug(
        CHECKPOINTS.current_event_id or "narration"
    )
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "narration.txt").write_text(narration, encoding="utf-8")
    attempts = []
    last_error = ""
    recoverable = (
        ComicFactoryError,
        GroqError,
        ValueError,
        TypeError,
        KeyError,
        OSError,
    )
    for cycle in range(1, AUDIO_REPAIR_CYCLES + 1):
        check_budget()
        attempt_dir = diagnostics / f"cycle_{cycle:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        print(f"→ Ses denemesi {cycle}/{AUDIO_REPAIR_CYCLES}")
        try:
            audio = generate_gacrux_voice(client, narration, cycle)
            shutil.copy2(audio, attempt_dir / "narration.wav")
            duration = media_duration(ffmpeg_path(), audio)
        except recoverable as error:
            last_error = f"{type(error).__name__}: {error}"
            attempts.append(
                {"cycle": cycle, "stage": "tts", "status": "error", "error": last_error}
            )
            save_json(diagnostics / "attempts.json", attempts)
            CHECKPOINTS.record(
                name="audio_generation",
                score=0,
                threshold=100,
                cycle=cycle,
                details=last_error,
            )
            continue
        for index, model in enumerate(models, 1):
            check_budget()
            if index > 1:
                print(f"→ Aynı ses yeniden çözümleniyor: {model}")
            attempt = {
                "cycle": cycle,
                "stage": "alignment",
                "model": model,
                "duration": duration,
            }
            prefix = f"asr_{index:02d}_{slug(model)}"
            score = 0.0
            try:
                whisper_words = transcribe_words(
                    audio,
                    narration,
                    model=model,
                    diagnostics_path=attempt_dir / f"{prefix}_raw.json",
                )
                aligned_words, metrics = align_narration_to_audio(
                    narration, whisper_words, duration
                )
                save_json(
                    attempt_dir / f"{prefix}_aligned.json",
                    {"metrics": metrics, "words": aligned_words},
                )
                score = float(metrics["score"])
                best_score = max(best_score, score)
                details = (
                    f"model={model} | coverage={metrics['coverage']:.2f}% | "
                    f"similarity={metrics['similarity']:.2f}% | timing={metrics['timing_coverage']:.2f}%"
                )
                attempt.update(
                    status="passed"
                    if score >= AUDIO_ALIGNMENT_THRESHOLD
                    else "below_threshold",
                    metrics=metrics,
                )
                last_error = details
            except recoverable as error:
                score = 0.0
                last_error = f"{type(error).__name__}: {error}"
                details = f"model={model} | {last_error}"
                attempt.update(status="error", error=last_error)
            attempts.append(attempt)
            save_json(diagnostics / "attempts.json", attempts)
            passed = CHECKPOINTS.record(
                name="audio_alignment",
                score=score,
                threshold=AUDIO_ALIGNMENT_THRESHOLD,
                cycle=cycle,
                details=details,
            )
            if not passed:
                continue
            shutil.copy2(audio, LATEST_AUDIO_FILE)
            save_json(
                LATEST_WORDS_FILE,
                {"provider": "groq", "model": model, "words": whisper_words},
            )
            save_json(ALIGNED_WORDS_FILE, {"metrics": metrics, "words": aligned_words})
            return (LATEST_AUDIO_FILE, whisper_words, aligned_words)
    raise AudioAlignmentError(
        f"Ses hizalaması {AUDIO_REPAIR_CYCLES} ses denemesinde tamamlanamadı. "
        f"En iyi puan={best_score:.2f}; gereken={AUDIO_ALIGNMENT_THRESHOLD:.2f}. "
        f"Konu elenmedi. Ses ve ham kelime zamanları audio_diagnostics klasöründe. Son durum: {last_error}"
    )


def ffmpeg_path():
    return core.ffmpeg_binary()


def run_ffmpeg(command, error_message):
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if process.returncode:
        raise ComicFactoryError(error_message + "\n" + process.stderr[-4000:])


def media_duration(ffmpeg, media):
    return float(core.inspect_media(media)["format"]["duration"])


def build_scene_timeline(storyboard, aligned_words, audio_duration_seconds):
    try:
        return core.scene_timeline(
            storyboard["scenes"], aligned_words, audio_duration_seconds, FPS
        )
    except ValueError as error:
        raise EventRejectedError(str(error)) from error


def search_image_queries(queries: list[str], each: int) -> list[dict[str, str]]:
    """Comic image aramalarını normalize eder."""
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            results = DDGS(timeout=15).images(
                query, region="us-en", safesearch="moderate", max_results=each
            )
        except Exception as error:
            print(f"! Image search skipped: {error}")
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
    return output


def global_image_search(
    event: dict[str, Any], storyboard: dict[str, Any]
) -> list[dict[str, str]]:
    """Event için geniş gerçek comic görsel havuzu arar."""
    series = clean(event.get("series"))
    issue = clean(event.get("issue"))
    title = clean(event.get("event_title"))
    queries = [
        f'"{series}" "{issue}" comic panels',
        f'"{series}" "{issue}" comic pages',
        f'"{series}" "{issue}" preview images',
        f'"{series}" "{issue}" review panels',
        f'"{title}" comic panels',
        f'"{title}" comic page',
    ]
    for scene in storyboard["scenes"]:
        visual = clean(scene.get("visual_description"))
        queries.append(f'"{series}" "{issue}" {visual[:90]}')
    return search_image_queries(queries, each=12)


def average_hash(image: Image.Image) -> str:
    """64-bit average hash oluşturur."""
    tiny = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(tiny.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join(("1" if pixel >= average else "0" for pixel in pixels))
    return f"{int(bits, 2):016x}"


def hash_distance(first: str, second: str) -> int:
    """İki perceptual hash arasındaki mesafeyi hesaplar."""
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def source_image_quality(image: Image.Image) -> float:
    """Kaynak görsel teknik kalite skorunu hesaplar."""
    width, height = image.size
    megapixels = width * height / 1000000
    minimum_side = min(width, height)
    entropy = image.convert("L").entropy()
    resolution_score = min(50.0, megapixels / 1.4 * 50.0)
    dimension_score = min(30.0, minimum_side / 900.0 * 30.0)
    entropy_score = min(20.0, max(0.0, (entropy - 3.0) / 4.5 * 20.0))
    return max(0.0, min(100.0, resolution_score + dimension_score + entropy_score))


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
    session.headers.update({"User-Agent": USER_AGENT})
    directory = CANDIDATE_DIR / event["id"] / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    hashes = list(known_hashes or [])
    output: list[ImageCandidate] = []
    for item in results[: max_candidates * 4]:
        check_budget()
        if len(output) >= max_candidates:
            break
        try:
            response = session.get(
                item["image_url"],
                headers={"Referer": item["source_page"]},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue
        if not 18000 <= len(response.content) <= 15000000:
            continue
        try:
            with Image.open(io.BytesIO(response.content)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception:
            continue
        width, height = image.size
        if width < 500 or height < 500 or width * height < 450000:
            continue
        phash = average_hash(image)
        if any((hash_distance(phash, existing) <= 5 for existing in hashes)):
            continue
        hashes.append(phash)
        candidate_id = f"c{start_index + len(output):04d}"
        local_file = directory / f"{candidate_id}.jpg"
        image.save(local_file, "JPEG", quality=96, optimize=True)
        output.append(
            ImageCandidate(
                candidate_id=candidate_id,
                local_file=str(local_file),
                source_page=item["source_page"],
                image_url=item["image_url"],
                title=item["title"],
                width=width,
                height=height,
                quality_score=round(source_image_quality(image), 2),
                perceptual_hash=phash,
            )
        )
    save_json(directory / "candidates.json", [asdict(item) for item in output])
    return output


def thumbnail_bytes(path: Path) -> bytes:
    """Gemini Vision için thumbnail üretir."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return buffer.getvalue()


def normalize_crop_box(crop_box: Any) -> list[float]:
    """Crop koordinatlarını normalize eder."""
    if not isinstance(crop_box, list) or len(crop_box) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    try:
        left, top, right, bottom = [float(value) for value in crop_box]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    left = max(0.0, min(0.9, left))
    top = max(0.0, min(0.9, top))
    right = max(left + 0.1, min(1.0, right))
    bottom = max(top + 0.1, min(1.0, bottom))
    return [left, top, right, bottom]


def rank_scene_candidates(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    candidates: list[ImageCandidate],
) -> list[RankedVisual]:
    """Bir sahne için gerçek panelleri relevance'a göre sıralar."""
    candidates = sorted(
        candidates, key=lambda candidate: candidate.quality_score, reverse=True
    )[:MAX_VISION_IMAGES]
    if not candidates:
        return []
    prompt = f"""\nPremium comic Visual Director'sın.\n\nEVENT:\n{clean(event.get("event_title"))}\n\nCOMIC:\n{clean(event.get("series"))}\n{clean(event.get("issue"))}\n\nNARRATION:\n{clean(scene_data.get("narration"))}\n\nO SIRADA EKRANDA GÖRÜLMESİ GEREKEN:\n{clean(scene_data.get("visual_description"))}\n\nAday görselleri tek tek incele.\n\nEn iyi 6 adayı sırala.\n\nrelevance_score:\n100 = tam anlatılan olay\n95 = aynı spesifik aksiyon açıkça görülüyor\n85 = doğru karakterler ama yanlış/eksik an\n70 = yalnız konu benziyor\n50 = zayıf\n0 = alakasız\n\n95+ kolay verme.\n\nCrop:\nDoğru comic panelini seç.\nAna karakter/aksiyon crop içinde kalmalı.\n\nSADECE JSON:\n{{\n  "ranked_visuals": [\n    {{\n      "candidate_id": "c0001",\n      "relevance_score": 97,\n      "crop_box": [0,0,1,1],\n      "reason": "..."\n    }}\n  ]\n}}\n"""
    contents: list[Any] = [prompt]
    for candidate in candidates:
        contents.append(
            f"CANDIDATE {candidate.candidate_id}\nTITLE: {candidate.title}\nSIZE: {candidate.width}x{candidate.height}\nQUALITY: {candidate.quality_score:.1f}"
        )
        contents.append(
            types.Part.from_bytes(
                data=thumbnail_bytes(Path(candidate.local_file)), mime_type="image/jpeg"
            )
        )

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.04, response_mime_type="application/json"
            ),
        )

    response = gemini_with_retry(
        request, operation_name=f"Scene Visual Ranking {scene_data['scene_number']}"
    )
    payload = parse_json_response(response.text or "{}")
    raw_items = payload.get("ranked_visuals", payload.get("items", []))
    ranked: list[RankedVisual] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        candidate_id = clean(item.get("candidate_id"))
        if not candidate_id:
            continue
        ranked.append(
            RankedVisual(
                candidate_id=candidate_id,
                relevance_score=int(item.get("relevance_score", 0) or 0),
                crop_box=normalize_crop_box(item.get("crop_box")),
                reason=clean(item.get("reason")),
            )
        )
    return ranked


def restore_crop(
    candidate: ImageCandidate, crop_box: list[float], output: Path
) -> tuple[Path, float]:
    """Comic panelini değiştirmeden crop/restoration uygular."""
    with Image.open(candidate.local_file) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    left, top, right, bottom = crop_box
    x1 = round(left * image.width)
    y1 = round(top * image.height)
    x2 = round(right * image.width)
    y2 = round(bottom * image.height)
    crop = image.crop((x1, y1, x2, y2))
    original_width = max(1, x2 - x1)
    original_height = max(1, y2 - y1)
    megapixels = original_width * original_height / 1000000
    minimum_side = min(original_width, original_height)
    entropy = crop.convert("L").entropy()
    technical_quality = (
        min(50.0, megapixels / 0.8 * 50.0)
        + min(30.0, minimum_side / 650.0 * 30.0)
        + min(20.0, max(0.0, (entropy - 3.0) / 4.5 * 20.0))
    )
    technical_quality = max(0.0, min(100.0, technical_quality))
    maximum_side = max(crop.size)
    if maximum_side < 2600:
        scale = min(2.4, 2600 / max(1, maximum_side))
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )

    crop = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=105, threshold=3))
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output, "PNG", optimize=True)
    return (output, technical_quality)


def evaluate_visual(
    client: genai.Client,
    scene_data: dict[str, Any],
    visual_file: Path,
    technical_quality: float,
) -> dict[str, Any]:
    """Final sahne panelini narration ve kalite açısından inceler."""
    prompt = f"""\nPremium comic Shorts Scene QC.\n\nNARRATION:\n{clean(scene_data.get("narration"))}\n\nEXPECTED VISUAL:\n{clean(scene_data.get("visual_description"))}\n\nFinal görseli değerlendir.\n\nvisual_match 0-100:\nNarration'daki spesifik olay gerçekten ekranda mı?\n\nimage_quality 0-100:\nNetlik, çözünürlük hissi, bozulma, artefact.\n\ncomposition 0-100:\nAna olay okunuyor mu?\nKarakter/aksiyon crop dışında mı?\n9:16 videoda kullanışlı mı?\n\nSOURCE TECHNICAL QUALITY:\n{technical_quality:.1f}\n\n95+ visual_match yalnız tam olay görünüyorsa ver.\n\nSADECE JSON:\n{{\n  "visual_match": 0,\n  "image_quality": 0,\n  "composition": 0,\n  "problems": ["..."],\n  "repair_instruction": "..."\n}}\n"""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=thumbnail_bytes(visual_file), mime_type="image/jpeg"
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.03, response_mime_type="application/json"
            ),
        )

    response = gemini_with_retry(
        request, operation_name=f"Scene Visual QC {scene_data['scene_number']}"
    )
    return parse_json_response(response.text or "{}")


def scene_specific_search(
    event: dict[str, Any], scene_data: dict[str, Any], feedback: str
) -> list[dict[str, str]]:
    """Başarısız sahne için daha spesifik panel araştırması yapar."""
    series = clean(event.get("series"))
    issue = clean(event.get("issue"))
    narration = clean(scene_data.get("narration"))
    visual = clean(scene_data.get("visual_description"))
    queries = [
        f'"{series}" "{issue}" {visual[:100]}',
        f'"{series}" "{issue}" {narration[:95]} panel',
        f'"{series}" "{issue}" comic page preview',
        f'"{series}" "{issue}" review scan',
    ]
    if feedback:
        queries.append(f'"{series}" "{issue}" {clean(feedback)[:90]}')
    return search_image_queries(queries, each=10)


def generate_reconstruction(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    references: list[Path],
    output: Path,
    feedback: str,
) -> Path:
    """Gerçek panel bulunamadığında referanslı AI reconstruction üretir."""
    prompt = f"\nCreate a premium vertical American comic-book illustration.\n\nEVENT:\n{clean(event.get('event_title'))}\n\nCOMIC:\n{clean(event.get('series'))}\n{clean(event.get('issue'))}\n\nEXACT NARRATED MOMENT:\n{clean(scene_data.get('visual_description'))}\n\nNARRATION:\n{clean(scene_data.get('narration'))}\n\nPREVIOUS QC FEEDBACK:\n{feedback or 'First reconstruction.'}\n\nUse references only for:\ncharacter appearance\ncostume\nera\nsetting\ncolor language\n\nCreate the exact missing event.\n\nRequirements:\npremium comic artwork\nhigh detail\ncinematic depth\ncorrect anatomy\nclear action\n9:16 composition\nno text\nno speech bubble\nno watermark\nno logo\nno issue number\nno UI\n\nCorrect every problem from PREVIOUS QC FEEDBACK.\nDo not copy the reference composition exactly.\n"
    inputs: list[dict[str, str]] = [{"type": "text", "text": prompt}]
    for reference in references[:3]:
        mime_type = "image/png" if reference.suffix.lower() == ".png" else "image/jpeg"
        inputs.append(
            {
                "type": "image",
                "data": base64.b64encode(reference.read_bytes()).decode("ascii"),
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
        request, operation_name=f"AI Reconstruction Scene {scene_data['scene_number']}"
    )
    if interaction.output_image is None:
        raise ComicFactoryError("AI reconstruction görsel döndürmedi.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(interaction.output_image.data))
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
        scene_number=int(scene_data["scene_number"]),
        narration=clean(scene_data.get("narration")),
        visual_description=clean(scene_data.get("visual_description")),
        story_role=clean(scene_data.get("story_role")),
        emphasis_words=[
            clean(word) for word in scene_data.get("emphasis_words", []) if clean(word)
        ],
        ranked_visuals=rankings,
        selected_candidate_id=selected_candidate_id,
        relevance_score=relevance_score,
        crop_box=crop_box,
        visual_source=visual_source,
        visual_file=str(visual_file),
        visual_match_score=visual_match,
        image_quality_score=image_quality,
        composition_score=composition,
    )


def resolve_scene_until_pass(
    client: genai.Client,
    event: dict[str, Any],
    scene_data: dict[str, Any],
    global_candidates: list[ImageCandidate],
    used_ids: set[str],
    asset_directory: Path,
    enable_ai_reconstruction: bool,
) -> Scene:
    """Sahne visual checkpointleri geçene kadar farklı çözüm dener."""
    number = int(scene_data["scene_number"])
    candidates = list(global_candidates)
    candidate_map = {item.candidate_id: item for item in candidates}
    known_hashes = [item.perceptual_hash for item in candidates]
    rankings = rank_scene_candidates(client, event, scene_data, candidates)
    attempted: set[str] = set()
    feedback = ""
    for cycle in range(1, VISUAL_REPAIR_CYCLES + 1):
        print(
            f"\n→ Scene {number:02d} real visual cycle {cycle}/{VISUAL_REPAIR_CYCLES}"
        )
        selection: RankedVisual | None = None
        for option in rankings:
            if option.candidate_id in attempted:
                continue
            if option.candidate_id in used_ids and option.relevance_score < 98:
                continue
            if option.candidate_id not in candidate_map:
                continue
            selection = option
            break
        if selection is not None:
            attempted.add(selection.candidate_id)
            candidate = candidate_map[selection.candidate_id]
            output = (
                asset_directory
                / f"scene_{number:02d}_{selection.candidate_id}_{cycle:02d}.png"
            )
            visual, technical_quality = restore_crop(
                candidate, selection.crop_box, output
            )
            evaluation = evaluate_visual(client, scene_data, visual, technical_quality)
            visual_match = float(evaluation.get("visual_match", 0) or 0)
            ai_quality = float(evaluation.get("image_quality", 0) or 0)
            image_quality = technical_quality * 0.45 + ai_quality * 0.55
            composition = float(evaluation.get("composition", 0) or 0)
            problems = "; ".join(
                (clean(item) for item in evaluation.get("problems", []) if clean(item))
            )
            match_pass = CHECKPOINTS.record(
                name=f"scene_{number:02d}_visual_match",
                score=visual_match,
                threshold=VISUAL_MATCH_THRESHOLD,
                cycle=cycle,
                details=problems,
            )
            quality_pass = CHECKPOINTS.record(
                name=f"scene_{number:02d}_image_quality",
                score=image_quality,
                threshold=IMAGE_QUALITY_THRESHOLD,
                cycle=cycle,
                details=f"source={technical_quality:.1f}, vision={ai_quality:.1f}",
            )
            composition_pass = CHECKPOINTS.record(
                name=f"scene_{number:02d}_composition",
                score=composition,
                threshold=COMPOSITION_THRESHOLD,
                cycle=cycle,
                details=problems,
            )
            if match_pass and quality_pass and composition_pass:
                used_ids.add(selection.candidate_id)
                return make_scene(
                    scene_data=scene_data,
                    rankings=rankings,
                    selected_candidate_id=selection.candidate_id,
                    relevance_score=selection.relevance_score,
                    crop_box=selection.crop_box,
                    visual_source="real_comic",
                    visual_file=visual,
                    visual_match=visual_match,
                    image_quality=image_quality,
                    composition=composition,
                )
            feedback = clean(evaluation.get("repair_instruction")) or problems
        if PANEL_DIR or cycle == VISUAL_REPAIR_CYCLES:
            continue
        search_results = scene_specific_search(event, scene_data, feedback)
        supplemental = download_candidates(
            event,
            search_results,
            max_candidates=MAX_SCENE_IMAGES,
            directory_name=f"scene_{number:02d}_search_{cycle:02d}",
            start_index=1000 + number * 100 + cycle * 20,
            known_hashes=known_hashes,
        )
        for candidate in supplemental:
            known_hashes.append(candidate.perceptual_hash)
            candidates.append(candidate)
            candidate_map[candidate.candidate_id] = candidate
        if supplemental:
            new_rankings = rank_scene_candidates(
                client, event, scene_data, supplemental + global_candidates[:8]
            )
            rankings = new_rankings + rankings
    if not enable_ai_reconstruction:
        raise EventRejectedError(
            f"Scene {number} için yeterli gerçek görsel bulunamadı."
        )
    references: list[Path] = []
    for option in rankings:
        candidate = candidate_map.get(option.candidate_id)
        if candidate is None:
            continue
        references.append(Path(candidate.local_file))
        if len(references) >= 3:
            break
    if not references:
        references = [Path(candidate.local_file) for candidate in global_candidates[:3]]
    feedback = (
        feedback or "Gerçek comic panelleri exact narration ile yeterince eşleşmedi."
    )
    for cycle in range(1, AI_RECONSTRUCTION_REPAIR_CYCLES + 1):
        print(f"🎨 Scene {number:02d} AI reconstruction cycle {cycle}")
        generated = asset_directory / f"scene_{number:02d}_ai_{cycle:02d}.jpg"
        generate_reconstruction(
            client, event, scene_data, references, generated, feedback
        )
        evaluation = evaluate_visual(client, scene_data, generated, 100.0)
        visual_match = float(evaluation.get("visual_match", 0) or 0)
        image_quality = float(evaluation.get("image_quality", 0) or 0)
        composition = float(evaluation.get("composition", 0) or 0)
        problems = "; ".join(
            (clean(item) for item in evaluation.get("problems", []) if clean(item))
        )
        match_pass = CHECKPOINTS.record(
            name=f"scene_{number:02d}_ai_visual_match",
            score=visual_match,
            threshold=VISUAL_MATCH_THRESHOLD,
            cycle=cycle,
            details=problems,
        )
        quality_pass = CHECKPOINTS.record(
            name=f"scene_{number:02d}_ai_image_quality",
            score=image_quality,
            threshold=IMAGE_QUALITY_THRESHOLD,
            cycle=cycle,
            details=problems,
        )
        composition_pass = CHECKPOINTS.record(
            name=f"scene_{number:02d}_ai_composition",
            score=composition,
            threshold=COMPOSITION_THRESHOLD,
            cycle=cycle,
            details=problems,
        )
        if match_pass and quality_pass and composition_pass:
            return make_scene(
                scene_data=scene_data,
                rankings=rankings,
                selected_candidate_id="AI",
                relevance_score=round(visual_match),
                crop_box=[0.0, 0.0, 1.0, 1.0],
                visual_source="ai_reconstruction",
                visual_file=generated,
                visual_match=visual_match,
                image_quality=image_quality,
                composition=composition,
            )
        feedback = clean(evaluation.get("repair_instruction")) or problems
    raise EventRejectedError(
        f"Scene {number} gerçek panel + AI reconstruction ile kalite checkpointlerini geçemedi."
    )


def build_quality_visuals(
    client, event, storyboard, max_images, enable_ai_reconstruction
):
    check_budget()
    if PANEL_DIR:
        candidates = local_candidates(PANEL_DIR)
    else:
        results = global_image_search(event, storyboard)
        candidates = download_candidates(
            event,
            results,
            max_candidates=max_images,
            directory_name="global",
            start_index=1,
        )
    if len(candidates) < min(4, len(storyboard["scenes"])):
        raise EventRejectedError(
            "Yeterli gerçek panel bulunamadı. Konuyu daralt veya kendi panel klasörünü kullan."
        )
    asset_directory = ASSET_DIR / event["id"]
    asset_directory.mkdir(parents=True, exist_ok=True)
    used_ids, scenes = (set(), [])
    for scene_data in storyboard["scenes"]:
        check_budget()
        scenes.append(
            resolve_scene_until_pass(
                client,
                event,
                scene_data,
                candidates,
                used_ids,
                asset_directory,
                enable_ai_reconstruction,
            )
        )
    return (scenes, candidates)


def compose_vertical(visual_file: Path) -> Image.Image:
    """Paneli temiz 9:16 video kompozisyonuna dönüştürür."""
    with Image.open(visual_file) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
    background = ImageOps.fit(
        source, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )
    background = background.filter(ImageFilter.GaussianBlur(radius=50))
    background = ImageEnhance.Brightness(background).enhance(0.28)
    background = ImageEnhance.Color(background).enhance(0.78)
    foreground = source.copy()
    ratio = source.width / max(1, source.height)
    if 0.48 <= ratio <= 0.7:
        foreground.thumbnail((WIDTH - 10, HEIGHT - 90), Image.Resampling.LANCZOS)
    else:
        foreground.thumbnail((WIDTH - 26, 1580), Image.Resampling.LANCZOS)
    canvas = background.copy()
    x = (WIDTH - foreground.width) // 2
    y = (HEIGHT - foreground.height) // 2 - 35
    y = max(40, y)
    canvas.paste(foreground, (x, y))
    return canvas


def build_contact_sheet(scenes: list[Scene]) -> Path:
    """Final Visual Director için contact sheet üretir."""
    columns = 4
    cell_width = 270
    cell_height = 480
    rows = math.ceil(len(scenes) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    for index, scene in enumerate(scenes):
        frame = compose_vertical(Path(scene.visual_file))
        frame.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = index % columns * cell_width
        y = index // columns * cell_height
        sheet.paste(frame, (x, y))
    output = WORK_DIR / "contact_sheet.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=94)
    return output


def review_full_visual_flow(
    client: genai.Client, scenes: list[Scene]
) -> dict[str, Any]:
    """Bütün videonun görsel akışını kontrol eder."""
    contact_sheet = build_contact_sheet(scenes)
    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\nNARRATION: {scene.narration}\nEXPECTED: {scene.visual_description}\nMATCH: {scene.visual_match_score:.1f}\nIMAGE QUALITY: {scene.image_quality_score:.1f}\nCOMPOSITION: {scene.composition_score:.1f}\nSOURCE: {scene.visual_source}"
            for scene in scenes
        )
    )
    prompt = f"""\nPremium comic Shorts Final Visual Director'sın.\n\nPLAN:\n{plan}\n\nContact sheet'i incele.\n\n0-100:\nvisual_story_match\nvisual_diversity\nimage_quality\ncrop_quality\nnarrative_flow\nprofessional_feel\nrewatch_value\noverall\n\n94+ overall yalnız gerçekten premium video için ver.\n\nKontrol et:\n- Sesin anlatacağı olayın görseli o sahnede mi?\n- Çok benzer paneller var mı?\n- Düşük kaliteli/upscale çamur görsel var mı?\n- Crop aksiyonu kesiyor mu?\n- Slideshow hissi çok mu güçlü?\n- AI reconstruction gerçek comic estetiğine uyuyor mu?\n\nSADECE JSON:\n{{\n  "visual_story_match": 0,\n  "visual_diversity": 0,\n  "image_quality": 0,\n  "crop_quality": 0,\n  "narrative_flow": 0,\n  "professional_feel": 0,\n  "rewatch_value": 0,\n  "overall": 0,\n  "weak_scenes": [3,7],\n  "problems": ["..."]\n}}\n"""

    def request() -> Any:
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=contact_sheet.read_bytes(), mime_type="image/jpeg"
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.05, response_mime_type="application/json"
            ),
        )

    response = gemini_with_retry(request, operation_name="Final Visual Director")
    return parse_json_response(response.text or "{}")


def run_final_visual_gate(
    client: genai.Client,
    event: dict[str, Any],
    storyboard: dict[str, Any],
    scenes: list[Scene],
    global_candidates: list[ImageCandidate],
    enable_ai_reconstruction: bool,
) -> list[Scene]:
    """Final görsel akışını zayıf sahneleri yeniden üreterek düzeltir."""
    asset_directory = ASSET_DIR / event["id"]
    for cycle in range(1, FINAL_VISUAL_REPAIR_CYCLES + 1):
        review = review_full_visual_flow(client, scenes)
        score = float(review.get("overall", 0) or 0)
        problems = "; ".join(
            (clean(item) for item in review.get("problems", []) if clean(item))
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
        if cycle == FINAL_VISUAL_REPAIR_CYCLES:
            break
        weak_numbers: set[int] = set()
        for raw in review.get("weak_scenes", []):
            try:
                number = int(raw)
                if 1 <= number <= len(scenes):
                    weak_numbers.add(number)
            except (TypeError, ValueError):
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
            weak_numbers.add(weakest.scene_number)
        for number in sorted(weak_numbers):
            scene_data = next(
                (
                    scene
                    for scene in storyboard["scenes"]
                    if int(scene["scene_number"]) == number
                )
            )
            used_ids = {
                scene.selected_candidate_id
                for scene in scenes
                if scene.scene_number != number and scene.selected_candidate_id != "AI"
            }
            replacement = resolve_scene_until_pass(
                client,
                event,
                scene_data,
                global_candidates,
                used_ids,
                asset_directory,
                enable_ai_reconstruction,
            )
            scenes[number - 1] = replacement
    raise EventRejectedError(
        "Final görsel akış bu event için premium kalite seviyesine ulaşamadı."
    )


def choose_motion_plan(
    client: genai.Client, scenes: list[Scene]
) -> dict[int, dict[str, Any]]:
    """Sahne anlamına uygun kamera hareketleri belirler."""
    plan = "\n\n".join(
        (
            f"SCENE {scene.scene_number}\nROLE: {scene.story_role}\nNARRATION: {scene.narration}\nVISUAL: {scene.visual_description}"
            for scene in scenes
        )
    )
    prompt = f"""\nPremium 9:16 comic Cinematic Director'sın.\n\nSCENES:\n{plan}\n\nmotion:\n{", ".join(ALLOWED_MOTIONS)}\n\ntransition:\n{", ".join(ALLOWED_TRANSITIONS)}\n\nKurallar:\n- aynı motion arka arkaya gelmesin\n- hareketler hafif ve premium olsun\n- comic paneli gereksiz zoom ile bozma\n- context sakin\n- escalation dinamik\n- twist/payoff impact kullanılabilir\n- cta sakin\n- sonraki scene görselini narration başlamadan gösterme\n- transition yalnız mevcut sahnenin sonunda fade olarak çalışsın\n\nSADECE JSON:\n{{\n  "scenes": [\n    {{\n      "scene_number": 1,\n      "motion": "slow_push",\n      "transition": "soft_black",\n      "transition_duration": 0.12\n    }}\n  ]\n}}\n"""
    payload = ask_gemini_json(
        client, prompt, temperature=0.16, operation_name="Cinematic Director"
    )
    output: dict[int, dict[str, Any]] = {}
    for item in payload.get("scenes", []):
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("scene_number", 0))
        except (TypeError, ValueError):
            continue
        output[number] = item
    return output


def apply_motion_plan(scenes: list[Scene], plan: dict[int, dict[str, Any]]) -> None:
    """Motion planını güvenli değerlerle scene'lere uygular."""
    previous_motion = ""
    for index, scene in enumerate(scenes):
        item = plan.get(scene.scene_number, {})
        motion = clean(item.get("motion"))
        if motion not in ALLOWED_MOTIONS:
            motion = ALLOWED_MOTIONS[index % len(ALLOWED_MOTIONS)]
        if motion == previous_motion:
            motion = ALLOWED_MOTIONS[
                (ALLOWED_MOTIONS.index(motion) + 1) % len(ALLOWED_MOTIONS)
            ]
        transition = clean(item.get("transition"))
        if transition not in ALLOWED_TRANSITIONS:
            transition = "cut"
        try:
            transition_duration = float(item.get("transition_duration", 0.12))
        except (TypeError, ValueError):
            transition_duration = 0.12
        if transition == "cut":
            transition_duration = 0.0
        else:
            transition_duration = max(0.08, min(0.18, transition_duration))
        scene.motion = motion
        scene.transition = transition
        scene.transition_duration = transition_duration
        previous_motion = motion


def build_quality_motion(client, scenes):
    """Validate concrete motion constraints instead of repeatedly self-scoring a plan."""
    plan = choose_motion_plan(client, scenes)
    apply_motion_plan(scenes, plan)
    previous = None
    for scene in scenes:
        if scene.motion not in ALLOWED_MOTIONS or scene.motion == previous:
            scene.motion = "slow_pull" if previous == "slow_push" else "slow_push"
        if scene.motion == "impact" and scene.story_role not in {
            "twist",
            "payoff",
            "escalation",
        }:
            scene.motion = "slow_push"
        if scene.transition not in {"cut", "soft_black", "dip_black"}:
            scene.transition = "cut"
        scene.transition_duration = min(max(0.0, scene.transition_duration), 0.12)
        previous = scene.motion
    CHECKPOINTS.record(
        name="motion_plan_valid",
        score=100,
        threshold=100,
        cycle=1,
        details="Hafif hareketler; kontrollü geçiş; geçerli sahne zamanları.",
        passed=True,
    )


def attach_timeline(scenes: list[Scene], timeline: list[tuple[float, float]]) -> None:
    """Exact audio timeline değerlerini sahnelere bağlar."""
    if len(scenes) != len(timeline):
        raise EventRejectedError("Scene count ve timeline uyuşmuyor.")
    for scene, (start, end) in zip(scenes, timeline, strict=True):
        scene.audio_start = start
        scene.audio_end = end


def create_subtitles(aligned_words, scenes, narration, offset):
    if narration.split() != [item["word"] for item in aligned_words]:
        raise EventRejectedError("Altyazı metni anlatımla eşleşmiyor.")
    output = core.create_captions(aligned_words, WORK_DIR / "precise.ass", offset)
    CHECKPOINTS.record(
        name="subtitle_safe_zone",
        score=100,
        threshold=100,
        cycle=1,
        details="Konuşulan kelime + sonraki kelime; taşma kontrolü geçti.",
        passed=True,
    )
    return output


def motion_filter(motion, frames=150):
    progress = f"on/{max(1, frames - 1)}"
    zooms = {
        "slow_push": f"1+0.045*{progress}",
        "slow_pull": f"1.045-0.045*{progress}",
        "impact": f"1+0.06*min(1,({progress})*2)",
        "hold": "1.0",
    }
    zoom = zooms.get(motion, "1.045")
    x, y = ("(iw-iw/zoom)/2", "(ih-ih/zoom)/2")
    if motion == "pan_left":
        x = f"(iw-iw/zoom)*(1-{progress})"
    elif motion == "pan_right":
        x = f"(iw-iw/zoom)*{progress}"
    elif motion == "vertical_scan":
        y = f"(ih-ih/zoom)*{progress}"
    return f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}"


def transition_filter(transition: str, duration: float, scene_duration: float) -> str:
    """Sahne sonunda yalnız mevcut paneli fade ederek sync'i korur."""
    if transition == "cut" or duration <= 0:
        return ""
    duration = min(duration, scene_duration / 4.0)
    start = max(0.0, scene_duration - duration)
    color = "white" if transition == "flash_white" else "black"
    return f",fade=t=out:st={start:.3f}:d={duration:.3f}:color={color}"


def render_scene(ffmpeg, scene, frame_file, output_file):
    frames = round(scene.audio_end * FPS) - round(scene.audio_start * FPS)
    if frames < 1:
        raise ComicFactoryError("Sahne süresi en az bir kare olmalı.")
    duration = frames / FPS
    filters = (
        f"scale={WIDTH}:{HEIGHT},"
        + motion_filter(scene.motion, frames)
        + transition_filter(scene.transition, scene.transition_duration, duration)
        + ",setsar=1,format=yuv420p"
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
            "-frames:v",
            str(frames),
            "-vf",
            filters,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-threads",
            "2",
            str(output_file),
        ],
        f"Sahne {scene.scene_number} render edilemedi.",
    )


def render_video(scenes, audio_file, subtitle_file):
    ffmpeg = ffmpeg_path()
    render_dir = WORK_DIR / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        image_path = render_dir / f"frame_{scene.scene_number:02d}.png"
        segment = render_dir / f"segment_{scene.scene_number:02d}.mp4"
        compose_vertical(Path(scene.visual_file)).save(image_path)
        render_scene(ffmpeg, scene, image_path, segment)
    concat_file = render_dir / "concat.txt"
    concat_file.write_text(
        "\n".join((f"file 'segment_{scene.scene_number:02d}.mp4'" for scene in scenes)),
        encoding="utf-8",
    )
    silent = render_dir / "silent.mp4"
    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(silent),
        ],
        "Sahneler birleştirilemedi.",
    )
    shutil.copy2(subtitle_file, render_dir / "captions.ass")
    archive = VIDEO_DIR / f"comic_{datetime.now(TZ):%Y%m%d_%H%M%S}.mp4"
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(silent.resolve()),
        "-i",
        str(audio_file.resolve()),
        "-vf",
        "ass=captions.ass",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-shortest",
        "-movflags",
        "+faststart",
        str(archive.resolve()),
    ]
    process = subprocess.run(
        command, cwd=render_dir, capture_output=True, text=True, timeout=900
    )
    if process.returncode:
        raise ComicFactoryError("Final render başarısız: " + process.stderr[-3000:])
    return archive


def technical_video_check(video, audio):
    return core.check_video(video, audio)


def render_until_pass(scenes, audio_file, subtitle_file):
    archive = render_video(scenes, audio_file, subtitle_file)
    metrics = technical_video_check(archive, audio_file)
    CHECKPOINTS.record(
        name="technical_video_qc",
        score=100 if metrics["passed"] else 0,
        threshold=100,
        cycle=1,
        details=json.dumps(metrics),
        passed=metrics["passed"],
    )
    if not metrics["passed"]:
        raise EventRejectedError("Video teknik kontrolden geçmedi; tanı dosyasına bak.")
    shutil.copy2(archive, LATEST_VIDEO_FILE)
    return archive


def save_script_metadata(
    event: dict[str, Any], storyboard: dict[str, Any], scenes: list[Scene]
) -> None:
    """Mevcut upload scriptleri için latest.json üretir."""
    hashtags = [clean(tag) for tag in storyboard.get("hashtags", []) if clean(tag)]
    if not any((tag.casefold() == "#shorts" for tag in hashtags)):
        hashtags.append("#Shorts")
    source_lines: list[str] = []
    seen_urls: set[str] = set()
    for source in event.get("sources", []):
        if not isinstance(source, dict):
            continue
        url = clean(source.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        source_lines.append(f"- {clean(source.get('name')) or 'Source'}: {url}")
    description = clean(storyboard.get("description"))
    full_description = description + "\n\n" + " ".join(hashtags)
    if source_lines:
        full_description += "\n\nKaynaklar:\n" + "\n".join(source_lines)
    save_json(
        LATEST_SCRIPT_FILE,
        {
            "event_id": event["id"],
            "generated_at": datetime.now(TZ).isoformat(),
            "script": {
                "title": clean(storyboard.get("title")),
                "description": description,
                "full_description": full_description,
                "hashtags": hashtags,
                "narration": clean(storyboard.get("narration")),
                "scene_narrations": [scene.narration for scene in scenes],
            },
        },
    )


def save_visual_manifest(event: dict[str, Any], scenes: list[Scene]) -> None:
    """Final scene planını kalite analizi için kaydeder."""
    directory = ASSET_DIR / event["id"]
    save_json(
        directory / "visual_manifest.json",
        {
            "event": {
                "event_title": event.get("event_title"),
                "series": event.get("series"),
                "issue": event.get("issue"),
            },
            "scenes": [asdict(scene) for scene in scenes],
        },
    )


def mark_used(event: dict[str, Any]) -> None:
    """Yalnız başarıyla video üretilmiş eventi kullanılmış işaretler."""
    payload = load_json(USED_EVENTS_FILE, {"events": []})
    if not isinstance(payload, dict):
        payload = {"events": []}
    events = payload.setdefault("events", [])
    key = event_key(event)
    if not any(
        (
            isinstance(item, dict) and clean(item.get("event_key")) == key
            for item in events
        )
    ):
        events.append(
            {
                "event_key": key,
                "event_title": clean(event.get("event_title")),
                "series": clean(event.get("series")),
                "issue": clean(event.get("issue")),
                "completed_at": datetime.now(TZ).isoformat(),
            }
        )
    save_json(USED_EVENTS_FILE, payload)


def build_single_event_video(
    client, event, *, max_images, subtitle_offset, enable_ai_reconstruction
):
    event = activate_event(event)
    storyboard = build_quality_storyboard(client, event)
    narration = clean(storyboard["narration"])
    audio_file, _, aligned_words = build_quality_audio(client, narration)
    duration = media_duration(ffmpeg_path(), audio_file)
    if not 15 <= duration <= 135:
        raise EventRejectedError(f"Anlatım süresi uygun değil: {duration:.1f} saniye.")
    timeline = build_scene_timeline(storyboard, aligned_words, duration)
    scenes, candidates = build_quality_visuals(
        client, event, storyboard, max_images, enable_ai_reconstruction
    )
    scenes = run_final_visual_gate(
        client, event, storyboard, scenes, candidates, enable_ai_reconstruction
    )
    attach_timeline(scenes, timeline)
    build_quality_motion(client, scenes)
    subtitle_file = create_subtitles(aligned_words, scenes, narration, subtitle_offset)
    archive = render_until_pass(scenes, audio_file, subtitle_file)
    save_script_metadata(event, storyboard, scenes)
    save_visual_manifest(event, scenes)
    build_contact_sheet(scenes)
    return archive


def configure(settings, topic="", panel_dir=None):
    global SETTINGS, TOPIC, PANEL_DIR, SCENE_COUNT, TARGET_SECONDS, BRIEF
    global STORY_THRESHOLD, AUDIO_ALIGNMENT_THRESHOLD, VISUAL_MATCH_THRESHOLD
    global IMAGE_QUALITY_THRESHOLD, COMPOSITION_THRESHOLD, FINAL_VISUAL_THRESHOLD
    global STORY_REPAIR_CYCLES_PER_EVENT, AUDIO_REPAIR_CYCLES, VISUAL_REPAIR_CYCLES
    global AI_RECONSTRUCTION_REPAIR_CYCLES, FINAL_VISUAL_REPAIR_CYCLES
    global \
        GEMINI_MODEL, \
        GEMINI_TTS_MODEL, \
        GEMINI_TTS_VOICE, \
        GEMINI_IMAGE_MODEL, \
        GROQ_MODEL
    global API_CALLS, STARTED_AT, CHECKPOINTS, MAX_GLOBAL_IMAGES, AUDIO_DIAGNOSTICS_DIR
    SETTINGS = settings
    TOPIC = topic.strip()
    PANEL_DIR = panel_dir
    SCENE_COUNT = settings.scene_count
    TARGET_SECONDS = settings.target_seconds
    BRIEF = settings.brief
    STORY_THRESHOLD = settings.story_threshold
    AUDIO_ALIGNMENT_THRESHOLD = settings.alignment_threshold
    VISUAL_MATCH_THRESHOLD = settings.visual_threshold
    IMAGE_QUALITY_THRESHOLD = settings.image_threshold
    COMPOSITION_THRESHOLD = settings.composition_threshold
    FINAL_VISUAL_THRESHOLD = settings.final_visual_threshold
    STORY_REPAIR_CYCLES_PER_EVENT = AUDIO_REPAIR_CYCLES = VISUAL_REPAIR_CYCLES = (
        settings.repair_attempts
    )
    AI_RECONSTRUCTION_REPAIR_CYCLES = FINAL_VISUAL_REPAIR_CYCLES = (
        settings.repair_attempts
    )
    MAX_GLOBAL_IMAGES = settings.max_images
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or settings.gemini_model
    GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL") or settings.tts_model
    GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE") or settings.tts_voice
    GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL") or settings.image_model
    GROQ_MODEL = os.getenv("GROQ_WHISPER_MODEL") or settings.whisper_model
    API_CALLS = 0
    STARTED_AT = time.monotonic()
    CHECKPOINTS = CheckpointManager()
    AUDIO_DIAGNOSTICS_DIR = AUDIO_DIR / "alignment_diagnostics"


def check_budget():
    if time.monotonic() - STARTED_AT > SETTINGS.max_minutes * 60:
        raise ComicFactoryError(
            "Üretim süre sınırına ulaştı. Sonuçlar ve tanı dosyaları saklandı."
        )
    if API_CALLS >= SETTINGS.max_api_calls:
        raise ComicFactoryError("Bu çalışma için Gemini istek sınırına ulaşıldı.")


def local_candidates(directory):
    candidates = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            candidates.append(
                ImageCandidate(
                    candidate_id=f"local_{len(candidates) + 1:04d}",
                    local_file=str(path.resolve()),
                    source_page="user_supplied",
                    image_url="",
                    title=path.stem,
                    width=image.width,
                    height=image.height,
                    quality_score=source_image_quality(image),
                    perceptual_hash=average_hash(image),
                )
            )
    return candidates
