"""Publish Comic Factory videos automatically as Instagram Reels."""

from __future__ import annotations
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any
import cloudinary
import cloudinary.uploader
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = BASE_DIR / "data" / "videos" / "latest.mp4"
DEFAULT_SCRIPT_PATH = BASE_DIR / "data" / "scripts" / "latest.json"
DEFAULT_HISTORY_PATH = BASE_DIR / "data" / "instagram" / "upload_history.json"
INSTAGRAM_API_VERSION = os.getenv("INSTAGRAM_API_VERSION", "v23.0")
INSTAGRAM_GRAPH_URL = f"https://graph.instagram.com/{INSTAGRAM_API_VERSION}"
MAX_CAPTION_LENGTH = 2200
MAX_HASHTAGS = 15
REQUEST_TIMEOUT_SECONDS = 60
PROCESSING_TIMEOUT_SECONDS = 600
PROCESSING_POLL_SECONDS = 5
CLOUDINARY_LARGE_FILE_LIMIT = 95 * 1024 * 1024


class InstagramUploadError(RuntimeError):
    """Raised when the Instagram publishing process fails."""


def require_environment_variable(name: str) -> str:
    """Return a required environment variable or fail with a clear message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise InstagramUploadError(f"Eksik ortam değişkeni: {name}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty object when unavailable."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Uyarı: {path} okunamadı: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_string(data: Any, candidate_keys: tuple[str, ...]) -> str:
    """Find the first non-empty string for any requested key recursively."""
    if isinstance(data, dict):
        for key in candidate_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            found = find_string(value, candidate_keys)
            if found:
                return found
    if isinstance(data, list):
        for value in data:
            found = find_string(value, candidate_keys)
            if found:
                return found
    return ""


def find_tags(data: Any) -> list[str]:
    """Find a tag list recursively in script metadata."""
    if isinstance(data, dict):
        for key in ("instagram_tags", "tags", "hashtags", "keywords"):
            value = data.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str) and value.strip():
                return [
                    item.strip() for item in re.split("[,;\\n]+", value) if item.strip()
                ]
        for value in data.values():
            tags = find_tags(value)
            if tags:
                return tags
    if isinstance(data, list):
        for value in data:
            tags = find_tags(value)
            if tags:
                return tags
    return []


def normalize_hashtag(tag: str) -> str:
    """Convert a tag into an Instagram hashtag."""
    cleaned = tag.strip().lstrip("#")
    cleaned = re.sub("[^\\w]+", "", cleaned, flags=re.UNICODE)
    if not cleaned:
        return ""
    return f"#{cleaned}"


def build_caption(script_path: Path) -> str:
    """Build an Instagram caption from Comic Factory metadata."""
    data = load_json(script_path)
    explicit_caption = find_string(data, ("instagram_caption", "reels_caption"))
    if explicit_caption:
        caption = explicit_caption
    else:
        title = find_string(data, ("title", "video_title", "youtube_title"))
        description = find_string(
            data, ("description", "video_description", "youtube_description")
        )
        parts = [part for part in (title, description) if part]
        caption = "\n\n".join(parts)
    if not caption:
        caption = "Çizgi roman tarihinin inanılmaz anlarından biri!"
    tags = find_tags(data)
    if not tags:
        tags = ["çizgiroman", "comics", "süperkahraman", "comicbooks", "reels"]
    hashtags: list[str] = []
    for tag in tags:
        hashtag = normalize_hashtag(tag)
        if hashtag and hashtag.lower() not in {item.lower() for item in hashtags}:
            hashtags.append(hashtag)
        if len(hashtags) >= MAX_HASHTAGS:
            break
    hashtag_text = " ".join(hashtags)
    if hashtag_text:
        available_length = MAX_CAPTION_LENGTH - len(hashtag_text) - 2
        caption = caption[: max(0, available_length)].rstrip()
        caption = f"{caption}\n\n{hashtag_text}"
    return caption[:MAX_CAPTION_LENGTH].rstrip()


def configure_cloudinary() -> None:
    """Configure the Cloudinary Python SDK from GitHub Secrets."""
    cloudinary.config(
        cloud_name=require_environment_variable("CLOUDINARY_CLOUD_NAME"),
        api_key=require_environment_variable("CLOUDINARY_API_KEY"),
        api_secret=require_environment_variable("CLOUDINARY_API_SECRET"),
        secure=True,
    )


def upload_video_to_cloudinary(video_path: Path) -> tuple[str, str]:
    """Upload a video temporarily and return its public ID and secure URL."""
    configure_cloudinary()
    unique_name = f"comic_factory_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    options: dict[str, Any] = {
        "resource_type": "video",
        "folder": "comic_factory_instagram_temp",
        "public_id": unique_name,
        "overwrite": False,
    }
    print("Cloudinary: video geçici olarak yükleniyor...")
    if video_path.stat().st_size >= CLOUDINARY_LARGE_FILE_LIMIT:
        result = cloudinary.uploader.upload_large(
            str(video_path), chunk_size=6000000, **options
        )
    else:
        result = cloudinary.uploader.upload(str(video_path), **options)
    public_id = str(result.get("public_id", "")).strip()
    secure_url = str(result.get("secure_url", "")).strip()
    if not public_id or not secure_url:
        raise InstagramUploadError("Cloudinary geçerli bir video URL'si döndürmedi.")
    print("Cloudinary: video hazır.")
    return (public_id, secure_url)


def delete_cloudinary_video(public_id: str) -> None:
    """Delete the temporary Cloudinary video."""
    if not public_id:
        return
    try:
        result = cloudinary.uploader.destroy(
            public_id, resource_type="video", invalidate=True
        )
        print(
            f"Cloudinary: geçici video silindi ({result.get('result', 'bilinmiyor')})."
        )
    except Exception as exc:
        print(f"Uyarı: Cloudinary geçici videosu silinemedi: {exc}")


def wait_until_video_is_public(video_url: str) -> None:
    """Wait until the Cloudinary video can be downloaded publicly."""
    print("Cloudinary CDN erişimi kontrol ediliyor...")
    for attempt in range(1, 11):
        try:
            with requests.get(video_url, stream=True, timeout=30) as response:
                if response.status_code == 200:
                    print("Cloudinary CDN: video erişilebilir.")
                    return
        except requests.RequestException:
            pass
        print(f"Cloudinary CDN henüz hazır değil ({attempt}/10)...")
        time.sleep(3)
    raise InstagramUploadError("Cloudinary videosuna herkese açık olarak erişilemiyor.")


def instagram_headers(access_token: str) -> dict[str, str]:
    """Create authorization headers for Instagram API requests."""
    return {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "ComicFactoryInstagramUploader/1.0",
    }


def response_error_message(response: requests.Response) -> str:
    """Extract a useful error message without exposing credentials."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "")).strip()
            code = error.get("code")
            subcode = error.get("error_subcode")
            details = []
            if code is not None:
                details.append(f"code={code}")
            if subcode is not None:
                details.append(f"subcode={subcode}")
            suffix = f" ({', '.join(details)})" if details else ""
            if message:
                return f"{message}{suffix}"
        return json.dumps(payload, ensure_ascii=False)[:1000]
    return str(payload)[:1000]


def request_json(
    method: str,
    url: str,
    *,
    access_token: str,
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    retries: int = 4,
) -> dict[str, Any]:
    """Perform an Instagram API request with retries."""
    if method.upper() == "POST" and url.endswith("/media_publish"):
        retries = 1
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=instagram_headers(access_token),
                data=data,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(attempt * 3)
                continue
            break
        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
            except ValueError as exc:
                raise InstagramUploadError(
                    "Instagram API geçersiz JSON yanıtı döndürdü."
                ) from exc
            if not isinstance(payload, dict):
                raise InstagramUploadError("Instagram API beklenmeyen yanıt döndürdü.")
            return payload
        last_error = response_error_message(response)
        if (
            response.status_code == 429 or response.status_code >= 500
        ) and attempt < retries:
            wait_seconds = attempt * 5
            print(
                f"Instagram API geçici hata verdi. {wait_seconds} saniye sonra tekrar deneniyor..."
            )
            time.sleep(wait_seconds)
            continue
        raise InstagramUploadError(
            f"Instagram API hatası: HTTP {response.status_code} - {last_error}"
        )
    raise InstagramUploadError(f"Instagram API bağlantısı başarısız: {last_error}")


def create_reel_container(
    account_id: str, access_token: str, video_url: str, caption: str
) -> str:
    """Create an Instagram Reel media container."""
    print("Instagram: Reel container oluşturuluyor...")
    url = f"{INSTAGRAM_GRAPH_URL}/{account_id}/media"
    payload = request_json(
        "POST",
        url,
        access_token=access_token,
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
        },
    )
    container_id = str(payload.get("id", "")).strip()
    if not container_id:
        raise InstagramUploadError("Instagram Reel container ID döndürmedi.")
    print(f"Instagram: container oluşturuldu ({container_id}).")
    return container_id


def wait_for_reel_processing(container_id: str, access_token: str) -> None:
    """Wait until Instagram finishes processing the Reel."""
    print("Instagram: video işleniyor...")
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        url = f"{INSTAGRAM_GRAPH_URL}/{container_id}"
        payload = request_json(
            "GET",
            url,
            access_token=access_token,
            params={"fields": "status_code,status"},
        )
        status_code = str(payload.get("status_code", "")).upper()
        status_text = str(payload.get("status", "")).strip()
        print(f"Instagram işlem durumu: {status_code or 'BİLİNMİYOR'}")
        if status_code == "FINISHED":
            return
        if status_code in {"ERROR", "EXPIRED"}:
            raise InstagramUploadError(
                f"Instagram videoyu işleyemedi: {status_text or status_code}"
            )
        time.sleep(PROCESSING_POLL_SECONDS)
    raise InstagramUploadError("Instagram video işleme süresi zaman aşımına uğradı.")


def publish_reel(account_id: str, access_token: str, container_id: str) -> str:
    """Publish a processed Instagram Reel container."""
    print("Instagram: Reel yayınlanıyor...")
    url = f"{INSTAGRAM_GRAPH_URL}/{account_id}/media_publish"
    payload = request_json(
        "POST", url, access_token=access_token, data={"creation_id": container_id}
    )
    media_id = str(payload.get("id", "")).strip()
    if not media_id:
        raise InstagramUploadError("Instagram yayınlanan medya ID'sini döndürmedi.")
    print(f"Instagram Reel başarıyla yayınlandı: {media_id}")
    return media_id


def load_history(path: Path) -> dict[str, Any]:
    """Load Instagram upload history."""
    history = load_json(path)
    uploads = history.get("uploads")
    if not isinstance(uploads, list):
        history["uploads"] = []
    return history


def already_uploaded(history: dict[str, Any], video_hash: str) -> bool:
    """Check whether the exact video was already published."""
    uploads = history.get("uploads", [])
    if not isinstance(uploads, list):
        return False
    for upload in uploads:
        if isinstance(upload, dict) and upload.get("sha256") == video_hash:
            return True
    return False


def record_upload(
    history_path: Path,
    history: dict[str, Any],
    *,
    video_path: Path,
    video_hash: str,
    media_id: str,
) -> None:
    """Record a successful Instagram upload."""
    uploads = history.setdefault("uploads", [])
    if not isinstance(uploads, list):
        uploads = []
        history["uploads"] = uploads
    uploads.append(
        {
            "sha256": video_hash,
            "video": video_path.name,
            "instagram_media_id": media_id,
            "uploaded_at_unix": int(time.time()),
        }
    )
    history["uploads"] = uploads[-100:]
    save_json(history_path, history)


def upload_reel(video_path: Path, script_path: Path, history_path: Path) -> str | None:
    """Run the complete Cloudinary-to-Instagram Reel publishing flow."""
    if not video_path.exists():
        raise InstagramUploadError(f"Video bulunamadı: {video_path}")
    if not video_path.is_file():
        raise InstagramUploadError(f"Video yolu dosya değil: {video_path}")
    if video_path.stat().st_size == 0:
        raise InstagramUploadError("Video dosyası boş.")
    access_token = require_environment_variable("INSTAGRAM_ACCESS_TOKEN")
    account_id = require_environment_variable("INSTAGRAM_ACCOUNT_ID")
    video_hash = file_sha256(video_path)
    history = load_history(history_path)
    if already_uploaded(history, video_hash):
        print("Bu video daha önce Instagram'a yüklenmiş. Tekrar paylaşılmayacak.")
        return None
    caption = build_caption(script_path)
    print("=" * 68)
    print("COMIC FACTORY - INSTAGRAM REELS")
    print("=" * 68)
    print(f"Video: {video_path}")
    print(f"Boyut: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Açıklama uzunluğu: {len(caption)} karakter")
    cloudinary_public_id = ""
    try:
        cloudinary_public_id, video_url = upload_video_to_cloudinary(video_path)
        wait_until_video_is_public(video_url)
        container_id = create_reel_container(
            account_id=account_id,
            access_token=access_token,
            video_url=video_url,
            caption=caption,
        )
        wait_for_reel_processing(container_id=container_id, access_token=access_token)
        media_id = publish_reel(
            account_id=account_id, access_token=access_token, container_id=container_id
        )
        record_upload(
            history_path,
            history,
            video_path=video_path,
            video_hash=video_hash,
            media_id=media_id,
        )
        print("=" * 68)
        print("INSTAGRAM REEL TAMAMLANDI")
        print("=" * 68)
        return media_id
    finally:
        if cloudinary_public_id:
            delete_cloudinary_video(cloudinary_public_id)
