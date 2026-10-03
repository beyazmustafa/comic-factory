from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
CLIENT_SECRET_FILE = PROJECT_DIRECTORY / "client_secret.json"
TOKEN_FILE = PROJECT_DIRECTORY / "token.json"
VIDEO_FILE = DATA_DIRECTORY / "videos" / "latest.mp4"
SCRIPT_FILE = DATA_DIRECTORY / "scripts" / "latest.json"
YOUTUBE_DIRECTORY = DATA_DIRECTORY / "youtube"
UPLOAD_HISTORY_FILE = YOUTUBE_DIRECTORY / "upload_history.json"
ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
CATEGORY_ID = "24"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
# Optional: watch-time / retention from YouTube Analytics. Older tokens without
# it keep working for uploads; learning simply gets fewer signals.
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"


class YouTubeUploaderError(RuntimeError):
    """YouTube yükleme işlemi başarısız olduğunda oluşur."""


def clean_text(value: Any) -> str:
    """Metni normalize eder."""
    return re.sub("\\s+", " ", str(value or "").strip())


def load_json(file_path: Path, default: Any = None) -> Any:
    """JSON dosyasını yükler."""
    if not file_path.exists():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise YouTubeUploaderError(f"JSON okunamadı: {file_path}") from error


def save_json(file_path: Path, payload: Any) -> None:
    """JSON dosyasını kaydeder."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_file.replace(file_path)


def validate_google_files() -> None:
    """Google OAuth dosyalarının varlığını kontrol eder."""
    if not CLIENT_SECRET_FILE.exists():
        raise YouTubeUploaderError("client_secret.json bulunamadı.")
    if os.getenv("CI") and (not TOKEN_FILE.exists()):
        raise YouTubeUploaderError("GitHub Actions içinde token.json bulunamadı.")


def credentials_have_scopes(credentials: Credentials) -> bool:
    """OAuth tokenının gerekli yetkilere sahip olduğunu kontrol eder."""
    try:
        return credentials.has_scopes(SCOPES)
    except Exception:
        return False


def save_credentials(credentials: Credentials) -> None:
    """OAuth tokenını kaydeder."""
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")


def load_credentials() -> Credentials:
    """OAuth kimlik bilgilerini yükler veya yeniler."""
    validate_google_files()
    credentials: Credentials | None = None
    if TOKEN_FILE.exists():
        try:
            credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
            try:
                granted = set(json.loads(TOKEN_FILE.read_text(encoding="utf-8-sig")).get("scopes") or [])
                if ANALYTICS_SCOPE in granted:
                    credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES + [ANALYTICS_SCOPE])
            except Exception:
                pass
        except Exception:
            credentials = None
    if credentials and credentials.valid and credentials_have_scopes(credentials):
        return credentials
    if (
        credentials
        and credentials.expired
        and credentials.refresh_token
        and credentials_have_scopes(credentials)
    ):
        try:
            credentials.refresh(Request())
            save_credentials(credentials)
            return credentials
        except RefreshError as error:
            if os.getenv("CI"):
                raise YouTubeUploaderError(
                    "YouTube OAuth token yenilenemedi. Yerel bilgisayarda tekrar yetkilendirip YOUTUBE_TOKEN_B64 secretını güncelle."
                ) from error
            credentials = None
    if os.getenv("CI"):
        raise YouTubeUploaderError(
            "GitHub Actions içinde geçerli YouTube OAuth tokenı bulunamadı."
        )
    print()
    print("Google YouTube yetkisi gerekiyor. Tarayıcı açılacak.")
    print()
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
    credentials = flow.run_local_server(
        port=0,
        open_browser=True,
        authorization_prompt_message="Google hesabını seçip YouTube iznini ver.",
        success_message="YouTube bağlantısı başarılı. Bu sekmeyi kapatabilirsin.",
    )
    save_credentials(credentials)
    return credentials


def create_youtube_client() -> Any:
    """YouTube Data API istemcisini oluşturur."""
    credentials = load_credentials()
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def create_analytics_client() -> Any | None:
    """YouTube Analytics istemcisi; token bu yetkiyi taşımıyorsa None."""
    credentials = load_credentials()
    if not credentials.has_scopes([ANALYTICS_SCOPE]):
        return None
    return build("youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False)


def video_retention(analytics: Any, video_ids: list[str], start_date: str, end_date: str) -> dict[str, dict]:
    """averageViewPercentage / averageViewDuration / views per video (lifetime window)."""
    if not video_ids:
        return {}
    response = (
        analytics.reports()
        .query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,averageViewDuration,averageViewPercentage,likes,shares",
            dimensions="video",
            filters="video==" + ",".join(video_ids[:200]),
        )
        .execute()
    )
    columns = [c["name"] for c in response.get("columnHeaders", [])]
    result = {}
    for row in response.get("rows", []):
        record = dict(zip(columns, row))
        result[str(record.get("video"))] = {
            "average_view_percentage": record.get("averageViewPercentage"),
            "average_view_duration": record.get("averageViewDuration"),
            "shares": record.get("shares"),
        }
    return result


def get_channel_info(youtube: Any) -> dict[str, str]:
    """Bağlı YouTube kanalının temel bilgilerini döndürür."""
    response = youtube.channels().list(part="snippet", mine=True).execute()
    items = response.get("items", [])
    if not items:
        raise YouTubeUploaderError("Bağlı YouTube kanalı bulunamadı.")
    item = items[0]
    return {
        "channel_id": clean_text(item.get("id", "")),
        "channel_title": clean_text(item.get("snippet", {}).get("title", "")),
    }


def load_video_metadata() -> tuple[str, str, list[str]]:
    """Comic Factory senaryosundan YouTube metadata üretir."""
    payload = load_json(SCRIPT_FILE)
    if not isinstance(payload, dict):
        raise YouTubeUploaderError("data\\scripts\\latest.json bulunamadı.")
    script = payload.get("script", {})
    if not isinstance(script, dict):
        raise YouTubeUploaderError("latest.json içinde script bulunamadı.")
    title = clean_text(script.get("title", ""))
    description = str(
        script.get("full_description", "") or script.get("description", "")
    ).strip()
    hashtags = script.get("hashtags", [])
    tags: list[str] = []
    if isinstance(hashtags, list):
        for hashtag in hashtags:
            tag = clean_text(hashtag).lstrip("#")
            if tag and tag not in tags:
                tags.append(tag)
    shorts_tags = ["shorts", "comics", "comic books", "superhero", "golden age comics", "comic recap"]
    for tag in shorts_tags:
        if tag not in tags:
            tags.append(tag)
    if not title:
        raise YouTubeUploaderError("Video başlığı bulunamadı.")
    # Shorts eligibility: vertical video + #Shorts in title/description.
    if "#shorts" not in title.lower():
        title = (title[:88].rstrip() + "..." if len(title) > 91 else title) + " #Shorts"
    if "#shorts" not in description.lower():
        description = "#Shorts #comics #superhero\n\n" + description
    if len(description) > 5000:
        description = description[:5000]
    return (title, description, tags)


def calculate_video_hash() -> str:
    """latest.mp4 SHA256 değerini hesaplar."""
    if not VIDEO_FILE.exists():
        raise YouTubeUploaderError("data\\videos\\latest.mp4 bulunamadı.")
    if VIDEO_FILE.stat().st_size == 0:
        raise YouTubeUploaderError("latest.mp4 boş.")
    digest = hashlib.sha256()
    with VIDEO_FILE.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_history() -> dict[str, Any]:
    """YouTube upload geçmişini yükler."""
    payload = load_json(UPLOAD_HISTORY_FILE, default={"uploads": []})
    if not isinstance(payload, dict):
        return {"uploads": []}
    if not isinstance(payload.get("uploads"), list):
        payload["uploads"] = []
    return payload


def find_previous_upload(video_hash: str) -> dict[str, Any] | None:
    """Aynı videonun daha önce yüklenip yüklenmediğini kontrol eder."""
    history = load_history()
    for upload in history["uploads"]:
        if not isinstance(upload, dict):
            continue
        if clean_text(upload.get("video_sha256", "")) == video_hash:
            return upload
    return None


def visibility() -> str:
    """YOUTUBE_VISIBILITY repo variable: unlisted (default), public or private."""
    value = clean_text(os.getenv("YOUTUBE_VISIBILITY", "")).lower() or "unlisted"
    if value not in {"public", "unlisted", "private"}:
        raise YouTubeUploaderError("YOUTUBE_VISIBILITY public, unlisted veya private olmalı.")
    return value


def save_upload_history(video_hash: str, video_id: str, title: str) -> None:
    """Başarılı yüklemeyi geçmişe kaydeder."""
    history = load_history()
    uploads = history.setdefault("uploads", [])
    if not isinstance(uploads, list):
        uploads = []
        history["uploads"] = uploads
    uploads.append(
        {
            "video_sha256": video_hash,
            "video_id": video_id,
            "title": title,
            "privacy_status": visibility(),
            "uploaded_at": datetime.now(ISTANBUL_TIMEZONE).isoformat(),
        }
    )
    history["uploads"] = uploads[-200:]
    save_json(UPLOAD_HISTORY_FILE, history)


def set_thumbnail(youtube: Any, video_id: str, path: Path) -> bool:
    """Custom thumbnail (hook card). Needs a phone-verified channel; a refusal
    is logged, never fatal — the upload itself already succeeded."""
    if not path.is_file():
        return False
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(path), mimetype="image/jpeg"),
        ).execute()
        print("Kapak görseli yüklendi.")
        return True
    except Exception as error:  # noqa: BLE001
        text = str(error)
        hint = (" (Kanal telefonla doğrulanmamış: YouTube Studio → Ayarlar → Kanal → Özellik uygunluğu)"
                if "403" in text or "forbidden" in text.casefold() else "")
        print(f"Kapak görseli yüklenemedi: {type(error).__name__}: {text[:200]}{hint}")
        return False


def upload_video(youtube: Any, title: str, description: str, tags: list[str]) -> str:
    """Videoyu YouTube'a YOUTUBE_VISIBILITY görünürlüğüyle yükler."""
    if not VIDEO_FILE.exists():
        raise YouTubeUploaderError("latest.mp4 bulunamadı.")
    snippet: dict[str, Any] = {
        "title": title,
        "description": description,
        "categoryId": CATEGORY_ID,
        "defaultLanguage": os.getenv("FACTORY_LANGUAGE", "en"),
        "defaultAudioLanguage": os.getenv("FACTORY_LANGUAGE", "en"),
    }
    if tags:
        snippet["tags"] = tags[:25]
    body = {
        "snippet": snippet,
        "status": {
            "privacyStatus": visibility(),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(
        str(VIDEO_FILE), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True
    )
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
        notifySubscribers=visibility() == "public",
    )
    print()
    print("=" * 78)
    print("YOUTUBE YÜKLEME")
    print("=" * 78)
    print()
    print(f"Başlık: {title}")
    print(f"Görünürlük: {visibility()}")
    print()
    response = None
    while response is None:
        upload_status, response = request.next_chunk(num_retries=3)
        if upload_status is not None:
            percentage = int(upload_status.progress() * 100)
            print(f"Yükleme: %{percentage}")
    video_id = clean_text(response.get("id", ""))
    if not video_id:
        raise YouTubeUploaderError("YouTube video ID döndürmedi.")
    return video_id
