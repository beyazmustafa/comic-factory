# superhero-shorts/youtube_uploader.py

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

CLIENT_SECRET_FILE = PROJECT_DIRECTORY / "client_secret.json"
TOKEN_FILE = PROJECT_DIRECTORY / "token.json"

VIDEO_FILE = DATA_DIRECTORY / "videos" / "latest.mp4"
SCRIPT_FILE = DATA_DIRECTORY / "scripts" / "latest.json"

YOUTUBE_DIRECTORY = DATA_DIRECTORY / "youtube"
UPLOAD_HISTORY_FILE = YOUTUBE_DIRECTORY / "upload_history.json"

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")

DAILY_UPLOAD_TIMES = (
    (12, 0),
    (20, 0),
)

MAX_LOOKAHEAD_DAYS = 365
MAX_CHANNEL_VIDEOS = 200

CATEGORY_ID = "24"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class YouTubeUploaderError(RuntimeError):
    """YouTube yükleme veya planlama hatası."""


def parse_arguments() -> argparse.Namespace:
    """Komut satırı seçeneklerini okur."""
    parser = argparse.ArgumentParser(
        description=(
            "Comic Shorts videosunu Türkiye saatiyle "
            "12:00 / 20:00 slotlarına planlar."
        )
    )

    parser.add_argument(
        "--check-slot",
        action="store_true",
        help="Sadece ilk boş slotu kontrol eder.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Aynı videonun tekrar yüklenmesine izin verir.",
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    """Metni normalize eder."""
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip(),
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
        raise YouTubeUploaderError(
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


def validate_google_files() -> None:
    """Google OAuth dosyalarını kontrol eder."""
    if not CLIENT_SECRET_FILE.exists():
        raise YouTubeUploaderError(
            "client_secret.json bulunamadı."
        )


def credentials_have_scopes(
    credentials: Credentials,
) -> bool:
    """Tokenın gerekli YouTube yetkilerine sahip olduğunu kontrol eder."""
    try:
        return credentials.has_scopes(
            SCOPES
        )
    except Exception:
        return False


def load_credentials() -> Credentials:
    """OAuth kimlik bilgilerini yükler veya yeniler."""
    validate_google_files()

    credentials: Credentials | None = None

    if TOKEN_FILE.exists():
        try:
            credentials = Credentials.from_authorized_user_file(
                str(TOKEN_FILE),
                SCOPES,
            )
        except Exception:
            credentials = None

    if (
        credentials
        and credentials.valid
        and credentials_have_scopes(credentials)
    ):
        return credentials

    if (
        credentials
        and credentials.expired
        and credentials.refresh_token
        and credentials_have_scopes(credentials)
    ):
        try:
            credentials.refresh(
                Request()
            )

            TOKEN_FILE.write_text(
                credentials.to_json(),
                encoding="utf-8",
            )

            return credentials

        except RefreshError:
            credentials = None

    print()
    print(
        "Google YouTube yetkisi gerekiyor. "
        "Tarayıcı birazdan açılacak."
    )
    print()

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        SCOPES,
    )

    credentials = flow.run_local_server(
        port=0,
        open_browser=True,
        authorization_prompt_message=(
            "Google hesabını seçip YouTube iznini ver."
        ),
        success_message=(
            "YouTube bağlantısı başarılı. "
            "Bu tarayıcı sekmesini kapatabilirsin."
        ),
    )

    TOKEN_FILE.write_text(
        credentials.to_json(),
        encoding="utf-8",
    )

    return credentials


def create_youtube_client() -> Any:
    """YouTube Data API istemcisini oluşturur."""
    credentials = load_credentials()

    return build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def parse_youtube_datetime(
    value: str,
) -> datetime | None:
    """YouTube tarihini datetime nesnesine çevirir."""
    value = clean_text(value)

    if not value:
        return None

    try:
        if value.endswith("Z"):
            value = (
                value[:-1]
                + "+00:00"
            )

        result = datetime.fromisoformat(
            value
        )

        if result.tzinfo is None:
            result = result.replace(
                tzinfo=timezone.utc
            )

        return result

    except ValueError:
        return None


def get_channel_info(
    youtube: Any,
) -> dict[str, str]:
    """Bağlı kanalın temel bilgilerini döndürür."""
    response = youtube.channels().list(
        part="snippet,contentDetails",
        mine=True,
    ).execute()

    items = response.get(
        "items",
        [],
    )

    if not items:
        raise YouTubeUploaderError(
            "Bağlı YouTube kanalı bulunamadı."
        )

    item = items[0]

    playlist_id = (
        item.get(
            "contentDetails",
            {},
        )
        .get(
            "relatedPlaylists",
            {},
        )
        .get(
            "uploads",
            "",
        )
    )

    if not playlist_id:
        raise YouTubeUploaderError(
            "Uploads playlist bulunamadı."
        )

    return {
        "channel_id": clean_text(
            item.get(
                "id",
                "",
            )
        ),
        "channel_title": clean_text(
            item.get(
                "snippet",
                {},
            ).get(
                "title",
                "",
            )
        ),
        "uploads_playlist_id": playlist_id,
    }


def get_recent_video_ids(
    youtube: Any,
    playlist_id: str,
) -> list[str]:
    """Kanalın son yüklediği video ID'lerini toplar."""
    video_ids: list[str] = []
    page_token: str | None = None

    while len(video_ids) < MAX_CHANNEL_VIDEOS:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()

        for item in response.get(
            "items",
            [],
        ):
            video_id = clean_text(
                item.get(
                    "contentDetails",
                    {},
                ).get(
                    "videoId",
                    "",
                )
            )

            if video_id:
                video_ids.append(
                    video_id
                )

            if len(video_ids) >= MAX_CHANNEL_VIDEOS:
                break

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return video_ids


def chunk_list(
    values: list[str],
    size: int,
) -> list[list[str]]:
    """Listeyi küçük gruplara böler."""
    return [
        values[
            index:index + size
        ]
        for index in range(
            0,
            len(values),
            size,
        )
    ]


def get_future_scheduled_videos(
    youtube: Any,
    video_ids: list[str],
) -> list[dict[str, Any]]:
    """Geleceğe planlanmış videoları bulur."""
    now_utc = datetime.now(
        timezone.utc
    )

    scheduled: list[
        dict[str, Any]
    ] = []

    for video_group in chunk_list(
        video_ids,
        50,
    ):
        if not video_group:
            continue

        response = youtube.videos().list(
            part="snippet,status",
            id=",".join(
                video_group
            ),
        ).execute()

        for item in response.get(
            "items",
            [],
        ):
            status = item.get(
                "status",
                {},
            )

            publish_at = parse_youtube_datetime(
                status.get(
                    "publishAt",
                    "",
                )
            )

            if publish_at is None:
                continue

            if publish_at <= now_utc:
                continue

            scheduled.append(
                {
                    "video_id": clean_text(
                        item.get(
                            "id",
                            "",
                        )
                    ),
                    "title": clean_text(
                        item.get(
                            "snippet",
                            {},
                        ).get(
                            "title",
                            "",
                        )
                    ),
                    "publish_at_utc": publish_at,
                    "publish_at_local": publish_at.astimezone(
                        ISTANBUL_TIMEZONE
                    ),
                }
            )

    scheduled.sort(
        key=lambda item: item[
            "publish_at_utc"
        ]
    )

    return scheduled


def slot_is_occupied(
    slot: datetime,
    scheduled_videos: list[dict[str, Any]],
) -> bool:
    """Slotun dolu olup olmadığını kontrol eder."""
    for video in scheduled_videos:
        scheduled_time = video[
            "publish_at_local"
        ]

        difference_seconds = abs(
            (
                scheduled_time
                - slot
            ).total_seconds()
        )

        if difference_seconds < 60:
            return True

    return False


def find_next_slot(
    scheduled_videos: list[dict[str, Any]],
) -> datetime:
    """İlk boş Türkiye 12:00 / 20:00 slotunu bulur."""
    now = datetime.now(
        ISTANBUL_TIMEZONE
    )

    for day_offset in range(
        MAX_LOOKAHEAD_DAYS
    ):
        target_date = (
            now.date()
            + timedelta(
                days=day_offset
            )
        )

        for hour, minute in DAILY_UPLOAD_TIMES:
            slot = datetime(
                year=target_date.year,
                month=target_date.month,
                day=target_date.day,
                hour=hour,
                minute=minute,
                second=0,
                tzinfo=ISTANBUL_TIMEZONE,
            )

            if slot <= now:
                continue

            if slot_is_occupied(
                slot,
                scheduled_videos,
            ):
                continue

            return slot

    raise YouTubeUploaderError(
        "365 gün içinde boş slot bulunamadı."
    )


def print_slot_result(
    channel_info: dict[str, str],
    scheduled_videos: list[dict[str, Any]],
    next_slot: datetime,
) -> None:
    """Slot sonucunu terminale yazdırır."""
    print()
    print("=" * 78)
    print("YOUTUBE - PLANLI YAYIN SLOT KONTROLÜ")
    print("=" * 78)
    print()
    print(
        "Kanal: "
        + channel_info[
            "channel_title"
        ]
    )
    print()

    if scheduled_videos:
        print(
            "Gelecekte planlanmış videolar:"
        )
        print()

        for video in scheduled_videos:
            local_time = video[
                "publish_at_local"
            ]

            line = (
                "- "
                + local_time.strftime(
                    "%d.%m.%Y %H:%M"
                )
                + " | "
                + video[
                    "title"
                ]
            )

            print(line)

    else:
        print(
            "Gelecekte planlanmış video bulunamadı."
        )

    print()
    print(
        "SONRAKİ BOŞ SLOT:"
    )
    print(
        next_slot.strftime(
            "%d.%m.%Y %H:%M"
        )
        + " Türkiye"
    )
    print()


def load_video_metadata() -> tuple[
    str,
    str,
    list[str],
]:
    """Comic Factory senaryosundan YouTube metadata üretir."""
    payload = load_json(
        SCRIPT_FILE
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise YouTubeUploaderError(
            "data\\scripts\\latest.json bulunamadı."
        )

    script = payload.get(
        "script",
        {},
    )

    if not isinstance(
        script,
        dict,
    ):
        raise YouTubeUploaderError(
            "latest.json içinde script bulunamadı."
        )

    title = clean_text(
        script.get(
            "title",
            "",
        )
    )

    description = str(
        script.get(
            "full_description",
            "",
        )
        or script.get(
            "description",
            "",
        )
    ).strip()

    hashtags = script.get(
        "hashtags",
        [],
    )

    tags: list[str] = []

    if isinstance(
        hashtags,
        list,
    ):
        for hashtag in hashtags:
            tag = clean_text(
                hashtag
            ).lstrip(
                "#"
            )

            if (
                tag
                and tag not in tags
            ):
                tags.append(
                    tag
                )

    if not title:
        raise YouTubeUploaderError(
            "Video başlığı bulunamadı."
        )

    if len(title) > 100:
        title = (
            title[:97].rstrip()
            + "..."
        )

    if len(description) > 5000:
        description = description[
            :5000
        ]

    return (
        title,
        description,
        tags,
    )


def calculate_video_hash() -> str:
    """latest.mp4 SHA256 değerini hesaplar."""
    if not VIDEO_FILE.exists():
        raise YouTubeUploaderError(
            "data\\videos\\latest.mp4 bulunamadı."
        )

    digest = hashlib.sha256()

    with VIDEO_FILE.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_history() -> dict[str, Any]:
    """Yerel upload geçmişini yükler."""
    payload = load_json(
        UPLOAD_HISTORY_FILE,
        default={
            "uploads": [],
        },
    )

    if not isinstance(
        payload,
        dict,
    ):
        return {
            "uploads": [],
        }

    if not isinstance(
        payload.get(
            "uploads"
        ),
        list,
    ):
        payload[
            "uploads"
        ] = []

    return payload


def find_previous_upload(
    video_hash: str,
) -> dict[str, Any] | None:
    """Aynı dosyanın daha önce yüklenip yüklenmediğini kontrol eder."""
    history = load_history()

    for upload in history[
        "uploads"
    ]:
        if not isinstance(
            upload,
            dict,
        ):
            continue

        if clean_text(
            upload.get(
                "video_sha256",
                "",
            )
        ) == video_hash:
            return upload

    return None


def save_upload_history(
    video_hash: str,
    video_id: str,
    title: str,
    slot: datetime,
) -> None:
    """Başarılı upload bilgisini kaydeder."""
    history = load_history()

    history[
        "uploads"
    ].append(
        {
            "video_sha256": video_hash,
            "video_id": video_id,
            "title": title,
            "publish_at": slot.isoformat(),
            "uploaded_at": datetime.now(
                ISTANBUL_TIMEZONE
            ).isoformat(),
        }
    )

    save_json(
        UPLOAD_HISTORY_FILE,
        history,
    )


def to_rfc3339_utc(
    local_datetime: datetime,
) -> str:
    """Yerel tarihi YouTube publishAt formatına dönüştürür."""
    utc_datetime = local_datetime.astimezone(
        timezone.utc
    )

    return (
        utc_datetime
        .replace(
            microsecond=0
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def upload_video(
    youtube: Any,
    title: str,
    description: str,
    tags: list[str],
    slot: datetime,
) -> str:
    """Video dosyasını planlı olarak YouTube'a yükler."""
    if not VIDEO_FILE.exists():
        raise YouTubeUploaderError(
            "latest.mp4 bulunamadı."
        )

    snippet: dict[str, Any] = {
        "title": title,
        "description": description,
        "categoryId": CATEGORY_ID,
        "defaultLanguage": "tr",
        "defaultAudioLanguage": "tr",
    }

    if tags:
        snippet[
            "tags"
        ] = tags[:25]

    body = {
        "snippet": snippet,
        "status": {
            "privacyStatus": "private",
            "publishAt": to_rfc3339_utc(
                slot
            ),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(VIDEO_FILE),
        mimetype="video/mp4",
        chunksize=8 * 1024 * 1024,
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
        notifySubscribers=True,
    )

    print()
    print("=" * 78)
    print("YOUTUBE YÜKLEME")
    print("=" * 78)
    print()
    print(
        "Başlık: "
        + title
    )
    print(
        "Planlanan yayın: "
        + slot.strftime(
            "%d.%m.%Y %H:%M"
        )
        + " Türkiye"
    )
    print()

    response = None

    while response is None:
        upload_status, response = request.next_chunk()

        if upload_status is not None:
            percentage = int(
                upload_status.progress()
                * 100
            )

            print(
                f"Yükleme: %{percentage}"
            )

    video_id = clean_text(
        response.get(
            "id",
            "",
        )
    )

    if not video_id:
        raise YouTubeUploaderError(
            "YouTube video ID döndürmedi."
        )

    return video_id


def main() -> None:
    """Slot kontrolünü veya YouTube upload işlemini çalıştırır."""
    arguments = parse_arguments()

    print()
    print("=" * 78)
    print("COMIC EVENTS - YOUTUBE UPLOADER")
    print("=" * 78)

    try:
        youtube = create_youtube_client()

        channel_info = get_channel_info(
            youtube
        )

        video_ids = get_recent_video_ids(
            youtube,
            channel_info[
                "uploads_playlist_id"
            ],
        )

        scheduled_videos = get_future_scheduled_videos(
            youtube,
            video_ids,
        )

        next_slot = find_next_slot(
            scheduled_videos
        )

        print_slot_result(
            channel_info,
            scheduled_videos,
            next_slot,
        )

        if arguments.check_slot:
            print(
                "✓ Sadece slot kontrolü yapıldı."
            )
            print(
                "✓ Video yüklenmedi."
            )
            return

        title, description, tags = load_video_metadata()

        video_hash = calculate_video_hash()

        previous_upload = find_previous_upload(
            video_hash
        )

        if (
            previous_upload is not None
            and not arguments.force
        ):
            print()
            print(
                "BU VIDEO DAHA ÖNCE YÜKLENMİŞ."
            )
            print(
                "Video ID: "
                + clean_text(
                    previous_upload.get(
                        "video_id",
                        "",
                    )
                )
            )
            print()
            print(
                "Tekrar yükleme yapılmadı."
            )
            return

        video_id = upload_video(
            youtube=youtube,
            title=title,
            description=description,
            tags=tags,
            slot=next_slot,
        )

        save_upload_history(
            video_hash=video_hash,
            video_id=video_id,
            title=title,
            slot=next_slot,
        )

        print()
        print("=" * 78)
        print("YOUTUBE PLANLAMASI BAŞARILI")
        print("=" * 78)
        print()
        print(
            "Video ID: "
            + video_id
        )
        print(
            "Yayın zamanı: "
            + next_slot.strftime(
                "%d.%m.%Y %H:%M"
            )
            + " Türkiye"
        )
        print()
        print(
            "Video planlanan saatte otomatik yayınlanacak."
        )

    except HttpError as error:
        print()
        print("=" * 78)
        print("YOUTUBE API HATASI")
        print("=" * 78)
        print()
        print(error)

        raise SystemExit(1)

    except Exception as error:
        print()
        print("=" * 78)
        print("HATA")
        print("=" * 78)
        print()
        print(
            f"{type(error).__name__}: {error}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
