from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .config import ROOT
from .core import file_hash, save_json


def prepare_youtube_credentials() -> None:
    """Use existing files or the repository's existing JSON/base64 secrets."""
    names = {
        "client_secret.json": (
            "YOUTUBE_CLIENT_SECRET_B64",
            "YOUTUBE_CLIENT_SECRETS_B64",
            "GOOGLE_CLIENT_SECRET_B64",
            "CLIENT_SECRET_B64",
            "YOUTUBE_CLIENT_SECRET_JSON",
            "YOUTUBE_CLIENT_SECRET",
        ),
        "token.json": (
            "YOUTUBE_TOKEN_B64",
            "YOUTUBE_CREDENTIALS_B64",
            "TOKEN_B64",
            "YOUTUBE_TOKEN_JSON",
            "YOUTUBE_TOKEN",
        ),
    }
    for filename, variables in names.items():
        destination = ROOT / filename
        for variable in variables:
            value = os.getenv(variable, "").strip()
            if not value:
                continue
            try:
                if value.startswith("{"):
                    payload = json.loads(value)
                else:
                    payload = json.loads(
                        base64.b64decode("".join(value.split()), validate=True).decode(
                            "utf-8-sig"
                        )
                    )
                if not isinstance(payload, dict):
                    raise ValueError()
            except (ValueError, UnicodeError) as error:
                raise ValueError(
                    f"{variable} geçerli JSON veya base64 JSON değil."
                ) from error
            save_json(destination, payload)
            try:
                destination.chmod(0o600)
            except OSError:
                pass
            break
    token = ROOT / "token.json"
    legacy = ROOT / "youtube_credentials.json"
    if not token.exists() and legacy.exists():
        payload = json.loads(legacy.read_text(encoding="utf-8-sig"))
        if (
            isinstance(payload, dict)
            and payload.get("refresh_token")
            and payload.get("client_id")
        ):
            save_json(token, payload)
    # A complete authorized-user token already contains the client ID and secret.
    client_file = ROOT / "client_secret.json"
    if not client_file.exists() and token.exists():
        payload = json.loads(token.read_text(encoding="utf-8-sig"))
        if payload.get("client_id") and payload.get("client_secret"):
            save_json(
                client_file,
                {
                    "installed": {
                        "client_id": payload["client_id"],
                        "client_secret": payload["client_secret"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": payload.get(
                            "token_uri", "https://oauth2.googleapis.com/token"
                        ),
                        "redirect_uris": ["http://localhost"],
                    }
                },
            )


def validate_run(directory: Path) -> tuple[dict, Path, Path]:
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "ready" or not manifest.get("technical_passed"):
        raise ValueError("Bu çalışma yayın için hazır değil.")
    review = directory / "quality_review.json"
    if (
        manifest.get("schema") != 2
        or manifest.get("quality_passed") is not True
        or not review.is_file()
        or file_hash(review) != manifest.get("quality_sha256")
        or json.loads(review.read_text()).get("passed") is not True
    ):
        raise ValueError("Bu videonun görüntü/ses kontrol kaydı eksik veya değişmiş.")
    video, metadata = directory / "video.mp4", directory / "metadata.json"
    for path, key in ((video, "video_sha256"), (metadata, "metadata_sha256")):
        if not path.is_file() or file_hash(path) != manifest.get(key):
            raise ValueError(f"Önizleme dosyası değişmiş veya eksik: {path.name}")
    script = json.loads(metadata.read_text(encoding="utf-8"))
    if not script.get("script", {}).get("title") or not script.get("script", {}).get(
        "narration"
    ):
        raise ValueError("Videonun başlık veya anlatım bilgisi eksik.")
    return manifest, video, metadata


def publish_run(directory: Path, platforms: str) -> dict:
    manifest, video, metadata = validate_run(directory)
    selected = ["youtube", "instagram"] if platforms == "both" else [platforms]
    if any(item not in {"youtube", "instagram"} for item in selected):
        raise ValueError("Platform youtube, instagram veya both olmalı.")
    journal_path = ROOT / "data" / "publishing" / (manifest["video_sha256"] + ".json")
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    results = {}
    for platform in selected:
        previous = journal.get(platform, {})
        if previous.get("status") == "success":
            print(f"{platform}: daha önce yüklendi, tekrar gönderilmedi.")
            results[platform] = previous
            continue
        if previous.get("status") in {"sending", "uncertain"}:
            results[platform] = {
                "status": "blocked",
                "error": "Önceki gönderimin sonucu belirsiz. Platform hesabını kontrol et; data/publishing kaydını buna göre düzelt.",
            }
            continue
        sending = False
        try:
            if platform == "youtube":
                from .publishers import youtube

                prepare_youtube_credentials()
                youtube.VIDEO_FILE, youtube.SCRIPT_FILE = video, metadata
                old = youtube.find_previous_upload(manifest["video_sha256"])
                if old:
                    identifier = old["video_id"]
                else:
                    client = youtube.create_youtube_client()
                    title, description, tags = youtube.load_video_metadata()
                    journal[platform] = {"status": "sending"}
                    save_json(journal_path, journal)
                    sending = True
                    identifier = youtube.upload_video(client, title, description, tags)
                    journal[platform] = {"status": "success", "id": identifier}
                    save_json(journal_path, journal)
                    youtube.save_upload_history(
                        manifest["video_sha256"], identifier, title
                    )
                    try:
                        from .learning import record_publication

                        record_publication(directory, "youtube", identifier)
                    except Exception as error:
                        print(f"Performans kaydı yazılamadı: {type(error).__name__}: {error}")
            else:
                from .publishers import instagram

                # Fail missing configuration before starting any publication.
                for key in (
                    "INSTAGRAM_ACCESS_TOKEN",
                    "INSTAGRAM_ACCOUNT_ID",
                    "CLOUDINARY_CLOUD_NAME",
                    "CLOUDINARY_API_KEY",
                    "CLOUDINARY_API_SECRET",
                ):
                    instagram.require_environment_variable(key)
                old = instagram.load_history(instagram.DEFAULT_HISTORY_PATH)
                if instagram.already_uploaded(old, manifest["video_sha256"]):
                    identifier = "previously_uploaded"
                else:
                    journal[platform] = {"status": "sending"}
                    save_json(journal_path, journal)
                    sending = True
                    identifier = instagram.upload_reel(
                        video, metadata, instagram.DEFAULT_HISTORY_PATH
                    )
            results[platform] = {"status": "success", "id": identifier}
        except Exception as error:
            if journal.get(platform, {}).get("status") == "success":
                results[platform] = journal[platform]
            else:
                results[platform] = {
                    "status": "uncertain" if sending else "failed",
                    "error": str(error),
                }
        journal[platform] = results[platform]
        save_json(journal_path, journal)
    save_json(directory / "publication.json", results)
    if any(item.get("status") != "success" for item in results.values()):
        raise RuntimeError(
            "Yükleme tamamlanmadı: "
            + "; ".join(
                f"{key}: {value.get('error', value['status'])}"
                for key, value in results.items()
                if value.get("status") != "success"
            )
        )
    return results
