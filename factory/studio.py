from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, Settings
from .core import file_hash, save_json


def select_and_build(engine, client, settings, topic: str):
    """Research and video attempts have separate finite limits."""
    rejected, queue = set(), []
    research_rounds = attempts = 0
    while attempts < (1 if topic else settings.max_events):
        engine.check_budget()
        if not queue:
            if research_rounds >= settings.max_research_rounds:
                break
            research_rounds += 1
            print(f"Konu araştırması {research_rounds}/{settings.max_research_rounds}")
            queue = engine.research_events(client, 4, rejected)
            if not queue:
                continue
        event = queue.pop(0)
        key = engine.event_key(event)
        if key in rejected:
            continue
        attempts += 1
        try:
            archive = engine.build_single_event_video(
                client,
                event,
                max_images=settings.max_images,
                subtitle_offset=settings.subtitle_offset,
                enable_ai_reconstruction=settings.ai_reconstruction,
            )
            return engine.load_active_event(), archive
        except engine.EventRejectedError as error:
            rejected.add(key)
            engine.record_rejected_event(event, str(error))
            print(f"Konu için üretim tamamlanamadı: {error}")
            if topic:
                raise  # A requested Thor story must never turn into a Batman video.
    raise RuntimeError(
        f"Uygun video üretilemedi: {research_rounds} araştırma, {attempts} üretim denemesi. Ayrıntılar tanı dosyalarında."
    )


def write_review(directory: Path, manifest: dict) -> None:
    storyboard = json.loads((directory / "storyboard.json").read_text())
    rows = "".join(
        f"<tr><td>{int(scene['scene_number'])}</td><td>{html.escape(scene['narration'])}</td><td>{html.escape(scene['visual_description'])}</td></tr>"
        for scene in storyboard["scenes"]
    )
    title = html.escape(manifest.get("title", "Comic Factory"))
    document = f"""<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{background:#11151d;color:#eee;font:17px/1.6 system-ui;max-width:1120px;margin:40px auto;padding:0 24px}}h1{{line-height:1.2}}.lead{{color:#f6c44c}}video{{max-height:720px;max-width:100%;background:#000;border-radius:16px}}img{{max-width:100%;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #303641;vertical-align:top}}small{{color:#aab4c5}}a{{color:#f6c44c}}</style>
<p class="lead">COMIC FACTORY · ÖNİZLEME</p><h1>{title}</h1>
<p><small>Çalışma: {html.escape(manifest["run_id"])} · Teknik kontrol geçti. Editoryal kaliteyi videoyu izleyerek değerlendir.</small></p>
<video controls preload="metadata" src="video.mp4"></video>
<h2>Sahne görünümü</h2><img src="contact_sheet.jpg" alt="Sahnelerin genel görünümü">
<h2>Senaryo</h2><table><tr><th>Sahne</th><th>Anlatım</th><th>Görsel</th></tr>{rows}</table>
<p><a href="metadata.json">Başlık ve kaynaklar</a> · <a href="checkpoints.json">Kontrol sonuçları</a></p></html>"""
    (directory / "review.html").write_text(document, encoding="utf-8")


def collect_diagnostics(engine, directory: Path) -> None:
    pairs = [
        (engine.CHECKPOINT_FILE, "checkpoints.json"),
        (engine.SCRIPT_DIR / "storyboard.json", "storyboard.json"),
        (engine.WORK_DIR / "contact_sheet.jpg", "contact_sheet.jpg"),
        (engine.ALIGNED_WORDS_FILE, "aligned_words.json"),
        (engine.LATEST_WORDS_FILE, "word_timestamps.json"),
        (engine.WORK_DIR / "precise.ass", "captions.ass"),
    ]
    for source, name in pairs:
        if source.exists():
            shutil.copy2(source, directory / name)


def build(settings: Settings, topic: str, panel_dir: Path | None = None) -> Path:
    from . import engine

    engine.configure(settings, topic, panel_dir)
    engine.ensure_dirs()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    directory = ROOT / "data" / "runs" / run_id
    directory.mkdir(parents=True)
    # Write attempts directly into this run so failures survive artifact collection.
    engine.AUDIO_DIAGNOSTICS_DIR = directory / "audio_diagnostics"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "building",
        "topic": topic,
        "settings": settings.to_dict(),
    }
    save_json(directory / "run.json", manifest)
    # These are derived outputs, never the durable event/upload history.
    for path in (
        engine.LATEST_SCRIPT_FILE,
        engine.LATEST_VIDEO_FILE,
        engine.LATEST_AUDIO_FILE,
        engine.LATEST_WORDS_FILE,
        engine.ALIGNED_WORDS_FILE,
        engine.SCRIPT_DIR / "storyboard.json",
    ):
        path.unlink(missing_ok=True)
    shutil.rmtree(engine.WORK_DIR, ignore_errors=True)
    engine.WORK_DIR.mkdir(parents=True)
    try:
        with engine.gemini_client() as client:
            event, archive = select_and_build(engine, client, settings, topic)
        engine.CHECKPOINTS.persist_lessons()
        engine.CHECKPOINTS.save()
        collect_diagnostics(engine, directory)
        shutil.copy2(archive, directory / "video.mp4")
        shutil.copy2(engine.LATEST_SCRIPT_FILE, directory / "metadata.json")
        shutil.copy2(engine.LATEST_AUDIO_FILE, directory / "narration.wav")
        asset_manifest = engine.ASSET_DIR / event["id"] / "visual_manifest.json"
        shutil.copy2(asset_manifest, directory / "scenes.json")
        metadata = json.loads((directory / "metadata.json").read_text())
        manifest.update(
            status="ready",
            title=metadata["script"]["title"],
            event=event,
            technical_passed=True,
            api_calls=engine.API_CALLS,
            video_sha256=file_hash(directory / "video.mp4"),
            metadata_sha256=file_hash(directory / "metadata.json"),
        )
        write_review(directory, manifest)
        save_json(directory / "run.json", manifest)
        engine.mark_used(event)
        save_json(
            ROOT / "data" / "latest_run.json",
            {"directory": str(directory.relative_to(ROOT)), "run_id": run_id},
        )
        if output_path := os.getenv("GITHUB_OUTPUT"):
            with open(output_path, "a", encoding="utf-8") as file:
                file.write(
                    f"run_directory={directory.relative_to(ROOT).as_posix()}\nrun_id={run_id}\n"
                )
        print(f"Video ve önizleme hazır: {directory}")
        return directory
    except BaseException as error:
        engine.CHECKPOINTS.persist_lessons()
        engine.CHECKPOINTS.save()
        collect_diagnostics(engine, directory)
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            api_calls=engine.API_CALLS,
        )
        save_json(directory / "run.json", manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comic Factory — tek üretim ve yayın ortamı"
    )
    parser.add_argument(
        "--check", action="store_true", help="Kurulum kontrolü; API çağrısı yapmaz"
    )
    parser.add_argument(
        "--topic", default="", help="İstenen çizgi roman olayı; boşsa otomatik"
    )
    parser.add_argument("--duration", type=int, help="Hedef saniye, 20–120")
    parser.add_argument("--scenes", type=int, help="Sahne sayısı, 4–20")
    parser.add_argument("--brief", help="Anlatım ve üslup isteği")
    parser.add_argument("--panel-dir", type=Path, help="Kendi çizgi roman görsellerin")
    parser.add_argument(
        "--allow-ai",
        action="store_true",
        default=None,
        help="Eksik panel için yapay görsel üretimine izin ver",
    )
    parser.add_argument(
        "--publish", action="store_true", help="Yeni üretilen videoyu yükle"
    )
    parser.add_argument(
        "--publish-run",
        type=Path,
        help="Önceden oluşturulan önizleme klasörünü yükle; tekrar üretmez",
    )
    parser.add_argument(
        "--platforms", choices=["youtube", "instagram", "both"], default="both"
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        settings = Settings.load(
            args.config,
            target_seconds=args.duration,
            scene_count=args.scenes,
            brief=args.brief,
            ai_reconstruction=args.allow_ai,
        )
        if args.check:
            from . import engine

            engine.configure(settings)
            if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
                raise ValueError("FFmpeg ve FFprobe kurulmalı.")
            from .core import caption_font

            caption_font(60)
            filters = engine.subprocess.run(
                [engine.ffmpeg_path(), "-filters"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if " ass " not in filters.stdout:
                raise ValueError("FFmpeg kurulumu ASS altyazı desteği içermiyor.")
            print("Kod, paketler, FFmpeg, FFprobe, Türkçe font ve ayarlar hazır.")
            for key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
                print(
                    f"{key}: {'mevcut' if os.getenv(key) else 'bu ortamda ayarlanmamış'}"
                )
            return 0
        if args.publish_run:
            from .publishing import publish_run

            publish_run(args.publish_run.resolve(), args.platforms)
            return 0
        if args.panel_dir and not args.panel_dir.is_dir():
            raise ValueError("Panel klasörü bulunamadı.")
        directory = build(
            settings, args.topic, args.panel_dir.resolve() if args.panel_dir else None
        )
        if args.publish:
            from .publishing import publish_run

            publish_run(directory, args.platforms)
        return 0
    except Exception as error:
        print(f"HATA: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
