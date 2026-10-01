import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from . import archive, research, panels, story, voice, render, quality
from .style import load_style
from .api import Api, FactoryError, SourceUnavailable
from .config import ROOT, VERSION, Settings
from .core import file_hash, save_json
from .publishing import publish_run
from .review import write_review
from .state import Checkpoints, signature, resume


def emit_directory(directory):
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"run_directory={directory.resolve()}\n")


def load_used():
    used = set()
    paths = list((ROOT / "data" / "events").rglob("*.json")) + list(
        (ROOT / "data" / "history").rglob("*.json")
    )
    for path in paths:
        try:
            data = json.loads(path.read_text())
            rows = data if isinstance(data, list) else data.get("events", [data])
            if isinstance(rows, dict):
                rows = list(rows.values())
            for item in rows:
                if isinstance(item, dict):
                    for key in ("id", "event_id", "title", "event_title", "identifier"):
                        if item.get(key):
                            used.add(str(item[key]).casefold())
        except (ValueError, TypeError, AttributeError):
            continue
    return sorted(used)


def generate(settings, topic="", resume_run=None, voice_only=False, api_factory=Api):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "data" / "runs" / stamp
    directory.mkdir(parents=True)
    emit_directory(directory)
    manifest = {
        "schema": 2,
        "version": VERSION,
        "status": "building",
        "stage": "setup",
        "topic": topic,
        "settings": settings.to_dict(),
        "run_id": os.getenv("GITHUB_RUN_ID", stamp),
    }
    cache = ROOT / "data" / "cache-v2"
    api = None

    def stage(name):
        manifest.update(stage=name, status="building")
        save_json(directory / "run.json", manifest)
        print("\n→ " + name, flush=True)

    try:
        if resume_run:
            previous = resume(Path(resume_run), directory)
            if topic and topic != previous.get("topic", ""):
                raise FactoryError(
                    "Devam ederken konu değiştirilemez; yeni konu için resume_run_id boş olmalı."
                )
            topic = previous.get("topic", "")
            manifest.update(topic=topic, resumed_from=previous.get("run_id"))
        save_json(directory / "run.json", manifest)
        if not os.getenv("GROQ_API_KEY", "").strip():
            raise FactoryError("GROQ_API_KEY bu çalışmaya aktarılmamış.")
        checkpoints = Checkpoints(directory)
        api = api_factory(settings, directory)
        stage("Sabit kurgu profili")
        style = load_style()
        save_json(directory / "editing_profile.json", style)
        if voice_only:
            stage("Türkçe ses karşılaştırması")
            voice.select_voice(api, style, cache)
            manifest.update(
                status="voice_ready", stage="Tamamlandı", api_calls=api.calls
            )
            save_json(directory / "run.json", manifest)
            return directory
        stage("Olay, gerçek sayfalar ve Türkçe senaryo")
        story_key = signature(
            [
                topic,
                settings.target_seconds,
                settings.max_shots,
                settings.panel_threshold,
                settings.gemini_model,
                style,
            ]
        )
        bundle = checkpoints.read("story", story_key)
        if bundle is None:
            used = load_used()
            from_archive = settings.source == "archive"
            candidates = (
                archive.shortlist(api, topic, used)
                if from_archive
                else research.shortlist(api, topic, used)
            )
            fetcher = research.Fetcher()
            failures = []
            try:
                for event in candidates:
                    print("Olay araştırılıyor: " + event["title"], flush=True)
                    try:
                        if from_archive:
                            pages, articles = archive.collect_pages(api, event)
                            inventory, facts = panels.catalog_archive(
                                api, event, pages, articles
                            )
                        else:
                            pages, articles = research.collect_pages(api, event, fetcher)
                            inventory, facts = panels.catalog(api, event, pages, articles)
                        script = story.create(api, event, inventory, facts, style)
                        bundle = {
                            "event": event,
                            "story": script,
                            "panels": inventory,
                            "facts": facts,
                        }
                        save_json(directory / "story_bundle.json", bundle)
                        files = [
                            directory / "story.json",
                            directory / "story_bundle.json",
                        ] + [
                            p
                            for p in (directory / "events" / event["id"]).rglob("*")
                            if p.is_file()
                        ]
                        checkpoints.save("story", story_key, bundle, files)
                        break
                    except SourceUnavailable as error:
                        failures.append({"event": event, "error": str(error)})
                        save_json(
                            directory / "research" / "rejected_candidates.json",
                            failures,
                        )
                if bundle is None:
                    raise SourceUnavailable(
                        "Aday olaylarda yeterli doğrulanmış panel/anlatım eşleşmesi bulunamadı; kaynak raporları saklandı."
                    )
            finally:
                fetcher.session.close()
        script, inventory = bundle["story"], bundle["panels"]
        story.validate_story(script, inventory, bundle["facts"], settings.max_shots)
        stage("Türkçe ses seçimi")
        voice_key = signature(
            [
                settings.voice,
                settings.tts_model,
                settings.whisper_model,
                settings.alignment_threshold,
                style,
            ]
        )
        selection = checkpoints.read("voice", voice_key)
        if selection is None:
            selection = voice.select_voice(api, style, cache)
            checkpoints.save(
                "voice",
                voice_key,
                selection,
                [directory / "voice_selection.json"]
                + [p for p in (directory / "voice_audition").rglob("*") if p.is_file()],
            )
        stage("Konuşma ve kelime zamanları")
        audio_key = signature(
            [
                script["narration"],
                selection["voice"],
                settings.tts_model,
                settings.whisper_model,
                settings.alignment_threshold,
                settings.chunk_words,
                style.get("narrator_delivery"),
            ]
        )
        timing = checkpoints.read("audio", audio_key)
        audio_path = directory / "narration.wav"
        if timing is None:
            audio_path, words, duration = voice.build_audio(
                api, script, selection["voice"], style, cache
            )
            timing = {"words": words, "duration": duration}
            checkpoints.save(
                "audio",
                audio_key,
                timing,
                [audio_path, directory / "aligned_words.json"],
            )
        stage("Dikey video ve eşzamanlı altyazı")
        music = Path(settings.music_file) if settings.music_file else None
        if music and not music.is_absolute():
            music = ROOT / music
        render_key = signature(
            [
                script,
                inventory,
                style,
                audio_key,
                settings.caption_offset,
                settings.music_gain_db,
                file_hash(music) if music and music.exists() else settings.music_file,
            ]
        )
        result = checkpoints.read("render", render_key)
        if result is None:
            video, technical = render.build(
                api,
                script,
                inventory,
                style,
                audio_path,
                timing["words"],
                timing["duration"],
            )
            result = {"technical": technical}
            checkpoints.save(
                "render",
                render_key,
                result,
                [
                    video,
                    directory / "captions.ass",
                    directory / "timeline.json",
                    directory / "technical_check.json",
                ],
            )
        video = directory / "video.mp4"
        stage("Bitmiş videonun görüntü ve ses kontrolü")
        quality_key = signature([file_hash(video), style, settings.gemini_model])
        report = checkpoints.read("quality", quality_key)
        if report is None:
            report = quality.review(api, video, script, style)
            corrected = (
                quality.adjusted_style(style, report) if not report["passed"] else None
            )
            if corrected:
                save_json(directory / "quality_review_before_repair.json", report)
                save_json(directory / "render_style.json", corrected)
                video, technical = render.build(
                    api,
                    script,
                    inventory,
                    corrected,
                    audio_path,
                    timing["words"],
                    timing["duration"],
                )
                result = {"technical": technical}
                checkpoints.save(
                    "render",
                    render_key,
                    result,
                    [
                        video,
                        directory / "captions.ass",
                        directory / "timeline.json",
                        directory / "technical_check.json",
                        directory / "render_style.json",
                    ],
                )
                quality_key = signature(
                    [file_hash(video), style, settings.gemini_model]
                )
                report = quality.review(api, video, script, style)
            if report["passed"]:
                checkpoints.save(
                    "quality", quality_key, report, [directory / "quality_review.json"]
                )
        if not report["passed"]:
            raise FactoryError(
                "Bitmiş videoda düzeltilmesi gereken noktalar var: "
                + ", ".join(report["failed_checks"])
                + ". Önizleme ve raporlar kaydedildi."
            )
        used_panels = {s["panel_id"] for s in script["shots"]}
        sources = sorted({p["source_url"] for p in inventory if p["id"] in used_panels})
        event = bundle["event"]
        licence = (
            "\nSayfalar: Internet Archive, kamu malı (public domain) olarak işaretlenmiş sayı."
            if settings.source == "archive"
            else ""
        )
        description = (
            script["description"]
            + f"\n\nÇizgi roman: {event['series']} #{event['issue']} ({event['year']})"
            + licence
            + "\nKaynaklar:\n"
            + "\n".join(sources)
        )
        hashtags = [
            "çizgiroman",
            "shorts",
            str(event.get("publisher", "comics")).lower().replace(" ", ""),
        ]
        metadata = {
            "script": {
                **script,
                "full_description": description,
                "hashtags": hashtags,
                "instagram_caption": script["title"] + "\n\n" + script["description"],
            },
            "sources": sources,
            "voice": selection,
            "editing_profile": style["profile_id"],
        }
        save_json(directory / "metadata.json", metadata)
        manifest.update(
            status="ready",
            stage="Tamamlandı",
            technical_passed=True,
            quality_passed=True,
            video_sha256=file_hash(video),
            metadata_sha256=file_hash(directory / "metadata.json"),
            quality_sha256=file_hash(directory / "quality_review.json"),
            api_calls=api.calls,
            duration=timing["duration"],
            voice=selection["voice"],
            event=event,
        )
        save_json(directory / "run.json", manifest)
        save_json(
            ROOT / "data" / "events" / "v2" / (event["id"] + ".json"),
            {**event, "run_id": manifest["run_id"], "status": "ready"},
        )
        if settings.source == "archive":
            archive.remember_issue(event, manifest["run_id"])
        return directory
    except Exception as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            api_calls=api.calls if api else 0,
        )
        save_json(directory / "run.json", manifest)
        (directory / "diagnostics").mkdir(exist_ok=True)
        (directory / "diagnostics" / "failure.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        write_review(directory)
        if api:
            api.close()


def check_setup():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FactoryError("FFmpeg ve FFprobe gerekli.")
    render.font({"font_style": "condensed_heavy"}, 70)
    Settings.load()
    print(f"Comic Factory {VERSION}: ayarlar, medya araçları ve Türkçe font hazır.")


def main(argv=None):
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Türkçe çizgi roman video stüdyosu")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--topic", default="")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--voice", choices=["auto", "Orus", "Gacrux", "Fenrir", "Puck"])
    parser.add_argument("--voice-test", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-run", type=Path)
    parser.add_argument(
        "--platforms", choices=["both", "youtube", "instagram"], default="both"
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            check_setup()
            return 0
        if args.publish_run:
            emit_directory(args.publish_run)
            publish_run(args.publish_run, args.platforms)
            return 0
        directory = generate(
            Settings.load(target_seconds=args.duration, voice=args.voice),
            args.topic.strip(),
            args.resume_run,
            args.voice_test,
        )
        print(f"Çalışma hazır: {directory / 'review.html'}", flush=True)
        if args.publish and not args.voice_test:
            publish_run(directory, args.platforms)
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"HATA: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
