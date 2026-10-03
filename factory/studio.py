import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from . import archive, famous, forge, learning, localize, research, panels, story, voice, render, quality
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
        stage("Öğrenme: oyun kitabı ve deney")
        api.run_id = manifest["run_id"]
        learned = learning.evolve(api, style)
        style = {**style, **learned}
        save_json(directory / "editing_profile.json", style)
        stage("Olay, gerçek sayfalar ve senaryo")
        # Playbook/experiment change every run; a resumed run must keep its story.
        story_key = signature(
            [
                topic,
                settings.target_seconds,
                settings.max_shots,
                settings.panel_threshold,
                settings.gemini_model,
                {k: v for k, v in style.items() if k not in ("playbook", "experiment")},
            ]
        )
        bundle = checkpoints.read("story", story_key)
        if bundle is None and resume_run and (directory / "story_bundle.json").exists():
            # A resumed run keeps its finished story even if the key format changed.
            try:
                candidate_bundle = json.loads((directory / "story_bundle.json").read_text(encoding="utf-8"))
                if all((directory / p["file"]).is_file() for p in candidate_bundle.get("panels", [])):
                    bundle = candidate_bundle
                    print("Önceki çalışmanın senaryosu kullanılıyor.", flush=True)
            except (ValueError, KeyError, TypeError):
                bundle = None
        if bundle is None:
            used = load_used()
            candidates = []
            effective = settings.source
            if effective == "mix":
                # Famous moments are what the audience clicks; the other
                # sources stay available for manual runs.
                effective = "famous"
                print(f"Kaynak (mix): {effective}", flush=True)
            if effective == "famous":
                candidates += famous.shortlist(api, topic, used)
            if effective == "studio":
                universe = forge.ensure_universe(api, note=api.note)
                event, draft, inventory, facts = forge.produce(api, universe, style, note=api.note)
                script = story.validate_story(draft, inventory, facts, settings.max_shots)
                script.update(event=event, panel_validation={"passed": True, "shots": [], "generated": True})
                save_json(directory / "story.json", script)
                bundle = {"event": event, "story": script, "panels": inventory, "facts": facts}
                save_json(directory / "story_bundle.json", bundle)
                checkpoints.save("story", story_key, bundle, [directory / "story.json", directory / "story_bundle.json"]
                                 + [p for p in (directory / "events" / event["id"]).rglob("*") if p.is_file()])
                candidates = None
            if candidates is not None and effective in {"auto", "web"}:
                try:
                    candidates += [{**e, "_source": "web"} for e in research.shortlist(api, topic, used)]
                except SourceUnavailable as error:
                    print(f"Önizleme kaynağı: {error}", flush=True)
                    if settings.source == "web":
                        raise
            if candidates is not None and effective in {"auto", "archive"}:
                try:
                    candidates += [{**e, "_source": "archive"} for e in archive.shortlist(api, topic, used)]
                except SourceUnavailable as error:
                    if not candidates:
                        raise
                    print(f"Arşiv kaynağı: {error}", flush=True)
            fetcher = research.Fetcher()
            failures = []
            try:
                for event in (candidates or []):
                    from_archive = event.get("_source") == "archive"
                    print("Olay araştırılıyor: " + event["title"], flush=True)
                    try:
                        if event.get("_source") == "famous":
                            inventory, facts = famous.collect(api, event, fetcher)
                        elif from_archive:
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
        if event.get("_source") == "studio":
            licence = "\nOriginal characters and artwork created for this channel."
        elif event.get("_source") == "famous":
            licence = "\nArtwork: official previews, covers and press images, shown for commentary and review. Characters and art © their publishers."
        elif event.get("_source") == "archive" or settings.source == "archive":
            licence = "\nPages: public-domain Golden Age issue scanned on the Internet Archive."
        else:
            licence = "\nPanels: official publisher previews and press coverage, used for commentary."
        description = (
            script["description"]
            + f"\n\nComic: {event['series']} #{event['issue']} ({event['year']})"
            + licence
            + "\nSources:\n"
            + "\n".join(sources)
        )
        hashtags = [
            "comics",
            "shorts",
            "superhero" if settings.channel_theme == "superheroes" else "comicbooks",
            str(event.get("publisher", "comics")).lower().replace(" ", ""),
        ]
        os.environ["FACTORY_LANGUAGE"] = settings.language
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
        if event.get("_source") == "archive" or settings.source == "archive":
            archive.remember_issue(event, manifest["run_id"])
        if event.get("_source") == "studio":
            forge.remember_episode(forge.load_universe() or {"heroes": [], "villains": []}, event, manifest["run_id"])
        extra = [c.strip() for c in str(settings.languages).split(",") if c.strip() and c.strip() != settings.language]
        localized = []
        for code in extra:
            stage(f"{code} sürümü")
            try:
                sub = localize.localize(api, directory, script, inventory, style, cache, code, metadata)
                localized.append(sub.name)
            except Exception as error:  # a missing edition never cancels the primary one
                print(f"{code} sürümü üretilemedi: {type(error).__name__}: {str(error)[:200]}", flush=True)
                save_json(directory / "diagnostics" / f"localize_{code}.json", {"error": str(error)[:1000]})
        manifest.update(localized=localized, stage="Tamamlandı", status="ready")
        save_json(directory / "run.json", manifest)
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


def publish_all(directory, platforms):
    """Primary edition to the chosen platforms, localized editions to YouTube."""
    results = publish_run(directory, platforms)
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for name in manifest.get("localized", []):
        sub = directory / name
        if not (sub / "run.json").is_file():
            continue
        previous = os.environ.get("FACTORY_LANGUAGE", "")
        try:
            os.environ["FACTORY_LANGUAGE"] = name.replace("lang_", "")
            publish_run(sub, "youtube")
        except Exception as error:
            print(f"{name} yayınlanamadı: {type(error).__name__}: {str(error)[:200]}", flush=True)
        finally:
            os.environ["FACTORY_LANGUAGE"] = previous
    return results


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
    parser.add_argument("--refresh-stats", action="store_true")
    parser.add_argument("--source", choices=["famous", "auto", "archive", "web", "studio", "mix"])
    parser.add_argument("--topic", default="")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--voice", choices=["auto", "Orus", "Gacrux", "Fenrir", "Puck", "Ahmet", "Emel"])
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
        if args.refresh_stats:
            rows = learning.refresh_stats()
            with_retention = sum(1 for r in rows if (r.get("stats") or {}).get("average_view_percentage") is not None)
            print(f"İstatistik güncellendi: {len(rows)} video, {with_retention} tanesinde retention var.")
            for r in rows:
                s = r.get("stats") or {}
                print(f"- {r.get('video_id')}: {s.get('views')} izlenme, retention %{s.get('average_view_percentage')}, {r['profile'].get('title','')[:50]}")
            return 0
        if args.publish_run:
            emit_directory(args.publish_run)
            publish_all(args.publish_run, args.platforms)
            return 0
        directory = generate(
            Settings.load(target_seconds=args.duration, voice=args.voice, source=args.source),
            args.topic.strip(),
            args.resume_run,
            args.voice_test,
        )
        print(f"Çalışma hazır: {directory / 'review.html'}", flush=True)
        if args.publish and not args.voice_test:
            publish_all(directory, args.platforms)
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"HATA: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
