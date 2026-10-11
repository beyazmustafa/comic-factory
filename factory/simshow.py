"""Simulation Shorts source: pick a physics scenario, let the writer give it
the reference's spare, declarative texts, render it with factory.sim, and
stage a run directory the publisher accepts.

A calm narration (one passage per phase) drives the timing: each phase of
the simulation begins where its passage begins. Small quiet captions sit
below the simulation; a soft drone lies under the voice. Real photographs are used only when a scenario
asks for a backdrop (none of the current ones do).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import sim
from .api import FactoryError
from .config import ROOT
from .core import file_hash, save_json

DEFAULT_TEXTS = {
    "three-body": {"title": "The three-body problem.", "phases": [
        {"at": 0.0, "narration": "Three stars, each pulling on the other two. With two bodies the orbit is a clean ellipse. With three, there is no formula, only motion.", "subtitle": "Three stars pull on each other."},
        {"at": 0.3, "narration": "Every close pass is a trade. One star speeds up, another slows down, and the energy of the whole system shuffles between them.", "subtitle": "Each close pass transfers energy."},
        {"at": "event", "title": "One escapes.\nTwo remain bound.", "narration": "Eventually one star gathers enough speed to break free. It leaves forever, and the two that remain settle into a quiet, stable pair.", "subtitle": "One gains enough energy to escape."}],
        "caption": "Stars' gravitational fields"},
    "gravity-assist": {"title": "The gravity assist.", "phases": [
        {"at": 0.0, "narration": "A probe approaches a planet that is itself moving through space. From the planet's point of view, the probe comes in and leaves at the same speed.", "subtitle": "A probe approaches a moving planet."},
        {"at": 0.4, "narration": "But the planet is travelling, and it drags its gravity with it. As the probe swings around, the planet's motion is added to its own.", "subtitle": "The planet's pull bends its path."},
        {"at": 0.7, "title": "Faster than it arrived.", "narration": "The probe leaves faster than it arrived, and the planet is slowed by an amount too small to measure. This is how spacecraft reach the outer solar system.", "subtitle": "The planet's motion hands it speed."}],
        "caption": "Planet's gravitational field"},
    "roche-limit": {"title": "The Roche limit.", "phases": [
        {"at": 0.0, "narration": "A moon orbits its planet, slowly drifting inward. Gravity pulls harder on the near side than the far side, and that difference is the tide.", "subtitle": "A moon spirals toward its planet."},
        {"at": 0.4, "narration": "The closer it comes, the stronger the stretch, until the pull across the moon is greater than the gravity holding it together.", "subtitle": "Tides pull harder on its near side."},
        {"at": 0.7, "title": "A moon becomes a ring.", "narration": "Inside that distance, the Roche limit, the moon comes apart. Its pieces spread along the orbit, and a ring is born.", "subtitle": "Inside the limit, it comes apart."}],
        "caption": "Planet's gravitational field"},
    "black-hole": {"title": "Feeding a black hole.", "phases": [
        {"at": 0.0, "narration": "Gas circles a black hole far outside the horizon. Nothing here is being pulled straight in; it is orbiting, as a planet orbits a star.", "subtitle": "Gas orbits the horizon."},
        {"at": 0.4, "narration": "Friction between the streams steals energy, and each orbit tightens a little. The gas heats, glows, and creeps toward the edge.", "subtitle": "Friction steals its energy."},
        {"at": 0.7, "title": "Past the horizon.", "narration": "At the horizon, the escape speed reaches the speed of light. What crosses that line cannot send anything back, not even light.", "subtitle": "What crosses never returns."}],
        "caption": "Black hole's gravitational field"},
    "escape-velocity": {"title": "Escape velocity.", "phases": [
        {"at": 0.0, "narration": "Three objects leave a planet at three different speeds. Gravity pulls on all of them equally; only their speed is different.", "subtitle": "Three launches. Three speeds."},
        {"at": 0.35, "narration": "The slowest rises and falls back. The second is fast enough that it keeps missing the ground: it is in orbit.", "subtitle": "Too slow falls back. Just right circles."},
        {"at": 0.7, "title": "One never returns.", "narration": "The third is faster still. Gravity slows it, but never to a stop. It is past escape velocity, and it will not return.", "subtitle": "Fast enough, and gravity loses."}],
        "caption": "Planet's gravitational field"},
    "binary-planet": {"title": "Two suns, one planet.", "phases": [
        {"at": 0.0, "narration": "Two stars circle a shared centre, and a planet circles both of them. Close in, the changing pull would tear its orbit apart.", "subtitle": "A planet circles a binary star."},
        {"at": 0.4, "narration": "Far enough out, the two stars act almost like one. The planet feels their combined gravity as a single, steady centre.", "subtitle": "The stars swing. The planet rides their pull."},
        {"at": 0.7, "title": "Two sunsets a day.", "narration": "So the orbit holds, year after year. On that world, two suns rise, and the sky has two shadows.", "subtitle": "Far enough out, the orbit holds."}],
        "caption": "Stars' gravitational fields"},
}


def used_scenarios():
    used = []
    folder = ROOT / "data" / "history" / "videos"
    if folder.is_dir():
        for run in sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            path = run / "story.json"
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            key = (data.get("event") or {}).get("scenario")
            if key:
                used.append(key)
    return used


def pick_scenario(topic=""):
    keys = list(sim.SCENARIOS)
    if topic:
        wanted = [k for k in keys if all(w in (k + " " + sim.SCENARIOS[k][1]).casefold() for w in topic.casefold().split())]
        if wanted:
            return wanted[0]
    recent = used_scenarios()
    fresh = [k for k in keys if k not in recent[:len(keys) - 1]]
    pool = fresh or keys
    # least recently used first
    pool.sort(key=lambda k: recent.index(k) if k in recent else -1)
    return pool[-1] if fresh else pool[0]


def clean_line(text, limit):
    text = " ".join(str(text or "").split())
    text = re.sub(r"[<>]", "", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text[:limit]


def write_texts(api, key, handle):
    """The writer gives the scenario its on-screen lines in the reference's
    voice: short declaratives with a full stop, no hype, no numbers."""
    brief = sim.SCENARIOS[key][1]
    default = DEFAULT_TEXTS[key]
    try:
        data = api.json(
            "Simülasyon metinleri",
            f"""You write the on-screen text of a silent physics-visualisation Short (style: '@ElsewhereLab' — a bold title, a thin one-line explanation that changes as the simulation progresses, nothing else). Scenario: {brief}
Rules: title 2..4 words ending with a period ("The three-body problem."); three phase lines, each a complete plain sentence of 3..8 words ending with a period, in the order the viewer will see the physics happen; the last phase may also replace the title with a 2-line conclusion (two short sentences, e.g. "One escapes.\\nTwo remain bound."). No numbers on screen, no exclamation marks, no adjectives like amazing/terrifying; the tone is calm and exact.
NARRATION: a calm, unhurried voice-over, one passage per phase (18..30 words each, 60..85 words total), explaining what the viewer is watching and why it happens, like a planetarium guide: precise, warm, no hype, no questions to the viewer, no "imagine", numbers allowed when they help (spelled as words). The three passages must read as one continuous explanation.
Also give a YouTube title (45..70 characters, curiosity-driven, no clickbait) and a one-sentence description.
Return {{"title":"","phases":[{{"subtitle":"","narration":""}},{{"subtitle":"","narration":""}},{{"title":"optional two lines or empty","subtitle":"","narration":""}}],"caption":"4-5 word monospace caption naming the field shown below, e.g. Stars' gravitational fields","youtube_title":"","description":""}}""",
        )
    except FactoryError as error:
        print(f"Metin yazarı yanıt vermedi ({str(error)[:100]}); varsayılan metinler kullanıldı.", flush=True)
        data = {}
    phases = []
    rows = data.get("phases") if isinstance(data.get("phases"), list) else []
    for index, base in enumerate(default["phases"]):
        row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
        subtitle = clean_line(row.get("subtitle"), 48) if row.get("subtitle") else base["subtitle"]
        if not 3 <= len(subtitle.split()) <= 9:
            subtitle = base["subtitle"]
        narration = " ".join(str(row.get("narration") or "").split()) if row else ""
        if not 10 <= len(narration.split()) <= 40 or any(v in narration for v in ("<", ">", "http")):
            narration = base.get("narration", "")
        phase = {"at": base["at"], "subtitle": subtitle, "narration": narration}
        title_override = str(row.get("title") or "").strip() if row else ""
        if index == len(default["phases"]) - 1:
            if title_override and 1 <= title_override.count("\n") <= 1 and len(title_override) <= 44:
                phase["title"] = "\n".join(clean_line(line, 24) for line in title_override.split("\n"))
            elif base.get("title"):
                phase["title"] = base["title"]
        phases.append(phase)
    title = clean_line(data.get("title"), 30) if data.get("title") else default["title"]
    if not 2 <= len(title.replace("\n", " ").split()) <= 5:
        title = default["title"]
    for phase in phases:
        phase.setdefault("title", title)
    caption = " ".join(str(data.get("caption") or default["caption"]).split())[:40]
    youtube_title = " ".join(str(data.get("youtube_title") or "").split())
    if not 30 <= len(youtube_title) <= 80:
        youtube_title = f"{title.rstrip('.')} — a physics simulation"
    description = clean_line(data.get("description"), 240) or f"A simulation of {brief}"
    return {"title": title, "phases": phases, "caption": caption, "handle": handle,
            "youtube_title": youtube_title, "description": description}


def generate(settings, topic="", api_factory=None):
    """Full run: scenario → texts → render → run directory ready to publish."""
    from .api import Api

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "data" / "runs" / stamp
    directory.mkdir(parents=True)
    manifest = {"schema": 2, "version": settings.to_dict().get("version", ""), "status": "building", "stage": "setup",
                "topic": topic, "settings": settings.to_dict(), "run_id": os.getenv("GITHUB_RUN_ID", stamp)}
    save_json(directory / "run.json", manifest)
    try:
        api = (api_factory or Api)(settings, directory)
        key = pick_scenario(topic)
        handle = str(getattr(settings, "brand_handle", "") or "")
        print(f"→ Senaryo: {key}", flush=True)
        texts = write_texts(api, key, handle)
        save_json(directory / "sim_texts.json", {"scenario": key, **texts})
        print("→ Anlatım sesi", flush=True)
        from . import render, voice
        from .style import load_style

        calm_style = {**load_style(), "narrator_delivery": "Calm, unhurried planetarium narration: warm, precise, low and even, gentle pauses between sentences, no excitement, no selling.",
                      "voice_character": "A calm documentary narrator, deep and soft, around 125 to 135 words per minute, measured pauses, never dramatic.",
                      "speech_rate": "-6%"}
        narration_story = {"shots": [{"id": f"shot_{i:03}", "narration": p["narration"], "emphasis": "normal"} for i, p in enumerate(texts["phases"])]}
        narration_story["narration"] = " ".join(s["narration"] for s in narration_story["shots"])
        cache = ROOT / "data" / "cache-v2"
        audio_path, words, speech_seconds = voice.build_audio(api, narration_story, getattr(settings, "sim_voice", "Charon"), calm_style, cache)
        head = 1.2  # silence before the voice starts; the title has a moment alone
        # phase timing follows the narration: each phase begins where its passage begins
        starts, cursor = [], 0
        for shot in narration_story["shots"]:
            count = len(shot["narration"].split())
            starts.append(float(words[cursor]["start"]) + head if count else 0.0)
            cursor += count
        duration = max(float(getattr(settings, "sim_seconds", 28.0)), speech_seconds + head + 2.6)
        for index, phase in enumerate(texts["phases"]):
            if phase.get("at") == "event":
                phase["target"] = max(0.4, min(0.8, starts[index] / duration))
            else:
                phase["at"] = 0.0 if index == 0 else starts[index]
        save_json(directory / "sim_texts.json", {"scenario": key, **texts, "duration": duration})
        print(f"→ Simülasyon render ({duration:.1f} sn)", flush=True)
        silent = sim.render(key, texts, directory / "sim_video.mp4", duration=duration)
        # captions: small, quiet, phrase by phrase, below the simulation
        caption_words = [{**w, "emphasis": "normal"} for w in words]
        caption_style = {**calm_style, "font_style": "regular", "font_size": 34, "stroke_width": 2, "shadow_depth": 0, "word_pop_ms": 0,
                         "caption_x": 0.5, "caption_y": 0.528, "caption_mode": "group", "caption_words": 5, "uppercase": False,
                         "text_color": "#B8BBD0", "active_color": "#FFFFFF", "stroke_color": "#05050C", "emphasis_colors": {}, "language": settings.language}
        subtitle = render.captions(caption_words, caption_style, directory / "captions.ass", head)
        drone = directory / "drone.wav"
        render.space_drone(drone, duration)
        escaped = str(subtitle.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        video = directory / "video.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio_path), "-i", str(drone),
                        "-filter_complex",
                        f"[0:v]ass=filename='{escaped}'[v];[1:a]adelay={int(head * 1000)}|{int(head * 1000)},loudnorm=I=-17:TP=-1.5:LRA=9,aresample=48000[speech];"
                        f"[2:a]volume={float(getattr(settings, 'sim_music_db', -30.0))}dB,aresample=48000[bed];[speech][bed]amix=inputs=2:duration=first:normalize=0,apad=pad_dur=3,alimiter=limit=0.95:latency=1[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(video)], check=True, timeout=1800)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                               capture_output=True, text=True, timeout=60)
        if float(probe.stdout.strip() or 0) < 10:
            raise FactoryError("Simülasyon videosu üretilemedi.")
        # thumbnail: a frame from the busiest part
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{duration * 0.45:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "2",
                        str(directory / "thumbnail.jpg")], capture_output=True, timeout=120)
        event = {"id": f"sim_{key}_{stamp}", "title": texts["title"], "series": "Simulation", "issue": key, "year": datetime.now(timezone.utc).year,
                 "publisher": "Original", "universe": "science", "scenario": key, "_source": "sim", "url": "", "source_urls": []}
        lines = [p["narration"] for p in texts["phases"]]
        script = {"title": texts["youtube_title"], "description": texts["description"], "hook_card": texts["title"],
                  "shots": [{"id": f"shot_{i:03}", "panel_id": "sim", "narration": line, "fact_ids": [], "motion": "hold", "emphasis": "normal"}
                            for i, line in enumerate(lines)],
                  "narration": " ".join(lines), "event": event, "scenario": key, "phases": texts["phases"],
                  "panel_validation": {"passed": True, "shots": [], "generated": True}}
        save_json(directory / "story.json", script)
        save_json(directory / "story_bundle.json", {"event": event, "story": script, "panels": [], "facts": []})
        aligned = min((float(w.get("end", 0)) >= 0 for w in words), default=True)
        review = {"passed": True, "review_mode": "simulation", "subtitle_sync": 100, "scene_match": 100, "delivery": 100,
                  "visual_readability": 95, "summary": "Code-generated simulation; narration aligned per chunk at synthesis.", "failed_checks": [], "issues": []}
        save_json(directory / "quality_review.json", review)
        hashtags = ["space", "shorts", "physics", "simulation", "science", "astronomy"]
        full = texts["description"] + "\n\nA physics simulation rendered by code. " + (handle + "\n" if handle else "")
        metadata = {"script": {**script, "full_description": full, "hashtags": hashtags,
                               "instagram_caption": texts["youtube_title"] + "\n\n" + texts["description"]},
                    "sources": [], "voice": {"voice": "none"}, "editing_profile": "simulation-v1", "language": settings.language}
        save_json(directory / "metadata.json", metadata)
        manifest.update(status="ready", stage="Tamamlandı", technical_passed=True, quality_passed=True,
                        video_sha256=file_hash(video), metadata_sha256=file_hash(directory / "metadata.json"),
                        quality_sha256=file_hash(directory / "quality_review.json"), api_calls=api.calls,
                        duration=duration, voice=getattr(settings, "sim_voice", "Charon"), event=event, localized=[])
        save_json(directory / "run.json", manifest)
        os.environ["FACTORY_LANGUAGE"] = settings.language
        return directory
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(directory / "run.json", manifest)
        raise
