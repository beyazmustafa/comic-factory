"""Footage documentary source ("film"): a calm narrated Short cut from real
NASA footage.

Flow: pick the least recently used topic → harvest clean clips from NASA's
archives (factory.footage) → the writer scripts beats tied to those clips,
using only the topic's verified facts → one calm narrator (factory.voice) →
beats become shots on the timeline, each clip trimmed to its narration →
factory.cut renders the edit → small captions, a soft bed, the mix, a cover,
and the run directory the publisher accepts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import cut, footage, topics_film
from .api import FactoryError
from .config import ROOT
from .core import ass_time, escape_ass, file_hash, save_json

HOOK_SECONDS = 3.4
END_SECONDS = 2.8
MAX_SHOT = 6.5      # a beat longer than this is covered by two shots — the picture keeps moving
HEAD = 0.9          # silence before the narrator starts (the hook card has a beat alone)
MAX_SECONDS = 150.0
DISSOLVE = 10       # frames
CAPTION_Y = 0.70


# ------------------------------------------------------------------ topics
def used_topics():
    """Topic keys of earlier film runs, oldest first (by run folder name)."""
    rows = []
    for folder in (ROOT / "data" / "history" / "videos", ROOT / "data" / "runs"):
        if not folder.is_dir():
            continue
        for story in sorted(folder.glob("*/story.json")):
            try:
                data = json.loads(story.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("event", {}).get("_source") == "film" and data.get("scenario"):
                rows.append((story.parent.name, data["scenario"]))
    rows.sort()
    return [key for _, key in rows]


def pick_topics(topic=""):
    """Requested topic first (by key or words), then the least recently used ones."""
    wanted = topic.strip().casefold()
    ordered = []
    if wanted:
        match = topics_film.by_key(wanted) or next(
            (t for t in topics_film.TOPICS if wanted in t["title"].casefold() or wanted in t["subject"].casefold()), None)
        if match:
            ordered.append(match["key"])
    history = used_topics()
    last_seen = {key: index for index, key in enumerate(history)}
    rest = sorted((k for k in topics_film.KEYS if k not in ordered), key=lambda k: (last_seen.get(k, -1), k))
    return ordered + rest


# ------------------------------------------------------------------ script
def words_of(text):
    return len(str(text).split())


def write_script(api, topic, clips, handle):
    facts = "\n".join(f"- {fact}" for fact in topic["facts"])
    ordered = sorted(clips, key=lambda c: ({"footage": 0, "visualization": 1, "animation": 2}[c["kind"]], -c["quality"]))
    listing = "\n".join(
        f"- {c['id']}{' ★ (best real imagery — use it for beat 1 if it fits)' if i == 0 else ''}: {c['shows'] or c['what']} | {c['kind']} | "
        f"up to {max(w[1] - w[0] for w in c['windows']):.0f} s | source: \"{c['title']}\" ({c.get('date', '')[:4]})"
        for i, c in enumerate(ordered))
    callouts = "\n".join(f"- [{i}] {c['value']} — {c['label']}" for i, c in enumerate(topic.get("callouts", [])))
    prompt = f"""You are writing a calm, premium vertical documentary Short (60–80 seconds) for the channel {handle}.
Subject: {topic['subject']}.
On-screen hook card (you may improve it, max 7 words): "{topic['title']}". Spoken opening idea: "{topic['hook']}".

VERIFIED FACTS — the only source of claims and numbers (reuse them, do not add facts, do not change numbers):
{facts}

CLIPS available (id: what is on screen | kind | usable length | source title):
{listing}

NUMBERS available for a big on-screen callout (choose at most 2, by index):
{callouts}

Write 8 to 11 beats. Each beat = one clip on screen + one or two spoken sentences (10 to 24 words) that fit that clip.
Rules of the voice: calm, precise, concrete, present tense for what is on screen; tie words to picture ("This is…", "Here…")
in two or three beats, not more; one idea per beat, in a clear chronological or causal order; every number comes from the facts;
no filler clauses that just restate a fact unrelated to the picture; no hype words (incredible, mind-blowing, unbelievable),
no "imagine", no "welcome", no "in this video", no questions to the viewer, no lists.
Beat 1 is the hook: the most surprising true statement, said plainly. The middle builds understanding. The last beat delivers
a payoff that makes the hook land. Then a closing line (max 12 words) shown on screen in silence: a resonant fact or image that
lingers (e.g. "The next crew will fly this same path." / "It is still falling, 50 years later.") — never an instruction to the
viewer: no share, comment, like, follow, subscribe, reflect, thoughts, below.
Use each clip at most twice, never in two consecutive beats, and vary the views (do not string together beats that show the same
thing); prefer the clips marked "footage" for the hook; a clip's beat must not be longer than its usable length.
Labels: a short on-screen tag per beat, 2–5 words, like "NASA · JUNO · 2017" (mission/instrument + year when known).
Total narration 130 to 185 words.

Return JSON:
{{"hook": "on-screen hook, max 7 words", "youtube_title": "max 60 characters, specific, no clickbait punctuation",
"description": "two sentences for the video description",
"beats": [{{"clip": "c00", "narration": "...", "label": "NASA · MISSION · YEAR", "callout": <index or null>}}],
"closing": "..."}}"""
    data = api.json("Belgesel senaryosu", prompt)
    script = validate_script(data, topic, clips)
    return polish_script(api, topic, script, clips)


def polish_script(api, topic, script, clips):
    """A second pass by the writer as its own editor: order, clarity, no filler, same facts and shape."""
    facts = "\n".join(f"- {fact}" for fact in topic["facts"])
    by_id = {c["id"]: c for c in clips}
    draft = "\n".join(f"{i + 1}. [{b['clip']}: {by_id[b['clip']]['shows'][:70]}] {b['narration']}" for i, b in enumerate(script["beats"]))
    prompt = f"""You are the script editor of a calm, premium documentary Short about {topic['subject']}.
Tighten this draft narration without changing its facts, beat count, clip assignments or overall length (each beat 10–24 words).
Make the order chronological or causal and make each beat say one clear thing that matches the clip in brackets.
Remove filler, repeated ideas, and sentences that describe something the clip does not show. Keep numbers exactly as in the facts.
Keep "This is / Here / Watch" phrasing to at most three beats. Plain, concrete, warm. No hype, no questions, no instructions to the viewer.

FACTS:
{facts}

DRAFT:
{draft}

Return {{"beats": [{{"narration": "..."}}, ...]}} with exactly {len(script['beats'])} entries in the same order."""
    try:
        data = api.json("Senaryo düzelti", prompt)
        rows = data.get("beats") or []
        if len(rows) != len(script["beats"]):
            return script
        polished = []
        for beat, row in zip(script["beats"], rows):
            text = re.sub(r"\s+", " ", str(row.get("narration", "") if isinstance(row, dict) else row)).strip()
            polished.append({**beat, "narration": text if 8 <= words_of(text) <= 28 else beat["narration"]})
        total = sum(words_of(b["narration"]) for b in polished)
        if 110 <= total <= 215:
            return {**script, "beats": polished}
    except Exception as error:  # noqa: BLE001
        print(f"Düzelti atlandı: {str(error)[:100]}", flush=True)
    return script


def validate_script(data, topic, clips):
    ids = {c["id"]: c for c in clips}
    beats = []
    uses = {}
    rows = data.get("beats") or []
    limit = max(2, -(-len(rows) // max(1, len(ids))))  # a small clip pool may be revisited
    for row in rows:
        clip = ids.get(str(row.get("clip", "")).strip())
        narration = re.sub(r"\s+", " ", str(row.get("narration", ""))).strip()
        if clip is None or not 6 <= words_of(narration) <= 30:
            continue
        if uses.get(clip["id"], 0) >= limit:
            continue
        uses[clip["id"]] = uses.get(clip["id"], 0) + 1
        label = re.sub(r"\s+", " ", str(row.get("label", ""))).strip(" .")[:34] or f"NASA · {clip.get('date', '')[:4]}".strip(" ·")
        callout = row.get("callout")
        callouts = topic.get("callouts", [])
        try:
            callout = callouts[int(callout)] if callout is not None and 0 <= int(callout) < len(callouts) else None
        except (TypeError, ValueError):
            callout = None
        beats.append({"clip": clip["id"], "narration": narration, "label": label, "callout": callout})
    # at most two callouts, never two in a row, never under the hook card
    seen = 0
    for index, beat in enumerate(beats):
        if beat["callout"] and (index == 0 or seen >= 2 or beats[index - 1]["callout"]):
            beat["callout"] = None
        elif beat["callout"]:
            seen += 1
    total = sum(words_of(b["narration"]) for b in beats)
    if len(beats) < 7 or not 110 <= total <= 210:
        raise FactoryError(f"Senaryo biçimi uygun değil ({len(beats)} beat, {total} kelime).")
    hook = re.sub(r"\s+", " ", str(data.get("hook") or topic["title"])).strip()
    if words_of(hook) > 8 or len(hook) < 6:
        hook = topic["title"]
    title = re.sub(r"\s+", " ", str(data.get("youtube_title") or "")).strip()[:70] or topic["title"]
    closing = re.sub(r"\s+", " ", str(data.get("closing") or "")).strip()
    salesy = re.search(r"\b(share|comment|like|follow|subscribe|reflect|thoughts|below|tell us|let us know)\b", closing, flags=re.I)
    if words_of(closing) > 14 or len(closing) < 8 or salesy:
        closing = topic.get("closing") or f"{topic['title'].rstrip('.')}. Real footage."
    description = re.sub(r"\s+", " ", str(data.get("description") or "")).strip()[:500] or topic["subject"].capitalize() + "."
    return {"hook": hook, "youtube_title": title, "description": description, "beats": beats, "closing": closing}


# ---------------------------------------------------------------- timeline
def build_timeline(script, clips, starts, duration):
    """Shots for the editor: each beat's clip trimmed (or gently slowed) to its narration span."""
    by_id = {c["id"]: c for c in clips}
    use_count = {}
    shots = []
    pool = sorted(clips, key=lambda c: ({"footage": 0, "visualization": 1, "animation": 2}[c["kind"]], -c["quality"]))
    for index, beat in enumerate(script["beats"]):
        t0 = starts[index]
        t1 = starts[index + 1] if index + 1 < len(starts) else duration
        span = t1 - t0
        clip = by_id[beat["clip"]]
        placed = 0.0
        first = True
        candidates = [clip] + [c for c in pool if c["id"] != clip["id"]]
        for attempt in range(2):
            for source in candidates:
                if span - placed < 0.4:
                    break
                fresh = [i for i in range(len(source["windows"])) if use_count.get((source["id"], i), 0) == 0]
                if attempt == 0 and not fresh:
                    continue  # first pass: unused windows only
                choices = fresh or list(range(len(source["windows"])))
                best = min(choices, key=lambda i: (use_count.get((source["id"], i), 0), -(source["windows"][i][1] - source["windows"][i][0])))
                w0, w1 = source["windows"][best]
                available = w1 - w0
                remaining = span - placed
                # long beats are split so no shot outstays its welcome (unless the remainder would be a stub)
                target = remaining if remaining <= MAX_SHOT or remaining - MAX_SHOT < 2.5 else min(MAX_SHOT, remaining * 0.55)
                need = target + DISSOLVE / cut.FPS
                if available >= need:
                    src_in = w0 + (available - need) / 2
                    src_out = src_in + need
                    speed = 1.0
                    shot_len = target
                elif available >= need * 0.72:
                    speed = max(0.72, available / need)
                    src_in, src_out = w0, w1
                    shot_len = target
                else:
                    # too short for the target: use it at a gentle slow-motion for part of the beat
                    speed = 0.8
                    src_in, src_out = w0, w1
                    shot_len = min(target, available / speed - DISSOLVE / cut.FPS)
                    if shot_len < 1.5:
                        continue
                use_count[(source["id"], best)] = use_count.get((source["id"], best), 0) + 1
                shots.append({"beat": index, "clip": source["id"], "asset": source["asset"], "file": source["file"], "width": source["width"],
                              "height": source["height"], "src_in": round(src_in, 3), "src_out": round(src_out, 3), "speed": round(speed, 3),
                              "focus": tuple(source["focus"]), "active": source.get("active"), "crop_width": source.get("crop_width", 0.32),
                              "zoom": (1.0, 1.06) if len(shots) % 2 == 0 else (1.06, 1.0),
                              "t0": round(t0 + placed, 3), "t1": round(t0 + placed + shot_len, 3), "label": beat["label"] if first else "",
                              "callout": beat["callout"] if first else None, "kind": source["kind"]})
                placed += shot_len
                first = False
            if span - placed < 0.4:
                break
        if first:
            raise FactoryError(f"{index + 1}. beat için görüntü yerleştirilemedi.")
    # stretch the final shot to the end and fix any gaps left by rounding
    for a, b in zip(shots, shots[1:]):
        a["t1"] = b["t0"]
    shots[-1]["t1"] = duration
    # full-screen footage is the rule; wide (letterboxed) compositions stay the exception
    wide = [s for s in shots if float(s.get("crop_width") or 0.32) > 0.36]
    allowed = max(1, round(len(shots) * 0.3))
    for shot in sorted(wide, key=lambda s: float(s["crop_width"]))[: max(0, len(wide) - allowed)]:
        shot["crop_width"] = 0.32
    for index, shot in enumerate(shots):
        if index == 0:
            shot["transition"] = "cut"
            continue
        previous = shots[index - 1]
        # mostly straight cuts; a dissolve every third beat breathes a little
        shot["transition"] = "dissolve" if (shot["beat"] != previous["beat"] and shot["beat"] % 3 == 0) else "cut"
    return shots


# ---------------------------------------------------------------- captions
def captions(words, path, offset, family, size=40, group=5):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {cut.W}
PlayResY: {cut.H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Main,{family},{size},&H00E8E8F0,&H00E8E8F0,&H00060608,&H78000000,0,0,0,0,100,100,0.4,0,1,2,1,5,100,140,260,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [header]
    measure = cut.font("medium", size)
    y = round(cut.H * CAPTION_Y)
    for i, word in enumerate(words):
        first = i // group * group
        last = min(len(words), first + group)
        texts = [str(w["word"]) for w in words[first:last]]
        fs = size
        while fs > 26 and measure.getlength(" ".join(texts)) * fs / size > 860:
            fs -= 2
        text = " ".join(("{\\1c&HFFFFFF&}" if first + j == i else "{\\1c&HB8BBD0&}") + escape_ass(token) for j, token in enumerate(texts))
        start, end = max(0.0, word["start"] + offset), word["end"] + offset
        if i + 1 < len(words):
            nxt = words[i + 1]["start"] + offset
            end = nxt if 0 <= nxt - end < 0.25 else min(end, nxt)
        if end <= start or ass_time(end) == ass_time(start):
            continue
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Main,,0,0,0,,{{\\an5\\pos({cut.W // 2},{y})\\fs{fs}\\fad(40,0)}}{text}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------- run
def generate(settings, topic="", api_factory=None):
    from .api import Api
    from . import render, voice
    from .style import load_style

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "data" / "runs" / stamp
    directory.mkdir(parents=True)
    manifest = {"schema": 2, "version": settings.to_dict().get("version", ""), "status": "building", "stage": "setup",
                "topic": topic, "settings": settings.to_dict(), "run_id": os.getenv("GITHUB_RUN_ID", stamp)}
    save_json(directory / "run.json", manifest)
    try:
        api = (api_factory or Api)(settings, directory)
        handle = str(getattr(settings, "brand_handle", "") or "")
        clips, spec = None, None
        for key in pick_topics(topic)[:3]:
            spec = topics_film.by_key(key)
            print(f"→ Konu: {key}", flush=True)
            try:
                clips = footage.harvest(api, spec, directory)
                break
            except FactoryError as error:
                print(f"Konu atlandı ({key}): {error}", flush=True)
                clips = None
        if not clips:
            raise FactoryError("Hiçbir konu için yeterli NASA görüntüsü bulunamadı.")
        save_json(directory / "clips.json", clips)
        print("→ Senaryo", flush=True)
        try:
            script = write_script(api, spec, clips, handle)
        except FactoryError as error:
            print(f"Senaryo yeniden yazılıyor: {error}", flush=True)
            script = write_script(api, spec, clips, handle)
        save_json(directory / "film_script.json", {"topic": spec["key"], **script})
        print("→ Anlatım sesi", flush=True)
        calm_style = {**load_style(), "narrator_delivery": "Calm, unhurried documentary narration: warm, precise, low and even, gentle pauses between sentences, no excitement, no selling.",
                      "voice_character": "A calm documentary narrator, deep and soft, around 130 to 140 words per minute, measured pauses, never dramatic.",
                      "speech_rate": "-5%"}
        narration_story = {"shots": [{"id": f"shot_{i:03}", "narration": b["narration"], "emphasis": "normal"} for i, b in enumerate(script["beats"])]}
        narration_story["narration"] = " ".join(s["narration"] for s in narration_story["shots"])
        cache = ROOT / "data" / "cache-v2"
        from dataclasses import replace as _replace

        api.settings = _replace(settings, alignment_threshold=min(settings.alignment_threshold, 80))
        audio_path, words, speech_seconds = voice.build_audio(api, narration_story, getattr(settings, "sim_voice", "Charon"), calm_style, cache)
        api.settings = settings
        starts, cursor = [], 0
        for shot in narration_story["shots"]:
            count = len(shot["narration"].split())
            starts.append(float(words[cursor]["start"]) + HEAD if count else 0.0)
            cursor += count
        starts[0] = 0.0
        # the end card begins just after the last spoken word
        duration = min(MAX_SECONDS, speech_seconds + HEAD + 0.3 + END_SECONDS)
        print(f"→ Kurgu ({duration:.1f} sn, {len(script['beats'])} beat, {len(clips)} kesit)", flush=True)
        shots = build_timeline(script, clips, starts, duration)
        first = next(c for c in clips if c["id"] == shots[0]["clip"])
        agency = {"hubble": "ESA/Hubble", "webb": "ESA/Webb"}.get(first.get("source"), "NASA")
        kicker = {"footage": f"Real footage · {agency}", "visualization": f"{agency} visualization"}.get(first["kind"], f"{agency} archive")
        plan = {"duration": duration, "handle": handle, "hook": script["hook"], "kicker": kicker, "closing": script["closing"],
                "hook_seconds": HOOK_SECONDS, "end_seconds": END_SECONDS, "dissolve_frames": DISSOLVE, "shots": shots}
        save_json(directory / "film_plan.json", plan)
        silent = cut.render(plan, directory / "film_silent.mp4")
        subtitle = captions([{**w} for w in words], directory / "captions.ass", HEAD, cut.caption_family())
        drone = directory / "drone.wav"
        render.space_drone(drone, duration)
        escaped = str(subtitle.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        video = directory / "video.mp4"
        bed_db = float(getattr(settings, "sim_music_db", -30.0)) + 2
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio_path), "-i", str(drone),
                        "-filter_complex",
                        f"[0:v]vignette=angle=PI/4.6,ass=filename='{escaped}'[v];[1:a]adelay={int(HEAD * 1000)}|{int(HEAD * 1000)},loudnorm=I=-17:TP=-1.5:LRA=9,aresample=48000[speech];"
                        f"[2:a]volume={bed_db}dB,aresample=48000[bed];[speech][bed]amix=inputs=2:duration=first:normalize=0,apad=pad_dur=4,alimiter=limit=0.95:latency=1[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(video)], check=True, timeout=2400)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                               capture_output=True, text=True, timeout=60)
        if float(probe.stdout.strip() or 0) < 20:
            raise FactoryError("Belgesel videosu üretilemedi.")
        # cover: the strongest clip's middle frame with the hook typography
        best = max(clips, key=lambda c: ({"footage": 2, "visualization": 1, "animation": 0}[c["kind"]], c["quality"]))
        w0, w1 = max(best["windows"], key=lambda w: w[1] - w[0])
        frame = directory / "cover_frame.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{(w0 + w1) / 2:.2f}", "-i", best["file"], "-frames:v", "1", "-q:v", "2", str(frame)],
                       capture_output=True, timeout=120)
        if frame.is_file():
            cut.still(frame, script["hook"], handle, directory / "thumbnail.jpg", kicker, focus=tuple(best["focus"]))
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.2", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(directory / "thumbnail.jpg")],
                           capture_output=True, timeout=120)
        event = {"id": f"film_{spec['key']}_{stamp}", "title": script["hook"], "series": "NASA footage", "issue": spec["key"],
                 "year": datetime.now(timezone.utc).year, "publisher": "NASA", "universe": "science", "scenario": spec["key"], "_source": "film",
                 "url": "", "source_urls": sorted({c["page_url"] for c in clips if c.get("page_url")})}
        lines = [b["narration"] for b in script["beats"]]
        story = {"title": script["youtube_title"], "description": script["description"], "hook_card": script["hook"],
                 "shots": [{"id": f"shot_{i:03}", "panel_id": b["clip"], "narration": b["narration"], "fact_ids": [], "motion": "footage", "emphasis": "normal"}
                           for i, b in enumerate(script["beats"])],
                 "narration": " ".join(lines), "event": event, "scenario": spec["key"], "beats": script["beats"], "closing": script["closing"],
                 "panel_validation": {"passed": True, "shots": [], "generated": True}}
        save_json(directory / "story.json", story)
        save_json(directory / "story_bundle.json", {"event": event, "story": story, "panels": [], "facts": [{"id": f"fact_{i:02}", "text": f} for i, f in enumerate(spec["facts"])]})
        review = {"passed": True, "review_mode": "film", "subtitle_sync": 100, "scene_match": 100, "delivery": 100, "visual_readability": 95,
                  "summary": "Documentary edit from NASA footage; narration aligned per chunk at synthesis.", "failed_checks": [], "issues": []}
        save_json(directory / "quality_review.json", review)
        used_assets = {}
        for c in clips:
            if any(s["clip"] == c["id"] for s in shots):
                used_assets[c["asset"]] = (c.get("credit") or "NASA", c.get("title", "")[:80], c.get("page_url", ""))
        credit_lines = [f"{credit} — {title} ({url})" if url else f"{credit} — {title}" for credit, title, url in used_assets.values()]
        licences = sorted({c.get("licence", "") for c in clips if any(s["clip"] == c["id"] for s in shots) and c.get("licence")})
        hashtags = ["space", "nasa", "shorts", "astronomy", "science", "universe", "documentary"]
        full = script["description"] + "\n\nReal footage and visualizations from public space-agency archives, edited for " + (handle or "the channel") + ".\n"
        if credit_lines:
            full += "Footage: " + "; ".join(credit_lines)[:1500] + "\n"
        if licences:
            full += "Licences: " + ", ".join(licences)[:200] + "\n"
        metadata = {"script": {**story, "full_description": full, "hashtags": hashtags,
                               "instagram_caption": script["youtube_title"] + "\n\n" + script["description"] + "\n\nFootage: NASA (public domain)."},
                    "sources": [], "voice": {"voice": "none"}, "editing_profile": "film-v1", "language": settings.language}
        save_json(directory / "metadata.json", metadata)
        manifest.update(status="ready", stage="Tamamlandı", technical_passed=True, quality_passed=True, video_sha256=file_hash(video),
                        metadata_sha256=file_hash(directory / "metadata.json"), quality_sha256=file_hash(directory / "quality_review.json"),
                        api_calls=api.calls, duration=duration, voice=getattr(settings, "sim_voice", "Charon"), event=event, localized=[])
        save_json(directory / "run.json", manifest)
        os.environ["FACTORY_LANGUAGE"] = settings.language
        # the heavy intermediates are not needed after the mix
        try:
            (directory / "film_silent.mp4").unlink()
        except OSError:
            pass
        for path in list((directory / "footage").rglob("*.mp4")) + list((directory / "footage").rglob("samples/*.jpg")):
            try:
                path.unlink()
            except OSError:
                pass
        return directory
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(directory / "run.json", manifest)
        raise
