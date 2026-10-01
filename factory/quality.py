import json
import math
import subprocess
from .core import ffmpeg_binary, inspect_media, save_json
from .api import FactoryError
from .style import validate_style

ADJUSTABLE = {
    "font_size",
    "stroke_width",
    "caption_y",
    "caption_x",
    "text_color",
    "active_color",
    "stroke_color",
    "background_color",
    "background",
    "transition",
    "transition_frames",
    "zoom_amount",
    "caption_mode",
    "caption_words",
    "uppercase",
    "panel_framing",
    "panel_max_width",
    "panel_max_height",
    "panel_center_y",
}


def validate_review(report, duration=None):
    failures = []
    for key in (
        "candidate_observed",
        "turkish_narration",
        "no_critical_errors",
    ):
        if report.get(key) is not True:
            failures.append(key)
    for key, minimum in (
        ("subtitle_sync", 90),
        ("scene_match", 90),
        ("delivery", 85),
        ("visual_readability", 85),
    ):
        value = report.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not minimum <= value <= 100
        ):
            failures.append(key)
    observations = report.get("observations")
    if (
        not isinstance(observations, list)
        or len(observations) < 3
        or any(
            not isinstance(row, dict)
            or not row.get("detail")
            or type(row.get("second")) not in (int, float)
            for row in observations
        )
    ):
        failures.append("observations")
    else:
        seconds = [row["second"] for row in observations]
        if any(not math.isfinite(s) or s < 0 for s in seconds) or len(set(seconds)) < 3:
            failures.append("observation_times")
        elif duration is not None and (
            max(seconds) > duration + 0.5
            or max(seconds) - min(seconds) < duration * 0.4
        ):
            failures.append("observation_coverage")
    report.update(
        passed=not failures, failed_checks=failures, subjective_assessment=True
    )
    return report


def review(api, video, story, style):
    duration = float(inspect_media(video)["format"]["duration"])
    prompt = f"""Watch/listen to the ENTIRE generated CANDIDATE. Evaluate actual decoded output, never plans. Explicitly set observed=false if inaccessible.
Evaluate panel framing, caption readability, pacing and Turkish narrator energy against the production settings below. Assess only this generated video.
SCRIPT {json.dumps(story["shots"], ensure_ascii=False)}
PRODUCTION SETTINGS {json.dumps(style, ensure_ascii=False)}
Check burned Turkish words vs heard speech including later scenes/joins, matching panel changes, cut-off faces/actions, Turkish glyph readability, audio artifacts, uncomfortable pauses and coherent payoff.
Return {{"candidate_observed":true,"turkish_narration":true,"no_critical_errors":true,"subtitle_sync":0,"scene_match":0,"delivery":0,"visual_readability":0,"observations":[{{"second":0,"detail":"Turkish concrete audible/visible observation"}}],"issues":[],"style_adjustments":{{}},"summary":"Turkish"}}.
Scores 0..100 are subjective assessments, not measured accuracy. At least SIX observations across start/middle/end of the candidate ({duration:.2f}s). If only layout/color/crop/zoom/transition issues exist, propose style_adjustments restricted to {sorted(ADJUSTABLE)}. Never hide a speech or source problem as a layout change."""
    try:
        report = api.video_json("Üretilen video kontrolü", prompt, video)
        report["review_mode"] = "video"
    except FactoryError as error:
        api.note(f"Video kontrolü Gemini'de yapılamadı ({str(error)[:120]}); kare incelemesine geçildi.")
        report = frame_review(api, video, story, style, duration)
    validate_review(report, duration)
    save_json(api.directory / "quality_review.json", report)
    return report


def extract_frames(video, duration, directory, count=5):
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        second = round(duration * (index + 0.5) / count, 2)
        path = directory / f"frame_{index:02}.jpg"
        subprocess.run(
            [ffmpeg_binary(), "-y", "-v", "error", "-ss", str(second), "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(path)],
            check=True, capture_output=True, timeout=120,
        )
        frames.append((second, path))
    return frames


def frame_review(api, video, story, style, duration):
    """Visual review from still frames plus measured speech alignment.

    Used only when no model can watch the video. Sync and delivery come from
    the Whisper alignment that was already measured for every audio chunk,
    scene match from the independent panel audit; the model judges only what
    a still frame can show (caption readability, framing, Turkish glyphs).
    """
    aligned = json.loads((api.directory / "aligned_words.json").read_text(encoding="utf-8"))
    chunk_scores = [c["score"] for c in aligned.get("chunks", []) if isinstance(c, dict)]
    sync = min(chunk_scores) if chunk_scores else 0
    audit = story.get("panel_validation", {}).get("shots", [])
    matches = [float(r.get("match_score", 0)) for r in audit if isinstance(r, dict)]
    scene = min(matches) if matches else 0
    frames = extract_frames(video, duration, api.directory / "diagnostics" / "frames")
    judged = api.json(
        "Kare incelemesi",
        f"""These are still frames from the generated vertical video at the listed seconds. Judge ONLY what is visible: Turkish caption readability (size, stroke, Turkish glyphs ÇĞİÖŞÜ), whether the comic panel fills the frame without cut faces or balloons, and overall layout. Production settings: {json.dumps(style, ensure_ascii=False)}
Return {{"visual_readability":0,"framing_ok":true,"observations":[{{"second":0,"detail":"Turkish concrete visible observation"}}],"issues":[],"style_adjustments":{{}}}}. One observation per frame; scores 0..100. style_adjustments restricted to {sorted(ADJUSTABLE)}.""",
        images=[(f"frame at {second}s", path) for second, path in frames],
    )
    observations = []
    for (second, _), row in zip(frames, judged.get("observations", [])):
        if isinstance(row, dict) and row.get("detail"):
            observations.append({"second": second, "detail": str(row["detail"])})
    try:
        readability = float(judged.get("visual_readability", 0))
    except (TypeError, ValueError):
        readability = 0
    return {
        "review_mode": "frames",
        "candidate_observed": bool(frames),
        "turkish_narration": sync >= api.settings.alignment_threshold,
        "no_critical_errors": judged.get("framing_ok") is True and sync >= api.settings.alignment_threshold,
        "subtitle_sync": sync,
        "scene_match": scene,
        "delivery": sync,
        "visual_readability": readability,
        "observations": observations,
        "issues": judged.get("issues", []),
        "style_adjustments": judged.get("style_adjustments", {}),
        "summary": "Video modeli erişilemediği için kare incelemesi ve ölçülen ses hizalaması kullanıldı.",
    }


def adjusted_style(style, report):
    if any(
        key in report.get("failed_checks", [])
        for key in (
            "candidate_observed",
            "turkish_narration",
            "subtitle_sync",
            "scene_match",
            "delivery",
        )
    ):
        return None
    changes = report.get("style_adjustments", {})
    if not isinstance(changes, dict) or not changes or set(changes) - ADJUSTABLE:
        return None
    try:
        result = validate_style({**style, **changes})
    except (FactoryError, ValueError, TypeError):
        return None
    return result if result != style else None
