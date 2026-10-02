"""Self-improvement loop.

Every published video leaves a record (what was made, how) and later gains
its YouTube numbers. Before each new video the model reads that history, the
previous video's quality review and the current playbook, rewrites the
playbook and chooses ONE deliberate experiment for the next video. The
story writer receives both, so each video is a measured step away from the
last one rather than a repeat of a fixed template.

Everything lives in data/history/ (committed by the workflow), so the loop
survives cache loss and is readable by a human.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .core import language_name, save_json



def history_dir() -> Path:
    return ROOT / "data" / "history"


def performance_path() -> Path:
    return history_dir() / "performance.json"


def playbook_path() -> Path:
    return history_dir() / "playbook.md"


def experiments_path() -> Path:
    return history_dir() / "experiments.json"

SEED_PLAYBOOK = """# Playbook (auto-evolving)
- Hook: the first sentence names the shocking event in <= 12 words; no "in this issue".
- Rhythm: 3-10 word beats, one idea per shot, average shot 2.5-4 s.
- Panels: lead with the most dramatic panel; faces and speech balloons must stay inside the crop.
- Emphasis: red for danger, green for reveals, cyan for turns; at most 1 in 4 shots.
- Ending: stop on the strongest image and consequence; no outro, no call to action.
- Title: curiosity-driven, 45-70 characters, names the hero or villain, never a lie.
"""


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_performance() -> list[dict]:
    rows = _load(performance_path(), [])
    return rows if isinstance(rows, list) else []


def load_playbook() -> str:
    try:
        return playbook_path().read_text(encoding="utf-8")
    except OSError:
        return SEED_PLAYBOOK


def load_experiments() -> list[dict]:
    rows = _load(experiments_path(), [])
    return rows if isinstance(rows, list) else []


def story_profile(script: dict, duration: float | None) -> dict:
    shots = script.get("shots", [])
    words = sum(len(s.get("narration", "").split()) for s in shots)
    emphasis = {}
    for shot in shots:
        emphasis[shot.get("emphasis", "normal")] = emphasis.get(shot.get("emphasis", "normal"), 0) + 1
    first = shots[0]["narration"] if shots else ""
    return {
        "shots": len(shots),
        "words": words,
        "duration": round(duration, 1) if duration else None,
        "words_per_minute": round(words / duration * 60) if duration else None,
        "hook": first,
        "hook_words": len(first.split()),
        "title": script.get("title", ""),
        "title_length": len(script.get("title", "")),
        "emphasis": emphasis,
    }


def record_publication(run_directory: Path, platform: str, video_id: str) -> None:
    """Called right after a successful upload; one row per video."""
    manifest = _load(run_directory / "run.json", {})
    metadata = _load(run_directory / "metadata.json", {})
    review = _load(run_directory / "quality_review.json", {})
    script = metadata.get("script", {})
    rows = load_performance()
    if any(r.get("video_id") == video_id for r in rows):
        return
    experiments = load_experiments()
    current = next((e for e in reversed(experiments) if e.get("run_id") == manifest.get("run_id")), None)
    rows.append(
        {
            "video_id": video_id,
            "platform": platform,
            "run_id": manifest.get("run_id"),
            "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": {k: manifest.get("event", {}).get(k) for k in ("title", "series", "issue", "year", "publisher")},
            "profile": story_profile(script, manifest.get("duration")),
            "voice": manifest.get("voice"),
            "review": {
                "mode": review.get("review_mode"),
                "scores": {k: review.get(k) for k in ("subtitle_sync", "scene_match", "delivery", "visual_readability")},
                "issues": review.get("issues", [])[:6],
                "summary": review.get("summary", ""),
            },
            "experiment": current.get("experiment") if current else None,
            "stats": {},
        }
    )
    save_json(performance_path(), rows)


def refresh_stats(note=print) -> list[dict]:
    """Pull view/like/comment counts for every published video (best effort)."""
    rows = load_performance()
    ids = [r["video_id"] for r in rows if r.get("platform") == "youtube" and r.get("video_id")]
    if not ids:
        return rows
    try:
        from .publishing import prepare_youtube_credentials
        from .publishers import youtube

        prepare_youtube_credentials()
        client = youtube.create_youtube_client()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for offset in range(0, len(ids), 50):
            batch = ids[offset : offset + 50]
            response = client.videos().list(part="statistics", id=",".join(batch)).execute()
            for item in response.get("items", []):
                stats = item.get("statistics", {})
                for row in rows:
                    if row.get("video_id") == item.get("id"):
                        published = row.get("published_at", now)
                        try:
                            hours = max(1.0, (datetime.fromisoformat(now) - datetime.fromisoformat(published)).total_seconds() / 3600)
                        except ValueError:
                            hours = 1.0
                        views = int(stats.get("viewCount", 0) or 0)
                        row["stats"] = {
                            "views": views,
                            "likes": int(stats.get("likeCount", 0) or 0),
                            "comments": int(stats.get("commentCount", 0) or 0),
                            "hours_live": round(hours, 1),
                            "views_per_hour": round(views / hours, 2),
                            "fetched_at": now,
                        }
                        history = row.setdefault("stats_history", [])
                        history.append({"h": round(hours, 1), "views": views, "likes": row["stats"]["likes"]})
                        del history[:-40]
                        row["stats"]["views_24h"] = views_at(history, 24)
                        row["stats"]["views_72h"] = views_at(history, 72)
        save_json(performance_path(), rows)
    except Exception as error:  # Stats are a bonus; production never waits on them.
        note(f"YouTube istatistikleri alınamadı ({type(error).__name__}: {str(error)[:120]}).")
    return rows


def views_at(history, hour):
    """Views at a fixed age, interpolated from snapshots; None until reached."""
    points = sorted((p["h"], p["views"]) for p in history if isinstance(p, dict))
    if not points or points[-1][0] < hour:
        return None
    previous = (0.0, 0)
    for h, v in points:
        if h >= hour:
            if h == previous[0]:
                return v
            ratio = (hour - previous[0]) / (h - previous[0])
            return round(previous[1] + (v - previous[1]) * ratio)
        previous = (h, v)
    return points[-1][1]


def score_of(row):
    """Comparable performance: views at 24h when known, else views per hour."""
    stats = row.get("stats") or {}
    if stats.get("views_24h") is not None:
        return float(stats["views_24h"]), "views_24h"
    return float(stats.get("views_per_hour") or 0), "views_per_hour"


def judge_experiments(rows, experiments):
    """Code-computed verdicts: the experiment video against the median of the
    three videos before it, on the same metric, once it is at least 12h old."""
    by_run = {r.get("run_id"): r for r in rows if r.get("platform") == "youtube" and not r.get("language")}
    ordered = [r for r in rows if r.get("platform") == "youtube" and not r.get("language")]
    changed = False
    for item in experiments:
        if item.get("verdict") in {"kept", "dropped"} and item.get("judged_by") == "numbers":
            continue
        row = by_run.get(item.get("run_id"))
        if not row or (row.get("stats") or {}).get("hours_live", 0) < 12:
            continue
        index = ordered.index(row)
        previous = ordered[max(0, index - 3) : index]
        if not previous:
            continue
        own, metric = score_of(row)
        baseline = sorted(score_of(r)[0] for r in previous)[len(previous) // 2]
        item["measured"] = {"metric": metric, "video": own, "baseline_median": baseline}
        item["verdict"] = "kept" if own >= baseline * 1.15 else ("dropped" if own < baseline * 0.85 else "neutral")
        item["judged_by"] = "numbers"
        changed = True
    return changed


def insights(rows):
    """Top vs bottom third, by the comparable score, on the levers the writer controls."""
    scored = [(score_of(r)[0], r) for r in rows if r.get("platform") == "youtube" and not r.get("language") and (r.get("stats") or {}).get("hours_live", 0) >= 6]
    if len(scored) < 4:
        return {}
    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, len(scored) // 3)
    top, bottom = [r for _, r in scored[:k]], [r for _, r in scored[-k:]]
    keys = ("duration", "shots", "words_per_minute", "hook_words", "title_length")

    def mean(group, key):
        values = [r["profile"].get(key) for r in group if isinstance(r["profile"].get(key), (int, float))]
        return round(sum(values) / len(values), 1) if values else None

    def emphasis_share(group):
        totals = sum(sum(r["profile"].get("emphasis", {}).values()) for r in group) or 1
        colored = sum(sum(v for k2, v in r["profile"].get("emphasis", {}).items() if k2 != "normal") for r in group)
        return round(colored / totals, 2)

    return {
        "top_titles": [r["profile"]["title"] for r in top],
        "bottom_titles": [r["profile"]["title"] for r in bottom],
        "top_means": {k: mean(top, k) for k in keys} | {"colored_emphasis_share": emphasis_share(top)},
        "bottom_means": {k: mean(bottom, k) for k in keys} | {"colored_emphasis_share": emphasis_share(bottom)},
        "metric": score_of(scored[0][1])[1],
    }


LEVERS = (
    "hook wording and length", "number of shots and target duration", "words per beat and words per minute",
    "which beats get red/green/cyan emphasis and how often", "title pattern", "story structure (what to open on, where the twist goes)",
    "which panels to favour (close-ups, action, reactions)", "ending beat",
)


def evolve(api, style: dict) -> dict:
    """Rewrite the playbook and pick this video's experiment; returns extras for the story prompt."""
    rows = refresh_stats(getattr(api, "note", print))
    playbook = load_playbook()
    experiments = load_experiments()
    if judge_experiments(rows, experiments):
        save_json(experiments_path(), experiments[-60:])
    numbers = insights(rows)
    ranked = sorted(rows, key=lambda r: r.get("stats", {}).get("views_per_hour", 0), reverse=True)
    history = [
        {
            "title": r.get("profile", {}).get("title"),
            "hook": r.get("profile", {}).get("hook"),
            "shots": r.get("profile", {}).get("shots"),
            "duration": r.get("profile", {}).get("duration"),
            "wpm": r.get("profile", {}).get("words_per_minute"),
            "emphasis": r.get("profile", {}).get("emphasis"),
            "stats": r.get("stats"),
            "review_issues": r.get("review", {}).get("issues"),
            "experiment": r.get("experiment"),
        }
        for r in ranked[:12]
    ]
    last_experiment = experiments[-1] if experiments else None
    previous_issues = rows[-1].get("review", {}) if rows else {}
    try:
        verdict = api.json(
            "Oyun kitabı güncelleme",
            f"""You are the showrunner of a YouTube Shorts channel that retells public-domain Golden Age {api.settings.channel_theme} comics in {language_name(api)}, panel by panel with an energetic narrator and word-by-word captions. Improve the next video using evidence only.
CURRENT PLAYBOOK:
{playbook}
PUBLISHED VIDEOS (best first by views per hour; stats may be empty for new videos):
{json.dumps(history, ensure_ascii=False)}
LAST VIDEO'S QUALITY REVIEW: {json.dumps(previous_issues, ensure_ascii=False)}
LAST EXPERIMENT (verdict computed from view counts by code when "judged_by" is "numbers"): {json.dumps(last_experiment, ensure_ascii=False)}
MEASURED DIFFERENCES between the best and worst third of videos (empty until enough data): {json.dumps(numbers, ensure_ascii=False)}
LEVERS YOU CONTROL (the only things a rule or experiment may change): {json.dumps(LEVERS)}. Rendering features that do not exist (caption colours per speaker, music, fonts, animations) must never appear in the playbook.
Rules: keep what the numbers support, drop what they contradict, fix what the review flagged. The section starting with "## Fixed rules" is written by the showrunner and must be copied into the new playbook VERBATIM and unchanged; your own rules go above it. Do not invent statistics. Choose exactly ONE new experiment for the next video — a concrete, checkable change in hook, pacing, emphasis, title style or structure that differs from the last experiment — so the following run can measure it. Never weaken accuracy: narration must still match the panels.
Return {{"playbook":"markdown, max 14 bullet lines, concrete and testable","experiment":{{"name":"short","change":"one sentence instruction to the writer","rationale":"one sentence"}},"verdict_on_last_experiment":"kept|dropped|unknown","notes":"one sentence"}}.""",
        )
        new_playbook = str(verdict.get("playbook") or "").strip()
        experiment = verdict.get("experiment") if isinstance(verdict.get("experiment"), dict) else None
        if new_playbook.count("\n") <= 30 and len(new_playbook) > 80:
            playbook_path().parent.mkdir(parents=True, exist_ok=True)
            fixed = ""
            if "## Fixed rules" in playbook:
                fixed = "\n\n## Fixed rules" + playbook.split("## Fixed rules", 1)[1].rstrip() + "\n"
            body = new_playbook.replace("# Playbook (auto-evolving)", "").split("## Fixed rules")[0].strip()
            playbook_path().write_text("# Playbook (auto-evolving)\n" + body + fixed, encoding="utf-8")
            playbook = load_playbook()
        if last_experiment and last_experiment.get("judged_by") != "numbers" and verdict.get("verdict_on_last_experiment") in {"kept", "dropped"}:
            last_experiment["verdict"] = verdict["verdict_on_last_experiment"]
            last_experiment["judged_by"] = "model"
        if experiment and experiment.get("change"):
            experiments.append(
                {
                    "run_id": getattr(api, "run_id", ""),
                    "chosen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "experiment": {k: str(experiment.get(k, ""))[:300] for k in ("name", "change", "rationale")},
                }
            )
        save_json(experiments_path(), experiments[-60:])
        save_json(api.directory / "learning.json", {"verdict": verdict, "history_rows": len(rows)})
    except Exception as error:
        getattr(api, "note", print)(f"Oyun kitabı güncellenemedi ({type(error).__name__}: {str(error)[:120]}); mevcut kurallar kullanılıyor.")
    current = experiments[-1]["experiment"] if experiments and experiments[-1].get("run_id") == getattr(api, "run_id", "") else None
    return {"playbook": playbook, "experiment": current}
