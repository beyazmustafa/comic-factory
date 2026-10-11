"""Editorial craft for the space channel: structure, variety and restraint.

The first videos all sounded the same: a number in every line, the same
basketball comparison, a flash on shot one, a counter on shot two. Devices
that are used every time stop being devices. This module gives the writer
a different story shape each video, a ledger of what the last videos did
(so it cannot repeat them), hard budgets for numbers and effects, and a
second pass by an editor model that reads the script as a script.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .config import ROOT
from .core import language_name

STORY_SHAPES = [
    ("journey", "A journey the viewer takes in first person, stage by stage, each stage worse than the last; the ending is where the journey leaves them."),
    ("countdown", "A clock is running from the first line; every beat is a moment on that clock; the last beat is zero."),
    ("mystery", "Open on a strange fact nobody can explain, show the three clues scientists found, land the explanation, then the unsettling implication."),
    ("witness", "Tell it through a real event and the people who saw it (a mission, a storm, a night in 1859), present tense, then what it means for us now."),
    ("myth-buster", "Open on what everybody believes, show why it is wrong with one vivid demonstration, replace it with the stranger truth."),
    ("scale", "Start at human size and keep changing the scale until the viewer feels small, then flip it: the one way we matter."),
    ("what-if", "One hypothetical, taken seriously: the first minute, the first day, the first year; consequences cascade in order."),
    ("inside", "Go inside the object layer by layer (surface, atmosphere, core), what the viewer sees, hears and feels at each layer, then out again."),
]

HOOK_TYPES = [
    ("claim", "a bold declarative claim that sounds impossible but is true"),
    ("question", "a direct question the viewer cannot answer and must know"),
    ("scene", "drop the viewer into the middle of the moment, present tense, no setup"),
    ("contradiction", "two facts that cannot both be true, placed side by side"),
    ("you", "address the viewer's own body or home directly"),
    ("number", "one astonishing number, stated plainly, then what it means"),
]

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
                "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion", "percent")


def has_number(text):
    lower = text.casefold()
    if re.search(r"\d", lower):
        return True
    words = re.findall(r"[a-z]+", lower)
    return any(w in NUMBER_WORDS for w in words)


def recent_scripts(limit=6):
    """story.json of the last published videos (newest first)."""
    folder = ROOT / "data" / "history" / "videos"
    if not folder.is_dir():
        return []
    runs = sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    scripts = []
    for run in runs:
        path = run / "story.json"
        if not path.is_file():
            continue
        try:
            scripts.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
        if len(scripts) >= limit:
            break
    return scripts


def ledger(scripts):
    """What the recent videos already did: shapes, hooks, comparisons, devices."""
    rows = []
    for script in scripts:
        shots = script.get("shots", [])
        craft = script.get("craft", {})
        comparisons = [s["narration"] for s in shots if re.search(r"\b(if the .+ were|the size of|as big as|like a|as hot as|as fast as)\b", s["narration"], re.I)]
        rows.append({
            "title": script.get("title", ""),
            "first_line": shots[0]["narration"] if shots else "",
            "shape": craft.get("shape"), "hook_type": craft.get("hook_type"),
            "comparisons": comparisons[:3],
            "graphics": [s.get("graphic", {}).get("type") for s in shots if s.get("graphic")],
            "effects": [s.get("effect") for s in shots if s.get("effect")],
            "numbers_share": round(sum(has_number(s["narration"]) for s in shots) / max(1, len(shots)), 2),
        })
    return rows


def choose_plan(scripts):
    """Rotate story shape and hook type away from the recent videos."""
    used_shapes = [s.get("craft", {}).get("shape") for s in scripts[:4]]
    used_hooks = [s.get("craft", {}).get("hook_type") for s in scripts[:3]]
    shapes = [s for s in STORY_SHAPES if s[0] not in used_shapes] or STORY_SHAPES
    hooks = [h for h in HOOK_TYPES if h[0] not in used_hooks] or HOOK_TYPES
    shape, hook = random.choice(shapes), random.choice(hooks)
    return {"shape": shape[0], "shape_brief": shape[1], "hook_type": hook[0], "hook_brief": hook[1]}


def craft_brief(plan, rows):
    """The paragraph that goes into the writer's prompt."""
    avoid_lines = "; ".join(f'"{r["first_line"]}"' for r in rows[:5] if r.get("first_line"))
    avoid_comp = "; ".join(c for r in rows[:5] for c in r.get("comparisons", [])[:1])
    return (
        f"STORY SHAPE for this video: {plan['shape'].upper()} — {plan['shape_brief']} "
        f"HOOK TYPE: {plan['hook_type'].upper()} — {plan['hook_brief']} "
        "RESTRAINT (this is what separates a documentary from a listicle): numbers in at most one shot in three, and only a number that changes how the viewer feels; "
        "no 'X times heavier/bigger' more than once per video; at most ONE scale comparison, and never one used recently; "
        "let sentences breathe (8..18 words), vary rhythm (a long sentence, then a four-word one); no purple adjectives ('fragile', 'massive', 'terrifying', 'deadly' are banned — show it instead); "
        "no 'surprisingly', 'yet', 'meanwhile' crutches; one idea per shot but shots should connect with cause ('because', 'so', 'until'). "
        "DEVICES are rare: at most 2 graphics and 2 effects per video, never both on the same shot, never on shot 1, never the same device type two videos in a row; impact words on at most 4 shots. "
        f"RECENT OPENINGS TO AVOID REPEATING: {avoid_lines or 'none'}. RECENT COMPARISONS TO AVOID: {avoid_comp or 'none'}. "
    )


def device_problems(shots, rows):
    """Hard budgets the writer must meet (feedback strings, empty when fine)."""
    problems = []
    graphics = [s for s in shots if s.get("graphic")]
    effects = [s for s in shots if s.get("effect")]
    impacts = [s for s in shots if s.get("impact_word")]
    numbers = sum(has_number(s["narration"]) for s in shots)
    if len(graphics) > 2:
        problems.append(f"{len(graphics)} graphics; at most 2.")
    if len(effects) > 2:
        problems.append(f"{len(effects)} effects; at most 2.")
    if len(impacts) > 4:
        problems.append(f"{len(impacts)} impact words; at most 4.")
    if shots and (shots[0].get("graphic") or shots[0].get("effect")):
        problems.append("Shot 1 must carry no graphic and no effect (the hook card is the opening device).")
    if any(s.get("graphic") and s.get("effect") for s in shots):
        problems.append("A shot may have a graphic or an effect, not both.")
    if numbers > max(3, len(shots) // 3 + 1):
        problems.append(f"Numbers in {numbers} of {len(shots)} shots; at most one shot in three may contain a number.")
    times = [s["narration"] for s in shots if re.search(r"\b(times|x)\s+(heavier|bigger|larger|more|hotter|faster|the size)", s["narration"], re.I)]
    if len(times) > 1:
        problems.append("'X times bigger/heavier' used more than once; keep one at most.")
    banned = [w for w in ("fragile", "massive", "terrifying", "deadly", "mind-blowing", "insane") if any(re.search(rf"\b{w}\b", s["narration"], re.I) for s in shots)]
    if banned:
        problems.append(f"Banned adjectives used: {banned}; show instead of telling.")
    previous = rows[0] if rows else {}
    if previous:
        if graphics and previous.get("graphics") and graphics[0]["graphic"]["type"] == previous["graphics"][0]:
            problems.append(f"The first graphic is '{previous['graphics'][0]}' again, same as the last video; use another type or none.")
    return problems


def editor_pass(api, value, event, plan):
    """A second model reads the script as a script and tightens it: same
    shots, same panels, same facts; better sentences. Fails safe."""
    shots = [{"id": s["id"], "narration": s["narration"]} for s in value["shots"]]
    try:
        data = api.json(
            "Senaryo editörü",
            f"""You are the script editor of a premium science channel (think the writing of Kurzgesagt or a BBC documentary, spoken, not written). Below is a draft Shorts narration about "{event['title']}". Story shape: {plan['shape']}. Hook type: {plan['hook_type']}.
Edit for a professional ear: remove clichés and listicle rhythm, cut needless numbers (keep the ones that carry the story), replace telling adjectives with concrete images, make beats connect with cause and consequence, vary sentence length, keep the first sentence a real hook and the last sentence a resonant close that echoes the first. Keep EXACTLY the same shot ids and count; keep every fact true; keep plain spoken {language_name(api)}; keep each shot 6..18 words; do not add new claims.
DRAFT: {json.dumps(shots, ensure_ascii=False)}
Return {{"shots":[{{"id":"","narration":""}}],"notes":"one line on what you changed"}}""",
            list_key="shots",
        )
    except Exception as error:  # noqa: BLE001 - the draft stands when the editor is unavailable
        print(f"Editör geçişi atlandı: {str(error)[:120]}", flush=True)
        return value
    rows = {r.get("id"): " ".join(str(r.get("narration", "")).split()) for r in data.get("shots", []) if isinstance(r, dict)}
    if len(rows) != len(value["shots"]):
        return value
    edited = []
    for shot in value["shots"]:
        text = rows.get(shot["id"], "")
        words = len(text.split())
        if not 5 <= words <= 20 or any(v in text for v in ("<", ">", "http")):
            return value  # one bad line: keep the draft as a whole
        impact = shot.get("impact_word", "")
        if impact and impact.casefold() not in {w.strip(".,!?;:\"'").casefold() for w in text.split()}:
            impact = ""
        edited.append({**shot, "narration": text, "impact_word": impact})
    value["shots"] = edited
    value["narration"] = " ".join(s["narration"] for s in edited)
    value["editor_notes"] = str(data.get("notes", ""))[:200]
    return value
