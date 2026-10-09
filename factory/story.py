from collections import Counter
import json
import re
from .api import SourceUnavailable
from .core import save_json, language_name
from .panels import verify_shots
from .research import clean


UNIT_WORDS = {
    "en": [
        (r"(?<=\d)\s*°\s*C\b", " degrees Celsius"), (r"(?<=\d)\s*°\s*F\b", " degrees Fahrenheit"), (r"(?<=\d)\s*°(?!\w)", " degrees"),
        (r"(?<=\d)\s*km/h\b", " kilometers per hour"), (r"(?<=\d)\s*km/s\b", " kilometers per second"), (r"(?<=\d)\s*m/s\b", " meters per second"),
        (r"(?<=\d)\s*mph\b", " miles per hour"), (r"(?<=\d)\s*km\b", " kilometers"), (r"(?<=\d)\s*kg\b", " kilograms"),
        (r"(?<=\d)\s*%", " percent"), (r"(?<=\d)\s*mSv\b", " millisieverts"), (r"(?<=\d)\s*ly\b", " light-years"),
        (r"(?<=\d)x(?=\s)", " times"), (r"\b(\d+)\s*-\s*(\d+)\b", r"\1 to \2"),
    ],
    "tr": [
        (r"(?<=\d)\s*°\s*C\b", " derece"), (r"(?<=\d)\s*km/s(?:a|aat)?\b", " kilometre saat"), (r"(?<=\d)\s*km\b", " kilometre"),
        (r"(?<=\d)\s*kg\b", " kilogram"), (r"(?<=\d)\s*%", " yüzde"), (r"%\s*(?=\d)", "yüzde "),
    ],
}


def spoken_form(text, language="en"):
    """Units written the way the narrator will say them, so the caption words
    and the Whisper transcript agree ("1,600 km/h" → "1,600 kilometers per hour")."""
    for pattern, replacement in UNIT_WORDS.get(language, UNIT_WORDS["en"]):
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def validate_story(value, panels, facts, max_shots=40, language="en"):
    pl, fl = {p["id"] for p in panels}, {f["id"] for f in facts}
    # Evidence repair: a small model often cites a fact id that does not exist.
    # Facts from the same page as the panel are the natural replacement.
    page_of = {p["id"]: p.get("page_id") for p in panels}
    facts_by_page = {}
    for fact in facts:
        facts_by_page.setdefault(fact.get("page_id"), []).append(fact["id"])
    repaired = 0
    shots = value.get("shots")
    if (
        not isinstance(shots, list)
        or not 4 <= len(shots) <= max_shots
        or not clean(value.get("title"))
    ):
        raise ValueError("Başlık veya sahne sayısı geçersiz.")
    result, usage, skipped = [], Counter(), []
    for i, row in enumerate(shots):
        if not isinstance(row, dict):
            raise ValueError("Geçersiz sahne.")
        text, ids = spoken_form(clean(row.get("narration")), language), row.get("fact_ids")
        if row.get("panel_id") not in pl or not 3 <= len(text.split()) <= 40:
            # One bad shot must not sink the draft: skip it, keep the rest.
            skipped.append(i + 1)
            continue
        ids = [k for k in ids if k in fl] if isinstance(ids, list) else []
        if not ids:
            ids = list(facts_by_page.get(page_of.get(row["panel_id"]), []))[:2] or [f["id"] for f in facts[:2]]
            repaired += 1
        if not ids:
            raise ValueError(f"Sahne {i + 1}: kaynak kanıtı eksik.")
        if row.get("motion") not in {"push", "pull", "left", "right", "up", "down", "hold"}:
            row["motion"] = "hold"
        if any(v in text for v in ("<", ">", "http://", "https://", chr(96) * 3)):
            raise ValueError("Anlatım düz metin olmalı.")
        usage[row["panel_id"]] += 1
        if usage[row["panel_id"]] > 3:
            skipped.append(i + 1)
            continue
        from .fx import EFFECT_TYPES, validate_graphic

        impact = clean(row.get("impact_word")).strip(".,!?;:\"'")
        if impact and (len(impact.split()) != 1 or impact.casefold() not in {w.strip(".,!?;:\"'").casefold() for w in text.split()}):
            impact = ""
        result.append(
            {
                "id": f"shot_{i:03}",
                "panel_id": row["panel_id"],
                "narration": text,
                "fact_ids": list(dict.fromkeys(ids)),
                "effect": row.get("effect") if row.get("effect") in EFFECT_TYPES else None,
                "graphic": validate_graphic(row.get("graphic")),
                "impact_word": impact,
                "motion": row["motion"],
                "emphasis": row.get("emphasis", "normal") if row.get("emphasis", "normal") in {"normal", "danger", "reveal", "turn"} else "normal",
            }
        )
    if len(result) < 4 or len(skipped) > len(shots) * 0.4:
        raise ValueError(f"Çok fazla geçersiz sahne: {skipped}.")
    if len(usage) < min(6, len(result)):
        raise ValueError("Yeterli farklı panel yok.")
    for index, row in enumerate(result):
        row["id"] = f"shot_{index:03}"
    return {
        "title": clean(value["title"]),
        "description": clean(value.get("description")),
        "hook_card": clean(value.get("hook_card"))[:48],
        "shots": result,
        "narration": " ".join(r["narration"] for r in result),
        "repaired_evidence": repaired,
        "skipped_shots": skipped,
    }


def polish_headline(api, value, event):
    """Second pass on the two things that decide the click: title and hook card."""
    try:
        data = api.json(
            "Başlık ve kanca",
            f"""You write titles for a top {language_name(api)} Shorts channel. Here is a finished narration:
{value["narration"][:1800]}
Current title: {value["title"]}
Current hook card: {value.get("hook_card", "")}
Write 5 candidate titles (40..70 characters, natural {language_name(api)}, name the subject, open a curiosity gap or state the shock, no clickbait lies, no ALL CAPS, no emojis, no colon-heavy SEO style) and 5 candidate hook cards (3..6 words, ALL CAPS, a complete punchy phrase a viewer reads in half a second, e.g. "YOU HAVE 15 SECONDS", "BANE BROKE THE BAT"; must be true to the first sentence). Then pick the best of each.
Return {{"titles":[],"hook_cards":[],"best_title":"","best_hook_card":""}}""",
        )
        title = clean(data.get("best_title"))
        hook = clean(data.get("best_hook_card"))
        if 30 <= len(title) <= 80 and not title.isupper():
            value["title"] = title[:100]
        if 2 <= len(hook.split()) <= 7 and len(hook) <= 48:
            value["hook_card"] = hook.upper() if hook.isascii() else hook
        value["headline_candidates"] = {"titles": data.get("titles", []), "hook_cards": data.get("hook_cards", [])}
    except Exception as error:  # noqa: BLE001 - polish is optional
        print(f"Başlık cilası atlandı: {str(error)[:120]}", flush=True)


def variety_problems(shots, panels, min_words=0):
    """Famous mode: the writer must spread beats over many distinct images."""
    kind = {p["id"]: p.get("kind", "interior") for p in panels}
    problems = []
    words = sum(len(s["narration"].split()) for s in shots)
    if min_words and words < min_words:
        problems.append(f"Narration too short: {words} words; write at least {min_words} words (70+ seconds): fuller sentences with cause, effect and a number, not bullet fragments.")
    repeats = [shots[i]["panel_id"] for i in range(1, len(shots)) if shots[i]["panel_id"] == shots[i - 1]["panel_id"]]
    if repeats:
        problems.append(f"Same panel used in consecutive shots: {sorted(set(repeats))}; change one of each pair.")
    usage = Counter(s["panel_id"] for s in shots)
    over = [p for p, n in usage.items() if n > 2]
    if over:
        problems.append(f"Panels used more than twice: {over}; each panel at most twice.")
    unique = len(usage)
    if unique < min(len(shots), max(8, round(len(shots) * 0.75))):
        problems.append(f"Only {unique} distinct panels for {len(shots)} shots; use at least {max(8, round(len(shots) * 0.75))} distinct panels.")
    covers = [s["panel_id"] for s in shots if kind.get(s["panel_id"]) == "cover"]
    if len(covers) > max(3, len(shots) // 4):
        problems.append(f"{len(covers)} shots use covers; at most {max(3, len(shots) // 4)} cover shots, prefer interior panels.")
    if shots and kind.get(shots[0]["panel_id"]) == "cover":
        problems.append("The opening shot must be an interior panel or promo art, not a cover.")
    return problems


def dedupe_consecutive(shots):
    """Last-attempt fallback: drop a shot that repeats the previous panel when
    the story can spare it (keeps at least 12 shots)."""
    kept = []
    for shot in shots:
        if kept and shot["panel_id"] == kept[-1]["panel_id"] and len(shots) >= 13 and len((kept[-1]["narration"] + " " + shot["narration"]).split()) <= 36:
            # merge the narration into the previous beat instead of showing the same image twice
            kept[-1] = {**kept[-1], "narration": kept[-1]["narration"] + " " + shot["narration"],
                        "fact_ids": list(dict.fromkeys(kept[-1]["fact_ids"] + shot["fact_ids"]))}
            continue
        kept.append(shot)
    for index, row in enumerate(kept):
        row["id"] = f"shot_{index:03}"
    return kept


def create(api, event, panels, facts, style):
    inventory = [
        {
            k: p.get(k)
            for k in (
                "id",
                "page_id",
                "reading_order",
                "characters",
                "action",
                "ocr",
                "narrative_fact",
                "confidence",
                "shot_type",
                "intensity",
                "kind",
            )
            if k in p
        }
        for p in panels
    ]
    famous = event.get("_source") in {"famous", "space"}
    space = event.get("_source") == "space"
    mode = (
        "MODE: a space/science story for a global Shorts audience, told like a thriller: what would ACTUALLY happen, step by step, with numbers the viewer can feel (seconds, degrees, kilometres per second), second person where it fits ('your blood', 'you have 15 seconds'). FACTS are the science; PANELS are real NASA/ESA/telescope photographs with a description each: for every beat pick the image whose visible content fits the sentence best (the Sun's surface for heat, a spacewalk for the body, a nebula for scale); never state something the image contradicts. Each panel carries confidence = relevance and intensity = how awe-inspiring it is; open on the most dramatic relevant image. Panels with kind=video are real moving footage: prefer them for action and transition beats (fire, flight, collapse), each at most twice. Never put two visually similar images back to back (two telescope dishes, two similar nebulae): alternate subjects so every cut feels new. No character names needed; name the object (the Sun, Jupiter, Betelgeuse) in the first sentence. LANGUAGE: simple, spoken English a 12-year-old gets on first hearing: short common words, one idea per sentence, no jargon (no 'millisieverts', 'perchlorates', 'tidal locking' without a plain explanation), numbers rounded and compared to things people know ('hotter than a pizza oven', 'faster than a bullet'). Every beat a complete sentence of 6..14 words, never a fragment. SCALE: at least one comparison that shrinks the cosmos to something in a kitchen or a street ('if the Sun were a basketball, Earth is a peppercorn'). LOOP ENDING: the final sentence must echo the first one so the video loops seamlessly (same image as the opening shot, a line that points back to the start, e.g. 'And it all begins again the moment you fall in'). "
        if space else
        "MODE: one world-famous Marvel/DC moment retold for people who half-remember it. FACTS are the storyline; PANELS are official art, previews and press images related to this storyline, NOT in story order and not one-per-sentence. For each beat choose the panel whose visible content fits best (same characters, matching mood or action; close-ups for emotional lines, wide shots for scale). A panel need not literally show the sentence, but never say something the image contradicts. Name the hero and villain in the first two sentences. Each panel carries confidence = how clearly it belongs to this exact storyline; use 75+ panels for the shocking beats and the opening, lower ones only as fillers. VARIETY IS MANDATORY: never the same panel in two consecutive shots, each panel at most twice, at least 75% of shots on distinct panels, covers (kind=cover) in at most a quarter of shots and never as the opening shot. "
        if famous else
        "MODE: complete public-domain Golden Age issue. Pick the ONE story in the issue that contains the hero's single most bizarre, shocking or dramatic moment (a death, betrayal, grotesque villain, impossible power, cruel twist). Use panels from that story only; ignore other stories in the issue. Open on that moment, then explain how it came to be and how it ends, beat by beat in panel order, present tense, like a top comics-recap Shorts narrator. Mention year/publisher at most once, late. "
        if event.get("identifier")
        else "MODE: one famous Marvel/DC superhero moment, using ONLY the official preview/review panels supplied. Open on the shocking moment itself, then the setup, then the consequence; present tense, hero and villain named; claim only what the panels show or the quoted sources state. "
    )
    feedback = ""
    desired = min(
        api.settings.max_shots,
        max(6, round(api.settings.target_seconds / style["mean_shot_seconds"])),
    )
    for attempt in range(api.settings.repair_attempts):
        draft = api.json(
            "Panellere bağlı anlatım",
            f"""Write an ORIGINAL {language_name(api)} narration using ONLY supplied facts and panels. Use an energetic natural {language_name(api)} voice and evidence-supported chronology.
{mode}
EVENT {json.dumps(event, ensure_ascii=False)}
FACTS {json.dumps(facts, ensure_ascii=False)}
PANELS {json.dumps(inventory, ensure_ascii=False)}
EDITING RHYTHM {style.get("story_structure", "")}
DELIVERY {style.get("narrator_delivery", "")}
LEARNED PLAYBOOK (follow; it is updated from real audience data):
{style.get("playbook", "")}
THIS VIDEO'S EXPERIMENT (apply exactly once, it will be measured): {json.dumps(style.get("experiment"), ensure_ascii=False)}
HOOK RULE: the very first sentence must state the single most shocking fact or event in <= 12 words, in present tense, naming the subject (hero, villain, or the object: the Sun, a black hole); never start with "Meanwhile", "In this issue", "Have you ever wondered" or scene-setting.
Use at least 18 shots when 18 or more distinct panels are available (a Short under 60 seconds feels thin); never pad with invented beats.
Target {api.settings.target_seconds} seconds, {round(api.settings.target_seconds * 1.8)}..{round(api.settings.target_seconds * 2.2)} {language_name(api)} words, around {desired} shots but at most {api.settings.max_shots}. Adapt duration to VERIFIED material, never invent scenes to fill time.
First line directly states the extraordinary event and matches the opening panel. Concrete cause/effect, coherent evidence-supported chronology, escalation, factual payoff. No generic intro, invented dialogue, filler, or subscribe CTA. Original proper-name spelling. Favor short 3..12 word beats, vary push/pull and vertical movement according to the visible action. Most shots use normal emphasis (yellow captions); use danger/reveal/turn only for meaningful story beats. Each shot 3..30 words, supplied panel_id and 1+ fact_ids. Each panel at most 3 times; at least 6 unique panels. Page IDs are not chronological page numbers. Unseen action cannot be claimed as visible; context must be explicit.
TITLE RULE: 40..70 characters, names the character, opens a curiosity gap or states the shock as a question ("Why Spider-Man Blames Himself for Gwen's Death"); no clickbait lies, no ALL CAPS, no emojis.
HOOK CARD: also return "hook_card": the 3..6 word ALL-CAPS line shown over the first frame for one second (e.g. "SPIDER-MAN KILLED HER?", "BANE BROKE THE BAT"); shocking, true, no spoiler beyond the first sentence.
DIRECTOR FIELDS per shot (the editor renders them automatically, use them where they land hardest, never on every shot):
- "impact_word": ONE word from that shot's narration that slams on screen bigger (the number, the death word, the object): use in roughly one shot in three.
- "effect": "shake" (impact, explosion, collision: picture shakes with a red flash), "blackout" (losing consciousness, dying, lights out: picture dims to black), "pressure" (being crushed, pulled in: hard push-in), "flash" (ignition, a star going off), or null. At most 4 effects per video, on the beats where the viewer's body is hit.
- "graphic": one of {{"type":"counter","value":465,"unit":"degrees","label":"Venus surface"}} (a number counting up as it is spoken), {{"type":"scale","big":"Sun","small":"Earth","ratio":109,"label":""}} (true-scale discs, use for the scale comparison), {{"type":"countdown","seconds":15,"label":"to live"}} (a live countdown with heartbeat, for "you have N seconds"), or null. 2..4 graphics per video, on shots whose narration states that exact number.
Return {{"title":"{language_name(api)}","description":"{language_name(api)}","hook_card":"3..6 WORDS","shots":[{{"panel_id":"","narration":"{language_name(api)}","fact_ids":[],"motion":"push|pull|left|right|up|down|hold","emphasis":"normal|danger|reveal|turn","impact_word":"","effect":null,"graphic":null}}]}}.
REPAIR FEEDBACK {feedback}""",
        )
        try:
            value = validate_story(draft, panels, facts, api.settings.max_shots, getattr(api.settings, "language", "en"))
        except (ValueError, TypeError, KeyError) as error:
            feedback = str(error)
            save_json(
                api.directory / "diagnostics" / f"story_{attempt + 1}.json",
                {"error": feedback, "draft": draft},
            )
            continue
        if famous:
            # Commentary illustrations: relevance was judged at collection time.
            # What is checked instead is variety: no panel twice in a row, no
            # panel more than twice, covers and promo art as a minority.
            problems = variety_problems(value["shots"], panels, min_words=round(api.settings.target_seconds * 1.2))
            if problems and attempt < api.settings.repair_attempts - 1:
                feedback = "; ".join(problems)
                save_json(api.directory / "diagnostics" / f"story_{attempt + 1}.json",
                          {"error": feedback, "draft": draft})
                continue
            if problems:
                value["shots"] = dedupe_consecutive(value["shots"])
                value["narration"] = " ".join(s["narration"] for s in value["shots"])
            if space and len(value["shots"]) >= 8:
                # Seamless loop: the last frame is the first frame.
                value["shots"][-1]["panel_id"] = value["shots"][0]["panel_id"]
                value["shots"][-1]["motion"] = "pull" if value["shots"][0].get("motion") == "push" else "push"
            polish_headline(api, value, event)
            value.update(event=event, panel_validation={"passed": True, "shots": [], "mode": "relevance",
                                                        "variety_warnings": problems})
            save_json(api.directory / "story.json", value)
            return value
        report = verify_shots(api, value["shots"], panels, facts)
        save_json(
            api.directory / "diagnostics" / f"panel_matches_{attempt + 1}.json", report
        )
        if report["passed"]:
            value.update(event=event, panel_validation=report)
            save_json(api.directory / "story.json", value)
            return value
        # Last attempt: drop the shots the auditor rejected instead of
        # discarding a whole issue over a few disputed sentences.
        failed = {f["shot_id"] for f in report["failures"]}
        kept = [s for s in value["shots"] if s["id"] not in failed]
        minimum = max(8, round(len(value["shots"]) * 0.7))
        if attempt == api.settings.repair_attempts - 1 and len(kept) >= minimum:
            try:
                trimmed = validate_story({**draft, "shots": kept}, panels, facts, api.settings.max_shots)
            except (ValueError, TypeError, KeyError):
                trimmed = None
            if trimmed:
                passed = [r for r in report["shots"] if isinstance(r, dict) and r.get("shot_id") not in failed]
                trimmed.update(
                    event=event,
                    panel_validation={"passed": True, "shots": passed, "failures": [],
                                      "dropped_shots": sorted(failed)},
                )
                print(f"Denetimi geçemeyen {len(failed)} sahne çıkarıldı; {len(kept)} sahne kaldı.", flush=True)
                save_json(api.directory / "story.json", trimmed)
                return trimmed
        feedback = json.dumps(report["failures"], ensure_ascii=False)
    raise SourceUnavailable(
        "Anlatım ile gerçek paneller yeterli kesinlikte eşleşmedi; raporlar kaydedildi."
    )
