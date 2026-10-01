from collections import Counter
import json
from .api import SourceUnavailable
from .core import save_json, language_name
from .panels import verify_shots
from .research import clean


def validate_story(value, panels, facts, max_shots=40):
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
        text, ids = clean(row.get("narration")), row.get("fact_ids")
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
        result.append(
            {
                "id": f"shot_{i:03}",
                "panel_id": row["panel_id"],
                "narration": text,
                "fact_ids": list(dict.fromkeys(ids)),
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
        "shots": result,
        "narration": " ".join(r["narration"] for r in result),
        "repaired_evidence": repaired,
        "skipped_shots": skipped,
    }


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
            )
        }
        for p in panels
    ]
    mode = (
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
HOOK RULE: the very first sentence must state the single most shocking event of the story in <= 12 words, in present tense, naming the hero or villain; never start with "Meanwhile", "In this issue" or scene-setting.
Target {api.settings.target_seconds} seconds, {round(api.settings.target_seconds * 1.8)}..{round(api.settings.target_seconds * 2.2)} {language_name(api)} words, around {desired} shots but at most {api.settings.max_shots}. Adapt duration to VERIFIED material, never invent scenes to fill time.
First line directly states the extraordinary event and matches the opening panel. Concrete cause/effect, coherent evidence-supported chronology, escalation, factual payoff. No generic intro, invented dialogue, filler, or subscribe CTA. Original proper-name spelling. Favor short 3..12 word beats, vary push/pull and vertical movement according to the visible action. Most shots use normal emphasis (yellow captions); use danger/reveal/turn only for meaningful story beats. Each shot 3..30 words, supplied panel_id and 1+ fact_ids. Each panel at most 3 times; at least 6 unique panels. Page IDs are not chronological page numbers. Unseen action cannot be claimed as visible; context must be explicit.
Return {{"title":"{language_name(api)}","description":"{language_name(api)}","shots":[{{"panel_id":"","narration":"{language_name(api)}","fact_ids":[],"motion":"push|pull|left|right|up|down|hold","emphasis":"normal|danger|reveal|turn"}}]}}.
REPAIR FEEDBACK {feedback}""",
        )
        try:
            value = validate_story(draft, panels, facts, api.settings.max_shots)
        except (ValueError, TypeError, KeyError) as error:
            feedback = str(error)
            save_json(
                api.directory / "diagnostics" / f"story_{attempt + 1}.json",
                {"error": feedback, "draft": draft},
            )
            continue
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
