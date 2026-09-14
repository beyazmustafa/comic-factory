import json
import math
from PIL import Image
from .api import SourceUnavailable
from .core import file_hash, save_json
from .research import clean


def box(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError("Geçersiz panel koordinatları.")
    l, t, r, b = value
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        raise ValueError("Panel görüntü sınırları dışında.")
    return value


def quote_present(quote, text):
    quote, text = clean(quote).casefold(), clean(text).casefold()
    return len(quote) >= 18 and quote in text


def confident(value, minimum):
    try:
        return minimum <= float(value) <= 100
    except (ValueError, TypeError):
        return False


def catalog(api, event, pages, articles):
    accepted, facts = [], []
    for offset in range(0, len(pages), 4):
        batch = pages[offset : offset + 4]
        context = [
            {
                "page_id": p["id"],
                "url": p["source_url"],
                "article_title": articles[p["source_id"]]["title"],
                "article_text": articles[p["source_id"]]["text"][:12000],
            }
            for p in batch
        ]
        data = api.json(
            "Gerçek sayfa ve paneller",
            f"""Read these ACTUAL page images and fetched article text. Target exact event: {json.dumps(event, ensure_ascii=False)}
SOURCES: {json.dumps(context, ensure_ascii=False)}
The source must explicitly identify the exact series/issue/year. Matching a character alone is insufficient. Reject covers, fan art, ads, unrelated issues and promotional collages.
Return {{"pages":[{{"page_id":"","belongs_to_issue":true,"source_evidence_quote":"exact >=18-character substring identifying the issue from article","page_role":"interior|cover|other","confidence":0,"panels":[{{"bbox":[left,top,right,bottom],"reading_order":1,"characters":[],"action":"only visible action in Turkish","ocr":"exact visible original dialogue or empty","narrative_fact":"Turkish supported fact","confidence":0}}],"facts":[{{"text":"Turkish event fact","quote":"exact article substring supporting it"}}]}}]}}.
All confidence values 0..100. Coordinates are fractions 0..1 in original image, x then y. Give 1..8 panels in reading order; never infer unseen action or missing ending.""",
            images=[(p["id"], api.directory / p["file"]) for p in batch],
        )
        lookup = {p["id"]: p for p in batch}
        for observed in data.get("pages", []):
            if not isinstance(observed, dict):
                continue
            page = lookup.get(observed.get("page_id"))
            if not page:
                continue
            article = articles[page["source_id"]]
            quote = observed.get("source_evidence_quote", "")
            if (
                observed.get("belongs_to_issue") is not True
                or observed.get("page_role") != "interior"
                or not confident(observed.get("confidence"), 85)
                or not quote_present(quote, article["text"])
            ):
                continue
            for fact in observed.get("facts", []):
                if (
                    isinstance(fact, dict)
                    and fact.get("text")
                    and quote_present(fact.get("quote", ""), article["text"])
                ):
                    facts.append(
                        {
                            "id": f"fact_{len(facts):03}",
                            "text": clean(fact["text"]),
                            "quote": clean(fact["quote"]),
                            "source_url": page["source_url"],
                            "source_id": page["source_id"],
                        }
                    )
            for item in observed.get("panels", [])[:8]:
                if not isinstance(item, dict):
                    continue
                try:
                    bounds = box(item.get("bbox"))
                    width, height = (
                        (bounds[2] - bounds[0]) * page["width"],
                        (bounds[3] - bounds[1]) * page["height"],
                    )
                    if (
                        not confident(
                            item.get("confidence"), api.settings.panel_threshold
                        )
                        or not clean(item.get("action"))
                        or min(width, height) < 180
                        or width * height < 70000
                    ):
                        continue
                except (ValueError, TypeError):
                    continue
                identifier = f"panel_{len(accepted):03}"
                path = (
                    api.directory
                    / "events"
                    / event["id"]
                    / "panels"
                    / (identifier + ".jpg")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(api.directory / page["file"]) as picture:
                    pixels = tuple(
                        round(v * (picture.width if i % 2 == 0 else picture.height))
                        for i, v in enumerate(bounds)
                    )
                    picture.crop(pixels).save(path, "JPEG", quality=97)
                accepted.append(
                    {
                        **item,
                        "id": identifier,
                        "page_id": page["id"],
                        "bbox": bounds,
                        "page_file": page["file"],
                        "file": str(path.relative_to(api.directory)),
                        "sha256": file_hash(path),
                        "source_url": page["source_url"],
                        "source_title": page["source_title"],
                        "source_evidence_quote": quote,
                    }
                )
    save_json(
        api.directory / "events" / event["id"] / "catalog.json",
        {"event": event, "panels": accepted, "facts": facts},
    )
    if len(accepted) < 6 or len(facts) < 3:
        raise SourceUnavailable(
            f"Yeterli doğrulanmış panel/kanıt yok: {len(accepted)} panel, {len(facts)} kanıt."
        )
    return accepted, facts


def verify_shots(api, shots, inventory, facts):
    pl = {p["id"]: p for p in inventory}
    fl = {f["id"]: f for f in facts}
    reports = []
    for offset in range(0, len(shots), 5):
        batch = shots[offset : offset + 5]
        data = [
            {
                "shot_id": s["id"],
                "narration": s["narration"],
                "evidence": [fl[k] for k in s["fact_ids"]],
            }
            for s in batch
        ]
        review = api.json(
            "Cümle-panel eşleşmesi",
            f"""Independently audit the Turkish narration against its ACTUAL cropped panel and provided evidence. Matching character alone is insufficient. Claimed action must be visible; context lines must clearly sound like context and be supported by quoted evidence. Check crop preserves faces/action/relevant dialogue.
{json.dumps(data, ensure_ascii=False)}
Return {{"shots":[{{"shot_id":"","match_score":0,"supported":true,"crop_ok":true,"reason":"Turkish specific explanation"}}]}}. Scores 0..100; 90+ only for clear specific matches.""",
            images=[
                (s["id"], api.directory / pl[s["panel_id"]]["file"]) for s in batch
            ],
        )
        reports.extend(review.get("shots", []))
    by_id = {r.get("shot_id"): r for r in reports if isinstance(r, dict)}
    failures = []
    for shot in shots:
        row = by_id.get(shot["id"], {})
        if (
            row.get("supported") is not True
            or row.get("crop_ok") is not True
            or not confident(row.get("match_score"), api.settings.panel_threshold)
        ):
            failures.append(
                {"shot_id": shot["id"], "reason": row.get("reason", "Doğrulama eksik.")}
            )
    return {"passed": not failures, "shots": reports, "failures": failures}
