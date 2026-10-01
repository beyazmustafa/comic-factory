import json
import math
from PIL import Image
from .api import SourceUnavailable
from .core import file_hash, save_json, language_name
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
Return {{"pages":[{{"page_id":"","belongs_to_issue":true,"source_evidence_quote":"exact >=18-character substring identifying the issue from article","page_role":"interior|cover|other","confidence":0,"panels":[{{"bbox":[left,top,right,bottom],"reading_order":1,"characters":[],"action":"only visible action in {language_name(api)}","ocr":"exact visible original dialogue or empty","narrative_fact":"{language_name(api)} supported fact","confidence":0}}],"facts":[{{"text":"{language_name(api)} event fact","quote":"exact article substring supporting it"}}]}}]}}.
All confidence values 0..100. Coordinates are fractions 0..1 in original image, x then y. Give 1..8 panels in reading order; never infer unseen action or missing ending.""",
            images=[(p["id"], api.directory / p["file"]) for p in batch],
            list_key="pages",
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
            f"""Independently audit the {language_name(api)} narration against its ACTUAL cropped panel and provided evidence. Matching character alone is insufficient. Claimed action must be visible; context lines must clearly sound like context and be supported by quoted evidence. Check crop preserves faces/action/relevant dialogue.
{json.dumps(data, ensure_ascii=False)}
Return {{"shots":[{{"shot_id":"","match_score":0,"supported":true,"crop_ok":true,"reason":"{language_name(api)} specific explanation"}}]}}. Scores 0..100; 90+ only for clear specific matches.""",
            images=[
                (s["id"], api.directory / pl[s["panel_id"]]["file"]) for s in batch
            ],
            list_key="shots",
        )
        rows = [r for r in review.get("shots", []) if isinstance(r, dict)]
        if rows and len(rows) == len(batch) and not all(r.get("shot_id") for r in rows):
            # Same order as sent: restore ids a small model dropped.
            for row, shot in zip(rows, batch):
                row.setdefault("shot_id", shot["id"])
        reports.extend(rows)
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


# ---------------------------------------------------------------------------
# Full-page issues (public domain archive): geometric panel segmentation.
# The model no longer guesses coordinates; it only describes numbered panels.
# ---------------------------------------------------------------------------

import numpy as np
from PIL import ImageDraw


def _light_mask(gray, paper):
    # Yellowed Golden Age paper with JPEG noise: be generous about "paper".
    return gray >= max(140, paper - 50)


def _gutter_runs(fraction, minimum_run, threshold=0.9, dark=None, reach=8):
    """Return (start, end) index ranges where almost every pixel is paper.

    With `dark` (per-row/column fraction of ink-dark pixels), a run counts only
    when a panel border line sits within `reach` pixels on at least one side;
    pale sky inside a panel has no border next to it and is left alone.
    """
    runs, start = [], None
    for index, value in enumerate(fraction):
        if value >= threshold:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= minimum_run:
                runs.append((start, index))
            start = None
    if start is not None and len(fraction) - start >= minimum_run:
        runs.append((start, len(fraction)))
    if dark is None:
        return runs
    kept = []
    for s, e in runs:
        before = dark[max(0, s - reach) : s].max() if s > 0 else 0.0
        after = dark[e : e + reach].max() if e < len(dark) else 0.0
        if max(before, after) >= 0.5:
            kept.append((s, e))
    return kept


def _content_box(mask):
    dark = ~mask
    rows, cols = np.where(dark.any(axis=1))[0], np.where(dark.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _split(mask, box, axis, depth, output, min_w, min_h, fallback=False, ink=None):
    l, t, r, b = box
    region = mask[t:b, l:r]
    if region.size == 0:
        return
    content = _content_box(region)
    if content is None:
        return
    l, t, r, b = l + content[0], t + content[1], l + content[2], t + content[3]
    region = mask[t:b, l:r]
    height, width = region.shape
    if depth >= 4:
        output.append((l, t, r, b))
        return
    fraction = region.mean(axis=1 if axis == 0 else 0)
    dark = ink[t:b, l:r].mean(axis=1 if axis == 0 else 0) if ink is not None else None
    length = height if axis == 0 else width
    runs = [
        (s, e)
        for s, e in _gutter_runs(fraction, max(2, length // 400), dark=dark)
        if s > (min_h if axis == 0 else min_w) * 0.5 and e < length - (min_h if axis == 0 else min_w) * 0.5
    ]
    if not runs:
        if not fallback and depth < 4:
            # No gutter this way: try the other direction once before accepting.
            _split(mask, (l, t, r, b), 1 - axis, depth, output, min_w, min_h, fallback=True, ink=ink)
        else:
            output.append((l, t, r, b))
        return
    cuts = [0] + [(s + e) // 2 for s, e in runs] + [length]
    for start, end in zip(cuts, cuts[1:]):
        child = (l, t + start, r, t + end) if axis == 0 else (l + start, t, l + end, b)
        _split(mask, child, 1 - axis, depth + 1, output, min_w, min_h, ink=ink)


def segment_page(picture, limit=12):
    """Split a scanned comic page into panel boxes (fractions, reading order)."""
    gray = np.asarray(picture.convert("L").resize(
        (720, max(1, round(picture.height * 720 / picture.width))), Image.Resampling.BILINEAR
    ), dtype=np.uint8)
    height, width = gray.shape
    paper = float(np.percentile(gray, 88))
    mask = _light_mask(gray, paper)
    ink = gray < max(60, paper - 110)  # panel border lines and lettering
    content = _content_box(mask)
    if content is None:
        return []
    boxes = []
    _split(mask, content, 0, 0, boxes, width * 0.12, height * 0.08, ink=ink)
    cleaned = []
    for l, t, r, b in boxes:
        w, h = r - l, b - t
        if w < width * 0.12 or h < height * 0.08 or w * h < width * height * 0.02:
            continue
        if w / h > 6 or h / w > 6:
            continue
        pad_x, pad_y = max(2, w // 60), max(2, h // 60)
        cleaned.append(
            (
                max(0, l - pad_x) / width,
                max(0, t - pad_y) / height,
                min(width, r + pad_x) / width,
                min(height, b + pad_y) / height,
            )
        )
    # Reading order: rows of similar top, then left to right.
    cleaned.sort(key=lambda bx: (round(bx[1] * 12), bx[0]))
    return [list(bx) for bx in cleaned[:limit]]


def page_layout_score(boxes):
    if not boxes:
        return 0.0
    return sum((r - l) * (b - t) for l, t, r, b in boxes)


def select_story_pages(pages, directory, maximum, minimum_run=4):
    """Choose the first contiguous run of panelled interior pages (skip cover/ads)."""
    annotated = []
    for page in pages:
        with Image.open(directory / page["file"]) as picture:
            boxes = segment_page(picture)
        coverage = page_layout_score(boxes)
        story = len(boxes) >= 2 and coverage >= 0.35
        annotated.append({**page, "panel_boxes": boxes, "coverage": coverage, "story_like": story})
    best, current = [], []
    for page in annotated[1:]:  # index 0 is almost always the cover
        if page["story_like"] or (current and len(page["panel_boxes"]) == 1 and page["coverage"] >= 0.5):
            current.append(page)
        else:
            if len(current) > len(best):
                best = current
            current = []
    if len(current) > len(best):
        best = current
    if len(best) < minimum_run:
        raise SourceUnavailable(
            f"Sayıda art arda yeterli panelli hikâye sayfası bulunamadı ({len(best)})."
        )
    return best[:maximum]


def _annotated_copy(directory, page, destination):
    with Image.open(directory / page["file"]) as source:
        picture = source.convert("RGB")
    draw = ImageDraw.Draw(picture)
    for number, (l, t, r, b) in enumerate(page["panel_boxes"], start=1):
        pixels = (l * picture.width, t * picture.height, r * picture.width, b * picture.height)
        draw.rectangle(pixels, outline="#FF0000", width=max(3, picture.width // 300))
        label = str(number)
        size = max(28, picture.width // 24)
        draw.rectangle((pixels[0], pixels[1], pixels[0] + size * 1.3, pixels[1] + size * 1.25), fill="#FF0000")
        draw.text((pixels[0] + size * 0.25, pixels[1] + size * 0.05), label, fill="#FFFFFF")
    destination.parent.mkdir(parents=True, exist_ok=True)
    picture.save(destination, "JPEG", quality=88)
    return destination


def catalog_archive(api, event, pages, articles):
    """Describe geometrically cut panels of a full public-domain issue."""
    story_pages = select_story_pages(pages, api.directory, api.settings.max_pages)
    article = articles["source_000"]
    accepted, facts = [], []
    root = api.directory / "events" / event["id"]
    for offset in range(0, len(story_pages), 3):
        batch = story_pages[offset : offset + 3]
        images = []
        for page in batch:
            images.append((page["id"], _annotated_copy(api.directory, page, root / "annotated" / (page["id"] + ".jpg"))))
        layout = {p["id"]: len(p["panel_boxes"]) for p in batch}
        data = api.json(
            "Arşiv sayfası panelleri",
            f"""These are ACTUAL interior pages of the public-domain issue {json.dumps({k: event.get(k) for k in ('title', 'series', 'issue', 'year', 'publisher')}, ensure_ascii=False)}. Panels are already cut and numbered with red labels (counts per page: {json.dumps(layout)}). Do not propose coordinates.
For every numbered panel describe ONLY what is visible. Transcribe visible dialogue/captions exactly (original language). Mark keep=false for ads, text pages, mislabelled boxes, half panels, boxes that cut faces/speech balloons, and boxes that contain MORE THAN ONE panel (a video shot must show exactly one panel).
Also list story facts established by these pages; each fact must quote exact visible text (>=18 characters) from a balloon or caption on these pages.
Return {{"pages":[{{"page_id":"","page_role":"interior|cover|ad|text|other","panels":[{{"number":1,"keep":true,"characters":[],"action":"visible action in {language_name(api)}","ocr":"exact visible text or empty","narrative_fact":"{language_name(api)} supported fact","confidence":0}}],"facts":[{{"text":"{language_name(api)} story fact","quote":"exact visible text"}}]}}]}}. Confidence 0..100.""",
            images=images,
            list_key="pages",
        )
        lookup = {p["id"]: p for p in batch}
        for observed in data.get("pages", []):
            if not isinstance(observed, dict):
                continue
            page = lookup.get(observed.get("page_id"))
            if not page or observed.get("page_role") != "interior":
                continue
            page_ocr = " ".join(
                clean(item.get("ocr")) for item in observed.get("panels", []) if isinstance(item, dict)
            )
            evidence_text = article["text"] + " " + page_ocr
            for item in observed.get("panels", []):
                if not isinstance(item, dict) or item.get("keep") is not True:
                    continue
                number = item.get("number")
                if type(number) is not int or not 1 <= number <= len(page["panel_boxes"]):
                    continue
                if not confident(item.get("confidence"), api.settings.panel_threshold) or not clean(item.get("action")):
                    continue
                bounds = page["panel_boxes"][number - 1]
                width = (bounds[2] - bounds[0]) * page["width"]
                height = (bounds[3] - bounds[1]) * page["height"]
                if min(width, height) < 180 or width * height < 70000:
                    continue
                identifier = f"panel_{len(accepted):03}"
                path = root / "panels" / (identifier + ".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(api.directory / page["file"]) as picture:
                    pixels = tuple(
                        round(v * (picture.width if i % 2 == 0 else picture.height))
                        for i, v in enumerate(bounds)
                    )
                    picture.crop(pixels).save(path, "JPEG", quality=97)
                accepted.append(
                    {
                        "id": identifier,
                        "page_id": page["id"],
                        "reading_order": number,
                        "characters": item.get("characters") if isinstance(item.get("characters"), list) else [],
                        "action": clean(item.get("action")),
                        "ocr": clean(item.get("ocr")),
                        "narrative_fact": clean(item.get("narrative_fact")),
                        "confidence": item.get("confidence"),
                        "bbox": bounds,
                        "page_file": page["file"],
                        "file": str(path.relative_to(api.directory)),
                        "sha256": file_hash(path),
                        "source_url": page["source_url"],
                        "source_title": page["source_title"],
                        "source_evidence_quote": article.get("public_domain_evidence", ""),
                    }
                )
            for fact in observed.get("facts", []):
                if isinstance(fact, dict) and clean(fact.get("text")) and quote_present(fact.get("quote", ""), evidence_text):
                    facts.append(
                        {
                            "id": f"fact_{len(facts):03}",
                            "text": clean(fact["text"]),
                            "quote": clean(fact["quote"]),
                            "source_url": page["source_url"],
                            "source_id": "source_000",
                            "page_id": page["id"],
                        }
                    )
    save_json(root / "catalog.json", {"event": event, "panels": accepted, "facts": facts,
                                      "story_pages": [p["id"] for p in story_pages]})
    if len(accepted) < 6 or len(facts) < 3:
        raise SourceUnavailable(
            f"Yeterli doğrulanmış panel/kanıt yok: {len(accepted)} panel, {len(facts)} kanıt."
        )
    return accepted, facts
