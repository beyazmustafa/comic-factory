"""Studio source: an original superhero universe drawn by image models.

Nothing here comes from an existing publisher. The first run asks the writer
model to invent a small universe (heroes, villains, a house art style) and
draws one character sheet per hero. Every later run writes a fresh "issue"
in that universe, then draws one panel per shot with the hero's character
sheet as a visual reference, so the cast stays consistent from video to
video. The universe and the list of published issues live in data/history/
(committed by the workflow), so continuity survives across runs.
"""

from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from google.genai import types
from PIL import Image

from .api import FactoryError, ModelUnavailable, ProviderOverloaded, SourceUnavailable
from .config import ROOT
from .core import file_hash, language_name, save_json

STYLE_PREFIX = "Comic book illustration, inked line art with bold black outlines, flat cel shading, halftone dots, printed comic page look."
STYLE_GUIDE = (
    "modern mainstream superhero comic book art, bold confident ink lines, dynamic cinematic composition, "
    "dramatic lighting, rich saturated colors with subtle halftone texture, detailed backgrounds, "
    "expressive faces, no text, no speech balloons, no captions, no watermark, no signature"
)
IMAGE_HINTS = ("image", "banana", "imagen")
IMAGE_EXCLUDE = ("embedding", "video", "veo", "tts", "audio")
POLLINATIONS = "https://image.pollinations.ai/prompt/"
PROTECTED_NAMES = (
    "spider", "batman", "superman", "wolverine", "hulk", "thor", "iron man", "captain america", "flash",
    "wonder woman", "green lantern", "aquaman", "deadpool", "venom", "joker", "thanos", "magneto", "x-men",
    "avengers", "justice league", "daredevil", "punisher", "black panther", "doctor strange", "captain marvel",
    "shazam", "blue beetle", "nightwing", "robin", "catwoman", "harley", "darkseid", "galactus", "silver surfer",
    "ghost rider", "moon knight", "hawkeye", "black widow", "scarlet witch", "vision", "ant-man", "wasp",
    "cyborg", "raven", "starfire", "supergirl", "batgirl", "hellboy", "spawn", "invincible",
)


def universe_path() -> Path:
    return ROOT / "data" / "history" / "universe.json"


def sheets_dir() -> Path:
    return ROOT / "data" / "history" / "universe"


def load_universe():
    try:
        data = json.loads(universe_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("heroes") else None
    except (OSError, ValueError):
        return None


def protected(name: str) -> bool:
    lower = name.casefold()
    return any(p in lower for p in PROTECTED_NAMES)


# ----------------------------------------------------------------- images
def image_models(api):
    """Image-capable Gemini models this key can use, newest first."""
    found = api.discover()
    names = [m for m in found.get("gemini_raw", []) if any(h in m.casefold() for h in IMAGE_HINTS)
             and not any(x in m.casefold() for x in IMAGE_EXCLUDE)]
    configured = str(getattr(api.settings, "image_model", "") or "").strip()
    ordered = ([configured] if configured else []) + sorted(names, reverse=True)
    return list(dict.fromkeys(m for m in ordered if m))


def gemini_image(api, prompt, references, aspect="3:4"):
    """One image from a Gemini image model; references are PIL images (character sheets)."""
    errors = []
    for model in [m for m in image_models(api) if m not in api.dead]:
        contents = [prompt]
        for picture in references:
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, "JPEG", quality=90)
            contents.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"))

        def call(model=model):
            options = dict(response_modalities=["IMAGE", "TEXT"])
            try:
                options["image_config"] = types.ImageConfig(aspect_ratio=aspect)
            except Exception:
                pass
            try:
                config = types.GenerateContentConfig(**options)
            except Exception:
                options.pop("image_config", None)
                config = types.GenerateContentConfig(**options)
            return api.client.models.generate_content(model=model, contents=contents, config=config)

        try:
            # Quota errors on free tiers are permanent for the day; one quick
            # attempt per model, then move on instead of sleeping for minutes.
            response = api.request(f"Panel çizimi ({model})", call, attempts=1)
        except Exception as error:  # noqa: BLE001 - any failure: next model
            api.dead.add(model)
            errors.append(f"{model}: {str(error)[:160]}")
            continue
        for candidate in getattr(response, "candidates", None) or []:
            for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None) and str(getattr(inline, "mime_type", "")).startswith("image/"):
                    data = inline.data if isinstance(inline.data, (bytes, bytearray)) else __import__("base64").b64decode(inline.data)
                    return Image.open(io.BytesIO(bytes(data))).convert("RGB"), model
        errors.append(f"{model}: görsel döndürmedi")
    raise ProviderOverloaded("; ".join(errors) or "Gemini görsel modeli yok.")


def pollinations_image(prompt, seed, width=1024, height=1408, note=None):
    """Keyless fallback generator (Flux). Less consistent, always available,
    but rate limited for anonymous callers: wait and retry with growing gaps."""
    base = POLLINATIONS + quote(prompt[:900])
    last = ""
    for attempt in range(6):
        model = "flux" if attempt % 2 == 0 else "turbo"
        url = f"{base}?width={width}&height={height}&seed={seed}&model={model}&nologo=true&enhance=false&safe=true"
        try:
            response = requests.get(url, timeout=(15, 150), headers={"User-Agent": "comic-factory/1.0"})
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            picture = Image.open(io.BytesIO(response.content)).convert("RGB")
            if picture.width < 256:
                raise RuntimeError("tiny image")
            # Anonymous renders carry a small logo strip at the bottom: crop it.
            picture = picture.crop((0, 0, picture.width, int(picture.height * 0.93)))
            time.sleep(4)  # stay under the anonymous rate limit for the next panel
            return picture
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {str(error)[:120]}"
            if note:
                note(f"Pollinations deneme {attempt + 1}: {last}")
            time.sleep(min(90, 12 * (attempt + 1)))
    raise SourceUnavailable(f"Yedek görsel üretici de yanıt vermedi ({last}).")


def draw(api, prompt, references, seed, note=None):
    try:
        picture, model = gemini_image(api, prompt, references)
        return picture, "gemini:" + model
    except (ProviderOverloaded, FactoryError) as error:
        if note:
            note(f"Gemini görsel modeli kullanılamadı ({str(error)[:600]}); Pollinations ile çizilecek.")
        return pollinations_image(prompt, seed, note=note), "pollinations"


# --------------------------------------------------------------- universe
def ensure_universe(api, note=print):
    """Create the universe once: cast, style, and one character sheet per hero."""
    universe = load_universe()
    if not universe:
        data = api.json(
            "Özgün evren",
            f"""Invent an ORIGINAL superhero universe for a YouTube Shorts channel that retells one dramatic "issue" per video in {language_name(api)}. It must not resemble any existing Marvel/DC/Image character in name, costume or powers.
Return {{"name":"universe name","tagline":"one line","style":"two sentences describing the house art style (modern, cinematic, gritty but colorful)","heroes":[{{"name":"","alias":"civilian name","powers":"","personality":"","visual":"VERY specific and stable: face, hair, skin, age, body, costume colors and shapes, emblem, cape yes/no, accessories — 60 words","weakness":""}} x4],"villains":[{{"name":"","powers":"","visual":"specific 40 words","motive":""}} x4],"setting":"city/world in 40 words","themes":["3 recurring story themes"]}}.
Names must be fresh (not a known hero/villain name). Keep visuals distinct from each other (different silhouettes and palettes).""",
        )
        heroes = [h for h in data.get("heroes", []) if isinstance(h, dict) and h.get("name") and h.get("visual") and not protected(h["name"])]
        villains = [v for v in data.get("villains", []) if isinstance(v, dict) and v.get("name") and v.get("visual") and not protected(v["name"])]
        if len(heroes) < 2 or len(villains) < 2:
            raise SourceUnavailable("Evren üretilemedi (yeterli özgün karakter yok).")
        universe = {
            "name": str(data.get("name", "Untitled Universe")),
            "tagline": str(data.get("tagline", "")),
            "style": str(data.get("style", "")),
            "setting": str(data.get("setting", "")),
            "themes": [str(t) for t in data.get("themes", [])][:5],
            "heroes": heroes[:4],
            "villains": villains[:4],
            "episodes": [],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Save before drawing: the cast must never be re-invented because one
        # character sheet failed to render.
        save_json(universe_path(), universe)
        note(f"Evren kuruldu: {universe['name']} — {', '.join(h['name'] for h in universe['heroes'])}")
    # Character sheets: draw the missing ones (resumable across runs).
    sheets_dir().mkdir(parents=True, exist_ok=True)
    changed = False
    for index, hero in enumerate(universe["heroes"]):
        sheet = ROOT / str(hero.get("sheet", "") or "")
        if hero.get("sheet") and sheet.is_file():
            continue
        api.check()
        prompt = (f"{STYLE_PREFIX} Character reference sheet of {hero['name']}: {hero['visual']}. Full body front view, three-quarter view and a close-up of the face, "
                  f"neutral grey background. {universe['style']} {STYLE_GUIDE}")
        try:
            picture, provider = draw(api, prompt, [], seed=1000 + index, note=note)
        except SourceUnavailable as error:
            note(f"Karakter kartı çizilemedi ({hero['name']}): {error}")
            continue
        path = sheets_dir() / (re.sub(r"[^a-z0-9]+", "-", hero["name"].casefold()).strip("-") + ".jpg")
        picture.save(path, "JPEG", quality=92)
        hero["sheet"] = str(path.relative_to(ROOT))
        hero["sheet_provider"] = provider
        changed = True
    if changed:
        save_json(universe_path(), universe)
    return universe


def next_hero(universe):
    counts = {h["name"]: 0 for h in universe["heroes"]}
    for episode in universe.get("episodes", []):
        counts[episode.get("hero", "")] = counts.get(episode.get("hero", ""), 0) + 1
    return min(universe["heroes"], key=lambda h: counts.get(h["name"], 0))


# ------------------------------------------------------------------ issue
def write_issue(api, universe, hero, style):
    previous = [e.get("title") for e in universe.get("episodes", [])][-12:]
    villain_names = [v["name"] for v in universe["villains"]]
    desired = min(api.settings.max_shots, max(18, round(api.settings.target_seconds / style["mean_shot_seconds"])))
    data = api.json(
        "Özgün sayı senaryosu",
        f"""Write issue #{len(universe.get("episodes", [])) + 1} of an original superhero comic for a YouTube Shorts recap in {language_name(api)}.
UNIVERSE: {json.dumps({k: universe[k] for k in ("name", "tagline", "setting", "themes", "style")}, ensure_ascii=False)}
HERO (lead): {json.dumps({k: hero.get(k) for k in ("name", "alias", "powers", "personality", "weakness", "visual")}, ensure_ascii=False)}
VILLAINS available: {json.dumps(universe["villains"], ensure_ascii=False)}
PREVIOUS ISSUES (do not repeat): {json.dumps(previous, ensure_ascii=False)}
PLAYBOOK: {style.get("playbook", "")}
EXPERIMENT: {json.dumps(style.get("experiment"), ensure_ascii=False)}
Build ONE shocking, self-contained story with a real twist and a costly ending (a death, betrayal, loss of power, impossible choice). Open on the shocking moment itself in <= 12 words, then show how it came to be, beat by beat, present tense, like a top comics-recap narrator; end on the consequence, no outro.
{desired} shots (18..{api.settings.max_shots}). Each shot: "narration" 3..18 words; "image" = a precise panel description for an illustrator (who is in frame BY NAME from the cast, action, expression, camera angle, location, lighting; 25-45 words; no text in image); "motion" push|pull|left|right|up|down|hold; "emphasis" normal|danger|reveal|turn (at most 1 in 4 colored).
Return {{"title":"{language_name(api)} curiosity title 45-70 chars with the hero's name","description":"{language_name(api)} one sentence","villain":"name from the cast","shots":[{{"narration":"","image":"","motion":"push","emphasis":"normal","characters":["names in frame"]}}]}}.""",
        list_key="shots",
    )
    shots = [s for s in data.get("shots", []) if isinstance(s, dict) and 3 <= len(str(s.get("narration", "")).split()) <= 20 and len(str(s.get("image", "")).split()) >= 8]
    if len(shots) < 12:
        raise SourceUnavailable(f"Senaryo yetersiz ({len(shots)} sahne).")
    return {
        "title": str(data.get("title") or f"{hero['name']} #{len(universe.get('episodes', [])) + 1}")[:100],
        "description": str(data.get("description") or ""),
        "villain": str(data.get("villain") or ""),
        "shots": shots[: api.settings.max_shots],
    }


def produce(api, universe, style, note=print):
    """Write an issue, draw its panels, return (event, script-like draft, inventory, facts)."""
    hero = next_hero(universe)
    issue = write_issue(api, universe, hero, style)
    number = len(universe.get("episodes", [])) + 1
    event = {
        "id": f"studio_{number:04}_{re.sub(r'[^a-z0-9]+', '-', hero['name'].casefold()).strip('-')}",
        "title": issue["title"],
        "series": universe["name"],
        "issue": str(number),
        "year": datetime.now(timezone.utc).year,
        "publisher": "Original",
        "universe": universe["name"],
        "hero": hero["name"],
        "villain": issue["villain"],
        "url": "https://github.com/beyazmustafa/comic-factory",
        "source_urls": [],
        "_source": "studio",
    }
    root = api.directory / "events" / event["id"]
    (root / "panels").mkdir(parents=True, exist_ok=True)
    references = []
    for cast in universe["heroes"]:
        sheet = ROOT / cast.get("sheet", "")
        if cast.get("sheet") and sheet.is_file():
            cast["_image"] = Image.open(sheet).convert("RGB")
    cast_lookup = {str(c.get("name", "")).casefold(): c for c in universe["heroes"] + universe.get("villains", []) if isinstance(c, dict)}
    inventory, facts, draft_shots = [], [], []
    for index, shot in enumerate(issue["shots"]):
        api.check()
        names = [str(n) for n in shot.get("characters", []) if isinstance(n, str)]
        refs = [cast_lookup[n.casefold()]["_image"] for n in names if n.casefold() in cast_lookup and cast_lookup[n.casefold()].get("_image")]
        if not refs and hero.get("_image"):
            refs = [hero["_image"]]
        descriptions = "; ".join(f"{n}: {cast_lookup[n.casefold()]['visual']}" for n in names if n.casefold() in cast_lookup)
        prompt = (f"{STYLE_PREFIX} {shot['image']} Characters must match the reference images exactly. "
                  f"Character designs: {descriptions}. {universe['style']} {STYLE_GUIDE}")
        picture, provider = draw(api, prompt, refs[:2], seed=number * 100 + index, note=note)
        identifier = f"panel_{index:03}"
        path = root / "panels" / (identifier + ".jpg")
        picture.save(path, "JPEG", quality=95)
        inventory.append({
            "id": identifier, "page_id": identifier, "reading_order": index + 1,
            "characters": names, "action": shot["narration"], "ocr": "", "narrative_fact": shot["narration"],
            "confidence": 100, "bbox": [0, 0, 1, 1], "page_file": str(path.relative_to(api.directory)),
            "file": str(path.relative_to(api.directory)), "sha256": file_hash(path),
            "source_url": event["url"], "source_title": universe["name"], "image_prompt": shot["image"],
            "image_provider": provider,
        })
        facts.append({"id": f"fact_{index:03}", "text": shot["narration"], "quote": shot["image"],
                      "source_url": event["url"], "source_id": "studio", "page_id": identifier})
        draft_shots.append({"panel_id": identifier, "narration": shot["narration"], "fact_ids": [f"fact_{index:03}"],
                            "motion": shot.get("motion", "push"), "emphasis": shot.get("emphasis", "normal")})
    for cast in universe["heroes"]:
        cast.pop("_image", None)
    save_json(root / "catalog.json", {"event": event, "panels": inventory, "facts": facts, "issue": issue})
    draft = {"title": issue["title"], "description": issue["description"], "shots": draft_shots}
    return event, draft, inventory, facts


def remember_episode(universe, event, run_id):
    universe.setdefault("episodes", []).append({
        "number": int(event["issue"]), "hero": event["hero"], "villain": event.get("villain", ""),
        "title": event["title"], "run_id": run_id,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    save_json(universe_path(), universe)
