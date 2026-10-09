"""Space & science source: "what would actually happen" stories over real
NASA / ESA / JWST imagery.

Why this source exists: it is the one format where a zero-budget channel can
have professional visuals. NASA's image library is public domain and served
through a free, keyless API; the European agencies publish under CC BY. The
pipeline picks a topic from a curated pool of questions people cannot scroll
past, gathers facts from the open web plus the model's own knowledge, writes
the script with a hook card, then fetches one real image per beat.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from urllib.parse import quote

import requests
from PIL import Image, ImageOps

from .api import SourceUnavailable
from .core import language_name, save_json
from .research import Fetcher, clean, search

NASA_API = "https://images-api.nasa.gov/search"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# (key, title question, image search terms, angle)
TOPICS = [
    # Ordered by scroll-stopping power; the picker favours the top of the list.
    ("black-hole-body", "What Happens to Your Body Inside a Black Hole", ["black hole", "accretion disk", "Sagittarius A*", "M87 black hole"], "you are stretched into a strand of atoms while time stops for everyone watching"),
    ("space-no-suit", "You Have 15 Seconds in Space Without a Suit", ["astronaut spacewalk", "EVA ISS", "astronaut helmet reflection"], "saliva boils, lungs rupture if you hold your breath, you do not freeze"),
    ("fall-into-sun", "What Happens If You Fall Into the Sun", ["sun surface SDO", "solar prominence", "solar flare", "corona"], "stage by stage: skin, bones, atoms; the 8 minutes nobody survives"),
    ("venus-death", "You Would Last 3 Seconds on Venus", ["Venus surface Magellan", "Venus clouds Akatsuki", "Venera"], "crushed by 92 atmospheres, cooked at 465°C, dissolved by sulfuric acid"),
    ("jupiter-fall", "There Is No Floor on Jupiter: You Fall Forever", ["Jupiter Juno", "Jupiter Great Red Spot", "Jupiter clouds Juno"], "pressure until you become metallic hydrogen"),
    ("earth-stops", "If Earth Stopped Spinning You Would Fly East at 1,600 km/h", ["Earth from space", "ISS Earth night lights", "Earth horizon"], "oceans migrate to the poles, atmosphere keeps moving"),
    ("sun-dies", "The Day the Sun Eats the Earth", ["red giant", "Helix Nebula", "planetary nebula", "white dwarf"], "oceans boil in 1 billion years, Mercury and Venus swallowed"),
    ("betelgeuse", "Betelgeuse Could Explode Tonight", ["Betelgeuse", "supernova remnant", "Crab Nebula", "Cassiopeia A"], "brighter than the full Moon, visible in daylight for months"),
    ("neutron-teaspoon", "A Teaspoon of Neutron Star Would Fall Through Earth", ["neutron star", "pulsar", "magnetar"], "a billion tonnes; it drills to the core"),
    ("magnetar", "The Magnetar Would Kill You From the Moon's Distance", ["magnetar", "neutron star artist", "SGR 1806-20"], "wipes credit cards at 100,000 km, dissolves you at 1,000 km"),
    ("gamma-ray-burst", "The Explosion That Could Sterilize Earth in Seconds", ["gamma ray burst", "hypernova", "neutron star merger"], "ozone gone, UV burns, mass extinction; WR 104"),
    ("moon-gone", "What If the Moon Disappeared Tonight", ["Moon full", "Moon surface LRO", "Earth and Moon from space"], "tides collapse, days shorten, Earth wobbles into ice ages"),
    ("asteroid-city", "What a City-Killer Asteroid Does in 10 Seconds", ["asteroid Bennu", "Chelyabinsk meteor", "impact crater", "DART impact"], "airburst, shockwave, 100 km of broken windows"),
    ("solar-storm", "The Solar Storm That Would Send Us Back to 1859", ["coronal mass ejection", "solar flare SDO", "aurora from ISS"], "Carrington event, transformers melt, months without power"),
    ("andromeda", "Andromeda Is Coming at 110 km/s", ["Andromeda galaxy", "galaxy collision", "Antennae galaxies"], "4.5 billion years, no two stars collide, the sky catches fire"),
    ("universe-end", "The 3 Ways the Universe Dies", ["JWST deep field", "Hubble ultra deep field", "galaxy cluster"], "heat death, big rip, big crunch; the last black hole evaporates"),
    ("pillars-gone", "The Pillars of Creation May Already Be Gone", ["Pillars of Creation JWST", "Eagle Nebula", "Pillars of Creation Hubble"], "6,500 light-years; we are watching the past"),
    ("voyager", "Voyager 1 Is Still Whispering From 24 Billion Km Away", ["Voyager spacecraft", "Voyager golden record", "pale blue dot"], "22 hours per message, 1977 technology, dies around 2030"),
    ("time-dilation", "Astronauts Come Back Younger Than You", ["ISS orbit", "astronaut ISS cupola", "GPS satellite"], "0.01 seconds a year; GPS fails in a day without Einstein"),
    ("europa-ocean", "There Is an Ocean Under Europa's Ice", ["Europa Juno", "Europa surface Galileo", "Europa Clipper"], "twice Earth's water, hydrothermal vents, the best place for alien life"),
    ("titan-fly", "On Titan You Could Fly by Flapping Your Arms", ["Titan Cassini", "Titan lakes radar", "Huygens Titan surface"], "methane rain, -180°C, seas of liquid gas"),
    ("io-volcano", "Io: The Moon Turning Itself Inside Out", ["Io Juno", "Io volcano plume", "Io Galileo"], "400 volcanoes, lava fountains 400 km high"),
    ("olympus-mons", "A Volcano So Big You Cannot See It Is a Mountain", ["Olympus Mons", "Mars volcano", "Valles Marineris"], "3x Everest, wider than France"),
    ("saturn-float", "Saturn Would Float in Your Bathtub", ["Saturn Cassini", "Saturn rings close", "Saturn hexagon"], "density below water, a hexagon storm 2x Earth"),
    ("mercury-extreme", "Mercury: Fried by Day, Frozen by Night", ["Mercury MESSENGER", "Mercury surface craters", "Mercury BepiColombo"], "430°C to -180°C, ice in the shadows"),
    ("oumuamua", "The Interstellar Object That Sped Up on Its Own", ["Oumuamua artist", "interstellar object", "comet Borisov"], "cigar shaped, no tail, accelerated"),
    ("great-attractor", "Something Is Pulling Our Galaxy at 600 km/s", ["Laniakea", "galaxy cluster", "Norma cluster"], "the Great Attractor, hidden behind the Milky Way"),
    ("pluto-heart", "Pluto Has a Beating Heart", ["Pluto New Horizons", "Pluto Sputnik Planitia", "Pluto mountains"], "nitrogen glaciers, water-ice mountains, a heart that drives the weather"),
    ("apollo-13", "The 90 Seconds That Nearly Killed Apollo 13", ["Apollo 13", "Apollo 13 service module damage", "Apollo mission control"], "an oxygen tank explodes 320,000 km from home"),
    ("challenger", "Challenger: 73 Seconds", ["Challenger launch", "Space Shuttle launch", "Challenger crew"], "an O-ring, a cold morning, the crew was likely alive until impact"),
    ("columbia", "Columbia Was Doomed at Launch", ["Columbia shuttle", "STS-107", "shuttle reentry"], "a briefcase-sized piece of foam, 16 days in orbit"),
    ("dark-forest", "Why Haven't We Heard From Aliens?", ["radio telescope", "Very Large Array", "exoplanet artist"], "Fermi paradox, dark forest, great filter"),
    ("kessler", "The Day Space Debris Traps Us on Earth", ["space debris", "satellite orbit", "ISS damage"], "Kessler syndrome, 36,000 pieces at 28,000 km/h"),
    ("enceladus", "Enceladus Is Spraying Its Ocean Into Space", ["Enceladus plumes", "Enceladus Cassini", "Saturn moon"], "geysers, salt, organic molecules"),
    ("red-dwarf", "Living Next to a Red Dwarf Would Be Hell", ["red dwarf flare", "Proxima Centauri", "TRAPPIST-1"], "flares strip atmospheres, one side always burning"),
    ("far-side", "There Is No Dark Side of the Moon", ["far side of the Moon", "Moon LRO", "Chang'e 4"], "tidal locking; the far side is just hidden"),
    ("sun-size", "The Sun Is Bigger Than You Think", ["Sun SDO", "solar eclipse corona", "Sun Earth comparison"], "1.3 million Earths, 8 minutes of light, 99.8% of all mass"),
    ("iss-death", "How the ISS Will Die", ["ISS", "ISS reentry", "Point Nemo"], "2030, Point Nemo, 400 tonnes of fire"),
    ("biggest-star", "A Star So Big a Plane Needs 1,100 Years to Circle It", ["VY Canis Majoris", "UY Scuti", "red supergiant"], "it would swallow Saturn"),
    ("deep-field", "This Photo Contains 10,000 Galaxies", ["Hubble ultra deep field", "JWST deep field", "SMACS 0723"], "a grain of sand of sky"),
    ("mars-colony", "What a Year on Mars Does to Your Body", ["Mars habitat", "Mars surface", "astronaut Mars artist"], "bone loss, radiation, 38% gravity, 24-minute delay"),
    ("sprites", "The Red Lightning That Shoots Upward", ["sprite lightning ISS", "red sprite", "upper atmosphere"], "sprites, blue jets, 90 km up"),
    ("tardigrade", "The Animal That Survives Space", ["tardigrade", "microgravity experiment", "space station lab"], "10 days in vacuum, frozen, boiled, irradiated"),
    ("earth-core", "Earth's Core Is as Hot as the Sun's Surface", ["Earth cutaway", "Earth magnetic field", "aurora"], "5,400°C, the magnetic shield that keeps us alive"),
    ("rogue-planet", "A Rogue Planet Could Pass Through the Solar System", ["rogue planet", "exoplanet artist", "Kuiper belt"], "orbits thrown into chaos, Earth flung into the dark"),
    ("white-hole", "What a White Hole Would Look Like", ["black hole artist", "quasar", "black hole jet"], "a black hole running backwards"),
    ("hurricane-orbit", "A Hurricane Seen From Orbit Is a 1,000 km Engine", ["hurricane from ISS", "hurricane satellite", "Earth storm"], "heat engine, eye 50 km wide"),
    ("supervolcano", "The Supervolcano Under Yellowstone", ["Yellowstone from space", "volcano eruption satellite", "caldera"], "600,000 year cycle, continent-wide ash"),
    ("antarctic-meteorites", "Why Antarctica Is Full of Meteorites", ["Antarctica meteorite", "Antarctica from space", "meteorite"], "black rocks on white ice"),
]

BAD_WORDS = ("logo", "chart", "diagram", "graph", "infographic", "poster", "portrait", "group photo", "briefing", "conference", "ceremony", "signing", "map")


def topic_id(key):
    return f"space_{key}"


def shortlist(api, topic, used):
    used = {str(u).casefold() for u in used}
    pool = [row for row in TOPICS if topic_id(row[0]) not in used and row[1].casefold() not in used]
    if topic:
        pool = [r for r in pool if all(w in (r[1] + " " + " ".join(r[2])).casefold() for w in topic.casefold().split())] or pool
    if not pool:
        raise SourceUnavailable("Uzay konu havuzundaki her şey kullanılmış.")
    # Weighted by list position: the most scroll-stopping topics come first.
    order = {row[0]: index for index, row in enumerate(TOPICS)}
    chosen, remaining = [], list(pool)
    while remaining and len(chosen) < 3:
        pick = random.choices(remaining, weights=[1.0 / (1 + order[r[0]] * 0.08) for r in remaining])[0]
        remaining.remove(pick)
        chosen.append(pick)
    events = []
    for key, title, terms, angle in chosen:
        events.append({
            "id": topic_id(key), "title": title, "publisher": "NASA/ESA", "series": "Space", "issue": key,
            "year": 2026, "universe": "science", "characters": [], "summary": angle, "famous_line": angle,
            "image_terms": terms, "source_urls": [], "url": "", "_source": "space",
        })
    save_json(api.directory / "research" / "candidates.json", {"requested_topic": topic, "events": events})
    return events


# ------------------------------------------------------------------ facts
def gather_facts(api, event, fetcher):
    queries = [f"{event['title']} explained", f"{event['title']} NASA facts", f"{event['summary']} science"]
    rows = search(queries, each=6)
    articles = []
    for row in rows[:12]:
        api.check()
        try:
            article = fetcher.article(row["url"])
        except (requests.RequestException, ValueError, OSError, KeyError):
            continue
        if len(article["text"]) < 400:
            continue
        articles.append({"url": article["url"], "title": article["title"], "text": article["text"][:9000]})
        if len(articles) == 4:
            break
    facts = []
    if articles:
        data = api.json(
            "Bilim gerçekleri",
            f"""From ONLY these fetched articles, list 12..20 concrete, numeric, vivid facts in {language_name(api)} for a video titled "{event['title']}" (angle: {event['summary']}). Prefer numbers (temperatures, speeds, distances, seconds to death), sequences (first this happens, then that) and comparisons a viewer can feel. Each fact needs an exact article substring (>= 20 characters) supporting it.
ARTICLES {json.dumps(articles, ensure_ascii=False)}
Return {{"facts":[{{"text":"","quote":"exact substring","url":"article url"}}]}}""",
            list_key="facts",
        )
        by_url = {a["url"]: a for a in articles}
        from .panels import quote_present

        for item in data.get("facts", []):
            if not isinstance(item, dict) or not clean(item.get("text")):
                continue
            article = by_url.get(item.get("url")) or next((a for a in articles if quote_present(item.get("quote", ""), a["text"])), None)
            if article and quote_present(item.get("quote", ""), article["text"]):
                facts.append({"id": f"fact_{len(facts):03}", "text": clean(item["text"]), "quote": clean(item.get("quote")),
                              "source_url": article["url"], "source_id": "article", "page_id": None})
    if len(facts) < 10:
        data = api.json(
            "Bilim bilgisi",
            f"""You are an astrophysicist writing for a YouTube Shorts channel. Topic: "{event['title']}" (angle: {event['summary']}).
List 14..20 accurate, widely established facts in {language_name(api)}, in the order a story would use them: the setup, what happens step by step (with numbers: temperatures, pressures, speeds, time), the comparisons that make it visceral, and the final consequence. Only mainstream, well-documented science; mark uncertainty inside the sentence when real scientists disagree.
Return {{"facts":[{{"text":""}}]}}""",
            list_key="facts",
        )
        for item in data.get("facts", []):
            text = clean(item.get("text")) if isinstance(item, dict) else clean(item)
            if text and 4 <= len(text.split()) <= 60:
                facts.append({"id": f"fact_{len(facts):03}", "text": text, "quote": "", "source_url": "",
                              "source_id": "general_knowledge", "page_id": None})
    if len(facts) < 8:
        raise SourceUnavailable("Konu için yeterli gerçek toplanamadı.")
    return facts, [a["url"] for a in articles]


# ----------------------------------------------------------------- images
def nasa_search(query, count=12):
    """NASA image library: public domain, keyless."""
    try:
        response = requests.get(NASA_API, params={"q": query, "media_type": "image", "page_size": count},
                                timeout=(10, 30), headers={"User-Agent": "comic-factory/1.0"})
        response.raise_for_status()
        items = response.json().get("collection", {}).get("items", [])
    except Exception as error:  # noqa: BLE001
        print(f"NASA araması ({query}): {type(error).__name__}", flush=True)
        return []
    rows = []
    for item in items:
        data = (item.get("data") or [{}])[0]
        links = item.get("links") or []
        href = next((l.get("href") for l in links if l.get("rel") == "preview"), None) or (links[0].get("href") if links else None)
        if not href:
            continue
        title = clean(data.get("title"))
        if any(b in title.casefold() for b in BAD_WORDS):
            continue
        nasa_id = data.get("nasa_id", "")
        # ~orig.jpg / ~large.jpg exist for most assets; the preview is ~thumb.
        large = re.sub(r"~(thumb|small|medium)\.(jpg|png)$", r"~large.\2", href)
        rows.append({"image_url": large, "fallback_url": href, "title": title,
                     "source_url": f"https://images.nasa.gov/details/{nasa_id}" if nasa_id else href,
                     "credit": clean(data.get("secondary_creator") or data.get("center") or "NASA"),
                     "description": clean(data.get("description"))[:300]})
    return rows


def commons_search(query, count=10):
    """Wikimedia Commons, kept to public-domain / CC BY files (attribution added)."""
    try:
        params = {"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
                  "gsrlimit": count, "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": 1600, "format": "json"}
        response = requests.get(COMMONS_API, params=params, timeout=(10, 30), headers={"User-Agent": "comic-factory/1.0 (youtube shorts)"})
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
    except Exception as error:  # noqa: BLE001
        print(f"Commons araması ({query}): {type(error).__name__}", flush=True)
        return []
    rows = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licence = clean(meta.get("LicenseShortName", {}).get("value", "")).casefold()
        if not (licence.startswith("pd") or licence.startswith("public domain") or licence in {"cc0", "cc by 4.0", "cc by 3.0", "cc by 2.0", "cc by-sa 4.0", "cc by-sa 3.0"}):
            continue
        if info.get("width", 0) < 1200:
            continue
        title = clean(page.get("title", "")).replace("File:", "")
        if any(b in title.casefold() for b in BAD_WORDS):
            continue
        rows.append({"image_url": info.get("thumburl") or info.get("url"), "fallback_url": info.get("url"), "title": title,
                     "source_url": info.get("descriptionurl", ""), "credit": clean(re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")))[:80] or "Wikimedia Commons",
                     "licence": licence, "description": clean(re.sub("<[^>]+>", "", meta.get("ImageDescription", {}).get("value", "")))[:300]})
    return rows


def dhash(picture, size=8):
    """Perceptual hash: near-identical photos collapse to the same bits."""
    gray = picture.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (1 if pixels[row * (size + 1) + col] > pixels[row * (size + 1) + col + 1] else 0)
    return bits


def download(api, event, rows, maximum, seen):
    download.signatures = []
    fetcher = Fetcher()
    root = api.directory / "events" / event["id"]
    (root / "pages").mkdir(parents=True, exist_ok=True)
    pages = []
    offset = len(list((root / "pages").glob("page_*.jpg")))
    try:
        for row in rows:
            if len(pages) >= maximum:
                break
            api.check()
            for url in (row.get("image_url"), row.get("fallback_url")):
                if not url:
                    continue
                try:
                    body, final = fetcher.get(url, 40_000_000)
                except Exception:  # noqa: BLE001
                    continue
                digest = hashlib.sha256(body).hexdigest()
                if digest in seen:
                    break
                try:
                    with Image.open(io.BytesIO(body)) as opened:
                        if min(opened.size) < 700:
                            continue
                        picture = ImageOps.exif_transpose(opened).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                if picture.width > 4000 or picture.height > 4000:
                    picture.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
                signature = dhash(picture)
                if any(bin(signature ^ other).count("1") <= 6 for other in download.signatures):
                    break  # near-duplicate of an image already kept (same dish, other angle)
                download.signatures.append(signature)
                seen.add(digest)
                identifier = f"page_{offset + len(pages):03}"
                path = root / "pages" / (identifier + ".jpg")
                picture.save(path, "JPEG", quality=94)
                pages.append({"id": identifier, "file": str(path.relative_to(api.directory)), "source_url": row.get("source_url") or final,
                              "image_url": final, "source_title": row.get("title", ""), "credit": row.get("credit", ""),
                              "licence": row.get("licence", "public domain (NASA)"), "caption": row.get("description", ""),
                              "query": row.get("query", ""), "width": picture.width, "height": picture.height, "sha256": digest})
                break
    finally:
        fetcher.session.close()
    return pages


def judge_images(api, event, pages):
    """Keep striking, relevant photographs; drop diagrams, portraits, text."""
    kept = []
    for offset in range(0, len(pages), 6):
        batch = pages[offset:offset + 6]
        api.check()
        try:
            data = api.json(
                "Görsel seçimi",
                f"""These images were fetched for a science video titled "{event['title']}" (angle: {event['summary']}).
For each image decide: usable (true only for a real photograph, telescope image, spacecraft image or high-quality scientific visualization that would look stunning full-screen on a phone; false for diagrams, charts, infographics, text slides, logos, group photos of people at desks, press conferences, low-quality or blurry images), relevance 0..100 to the topic, drama 0..100 (how awe-inspiring it is), focus [x,y] fractions of the most interesting point, and a one-sentence {language_name(api)} description of what is visible.
Return {{"images":[{{"page_id":"","usable":true,"relevance":0,"drama":0,"focus":[0.5,0.5],"description":""}}]}}""",
                images=[(p["id"], api.directory / p["file"]) for p in batch],
                list_key="images",
            )
        except Exception as error:  # noqa: BLE001
            print(f"Görsel seçimi atlandı: {str(error)[:120]}", flush=True)
            kept.extend({**p, "relevance": 60, "drama": 60, "focus": [0.5, 0.5], "description": p.get("caption", "")} for p in batch)
            continue
        rows = [r for r in data.get("images", []) if isinstance(r, dict)]
        if rows and len(rows) == len(batch) and not all(r.get("page_id") for r in rows):
            for row, page in zip(rows, batch):
                row.setdefault("page_id", page["id"])
        by_id = {r.get("page_id"): r for r in rows}
        for page in batch:
            row = by_id.get(page["id"])
            if not row:
                continue
            usable = row.get("usable") is True or str(row.get("usable")).casefold() == "true"
            try:
                relevance, drama = float(row.get("relevance", 0)), float(row.get("drama", 0))
            except (TypeError, ValueError):
                relevance, drama = 0, 0
            if not usable or relevance < 50 or drama < 40:
                continue
            focus = row.get("focus")
            if not (isinstance(focus, list) and len(focus) == 2):
                focus = [0.5, 0.5]
            try:
                focus = [min(1.0, max(0.0, float(focus[0]))), min(1.0, max(0.0, float(focus[1])))]
            except (TypeError, ValueError):
                focus = [0.5, 0.5]
            kept.append({**page, "relevance": relevance, "drama": drama, "focus": focus,
                         "description": clean(row.get("description")) or page.get("caption", "")})
    kept.sort(key=lambda p: -(p["relevance"] * 0.6 + p["drama"] * 0.4))
    return kept


def collect(api, event, fetcher):
    facts, article_urls = gather_facts(api, event, fetcher)
    seen, rows = set(), []
    for term in event.get("image_terms", [])[:4]:
        api.check()
        for row in nasa_search(term, 10):
            rows.append({**row, "query": term})
    for term in event.get("image_terms", [])[:2]:
        for row in commons_search(term, 6):
            rows.append({**row, "query": term})
    random.shuffle(rows)
    # Round-robin over search terms so one term cannot fill the whole budget.
    by_term = {}
    for row in rows:
        by_term.setdefault(row["query"], []).append(row)
    ordered = []
    while any(by_term.values()):
        for term in list(by_term):
            if by_term[term]:
                ordered.append(by_term[term].pop(0))
    pages = download(api, event, ordered, 36, seen)
    print(f"Uzay görselleri: {len(ordered)} aday, {len(pages)} indirildi.", flush=True)
    if len(pages) < 6:
        raise SourceUnavailable("Konu için yeterli görsel bulunamadı.")
    judged = judge_images(api, event, pages)
    if len(judged) < 6:
        raise SourceUnavailable(f"Konu için yeterli kullanılabilir görsel yok ({len(judged)}/{len(pages)}).")
    root = api.directory / "events" / event["id"]
    (root / "panels").mkdir(parents=True, exist_ok=True)
    inventory = []
    for page in judged[:28]:
        src = api.directory / page["file"]
        inventory.append({
            "id": page["id"] + "_p00", "page_id": page["id"], "reading_order": 1, "characters": [],
            "action": page["description"], "ocr": "", "narrative_fact": "", "confidence": round(page["relevance"]),
            "intensity": round(page["drama"]), "focus": page["focus"], "framing": "fill",
            "bbox": [0, 0, 1, 1], "page_file": page["file"], "file": page["file"],
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "source_url": page["source_url"], "source_title": page.get("source_title", ""),
            "credit": page.get("credit", ""), "licence": page.get("licence", ""), "kind": "photo",
        })
    event["source_urls"] = list(dict.fromkeys(article_urls + [p["source_url"] for p in judged]))[:14]
    event["url"] = event["source_urls"][0] if event["source_urls"] else ""
    save_json(root / "catalog.json", {"event": event, "panels": inventory, "facts": facts})
    return inventory, facts
