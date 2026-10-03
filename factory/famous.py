"""Famous-moment source: the events everyone knows, told as a recap.

The channel's audience stops scrolling for a character they recognise and a
moment they half-remember ("wait, Spider-Man's web did WHAT?"). This source
starts from a curated list of the most famous Marvel/DC moments, researches
the facts on the open web, collects official art, previews and press images
that depict the storyline, and hands the story writer a loose inventory of
related panels instead of demanding a page-exact match. Panels are commentary
illustrations here, so the audit only requires relevance, never literal
depiction of each sentence.

No full-issue scans are used: images come from search results (publisher
previews, press coverage, promotional art), are cropped to single panels
where a page layout is detected, and every image keeps its source URL for the
description credits.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re

import requests
from PIL import Image, ImageOps

from .api import SourceUnavailable
from .core import language_name, save_json
from .panels import quote_present, segment_page
from .research import Fetcher, clean, search

# (key, event, series, issue, year, publisher, characters, famous line)
# Ordered roughly by global recognisability; the picker weights the top.
MOMENTS = [
    ("gwen-stacy", "The Night Gwen Stacy Died", "The Amazing Spider-Man", "121-122", 1973, "Marvel",
     ["Spider-Man", "Gwen Stacy", "Green Goblin"], "Spider-Man's own web snaps Gwen Stacy's neck"),
    ("death-of-superman", "The Death of Superman", "Superman", "75", 1992, "DC",
     ["Superman", "Doomsday", "Lois Lane"], "Superman dies in Lois Lane's arms after killing Doomsday"),
    ("knightfall", "Knightfall: Bane Breaks the Bat", "Batman", "497", 1993, "DC",
     ["Batman", "Bane"], "Bane breaks Batman's back over his knee"),
    ("death-in-the-family", "A Death in the Family", "Batman", "426-429", 1988, "DC",
     ["Batman", "Jason Todd", "Joker"], "The Joker beats Robin with a crowbar and readers voted to let him die"),
    ("killing-joke", "The Killing Joke", "Batman: The Killing Joke", "1", 1988, "DC",
     ["Joker", "Barbara Gordon", "Batman", "Commissioner Gordon"], "The Joker shoots Barbara Gordon through the spine"),
    ("infinity-gauntlet", "The Infinity Gauntlet Snap", "The Infinity Gauntlet", "1", 1991, "Marvel",
     ["Thanos", "Mephisto", "Silver Surfer", "Avengers"], "Thanos erases half of all life with a snap"),
    ("dark-phoenix", "The Dark Phoenix Saga", "Uncanny X-Men", "135-137", 1980, "Marvel",
     ["Jean Grey", "Phoenix", "Cyclops", "Wolverine"], "Jean Grey devours a star and kills five billion people"),
    ("civil-war-unmask", "Civil War: Spider-Man Unmasks", "Civil War", "2", 2006, "Marvel",
     ["Spider-Man", "Iron Man", "Captain America"], "Spider-Man reveals he is Peter Parker on live television"),
    ("secret-empire", "Captain America: Hail Hydra", "Captain America: Steve Rogers", "1", 2016, "Marvel",
     ["Captain America", "Red Skull", "Hydra"], "Captain America says 'Hail Hydra' and throws his ally from a plane"),
    ("old-man-logan", "Old Man Logan", "Wolverine", "66-72", 2008, "Marvel",
     ["Wolverine", "Hawkeye", "Hulk", "Mysterio"], "Wolverine slaughters every X-Man believing they were villains"),
    ("crisis-supergirl", "Crisis on Infinite Earths: Death of Supergirl", "Crisis on Infinite Earths", "7", 1985, "DC",
     ["Supergirl", "Superman", "Anti-Monitor"], "Supergirl dies saving Superman from the Anti-Monitor"),
    ("crisis-flash", "Crisis on Infinite Earths: Death of the Flash", "Crisis on Infinite Earths", "8", 1985, "DC",
     ["Flash", "Barry Allen", "Anti-Monitor"], "Barry Allen runs so fast he turns to dust saving the universe"),
    ("marvel-zombies", "Marvel Zombies", "Marvel Zombies", "1-5", 2005, "Marvel",
     ["Zombie Spider-Man", "Hulk", "Silver Surfer", "Galactus"], "Zombie Avengers eat Galactus and gain his power"),
    ("kravens-last-hunt", "Kraven's Last Hunt", "Web of Spider-Man", "31", 1987, "Marvel",
     ["Spider-Man", "Kraven the Hunter", "Vermin"], "Kraven shoots Spider-Man, buries him alive, then kills himself"),
    ("born-again", "Daredevil: Born Again", "Daredevil", "227-233", 1986, "Marvel",
     ["Daredevil", "Kingpin", "Karen Page"], "Kingpin learns Daredevil's identity and destroys his whole life"),
    ("elektra-dies", "Elektra Killed by Bullseye", "Daredevil", "181", 1982, "Marvel",
     ["Elektra", "Bullseye", "Daredevil"], "Bullseye impales Elektra with her own sai"),
    ("wolverine-adamantium", "Fatal Attractions: Magneto Rips Out Wolverine's Adamantium", "X-Men", "25", 1993, "Marvel",
     ["Wolverine", "Magneto", "Professor X"], "Magneto tears the adamantium out of Wolverine's bones"),
    ("house-of-m", "House of M: No More Mutants", "House of M", "7-8", 2005, "Marvel",
     ["Scarlet Witch", "Magneto", "Wolverine"], "Scarlet Witch whispers three words and erases almost every mutant"),
    ("emerald-twilight", "Emerald Twilight: Hal Jordan Becomes Parallax", "Green Lantern", "48-50", 1994, "DC",
     ["Hal Jordan", "Green Lantern", "Sinestro", "Kilowog"], "Hal Jordan murders the Green Lantern Corps and becomes Parallax"),
    ("identity-crisis", "Identity Crisis", "Identity Crisis", "1-7", 2004, "DC",
     ["Elongated Man", "Sue Dibny", "Doctor Light", "Zatanna"], "The Justice League wiped a villain's mind and hid it for years"),
    ("flashpoint", "Flashpoint: Thomas Wayne Batman", "Flashpoint", "1-5", 2011, "DC",
     ["Flash", "Batman", "Thomas Wayne", "Reverse-Flash"], "Flash changes history and Bruce Wayne's father is now Batman"),
    ("red-son", "Superman: Red Son", "Superman: Red Son", "1-3", 2003, "DC",
     ["Superman", "Batman", "Lex Luthor", "Wonder Woman"], "Superman lands in the Soviet Union and rules the world"),
    ("death-of-captain-america", "The Death of Captain America", "Captain America", "25", 2007, "Marvel",
     ["Captain America", "Sharon Carter", "Crossbones", "Red Skull"], "Captain America is shot dead on courthouse steps by his own lover"),
    ("world-war-hulk", "World War Hulk", "World War Hulk", "1-5", 2007, "Marvel",
     ["Hulk", "Iron Man", "Doctor Strange", "Sentry"], "Hulk returns from space to make the Illuminati pay"),
    ("planet-hulk", "Planet Hulk", "Incredible Hulk", "92-105", 2006, "Marvel",
     ["Hulk", "Silver Surfer", "Caiera"], "The Illuminati shoot Hulk into space and he becomes a gladiator king"),
    ("court-of-owls", "Batman: Court of Owls", "Batman", "1-11", 2011, "DC",
     ["Batman", "Court of Owls", "Talon"], "A secret society has ruled Gotham for centuries and Batman never knew"),
    ("injustice", "Injustice: Superman Kills the Joker", "Injustice: Gods Among Us", "1", 2013, "DC",
     ["Superman", "Joker", "Lois Lane", "Batman"], "Joker tricks Superman into killing Lois and Superman punches through Joker's chest"),
    ("thanos-wins", "Thanos Wins", "Thanos", "13-18", 2017, "Marvel",
     ["Thanos", "Cosmic Ghost Rider", "Silver Surfer"], "An ancient Thanos has killed everyone and a Frank Castle Ghost Rider serves him"),
    ("maximum-carnage", "Maximum Carnage", "Spider-Man", "35", 1993, "Marvel",
     ["Carnage", "Venom", "Spider-Man", "Shriek"], "Carnage turns Manhattan into a massacre and Spider-Man teams with Venom"),
    ("deadpool-kills", "Deadpool Kills the Marvel Universe", "Deadpool Kills the Marvel Universe", "1-4", 2012, "Marvel",
     ["Deadpool", "Spider-Man", "Hulk", "Professor X"], "Deadpool realises he is a comic character and kills every hero"),
    ("days-of-future-past", "Days of Future Past", "Uncanny X-Men", "141-142", 1981, "Marvel",
     ["Wolverine", "Kitty Pryde", "Sentinels", "Storm"], "Sentinels vaporise Wolverine in a future where mutants are hunted"),
    ("mutant-massacre", "The Mutant Massacre", "Uncanny X-Men", "211", 1986, "Marvel",
     ["X-Men", "Marauders", "Sabretooth", "Angel"], "The Marauders slaughter the Morlocks and crucify Angel"),
    ("back-in-black", "Back in Black: Aunt May Shot", "The Amazing Spider-Man", "539-543", 2007, "Marvel",
     ["Spider-Man", "Aunt May", "Kingpin"], "A sniper hired by Kingpin shoots Aunt May and Spider-Man goes dark"),
    ("wonder-woman-maxwell-lord", "Wonder Woman Kills Maxwell Lord", "Wonder Woman", "219", 2005, "DC",
     ["Wonder Woman", "Maxwell Lord", "Superman"], "Wonder Woman snaps Maxwell Lord's neck on live broadcast"),
    ("blackest-night", "Blackest Night", "Blackest Night", "1-8", 2009, "DC",
     ["Green Lantern", "Black Hand", "Nekron", "Flash"], "Dead heroes rise as Black Lanterns to eat their friends' hearts"),
    ("avengers-disassembled", "Avengers Disassembled", "Avengers", "500-503", 2004, "Marvel",
     ["Scarlet Witch", "Vision", "Hawkeye", "Ant-Man"], "Scarlet Witch's breakdown kills Vision, Hawkeye and Ant-Man in one day"),
    ("ragnarok-thor", "Thor: Ragnarok (2004)", "Thor", "80-85", 2004, "Marvel",
     ["Thor", "Loki", "Odin", "Surtur"], "Thor lets all of Asgard die to break the cycle of Ragnarok"),
    ("batman-rip", "Batman R.I.P.", "Batman", "676-681", 2008, "DC",
     ["Batman", "Doctor Hurt", "Joker"], "Batman is driven insane and buried alive by the Black Glove"),
    ("age-of-apocalypse", "Age of Apocalypse", "X-Men: Alpha", "1", 1995, "Marvel",
     ["Magneto", "Apocalypse", "Wolverine", "Professor X"], "Professor X dies before founding the X-Men and Apocalypse conquers America"),
    ("clone-saga", "The Clone Saga", "The Amazing Spider-Man", "394", 1994, "Marvel",
     ["Spider-Man", "Ben Reilly", "Scarlet Spider", "Norman Osborn"], "Peter Parker is told he is the clone and Ben Reilly is the real Spider-Man"),
    ("demon-in-a-bottle", "Demon in a Bottle", "Iron Man", "120-128", 1979, "Marvel",
     ["Iron Man", "Tony Stark", "Justin Hammer"], "Tony Stark's armour is hijacked and he drinks himself into ruin"),
    ("doomsday-first-fight", "Doomsday Arrives", "Superman: The Man of Steel", "18-19", 1992, "DC",
     ["Doomsday", "Justice League", "Superman"], "Doomsday beats the entire Justice League with one hand tied behind his back"),
    ("secret-wars-2015", "Secret Wars: God Emperor Doom", "Secret Wars", "1-9", 2015, "Marvel",
     ["Doctor Doom", "Reed Richards", "Thanos", "Black Panther"], "Doctor Doom becomes God and rules what is left of every universe"),
    ("king-in-black", "King in Black", "King in Black", "1-5", 2020, "Marvel",
     ["Knull", "Venom", "Eddie Brock", "Sentry"], "Knull rips Sentry in half and drowns Earth in symbiotes"),
    ("batman-who-laughs", "The Batman Who Laughs", "Dark Nights: Metal", "1-6", 2017, "DC",
     ["Batman", "Batman Who Laughs", "Joker"], "A Batman who killed the Joker becomes a Joker with Batman's mind"),
    ("spider-man-no-more", "Spider-Man No More", "The Amazing Spider-Man", "50", 1967, "Marvel",
     ["Spider-Man", "Peter Parker", "Kingpin"], "Peter Parker throws his costume in the trash and quits"),
    ("superior-spider-man", "Superior Spider-Man: Doc Ock Steals Peter's Body", "The Amazing Spider-Man", "700", 2012, "Marvel",
     ["Spider-Man", "Doctor Octopus", "Peter Parker"], "Doctor Octopus swaps minds with Peter Parker and lets him die"),
    ("the-long-halloween", "The Long Halloween", "Batman: The Long Halloween", "1-13", 1996, "DC",
     ["Batman", "Harvey Dent", "Two-Face", "Catwoman"], "A killer murders someone every holiday while Harvey Dent becomes Two-Face"),
    ("hush", "Batman: Hush", "Batman", "608-619", 2002, "DC",
     ["Batman", "Hush", "Catwoman", "Riddler"], "Batman's childhood friend wraps himself in bandages to destroy him"),
    ("spider-verse", "Spider-Verse", "The Amazing Spider-Man", "9-15", 2014, "Marvel",
     ["Spider-Man", "Morlun", "Spider-Gwen", "Miles Morales"], "The Inheritors hunt and eat every Spider-Man across the multiverse"),
]


def moment_id(key):
    return f"famous_{key}"


def shortlist(api, topic, used):
    """Pick unused famous moments, favouring the most recognisable ones."""
    used = {str(u).casefold() for u in used}
    pool = []
    for index, row in enumerate(MOMENTS):
        key, event, series, issue, year, publisher, characters, line = row
        if moment_id(key) in used or event.casefold() in used:
            continue
        if topic and not all(w in (event + " " + series + " " + " ".join(characters)).casefold() for w in topic.casefold().split()):
            continue
        pool.append((index, row))
    if not pool:
        raise SourceUnavailable("Ünlü olay listesindeki her şey kullanılmış.")
    # Weighted sample: list order is popularity; earlier rows are picked more often.
    weights = [1.0 / (1 + index * 0.08) for index, _ in pool]
    chosen, remaining = [], list(zip(pool, weights))
    while remaining and len(chosen) < 3:
        pick = random.choices(remaining, weights=[w for _, w in remaining])[0]
        remaining.remove(pick)
        chosen.append(pick[0][1])
    events = []
    for key, event, series, issue, year, publisher, characters, line in chosen:
        events.append({
            "id": moment_id(key),
            "title": event,
            "publisher": publisher,
            "series": series,
            "issue": issue,
            "year": year,
            "universe": publisher,
            "characters": characters,
            "summary": line,
            "famous_line": line,
            "source_urls": [],
            "url": "",
            "_source": "famous",
        })
    save_json(api.directory / "research" / "candidates.json", {"requested_topic": topic, "events": events})
    return events


# ------------------------------------------------------------------ facts
def gather_facts(api, event, fetcher):
    """Facts from fetched articles (quoted) plus the model's own knowledge of a
    famous storyline, clearly labelled, so the narration never starves."""
    queries = [f"{event['title']} {event['series']} {event['issue']} comic explained",
               f"{event['title']} comic storyline summary"]
    rows = search(queries, each=6)
    articles = []
    for row in rows[:10]:
        api.check()
        try:
            article = fetcher.article(row["url"])
        except (requests.RequestException, ValueError, OSError, KeyError):
            continue
        if len(article["text"]) < 300:
            continue
        articles.append({"url": article["url"], "title": article["title"], "text": article["text"][:9000]})
        if len(articles) == 4:
            break
    facts = []
    if articles:
        data = api.json(
            "Olay gerçekleri",
            f"""From ONLY these fetched articles, list 10..18 concrete facts about the comic event {json.dumps({k: event[k] for k in ('title', 'series', 'issue', 'year', 'publisher', 'characters')}, ensure_ascii=False)} in {language_name(api)}: who does what to whom, in story order, with the shocking beats, the setup and the aftermath. Each fact needs an exact article substring (>= 20 characters) that supports it.
ARTICLES {json.dumps(articles, ensure_ascii=False)}
Return {{"facts":[{{"text":"","quote":"exact substring","url":"article url"}}]}}""",
            list_key="facts",
        )
        by_url = {a["url"]: a for a in articles}
        for item in data.get("facts", []):
            if not isinstance(item, dict) or not clean(item.get("text")):
                continue
            article = by_url.get(item.get("url")) or next(
                (a for a in articles if quote_present(item.get("quote", ""), a["text"])), None)
            if article and quote_present(item.get("quote", ""), article["text"]):
                facts.append({"id": f"fact_{len(facts):03}", "text": clean(item["text"]),
                              "quote": clean(item.get("quote")), "source_url": article["url"],
                              "source_id": "article", "page_id": None})
    if len(facts) < 10:
        data = api.json(
            "Olay bilgisi",
            f"""You know this famous comic storyline well: {json.dumps({k: event[k] for k in ('title', 'series', 'issue', 'year', 'publisher', 'characters', 'famous_line')}, ensure_ascii=False)}.
List 14..20 concrete, widely documented facts in {language_name(api)} in story order: the setup, the shocking moment itself (be specific and vivid), what the characters do and say, the aftermath and why fans still talk about it. Only facts you are confident are accurate for this exact storyline; no speculation, no movie versions.
Return {{"facts":[{{"text":""}}]}}""",
            list_key="facts",
        )
        for item in data.get("facts", []):
            text = clean(item.get("text")) if isinstance(item, dict) else clean(item)
            if text and 4 <= len(text.split()) <= 60:
                facts.append({"id": f"fact_{len(facts):03}", "text": text, "quote": "",
                              "source_url": "", "source_id": "general_knowledge", "page_id": None})
    if len(facts) < 8:
        raise SourceUnavailable("Olay için yeterli gerçek toplanamadı.")
    return facts, [a["url"] for a in articles]


# ----------------------------------------------------------------- images
BAD_HOSTS = ("pinterest.", "pinimg.com", "tiktok.", "facebook.", "instagram.")


def collect_images(api, event, fetcher, maximum):
    """Download candidate art for the storyline from image search results."""
    characters = " ".join(event["characters"][:2])
    queries = [
        f"{event['title']} comic panel",
        f"{event['series']} {event['issue']} comic page {characters}",
        f"{event['title']} comic art {event['publisher']}",
        f"{event['series']} {event['issue']} preview",
    ]
    rows = search(queries, images=True, each=14)
    rows = [r for r in rows if r.get("image_url") and not any(h in r["image_url"] for h in BAD_HOSTS)]
    root = api.directory / "events" / event["id"]
    (root / "pages").mkdir(parents=True, exist_ok=True)
    pages, seen = [], set()
    for row in rows:
        if len(pages) >= maximum:
            break
        api.check()
        try:
            body, url = fetcher.get(row["image_url"], 12_000_000)
            digest = hashlib.sha256(body).hexdigest()
            if digest in seen:
                continue
            with Image.open(io.BytesIO(body)) as opened:
                if min(opened.size) < 480 or opened.width * opened.height > 35_000_000:
                    continue
                picture = ImageOps.exif_transpose(opened).convert("RGB")
            if not 0.3 <= picture.width / picture.height <= 2.6:
                continue
            identifier = f"page_{len(pages):03}"
            path = root / "pages" / (identifier + ".jpg")
            picture.save(path, "JPEG", quality=95)
            pages.append({"id": identifier, "file": str(path.relative_to(api.directory)),
                          "source_url": row.get("url") or url, "image_url": url,
                          "source_title": row.get("title", ""), "width": picture.width,
                          "height": picture.height, "sha256": digest})
            seen.add(digest)
        except Exception:
            continue
    save_json(root / "pages.json", pages)
    if len(pages) < 4:
        raise SourceUnavailable("Bu olay için yeterli görsel bulunamadı.")
    return pages


def judge_images(api, event, pages):
    """Keep drawn comic art that depicts this storyline's characters."""
    kept = []
    for offset in range(0, len(pages), 6):
        batch = pages[offset:offset + 6]
        api.check()
        try:
            data = api.json(
                "Görsel uygunluğu",
                f"""These images were found by searching for the comic storyline {json.dumps({k: event[k] for k in ('title', 'series', 'issue', 'publisher', 'characters')}, ensure_ascii=False)}.
For each image decide: is_comic_art (true only for drawn comic book art: interior panels, pages, covers or official illustrations; false for photos, movie/TV stills, cosplay, toys, video games, fan 3D renders, memes), relevance 0..100 to this storyline and its characters (same characters in costume = 60+, the famous scene itself = 90+), text_heavy (true if large watermarks, memes captions or a text-dominated page), characters visible, and a one-sentence description of the visible action in {language_name(api)}.
Return {{"images":[{{"page_id":"","is_comic_art":true,"relevance":0,"text_heavy":false,"characters":[],"description":""}}]}}""",
                images=[(p["id"], api.directory / p["file"]) for p in batch],
                list_key="images",
            )
        except Exception as error:  # noqa: BLE001 - vision judge is advisory
            print(f"Görsel değerlendirmesi atlandı: {str(error)[:120]}", flush=True)
            for page in batch:
                kept.append({**page, "relevance": 50, "description": "", "characters": []})
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
            try:
                relevance = float(row.get("relevance", 0))
            except (TypeError, ValueError):
                relevance = 0
            if row.get("is_comic_art") is True and not row.get("text_heavy") and relevance >= 55:
                kept.append({**page, "relevance": relevance, "description": clean(row.get("description")),
                             "characters": [str(c) for c in row.get("characters", []) if isinstance(c, str)]})
    kept.sort(key=lambda p: p["relevance"], reverse=True)
    return kept


def cut_panels(api, event, pages):
    """Single panels from multi-panel pages; whole image otherwise."""
    root = api.directory / "events" / event["id"]
    (root / "panels").mkdir(parents=True, exist_ok=True)
    inventory = []
    for page in pages:
        with Image.open(api.directory / page["file"]) as source:
            picture = source.convert("RGB")
        boxes = segment_page(picture)
        crops = boxes[:6] if len(boxes) >= 3 else [[0, 0, 1, 1]]
        for order, box in enumerate(crops):
            left, top, right, bottom = box
            crop = picture.crop((round(left * picture.width), round(top * picture.height),
                                 round(right * picture.width), round(bottom * picture.height)))
            if min(crop.size) < 220:
                continue
            identifier = f"{page['id']}_p{order:02}"
            path = root / "panels" / (identifier + ".jpg")
            crop.save(path, "JPEG", quality=95)
            inventory.append({
                "id": identifier, "page_id": page["id"], "reading_order": order + 1,
                "characters": page.get("characters", []), "action": page.get("description", ""),
                "ocr": "", "narrative_fact": "", "confidence": round(page.get("relevance", 50)),
                "bbox": [left, top, right, bottom], "page_file": page["file"],
                "file": str(path.relative_to(api.directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_url": page["source_url"], "source_title": page.get("source_title", ""),
            })
    return inventory


def describe_panels(api, inventory):
    """One line per cropped panel so the writer can place each beat well."""
    for offset in range(0, len(inventory), 8):
        batch = inventory[offset:offset + 8]
        api.check()
        try:
            data = api.json(
                "Panel açıklamaları",
                f"""Describe each cropped comic panel in one vivid {language_name(api)} sentence: who is visible (use the character names if recognisable), the action, the mood (shock, grief, rage, triumph), and whether it is a close-up, medium or wide shot.
Return {{"panels":[{{"id":"","description":"","characters":[],"shot_type":"close|medium|wide","intensity":0}}]}} with intensity 0..100 (how dramatic the image is).""",
                images=[(p["id"], api.directory / p["file"]) for p in batch],
                list_key="panels",
            )
        except Exception as error:  # noqa: BLE001
            print(f"Panel açıklaması atlandı: {str(error)[:120]}", flush=True)
            continue
        rows = [r for r in data.get("panels", []) if isinstance(r, dict)]
        if rows and len(rows) == len(batch) and not all(r.get("id") for r in rows):
            for row, panel in zip(rows, batch):
                row.setdefault("id", panel["id"])
        by_id = {r.get("id"): r for r in rows}
        for panel in batch:
            row = by_id.get(panel["id"])
            if not row:
                continue
            if clean(row.get("description")):
                panel["action"] = clean(row["description"])
            if isinstance(row.get("characters"), list):
                panel["characters"] = [str(c) for c in row["characters"] if isinstance(c, str)] or panel["characters"]
            panel["shot_type"] = row.get("shot_type", "medium")
            try:
                panel["intensity"] = max(0, min(100, int(float(row.get("intensity", 50)))))
            except (TypeError, ValueError):
                panel["intensity"] = 50
    return inventory


def collect(api, event, fetcher):
    """Everything the writer needs: (inventory, facts). Also fills event['source_urls']."""
    facts, article_urls = gather_facts(api, event, fetcher)
    pages = collect_images(api, event, fetcher, max(16, api.settings.max_pages + 8))
    judged = judge_images(api, event, pages)
    if len(judged) < 4:
        raise SourceUnavailable(f"Olayla ilgili yeterli çizgi roman görseli bulunamadı ({len(judged)}).")
    inventory = describe_panels(api, cut_panels(api, event, judged[:18]))
    if len(inventory) < 6:
        raise SourceUnavailable("Yeterli panel kesilemedi.")
    event["source_urls"] = list(dict.fromkeys(article_urls + [p["source_url"] for p in judged]))[:12]
    event["url"] = event["source_urls"][0] if event["source_urls"] else ""
    save_json(api.directory / "events" / event["id"] / "catalog.json",
              {"event": event, "panels": inventory, "facts": facts})
    return inventory, facts


def hook_card_text(script, event):
    """3..6 word title-card line; the model's own if valid, else derived."""
    raw = clean(script.get("hook_card"))
    words = re.findall(r"[^\s]+", raw)
    if 2 <= len(words) <= 7 and len(raw) <= 48:
        return raw
    line = clean(event.get("famous_line") or script.get("title") or "")
    words = line.split()[:6]
    return " ".join(words)
