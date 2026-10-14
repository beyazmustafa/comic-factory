"""Real footage for the film source: NASA's image library and NASA's
Scientific Visualization Studio, both public domain.

Pipeline per topic:
  search → rank candidates by title/description → resolve renditions and
  duration → sample frames along the whole video → one look by the vision
  model per asset (is it real footage of the subject, which stretches are
  clean, where is the subject) → cut those stretches from the best rendition
  → find hard cuts and black frames inside them so the editor only uses
  continuous, lit footage.

Network calls carry their own timeouts and never raise past `harvest`; a
topic without enough usable footage raises FactoryError so the caller can
move on to another topic.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .api import FactoryError

NASA_API = "https://images-api.nasa.gov/search"
SVS_SEARCH = "https://svs.gsfc.nasa.gov/api/search/"
SVS_ITEM = "https://svs.gsfc.nasa.gov/api/{id}"
HEADERS = {"User-Agent": "comic-factory/2.0 (short documentaries from public-domain NASA footage)"}
FFMPEG_NET = ["-rw_timeout", "40000000", "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "8",
              "-user_agent", HEADERS["User-Agent"]]


def net_options(url):
    """ffmpeg's http options are rejected for local files, so only pass them for URLs."""
    return list(FFMPEG_NET) if str(url).startswith(("http://", "https://")) else []
# Titles that announce talking heads, live events or compilations.
SKIP_TITLE = ("interview", "briefing", "press conference", "news conference", "b-roll", "broll", "podcast", "webinar",
              "q&a", "town hall", "teleconference", "newscast", "360 video", "trailer", "highlight", "this week",
              "science casts", "sciencecast", "we asked", "asl", "en español", "remarks", "speech", "testimony",
              "hearing", "award", "year in review", "students", "ask nasa", "episode", "live:", "livestream",
              "ceremony", "panel", "lecture", "tutorial", "message from", "greeting", "crew news", "space to ground",
              "nasa explorers", "talks", "chat", "conversation")
MAX_ASSET_SECONDS = 15 * 60
MIN_ASSET_SECONDS = 6
MAX_PIXELS = 4096 * 2304


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def https(url):
    return re.sub(r"^http://", "https://", str(url or ""))


# ------------------------------------------------------------------ search
def nasa_search(query, count=12):
    """NASA image library videos: public domain; each asset has a manifest of renditions."""
    try:
        response = requests.get(NASA_API, params={"q": query, "media_type": "video", "page_size": count},
                                timeout=(10, 30), headers=HEADERS)
        response.raise_for_status()
        items = response.json().get("collection", {}).get("items", [])
    except Exception as error:  # noqa: BLE001
        print(f"NASA video araması ({query}): {type(error).__name__}: {str(error)[:80]}", flush=True)
        return []
    rows = []
    for item in items:
        data = (item.get("data") or [{}])[0]
        title = clean(data.get("title"))
        if not title or any(word in title.casefold() for word in SKIP_TITLE):
            continue
        nasa_id = clean(data.get("nasa_id"))
        manifest = item.get("href")
        if not manifest or not nasa_id:
            continue
        links = item.get("links") or []
        thumb = next((l.get("href") for l in links if l.get("rel") == "preview"), "")
        rows.append({"id": f"nasa:{nasa_id}", "source": "nasa", "title": title, "query": query,
                     "description": clean(data.get("description"))[:400], "date": clean(data.get("date_created"))[:10],
                     "credit": clean(data.get("secondary_creator") or data.get("center") or "NASA")[:80],
                     "page_url": f"https://images.nasa.gov/details/{nasa_id}", "manifest": https(manifest), "thumb": https(thumb),
                     "keywords": [clean(k) for k in (data.get("keywords") or [])][:12]})
    return rows


def svs_search(query, count=10):
    """NASA Scientific Visualization Studio: 4K visualizations and ISS/satellite footage."""
    try:
        response = requests.get(SVS_SEARCH, params={"search": query}, timeout=(10, 30), headers=HEADERS)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as error:  # noqa: BLE001
        print(f"SVS araması ({query}): {type(error).__name__}: {str(error)[:80]}", flush=True)
        return []
    rows = []
    for item in results[:count]:
        title = clean(item.get("title"))
        identifier = item.get("id")
        if not title or not identifier or any(word in title.casefold() for word in SKIP_TITLE):
            continue
        rows.append({"id": f"svs:{identifier}", "source": "svs", "title": title, "query": query,
                     "description": clean(re.sub(r"<[^>]+>", " ", str(item.get("description") or "")))[:400],
                     "date": clean(item.get("release_date"))[:10], "credit": "NASA's Scientific Visualization Studio",
                     "page_url": f"https://svs.gsfc.nasa.gov/{identifier}", "svs_id": identifier, "thumb": "",
                     "keywords": [clean(k) for k in (item.get("keywords") or [])][:12] if isinstance(item.get("keywords"), list) else []})
    return rows


# -------------------------------------------------------------- renditions
def _parse_duration(value):
    text = clean(value)
    if not text:
        return None
    match = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", text)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|seconds)?$", text)
    if match:
        return float(match.group(1))
    return None


def _rendition_tag(url):
    name = url.casefold()
    for tag in ("orig", "large", "medium", "small", "mobile", "preview"):
        if f"~{tag}" in name:
            return tag
    return "other"


def nasa_renditions(asset):
    """Read the asset manifest (+ metadata.json) → mp4 renditions, duration, size."""
    response = requests.get(asset["manifest"], timeout=(10, 30), headers=HEADERS)
    response.raise_for_status()
    files = response.json()
    if not isinstance(files, list):
        raise ValueError("manifest is not a list")
    files = [https(f) for f in files if isinstance(f, str)]
    movies = [f for f in files if f.casefold().endswith(".mp4")]
    renditions = [{"url": f, "tag": _rendition_tag(f), "width": None, "height": None} for f in movies]
    duration, width, height = None, None, None
    meta = next((f for f in files if f.casefold().endswith("metadata.json")), None)
    if meta:
        try:
            info = requests.get(meta, timeout=(10, 30), headers=HEADERS).json()
            for key, value in (info or {}).items():
                lower = str(key).casefold()
                if duration is None and lower.endswith("duration"):
                    duration = _parse_duration(value)
                if lower.endswith("imagesize") and isinstance(value, str) and "x" in value:
                    try:
                        width, height = (int(v) for v in value.casefold().split("x")[:2])
                    except ValueError:
                        pass
                if lower.endswith("imagewidth") and width is None:
                    width = int(value) if str(value).isdigit() else width
                if lower.endswith("imageheight") and height is None:
                    height = int(value) if str(value).isdigit() else height
        except Exception:  # noqa: BLE001
            pass
    for r in renditions:
        if r["tag"] == "orig":
            r["width"], r["height"] = width, height
    order = {"orig": 0, "large": 1, "medium": 2, "small": 3, "mobile": 4, "preview": 5, "other": 6}
    renditions.sort(key=lambda r: order[r["tag"]])
    return renditions, duration


def svs_renditions(asset):
    response = requests.get(SVS_ITEM.format(id=asset["svs_id"]), timeout=(10, 40), headers=HEADERS)
    response.raise_for_status()
    item = response.json()
    renditions = []
    for group in item.get("media_groups") or []:
        for entry in group.get("items") or []:
            instance = entry.get("instance") or {}
            url = https(instance.get("url"))
            name = url.casefold()
            if instance.get("media_type") != "Movie" or not name.endswith(".mp4"):
                continue
            if any(bad in name for bad in ("prores", "_print", "youtube", "twitter", "instagram", "facebook", "tiktok")):
                continue
            width, height = instance.get("width") or 0, instance.get("height") or 0
            if width * height > MAX_PIXELS or not height:
                continue
            renditions.append({"url": url, "width": int(width), "height": int(height),
                               "tag": "orig" if height >= 1080 else "small", "caption": clean(entry.get("caption"))[:200]})
    asset["description"] = asset.get("description") or clean(re.sub(r"<[^>]+>", " ", str(item.get("description") or "")))[:400]
    credits = item.get("main_credits") or ""
    if isinstance(credits, list):
        credits = ", ".join(clean(c.get("name") if isinstance(c, dict) else c) for c in credits[:3])
    asset["credit"] = clean(credits)[:80] or asset["credit"]
    # biggest first, but never above 4K
    renditions.sort(key=lambda r: (-(r["width"] * r["height"]), r["url"]))
    # dedupe near-identical resolutions keeping the first
    seen, unique = set(), []
    for r in renditions:
        key = (r["width"], r["height"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique, None


def probe(url, timeout=90):
    """ffprobe over https: duration, size, fps, audio — reads only the headers."""
    command = ["ffprobe", "-v", "error", *net_options(url), "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate",
               "-of", "json", url]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise ValueError(f"ffprobe: {result.stderr.strip()[-160:]}")
    data = json.loads(result.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    fps = 30.0
    try:
        num, den = (video.get("avg_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        pass
    return {"duration": float((data.get("format") or {}).get("duration") or 0), "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0), "fps": fps, "audio": audio}


def resolve(asset):
    """Fill renditions, pick the best (≤4K) and a small one for sampling; probe when needed."""
    renditions, duration = (nasa_renditions if asset["source"] == "nasa" else svs_renditions)(asset)
    if not renditions:
        raise ValueError("no mp4 rendition")
    best = renditions[0]
    small = next((r for r in renditions if r["tag"] in ("medium", "small", "mobile")), None)
    if asset["source"] == "svs":
        small = next((r for r in reversed(renditions) if r["height"] and r["height"] <= 720), None) or best
    small = small or best
    info = probe(best["url"])
    if info["duration"] <= 0:
        raise ValueError("no duration")
    asset.update(best=best["url"], small=small["url"], duration=info["duration"], width=info["width"], height=info["height"],
                 fps=info["fps"], audio=info["audio"], renditions=renditions)
    return asset


# ---------------------------------------------------------------- sampling
def grab_frame(url, seconds, path, width=640):
    command = ["ffmpeg", "-v", "error", "-y", *net_options(url), "-ss", f"{seconds:.2f}", "-i", url, "-frames:v", "1",
               "-vf", f"scale={width}:-2", "-q:v", "4", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
    if result.returncode != 0 or not path.is_file():
        raise ValueError(f"frame {seconds:.1f}s: {result.stderr.strip()[-120:]}")
    return path


def sample_times(duration, count):
    margin = min(2.0, duration * 0.08)
    span = max(0.5, duration - 2 * margin)
    return [margin + span * (i + 0.5) / count for i in range(count)]


def sample(asset, directory, workers=4):
    """Evenly spaced frames across the small rendition (seeks only fetch what they need)."""
    duration = asset["duration"]
    count = max(8, min(16, int(round(duration / 6))))
    times = sample_times(duration, count)
    directory.mkdir(parents=True, exist_ok=True)
    frames = [None] * count
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(grab_frame, asset["small"], t, directory / f"s{i:02}.jpg"): i for i, t in enumerate(times)}
        for job in as_completed(jobs):
            i = jobs[job]
            try:
                frames[i] = (times[i], job.result())
            except Exception as error:  # noqa: BLE001
                print(f"Kare alınamadı ({asset['id']} @{times[i]:.0f}s): {str(error)[:80]}", flush=True)
    frames = [f for f in frames if f]
    if len(frames) < max(4, count // 2):
        raise ValueError("too few frames")
    return frames


def _sheet_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(frames, path, columns=3, tile=(400, 225)):
    """Numbered grid of the sampled frames with their timestamps."""
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile[0], rows * (tile[1] + 28)), (12, 12, 16))
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(20)
    for index, (seconds, frame_path) in enumerate(frames):
        with Image.open(frame_path) as picture:
            picture = picture.convert("RGB")
            picture.thumbnail(tile)
            x = (index % columns) * tile[0] + (tile[0] - picture.width) // 2
            y = (index // columns) * (tile[1] + 28) + 28
            sheet.paste(picture, (x, y))
        draw.text(((index % columns) * tile[0] + 8, (index // columns) * (tile[1] + 28) + 4),
                  f"#{index}  {int(seconds // 60)}:{int(seconds % 60):02}", font=font, fill=(255, 220, 90))
    sheet.save(path, "JPEG", quality=88)
    return path


# ----------------------------------------------------------------- judging
def rank_candidates(api, topic, candidates, keep=10):
    """Cheap text pass: which search hits probably contain real footage of the subject."""
    if len(candidates) <= keep:
        return candidates
    listing = "\n".join(f"[{i}] ({c['source']}, {c.get('date', '')}) {c['title']} — {c['description'][:160]}"
                        for i, c in enumerate(candidates))
    prompt = f"""We are editing a calm, high-end vertical documentary Short about {topic['subject']}.
Below are video search results from NASA's archives. Choose the {keep} most likely to contain REAL, beautiful, full-frame footage or
high-quality visualizations of the subject: spacecraft camera footage, telescope imagery, orbital views, science visualizations.
Reject anything that sounds like a talking-head interview, a briefing, a news package with presenters, a compilation of many topics,
or an explainer built from slides and on-screen text.
Return {{"keep":[indices in order of preference]}}.

{listing}"""
    try:
        data = api.json("Görüntü adayları", prompt)
        chosen = [candidates[int(i)] for i in data.get("keep", []) if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(candidates)]
        if chosen:
            return chosen[:keep]
    except Exception as error:  # noqa: BLE001
        print(f"Aday sıralaması atlandı: {str(error)[:100]}", flush=True)
    return candidates[:keep]


def judge_asset(api, topic, asset, frames, sheet):
    """One look at the contact sheet: is this real footage, which stretches are clean, where is the subject."""
    count = len(frames)
    prompt = f"""You are the picture editor of a calm, premium vertical documentary Short about {topic['subject']}.
This contact sheet shows {count} frames (#0..#{count - 1}, with timestamps) sampled evenly from one NASA video:
title: "{asset['title']}" ({asset.get('date', '')}); description: "{asset.get('description', '')[:300]}".

Decide which stretches are usable as full-screen footage on a phone. A stretch is usable only if EVERY sampled frame in it:
- shows the subject or closely related real imagery (spacecraft/telescope footage, orbital views, science visualization),
- has NO burned-in text, captions, logos, title cards, charts, diagrams, slides or watermarks,
- has NO presenters, interviewees, studio or control-room scenes, hands, or people talking,
- is not a black/white blank, not a multi-panel collage, not a blurry or tiny inset.
Group consecutive usable frames into stretches (start index, end index, inclusive). Prefer stretches with the most striking,
cinematic imagery. For each stretch give the focus point of the main subject as fractions [x, y] of the frame (0..1, from top-left)
— the vertical crop will be centred there — and say in a few words what is on screen.
Also classify the imagery: "footage" (a real camera or telescope image sequence), "visualization" (rendered from real data, e.g. a
flyover built from spacecraft measurements) or "animation" (an artist's render of hardware or events).
Return {{"usable": true|false, "quality": 1-10, "kind": "footage"|"visualization"|"animation", "what": "one line: what this video shows",
"has_text": true|false, "has_people": true|false, "stretches": [{{"start": <index>, "end": <index>, "focus": [x, y], "shows": "..."}}]}}."""
    data = api.json(f"Görüntü değerlendirme {asset['id']}", prompt, images=[("contact sheet", sheet)])
    stretches = []
    for row in data.get("stretches") or []:
        try:
            a, b = int(row.get("start")), int(row.get("end"))
        except (TypeError, ValueError):
            continue
        if not (0 <= a <= b < count):
            continue
        focus = row.get("focus") or [0.5, 0.5]
        try:
            fx, fy = (min(0.95, max(0.05, float(v))) for v in focus[:2])
        except (TypeError, ValueError):
            fx, fy = 0.5, 0.5
        stretches.append({"start": a, "end": b, "focus": [fx, fy], "shows": clean(row.get("shows"))[:120]})
    kind = str(data.get("kind") or "footage").casefold()
    kind = kind if kind in ("footage", "visualization", "animation") else "footage"
    return {"usable": bool(data.get("usable")) and bool(stretches), "quality": int(data.get("quality") or 0), "kind": kind,
            "what": clean(data.get("what"))[:160], "has_text": bool(data.get("has_text")), "has_people": bool(data.get("has_people")),
            "stretches": stretches}


# ------------------------------------------------------------- extraction
def extract(url, start, end, path, margin=1.0):
    """Cut a stretch from the best rendition. Re-encoding (rather than stream
    copy) makes the cut frame-accurate, so clip time = stretch time + margin."""
    begin = max(0.0, start - margin)
    length = (end - start) + 2 * margin
    command = ["ffmpeg", "-v", "error", "-y", *net_options(url), "-ss", f"{begin:.3f}", "-i", url, "-t", f"{length:.3f}",
               "-map", "0:v:0", "-an", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode != 0 or not path.is_file() or path.stat().st_size < 20_000:
        raise ValueError(f"extract: {result.stderr.strip()[-160:]}")
    return path


def analyse(path):
    """Hard cuts and black stretches inside a local clip, plus its real duration/size."""
    info = probe(str(path), timeout=60)
    duration = info["duration"]
    cuts = []
    result = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", "scale=320:-2,select='gt(scene,0.42)',showinfo",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
    for match in re.finditer(r"pts_time:\s*([0-9.]+)", result.stderr or ""):
        cuts.append(float(match.group(1)))
    black = []
    result = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", "scale=320:-2,blackdetect=d=0.3:pic_th=0.96:pix_th=0.12",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
    for match in re.finditer(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)", result.stderr or ""):
        black.append((float(match.group(1)), float(match.group(2))))
    # continuous lit windows
    edges = sorted({0.0, duration, *cuts, *[b for pair in black for b in pair]})
    windows = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 1.5:
            continue
        if any(bs <= a + 0.05 and b - 0.05 <= be for bs, be in black):
            continue
        windows.append((round(a, 2), round(b, 2)))
    return {"duration": duration, "width": info["width"], "height": info["height"], "fps": info["fps"], "cuts": cuts,
            "black": black, "windows": windows}


# ------------------------------------------------------------------ driver
def candidates_for(topic):
    rows, seen = [], set()
    for query in topic.get("nasa", [])[:3]:
        for row in nasa_search(query):
            if row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
    for query in topic.get("svs", [])[:2]:
        for row in svs_search(query):
            if row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
    return rows


def harvest(api, topic, directory, want=8, minimum=4, log=print):
    """Return usable clips for the topic: [{asset, file, start, end, focus, shows, windows, ...}]."""
    root = Path(directory) / "footage"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    candidates = candidates_for(topic)
    log(f"Görüntü adayları: {len(candidates)} ({topic['key']})")
    if not candidates:
        raise FactoryError("NASA arşivinde aday video bulunamadı.")
    ranked = rank_candidates(api, topic, candidates, keep=10)
    resolved = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(resolve, dict(asset)): asset for asset in ranked}
        for job in as_completed(jobs):
            asset = jobs[job]
            try:
                item = job.result()
            except Exception as error:  # noqa: BLE001
                log(f"Çözümlenemedi {asset['id']}: {type(error).__name__}: {str(error)[:90]}")
                continue
            if not (MIN_ASSET_SECONDS <= item["duration"] <= MAX_ASSET_SECONDS):
                log(f"Süre uygun değil {item['id']}: {item['duration']:.0f} sn")
                continue
            if item["height"] and item["height"] < 700:
                log(f"Düşük çözünürlük {item['id']}: {item['width']}x{item['height']}")
                continue
            resolved.append(item)
    order = {a["id"]: i for i, a in enumerate(ranked)}
    resolved.sort(key=lambda a: order.get(a["id"], 99))
    log(f"Çözümlenen video: {len(resolved)}")
    clips, judged = [], []
    for asset in resolved:
        if len(clips) >= want or time.monotonic() - started > 14 * 60:
            break
        api.check()
        folder = root / re.sub(r"[^a-z0-9]+", "_", asset["id"].casefold())
        try:
            frames = sample(asset, folder / "samples")
            sheet = contact_sheet(frames, folder / "sheet.jpg")
            verdict = judge_asset(api, topic, asset, frames, sheet)
        except Exception as error:  # noqa: BLE001
            log(f"Değerlendirilemedi {asset['id']}: {type(error).__name__}: {str(error)[:90]}")
            continue
        judged.append({"id": asset["id"], "title": asset["title"], **{k: verdict[k] for k in ("usable", "quality", "kind", "what", "has_text", "has_people")},
                       "stretches": verdict["stretches"]})
        log(f"{asset['id']}: {'kullanılır' if verdict['usable'] else 'elendi'} ({verdict['quality']}/10) {verdict['what'][:70]}")
        if not verdict["usable"] or verdict["quality"] < 5:
            continue
        spacing = asset["duration"] / max(1, len(frames))
        taken = 0
        for stretch in sorted(verdict["stretches"], key=lambda s: -(s["end"] - s["start"])):
            if taken >= 3 or len(clips) >= want:
                break
            start = frames[stretch["start"]][0] - spacing * 0.5
            end = frames[stretch["end"]][0] + spacing * 0.5
            start, end = max(0.0, start), min(asset["duration"], end)
            if end - start < 3.0:
                continue
            # at most ~24 s per stretch; long stretches are cut from their middle
            if end - start > 24:
                centre = (start + end) / 2
                start, end = centre - 12, centre + 12
            clip_path = folder / f"clip_{taken}.mp4"
            try:
                extract(asset["best"], start, end, clip_path)
                analysis = analyse(clip_path)
            except Exception as error:  # noqa: BLE001
                log(f"Kesit alınamadı {asset['id']} {start:.0f}-{end:.0f}: {str(error)[:90]}")
                continue
            usable = [w for w in analysis["windows"] if w[1] - w[0] >= 2.5]
            if not usable:
                continue
            clips.append({"id": f"c{len(clips):02}", "asset": asset["id"], "title": asset["title"], "date": asset.get("date", ""),
                          "credit": asset.get("credit", ""), "page_url": asset.get("page_url", ""), "file": str(clip_path),
                          "source_start": start, "duration": analysis["duration"], "width": analysis["width"], "height": analysis["height"],
                          "fps": analysis["fps"], "windows": usable, "focus": stretch["focus"], "shows": stretch["shows"] or verdict["what"],
                          "quality": verdict["quality"], "kind": verdict["kind"], "what": verdict["what"]})
            taken += 1
    (root / "judged.json").write_text(json.dumps({"candidates": [{k: c.get(k) for k in ("id", "title", "date", "source")} for c in candidates],
                                                  "ranked": [c["id"] for c in ranked], "judged": judged}, indent=1, ensure_ascii=False), encoding="utf-8")
    seconds = sum(max(w[1] - w[0] for w in c["windows"]) for c in clips)
    log(f"Kullanılır kesit: {len(clips)} ({seconds:.0f} sn temiz görüntü)")
    if len(clips) < minimum or seconds < 30:
        raise FactoryError(f"Yeterli temiz görüntü yok ({len(clips)} kesit, {seconds:.0f} sn).")
    return clips
