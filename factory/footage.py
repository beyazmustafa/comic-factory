"""Real footage for the film source.

Sources (all free to reuse with credit, most public domain):
  nasa     NASA Image and Video Library (images.nasa.gov) — JPL/GSFC/JSC videos
  svs      NASA Scientific Visualization Studio — 4K visualizations, Earth/Sun/Moon
  hubble   ESA/Hubble video archive (CC BY 4.0) — zooms, flythroughs, timelapses
  webb     ESA/Webb video archive (CC BY 4.0)
  sdo      Solar Dynamics Observatory daily movies — the real Sun, day by day
  commons  Wikimedia Commons videos (public domain / CC BY only), mostly NASA mirrors

Pipeline per topic:
  search → rank candidates by title/description → resolve renditions and
  duration → sample frames along the whole video, OCR them → one look by the
  vision model per asset (relevant? which stretches are clean? where is the
  subject?) → cut those stretches from the best rendition → find hard cuts,
  black frames and burned-in text inside them so the editor only uses
  continuous, lit, clean footage.

Network calls carry their own timeouts and never raise past `harvest`; a
topic without enough usable footage raises FactoryError so the caller can
move on to another topic.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .api import FactoryError

NASA_API = "https://images-api.nasa.gov/search"
SVS_SEARCH = "https://svs.gsfc.nasa.gov/api/search/"
SVS_ITEM = "https://svs.gsfc.nasa.gov/api/{id}"
ESA_SITES = {"hubble": "https://esahubble.org", "webb": "https://esawebb.org"}
ESA_CDN = {"hubble": "https://cdn.esahubble.org", "webb": "https://cdn.esawebb.org"}
SDO_DAILY = "https://sdo.gsfc.nasa.gov/assets/img/dailymov/{y}/{m}/{d}/{y}{m}{d}_1024_{channel}.mp4"
SDO_CHANNELS = (("0304", "304 Å, the chromosphere"), ("0171", "171 Å, coronal loops"), ("0131", "131 Å, flaring regions"),
                ("0193", "193 Å, the corona"), ("0094", "94 Å, flares"))
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "comic-factory/2.0 (short documentaries from public-domain space footage; contact via github)"}
FFMPEG_NET = ["-rw_timeout", "40000000", "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "8",
              "-user_agent", HEADERS["User-Agent"]]
# Titles that announce talking heads, live events or compilations.
SKIP_TITLE = ("interview", "briefing", "press conference", "news conference", "b-roll", "broll", "podcast", "webinar",
              "q&a", "town hall", "teleconference", "newscast", "360 video", "trailer", "highlight", "this week",
              "science casts", "sciencecast", "we asked", "asl", "en español", "remarks", "speech", "testimony",
              "hearing", "award", "year in review", "students", "ask nasa", "episode", "live:", "livestream", "live shots",
              "ceremony", "panel", "lecture", "tutorial", "message from", "greeting", "crew news", "space to ground",
              "nasa explorers", "talks", "chat", "conversation", "hubblecast", "webb cast", "commentary", "coverage",
              "science live", "calendar", "valentine", "infographic", "challenge", "mission overview", "what's up")
SVS_TYPES = {"Visualization": 0, "Animation": 1, "Produced Video": 2}
MAX_ASSET_SECONDS = 45 * 60
COMMONS_MAX_BYTES = 450_000_000
MIN_ASSET_SECONDS = 6
MAX_PIXELS = 4096 * 2304
OCR_WORD = re.compile(r"^[A-Za-z][A-Za-z'’-]{2,}[.,!?:;]*$")


def clean(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def https(url):
    return re.sub(r"^http://", "https://", str(url or ""))


def net_options(url):
    """ffmpeg's http options are rejected for local files, so only pass them for URLs."""
    return list(FFMPEG_NET) if str(url).startswith(("http://", "https://")) else []


def skip_title(title):
    lower = title.casefold()
    return any(word in lower for word in SKIP_TITLE)


# ------------------------------------------------------------------ search
def nasa_search(query, count=30):
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
        if not title or skip_title(title):
            continue
        nasa_id = clean(data.get("nasa_id"))
        manifest = item.get("href")
        if not manifest or not nasa_id:
            continue
        rows.append({"id": f"nasa:{nasa_id}", "source": "nasa", "title": title, "query": query,
                     "description": clean(data.get("description"))[:400], "date": clean(data.get("date_created"))[:10],
                     "credit": clean(data.get("secondary_creator") or data.get("center") or "NASA")[:80], "licence": "public domain (NASA)",
                     "page_url": f"https://images.nasa.gov/details/{nasa_id}", "manifest": https(manifest)})
    return rows


def svs_search(query, count=12):
    """NASA Scientific Visualization Studio: visualizations and animations first, produced videos last."""
    try:
        response = requests.get(SVS_SEARCH, params={"search": query}, timeout=(10, 30), headers=HEADERS)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as error:  # noqa: BLE001
        print(f"SVS araması ({query}): {type(error).__name__}: {str(error)[:80]}", flush=True)
        return []
    rows = []
    for item in results:
        title = clean(item.get("title"))
        identifier = item.get("id")
        kind = str(item.get("result_type") or "")
        if not title or not identifier or kind not in SVS_TYPES or skip_title(title):
            continue
        rows.append({"id": f"svs:{identifier}", "source": "svs", "title": title, "query": query, "svs_type": kind,
                     "description": clean(re.sub(r"<[^>]+>", " ", str(item.get("description") or "")))[:400],
                     "date": clean(item.get("release_date"))[:10], "credit": "NASA's Scientific Visualization Studio",
                     "licence": "public domain (NASA)", "page_url": f"https://svs.gsfc.nasa.gov/{identifier}", "svs_id": identifier})
    rows.sort(key=lambda r: SVS_TYPES[r["svs_type"]])
    return rows[:count]


def esa_search(site, query, count=8):
    """ESA/Hubble and ESA/Webb video archives (CC BY 4.0): ids scraped from the search page."""
    base = ESA_SITES[site]
    try:
        response = requests.get(f"{base}/videos/", params={"search": query}, timeout=(10, 30), headers=HEADERS)
        response.raise_for_status()
        page = response.text
    except Exception as error:  # noqa: BLE001
        print(f"ESA/{site} araması ({query}): {type(error).__name__}: {str(error)[:80]}", flush=True)
        return []
    rows, seen = [], set()
    for match in re.finditer(r'href="/videos/([a-z0-9_-]+)/"[^>]*>(.*?)</a>', page, flags=re.I | re.S):
        identifier, inner = match.group(1), match.group(2)
        if identifier in seen or identifier in ("archive", "feed", "viewall"):
            continue
        title = clean(re.sub(r"<[^>]+>", " ", inner))
        if not title or len(title) < 4:
            continue
        seen.add(identifier)
        if skip_title(title) or identifier.startswith(("hubblecast", "webbcast")):
            continue
        rows.append({"id": f"{site}:{identifier}", "source": site, "title": title, "query": query, "description": "",
                     "date": "", "credit": f"ESA/{site.capitalize()}", "licence": "CC BY 4.0",
                     "page_url": f"{base}/videos/{identifier}/", "esa_id": identifier})
        if len(rows) >= count:
            break
    return rows


def sdo_candidates(dates):
    """The real Sun from SDO: one daily movie per wavelength for each listed date (public domain)."""
    rows = []
    for date in dates:
        y, m, d = str(date).split("-")
        for channel, label in SDO_CHANNELS:
            rows.append({"id": f"sdo:{y}{m}{d}_{channel}", "source": "sdo", "title": f"SDO {label} — {date}", "query": "sdo",
                         "description": f"Solar Dynamics Observatory daily movie, {label}, {date}.", "date": str(date),
                         "credit": "NASA/SDO", "licence": "public domain (NASA)", "page_url": "https://sdo.gsfc.nasa.gov/data/",
                         "sdo_url": SDO_DAILY.format(y=y, m=m, d=d, channel=channel), "trim_bottom": 0.08})
    return rows


def commons_search(query, count=10):
    """Wikimedia Commons videos; licence is checked when the asset is resolved."""
    try:
        hits = commons_get({"action": "query", "list": "search", "srsearch": f"{query} filetype:video",
                            "srnamespace": 6, "srlimit": count, "format": "json"}).get("query", {}).get("search", [])
    except Exception as error:  # noqa: BLE001
        print(f"Commons araması ({query}): {type(error).__name__}: {str(error)[:80]}", flush=True)
        return []
    rows = []
    for hit in hits:
        title = clean(hit.get("title"))
        if not title.startswith("File:"):
            continue
        name = title[5:]
        shown = re.sub(r"\.(webm|ogv|mp4|mov)$", "", name, flags=re.I).replace("_", " ")
        if skip_title(shown):
            continue
        rows.append({"id": f"commons:{name}", "source": "commons", "title": shown, "query": query,
                     "description": clean(re.sub(r"<[^>]+>", " ", str(hit.get("snippet") or "")))[:300], "date": "",
                     "credit": "", "licence": "", "page_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                     "commons_title": title})
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
    """Read the asset manifest (+ metadata.json) → mp4 renditions, best first."""
    response = requests.get(asset["manifest"], timeout=(10, 30), headers=HEADERS)
    response.raise_for_status()
    files = response.json()
    if not isinstance(files, list):
        raise ValueError("manifest is not a list")
    files = [https(f) for f in files if isinstance(f, str)]
    movies = [f for f in files if f.casefold().endswith(".mp4")]
    renditions = [{"url": f, "tag": _rendition_tag(f), "width": 0, "height": 0} for f in movies]
    order = {"orig": 0, "large": 1, "medium": 2, "small": 3, "mobile": 4, "preview": 5, "other": 6}
    renditions.sort(key=lambda r: order[r["tag"]])
    small = next((r for r in renditions if r["tag"] in ("medium", "small", "mobile")), None)
    return renditions, small


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
            if any(bad in name for bad in ("prores", "_print", "youtube", "twitter", "instagram", "facebook", "tiktok", "square", "vertical")):
                continue
            width, height = int(instance.get("width") or 0), int(instance.get("height") or 0)
            if width * height > MAX_PIXELS or not height:
                continue
            renditions.append({"url": url, "width": width, "height": height, "tag": "orig" if height >= 1080 else "small"})
    asset["description"] = asset.get("description") or clean(re.sub(r"<[^>]+>", " ", str(item.get("description") or "")))[:400]
    credits = item.get("main_credits") or ""
    if isinstance(credits, list):
        credits = ", ".join(clean(c.get("name") if isinstance(c, dict) else c) for c in credits[:3])
    asset["credit"] = clean(credits)[:80] or asset["credit"]
    renditions.sort(key=lambda r: (-(r["width"] * r["height"]), r["url"]))
    seen, unique = set(), []
    for r in renditions:
        key = (r["width"], r["height"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    small = next((r for r in reversed(unique) if r["height"] <= 720), None)
    return unique, small


def _head_ok(url):
    try:
        response = requests.head(url, timeout=(10, 20), headers=HEADERS, allow_redirects=True)
        return response.status_code == 200 and int(response.headers.get("Content-Length", "1") or 1) > 100_000
    except requests.RequestException:
        return False


ESA_FOLDERS = {"ultra_hd": (3840, 2160), "hd_1080p25_screen": (1920, 1080), "hd_and_apple": (1280, 720),
               "medium_podcast": (640, 360), "large_qt": (720, 405)}


def esa_renditions(asset):
    site, identifier = asset["source"], asset["esa_id"]
    try:
        page = requests.get(asset["page_url"], timeout=(10, 30), headers=HEADERS).text
    except requests.RequestException:
        page = ""
    title = re.search(r"<h1[^>]*>(.*?)</h1>", page, flags=re.S)
    if title:
        asset["title"] = clean(re.sub(r"<[^>]+>", " ", title.group(1)))[:120] or asset["title"]
    credit = re.search(r"Credit:?\s*</?[^>]*>?\s*(.*?)</(?:p|div|td)>", page, flags=re.S | re.I)
    if credit:
        asset["credit"] = clean(re.sub(r"<[^>]+>", " ", credit.group(1)))[:120] or asset["credit"]
    date = re.search(r"Release date:?.*?(\d{1,2} \w+ \d{4})", page, flags=re.S | re.I)
    if date:
        asset["date"] = date.group(1)
    description = re.search(r'<meta name="description" content="([^"]*)"', page)
    if description:
        asset["description"] = clean(description.group(1))[:400]
    # every archive link the page offers, by folder
    found = {}
    for match in re.finditer(r"https?://[a-z0-9.]*(?:esahubble|esawebb)\.org/(?:media/)?archives/videos/([a-z0-9_]+)/([A-Za-z0-9_.-]+\.(?:mp4|m4v|mov))", page):
        folder, name = match.group(1), match.group(2)
        if folder in ESA_FOLDERS and folder not in found:
            found[folder] = https(match.group(0))
    cdn = ESA_CDN[site]
    if "hd_1080p25_screen" not in found and _head_ok(f"{cdn}/archives/videos/hd_1080p25_screen/{identifier}.mp4"):
        found["hd_1080p25_screen"] = f"{cdn}/archives/videos/hd_1080p25_screen/{identifier}.mp4"
    renditions = []
    for folder in ("ultra_hd", "hd_1080p25_screen", "hd_and_apple"):
        if folder in found:
            width, height = ESA_FOLDERS[folder]
            renditions.append({"url": found[folder], "width": width, "height": height, "tag": "orig" if height >= 1080 else "small"})
    small = None
    for folder in ("medium_podcast", "large_qt", "hd_and_apple"):
        if folder in found:
            width, height = ESA_FOLDERS[folder]
            small = {"url": found[folder], "width": width, "height": height, "tag": "small"}
            break
    return renditions, small


def sdo_renditions(asset):
    url = asset["sdo_url"]
    if not _head_ok(url):
        return [], None
    return [{"url": url, "width": 1024, "height": 1024, "tag": "orig"}], None


_commons_lock = threading.Lock()


def commons_get(params):
    """One Commons API call at a time, with a pause and one retry on 429."""
    with _commons_lock:
        for attempt in range(2):
            response = requests.get(COMMONS_API, params=params, timeout=(10, 40), headers=HEADERS)
            if response.status_code == 429 and attempt == 0:
                time.sleep(6)
                continue
            response.raise_for_status()
            time.sleep(1.0)
            return response.json()
    return {}


_download_lock = threading.Lock()


def download(url, path, maximum):
    """Stream a file to disk with our User-Agent, one at a time with a pause
    (Wikimedia throttles bulk downloads from shared runners; a 429 is retried once)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock:
        for attempt in range(2):
            with requests.get(url, stream=True, timeout=(15, 180), headers=HEADERS) as response:
                if response.status_code == 429 and attempt == 0:
                    time.sleep(25)
                    continue
                response.raise_for_status()
                if int(response.headers.get("Content-Length", "0") or 0) > maximum:
                    raise ValueError("file too large")
                size = 0
                with path.open("wb") as handle:
                    for chunk in response.iter_content(1 << 16):
                        handle.write(chunk)
                        size += len(chunk)
                        if size > maximum:
                            raise ValueError("file too large")
            time.sleep(2.0)
            return path
    raise ValueError("download refused (429)")


def commons_renditions(asset):
    data = commons_get({"action": "query", "titles": asset["commons_title"], "prop": "videoinfo",
                        "viprop": "url|size|derivatives|extmetadata|mediatype", "format": "json"})
    pages = data.get("query", {}).get("pages", {})
    info = next((p.get("videoinfo", [{}])[0] for p in pages.values() if p.get("videoinfo")), None)
    if not info or info.get("mediatype") != "VIDEO":
        raise ValueError("not a video")
    meta = info.get("extmetadata") or {}
    licence = clean((meta.get("LicenseShortName") or {}).get("value"))
    lower = licence.casefold()
    if not (("public domain" in lower or lower in ("pd", "cc0")) or (lower.startswith("cc by") and "sa" not in lower.split("by", 1)[1])):
        raise ValueError(f"licence not reusable: {licence or 'unknown'}")
    asset["licence"] = licence
    artist = clean(re.sub(r"<[^>]+>", " ", (meta.get("Artist") or {}).get("value") or ""))[:80]
    credit = clean(re.sub(r"<[^>]+>", " ", (meta.get("Credit") or {}).get("value") or ""))[:80]
    asset["credit"] = artist or credit or "Wikimedia Commons"
    asset["date"] = clean((meta.get("DateTimeOriginal") or {}).get("value"))[:10]
    asset["description"] = asset.get("description") or clean(re.sub(r"<[^>]+>", " ", (meta.get("ImageDescription") or {}).get("value") or ""))[:400]
    renditions = []
    original = {"url": str(info.get("url") or "").split("?")[0], "width": int(info.get("width") or 0), "height": int(info.get("height") or 0), "tag": "orig"}
    if original["url"] and original["width"] * original["height"] <= MAX_PIXELS:
        renditions.append(original)
    for d in info.get("derivatives") or []:
        src, width, height = str(d.get("src") or "").split("?")[0], int(d.get("width") or 0), int(d.get("height") or 0)
        if not src or not height or not src.casefold().endswith((".webm", ".mp4")):
            continue
        if width * height <= MAX_PIXELS:
            renditions.append({"url": src, "width": width, "height": height, "tag": "orig" if height >= 1080 else "small"})
    renditions.sort(key=lambda r: -(r["width"] * r["height"]))
    small = next((r for r in reversed(renditions) if 300 <= r["height"] <= 720), None)
    return renditions, small


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


RESOLVERS = {"nasa": nasa_renditions, "svs": svs_renditions, "hubble": esa_renditions, "webb": esa_renditions,
             "sdo": sdo_renditions, "commons": commons_renditions}


def resolve(asset, directory=None):
    """Fill renditions, pick the best (≤4K) and a small one for sampling; probe the best.
    Commons files are downloaded whole (Wikimedia refuses ffmpeg's piecemeal requests)."""
    renditions, small = RESOLVERS[asset["source"]](asset)
    if not renditions:
        raise ValueError("no usable rendition")
    best = renditions[0]
    small = small or best
    if asset["source"] == "commons" and directory is not None:
        folder = Path(directory) / "downloads"
        local_best = download(best["url"], folder / (re.sub(r"[^a-z0-9]+", "_", asset["id"].casefold())[:50] + "_best" + Path(best["url"].split("?")[0]).suffix), COMMONS_MAX_BYTES)
        local_small = local_best
        if small["url"] != best["url"]:
            try:
                local_small = download(small["url"], folder / (re.sub(r"[^a-z0-9]+", "_", asset["id"].casefold())[:50] + "_small" + Path(small["url"].split("?")[0]).suffix), 120_000_000)
            except Exception:  # noqa: BLE001
                local_small = local_best
        best, small = {**best, "url": str(local_best)}, {**small, "url": str(local_small)}
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


def ocr_words(path):
    """Confident words tesseract reads in a frame (empty when tesseract is missing)."""
    if not shutil.which("tesseract"):
        return []
    try:
        result = subprocess.run(["tesseract", str(path), "stdout", "--psm", "11", "-l", "eng", "tsv"],
                                capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return []
    words = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            confidence = float(parts[10])
        except ValueError:
            continue
        word = parts[11].strip()
        if confidence >= 70 and OCR_WORD.match(word):
            words.append(word)
    return words


def has_text(path):
    return len(ocr_words(path)) >= 2


_face_net = None
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def _face_detector():
    """OpenCV's SSD face detector (res10) when its two model files are present; Haar cascades
    are far too noisy on craters and plasma, so without the model no face check runs."""
    global _face_net
    if _face_net is None:
        _face_net = False
        prototxt, weights = MODEL_DIR / "deploy.prototxt", MODEL_DIR / "res10.caffemodel"
        if prototxt.is_file() and weights.is_file() and weights.stat().st_size > 1_000_000:
            try:
                import cv2

                _face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
            except Exception:  # noqa: BLE001
                _face_net = False
    return _face_net or None


_face_lock = threading.Lock()


def has_face(path, threshold=0.6):
    """A presenter, interviewee or crowd: a confident face of a plausible size."""
    net = _face_detector()
    if net is None:
        return False
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return False
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        with _face_lock:
            net.setInput(blob)
            detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < threshold:
                continue
            x1, y1, x2, y2 = detections[0, 0, i, 3:7]
            if (x2 - x1) * width >= width * 0.05 and (y2 - y1) * height >= height * 0.05:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def is_blank(path):
    """Nearly black (or nearly white) frame with nothing in it."""
    try:
        with Image.open(path) as picture:
            grey = picture.convert("L").resize((64, 36))
            pixels = list(grey.getdata())
        mean = sum(pixels) / len(pixels)
        lit = sum(1 for v in pixels if v > 40) / len(pixels)
        # nothing in it, or only a few bright specks in the dark (a distant dot is not a shot)
        return (mean < 8 and max(pixels) < 60) or (mean < 15 and lit < 0.02) or mean > 245
    except Exception:  # noqa: BLE001
        return False


def unclean(path, trim_bottom=0.0):
    """Frame carries text or a face, or is blank → never usable. `trim_bottom` ignores a
    known timestamp strip (SDO daily movies stamp the date along the bottom edge)."""
    if is_blank(path):
        return True
    if not trim_bottom:
        return has_text(path) or has_face(path)
    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        trimmed = picture.crop((0, 0, picture.width, int(picture.height * (1 - trim_bottom))))
        scratch = Path(str(path) + ".trim.jpg")
        trimmed.save(scratch, "JPEG", quality=90)
    try:
        return has_text(scratch) or has_face(scratch)
    finally:
        scratch.unlink(missing_ok=True)


def sample(asset, directory, workers=4):
    """Evenly spaced frames across the small rendition (seeks only fetch what they need), each OCR-checked."""
    duration = asset["duration"]
    count = max(8, min(24, int(round(duration / 6))))
    times = sample_times(duration, count)
    directory.mkdir(parents=True, exist_ok=True)
    frames = [None] * count
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(grab_frame, asset["small"], t, directory / f"s{i:02}.jpg"): i for i, t in enumerate(times)}
        for job in as_completed(jobs):
            i = jobs[job]
            try:
                frames[i] = (times[i], job.result())
            except Exception as error:  # noqa: BLE001
                failures.append(str(error)[-90:])
    frames = [f for f in frames if f]
    if len(frames) < max(4, count // 2):
        raise ValueError(f"too few frames ({len(frames)}/{count}): {failures[0] if failures else ''}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        flags = list(pool.map(lambda f: unclean(f[1], asset.get("trim_bottom", 0.0)), frames))
        faces = list(pool.map(lambda f: has_face(f[1]), frames))
    asset["face_frames"] = sum(1 for f in faces if f)
    return frames, flags


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
    """Cheap text pass: which search hits probably contain real footage of the subject; irrelevant ones are dropped."""
    listing = "\n".join(f"[{i}] ({c['source']}, {c.get('date', '')[:4]}) {c['title']} — {c['description'][:160]}"
                        for i, c in enumerate(candidates))
    prompt = f"""We are editing a calm, high-end vertical documentary Short about {topic['subject']}.
Below are video search results from space agency archives. Choose up to {keep}, in order of preference, that most likely contain
REAL, beautiful, full-frame footage or high-quality visualizations OF THIS SUBJECT: spacecraft camera footage, telescope imagery,
orbital views, science visualizations.
Drop anything off-subject (another planet, mission or phenomenon), anything that sounds like a talking-head interview, a
briefing, a news package with presenters, a compilation of many topics, or an explainer built from slides and on-screen text.
Return {{"keep":[indices in order of preference]}} — fewer is fine, an empty list if nothing fits.

{listing}"""
    try:
        data = api.json("Görüntü adayları", prompt)
        chosen = [candidates[int(i)] for i in data.get("keep", []) if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(candidates)]
        return chosen[:keep]
    except Exception as error:  # noqa: BLE001
        print(f"Aday sıralaması atlandı: {str(error)[:100]}", flush=True)
    return candidates[:keep]


def judge_asset(api, topic, asset, frames, flags, sheet):
    """One look at the contact sheet: relevant? which stretches are clean? where is the subject?"""
    count = len(frames)
    flagged = [i for i, f in enumerate(flags) if f]
    ocr_note = (f"\nAn automatic pass found on-screen text or a person in frames {flagged}; those frames are excluded whatever you decide." if flagged else "")
    prompt = f"""You are the picture editor of a calm, premium vertical documentary Short about {topic['subject']}.
This contact sheet shows {count} frames (#0..#{count - 1}, with timestamps) sampled evenly from one video:
title: "{asset['title']}" ({asset.get('date', '')}); description: "{asset.get('description', '')[:300]}".{ocr_note}

First decide relevance: the video must actually show {topic['subject']} (or its direct subject matter). A video about a
different planet, mission or phenomenon is NOT relevant even if it is beautiful.
If any sampled frame shows a presenter, an interview or a studio, this is a produced programme: mark usable=false unless the
remaining frames are clearly raw mission footage with nothing added. When in doubt, reject.
Then decide which stretches are usable as full-screen footage on a phone. A stretch is usable only if EVERY sampled frame in it:
- shows the subject or closely related real imagery (spacecraft/telescope footage, orbital views, science visualization),
- has NO burned-in text, captions, subtitles, logos, title cards, charts, diagrams, slides, grid or coordinate lines, arrows or watermarks,
- has NO presenters, interviewees, studio, lab, clean-room or control-room scenes, hands, or people talking (indoor scenes on
  Earth are not usable unless the subject itself is that hardware being built),
- is not a black/white blank, not a multi-panel collage, not a blurry or tiny inset.
Group consecutive usable frames into stretches (start index, end index, inclusive). Prefer the most striking, cinematic imagery.
For each stretch give the focus point of the main subject as fractions [x, y] of the frame (0..1, from top-left) — the vertical
crop will be centred there — and the fraction of the frame WIDTH the composition needs: 0.32 (a phone-shaped slice around the
focus, the default — full-screen footage looks best) unless that slice would cut away something essential, e.g. a spacecraft on one
side and a planet on the other; then 0.6 or 1.0. Say in a few words what is on screen.
Also classify the imagery: "footage" (a real camera or telescope image sequence), "visualization" (rendered from real data, e.g. a
flyover built from spacecraft measurements) or "animation" (an artist's render of hardware or events).
Return {{"relevant": true|false, "usable": true|false, "quality": 1-10, "kind": "footage"|"visualization"|"animation",
"what": "one line: what this video shows", "text_frames": [indices with any on-screen text], "people_frames": [indices with people],
"stretches": [{{"start": <index>, "end": <index>, "focus": [x, y], "width": 0.32-1.0, "shows": "...", "score": 1-10 for sharpness and beauty}}]}}."""
    data = api.json(f"Görüntü değerlendirme {asset['id']}", prompt, images=[("contact sheet", sheet)])
    excluded = set(flagged)
    for key in ("text_frames", "people_frames"):
        for value in data.get(key) or []:
            if str(value).isdigit() and 0 <= int(value) < count:
                excluded.add(int(value))
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
        try:
            score = int(row.get("score") or 7)
        except (TypeError, ValueError):
            score = 7
        if score < 5:
            continue
        try:
            crop_width = min(1.0, max(0.32, float(row.get("width") or 0.32)))
        except (TypeError, ValueError):
            crop_width = 0.32
        # split the stretch around excluded frames
        run = []
        for index in list(range(a, b + 1)) + [None]:
            if index is not None and index not in excluded:
                run.append(index)
                continue
            if run:
                stretches.append({"start": run[0], "end": run[-1], "focus": [fx, fy], "width": crop_width, "shows": clean(row.get("shows"))[:120], "score": score})
            run = []
    kind = str(data.get("kind") or "footage").casefold()
    kind = kind if kind in ("footage", "visualization", "animation") else "footage"
    relevant = bool(data.get("relevant", True))
    return {"relevant": relevant, "usable": relevant and bool(data.get("usable")) and bool(stretches),
            "quality": int(data.get("quality") or 0), "kind": kind, "what": clean(data.get("what"))[:160],
            "excluded": sorted(excluded), "stretches": stretches}


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


def text_intervals(path, duration, step=1.0, pad=0.8, trim_bottom=0.0):
    """Seconds of a local clip whose frames carry readable text or a face (one check per second)."""
    if not shutil.which("tesseract") and _face_detector() is None:
        return []
    folder = Path(path).with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", f"fps=1/{step},scale=640:-2", "-q:v", "5",
                    str(folder / "t%03d.jpg")], capture_output=True, timeout=600)
    frames = sorted(folder.glob("t*.jpg"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        flags = list(pool.map(lambda f: unclean(f, trim_bottom), frames))
    intervals = []
    for index, flag in enumerate(flags):
        if flag:
            centre = index * step + step / 2
            intervals.append((max(0.0, centre - step / 2 - pad), min(duration, centre + step / 2 + pad)))
    for frame in frames:
        frame.unlink(missing_ok=True)
    try:
        folder.rmdir()
    except OSError:
        pass
    return intervals


def active_area(path, width, height, duration):
    """Letterbox/pillarbox bars → the picture's real rectangle (x, y, w, h), or None for a full frame."""
    start = max(0.0, min(duration - 2.0, duration * 0.3))
    result = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{start:.2f}", "-i", str(path), "-t", "2.5", "-vf", "cropdetect=48:4:0",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=300)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr or "")
    if not found:
        return None
    w, h, x, y = (int(v) for v in found[-1])
    if w < width * 0.5 or h < height * 0.5:
        return None  # a mostly dark frame, not bars
    if w >= width * 0.96 and h >= height * 0.96:
        return None
    return [x, y, w, h]


def analyse(path, trim_bottom=0.0):
    """Hard cuts, black stretches and on-screen text inside a local clip, plus its real duration/size."""
    info = probe(str(path), timeout=60)
    duration = info["duration"]
    cuts = []
    result = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", "scale=320:-2,select='gt(scene,0.42)',showinfo",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
    for match in re.finditer(r"pts_time:\s*([0-9.]+)", result.stderr or ""):
        cuts.append(float(match.group(1)))
    blocked = []
    result = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", "scale=320:-2,blackdetect=d=0.3:pic_th=0.999:pix_th=0.05",
                             "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=600)
    for match in re.finditer(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)", result.stderr or ""):
        blocked.append((float(match.group(1)), float(match.group(2))))
    text = text_intervals(path, duration, trim_bottom=trim_bottom)
    blocked += text
    active = active_area(path, info["width"], info["height"], duration)
    edges = sorted({0.0, duration, *cuts, *[b for pair in blocked for b in pair]})
    windows = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 1.5:
            continue
        if any(bs <= a + 0.05 and b - 0.05 <= be for bs, be in blocked):
            continue
        windows.append((round(a, 2), round(b, 2)))
    return {"duration": duration, "width": info["width"], "height": info["height"], "fps": info["fps"], "cuts": cuts,
            "black": blocked, "text": text, "windows": windows, "active": active}


# ------------------------------------------------------------------ driver
def candidates_for(topic, commons=False):
    """Search hits for the topic; Wikimedia Commons only on request (its servers
    throttle shared runners hard, so it is a second phase when the rest is thin)."""
    rows, seen = [], set()

    def add(items):
        for row in items:
            if row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)

    for query in topic.get("nasa", [])[:4]:
        add(nasa_search(query))
    for query in topic.get("svs", [])[:3]:
        add(svs_search(query))
    for site in ("hubble", "webb"):
        for query in topic.get(site, [])[:2]:
            add(esa_search(site, query))
    if topic.get("sdo"):
        add(sdo_candidates(topic["sdo"][:2]))
    if commons:
        commons_rows = []
        for query in topic.get("commons", [])[:2]:
            commons_rows += commons_search(query)
        add(commons_rows[:4])
    return rows


def harvest(api, topic, directory, want=8, minimum=4, log=print):
    """Return usable clips for the topic: [{asset, file, start, end, focus, shows, windows, ...}]."""
    root = Path(directory) / "footage"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    clips, judged, ranked_ids, candidate_rows = [], [], [], []
    for phase, use_commons in enumerate((False, True)):
        if phase and (len(clips) >= minimum + 1 or not topic.get("commons")):
            break
        candidates = [c for c in candidates_for(topic, commons=use_commons) if c["source"] == "commons" or not phase]
        if phase:
            candidates = [c for c in candidates if c["source"] == "commons"]
        candidate_rows += candidates
        log(f"Görüntü adayları: {len(candidates)} ({topic['key']}{', Commons' if phase else ''})")
        if not candidates:
            continue
        ranked = rank_candidates(api, topic, candidates, keep=12 if not phase else 4)
        ranked_ids += [c["id"] for c in ranked]
        log(f"Sıralanan aday: {len(ranked)}")
        resolved = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs = {pool.submit(resolve, dict(asset), root): asset for asset in ranked}
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
        for asset in resolved:
            if len(clips) >= want or time.monotonic() - started > 18 * 60:
                break
            api.check()
            folder = root / re.sub(r"[^a-z0-9]+", "_", asset["id"].casefold())[:60]
            try:
                frames, flags = sample(asset, folder / "samples")
                sheet = contact_sheet(frames, folder / "sheet.jpg")
                verdict = judge_asset(api, topic, asset, frames, flags, sheet)
            except Exception as error:  # noqa: BLE001
                log(f"Değerlendirilemedi {asset['id']}: {type(error).__name__}: {str(error)[:90]}")
                continue
            judged.append({"id": asset["id"], "title": asset["title"], **{k: verdict[k] for k in ("relevant", "usable", "quality", "kind", "what", "excluded")},
                           "stretches": verdict["stretches"]})
            state = "kullanılır" if verdict["usable"] else ("konu dışı" if not verdict["relevant"] else "elendi")
            log(f"{asset['id']}: {state} ({verdict['quality']}/10, {verdict['kind']}) {verdict['what'][:70]}")
            if not verdict["usable"] or verdict["quality"] < 5:
                continue
            if asset.get("face_frames", 0) > len(frames) * 0.25 or len(verdict["excluded"]) > len(frames) * 0.6:
                log(f"{asset['id']}: üretilmiş program gibi ({asset.get('face_frames', 0)} karede insan, {len(verdict['excluded'])}/{len(frames)} kare elendi); atlandı")
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
                if end - start > 24:
                    centre = (start + end) / 2
                    start, end = centre - 12, centre + 12
                clip_path = folder / f"clip_{taken}.mp4"
                try:
                    extract(asset["best"], start, end, clip_path)
                    analysis = analyse(clip_path, asset.get("trim_bottom", 0.0))
                    if asset.get("trim_bottom") and not analysis["active"]:
                        analysis["active"] = [0, 0, analysis["width"], int(analysis["height"] * (1 - asset["trim_bottom"]))]
                except Exception as error:  # noqa: BLE001
                    log(f"Kesit alınamadı {asset['id']} {start:.0f}-{end:.0f}: {str(error)[:90]}")
                    continue
                usable = [w for w in analysis["windows"] if w[1] - w[0] >= 2.5]
                if not usable:
                    log(f"Kesitte temiz pencere yok {asset['id']} {start:.0f}-{end:.0f} (yazı: {len(analysis['text'])}, kesme: {len(analysis['cuts'])})")
                    continue
                clips.append({"id": f"c{len(clips):02}", "asset": asset["id"], "source": asset["source"], "title": asset["title"],
                              "date": asset.get("date", ""), "credit": asset.get("credit", ""), "licence": asset.get("licence", ""),
                              "page_url": asset.get("page_url", ""), "file": str(clip_path), "source_start": start,
                              "duration": analysis["duration"], "width": analysis["width"], "height": analysis["height"], "fps": analysis["fps"],
                              "windows": usable, "focus": stretch["focus"], "crop_width": stretch.get("width", 0.32), "shows": stretch["shows"] or verdict["what"],
                              "active": analysis["active"], "quality": min(verdict["quality"], stretch.get("score", verdict["quality"])),
                              "kind": verdict["kind"], "what": verdict["what"]})
                taken += 1
    (root / f"judged_{topic['key']}.json").write_text(json.dumps({"candidates": [{k: c.get(k) for k in ("id", "title", "date", "source")} for c in candidate_rows],
                                                                  "ranked": ranked_ids, "judged": judged}, indent=1, ensure_ascii=False), encoding="utf-8")
    seconds = sum(max(w[1] - w[0] for w in c["windows"]) for c in clips)
    log(f"Kullanılır kesit: {len(clips)} ({seconds:.0f} sn temiz görüntü)")
    if len(clips) < minimum or seconds < 30:
        raise FactoryError(f"Yeterli temiz görüntü yok ({len(clips)} kesit, {seconds:.0f} sn).")
    return clips
