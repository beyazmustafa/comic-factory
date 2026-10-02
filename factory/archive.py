"""Public-domain comic issues from the Internet Archive.

The reference format (one real issue, read panel by panel with narration) only
works with complete issues. Internet Archive hosts thousands of Golden Age
issues scanned from the Digital Comic Museum / Comic Book Plus communities,
but its "comics" collection also holds copyrighted material (Disney, MAD, ...).
This module therefore accepts an item ONLY when its metadata declares a public
domain license and the publication date is before 1964; both facts are saved
beside the pages as evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageOps

from .api import SourceUnavailable
from .config import ROOT
from .core import save_json, language_name
from .research import clean, public_url

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/"
DOWNLOAD_URL = "https://archive.org/download/"
DETAILS_URL = "https://archive.org/details/"
MAX_YEAR = 1963  # US works first published before 1964 needed renewal; DCM verifies.
MAX_DOWNLOAD = 400_000_000
PAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".jp2", ".webp", ".gif", ".bmp", ".tif", ".tiff")
PD_MARKERS = ("publicdomain", "public domain", "public-domain", "pdm", "cc0")
# Words that mark a Golden Age superhero issue (public-domain heroes and generic hero titles).
HERO_MARKERS = (
    "superhero", "super hero", "super-hero", "super-man", "superman", "hero", "heroes", "black terror",
    "fighting yank", "daredevil", "samson", "the flame", "blue beetle", "amazing man", "amazing-man",
    "wonder comics", "wonderman", "captain", "mystery men", "cat-man", "catman", "black owl", "green lama",
    "miss masque", "hangman", "silver streak", "crimebuster", "bulletman", "spy smasher", "ibis",
    "mr. scarlet", "minute-man", "pyroman", "doll man", "blackhawk", "kid eternity", "airboy", "skyman",
    "liberty", "patriot", "invincible", "magno", "blue bolt", "dynamic man", "steel sterling",
    "rocketman", "fantoman", "stuntman", "phantom lady", "lady luck", "the mask", "red mask",
    "exciting comics", "startling comics", "thrilling comics", "fight comics", "big shot", "smash comics",
    "crack comics", "hit comics", "prize comics", "pep comics", "zip comics", "top-notch", "blue ribbon",
    "mystic comics", "speed comics", "champ comics", "green mask", "yellowjacket", "the owl", "boy heroes",
)

# Golden Age publishers that closed before 1964 and whose issues were, as a
# rule, never copyright-renewed; the Digital Comic Museum catalogue is built on
# these. A hit here is "inferred" evidence and is recorded as such.
LAPSED_PUBLISHERS = (
    "ace", "ace magazines", "ace comics", "fox", "fox feature", "fiction house",
    "lev gleason", "gleason", "charlton", "nedor", "standard", "better publications",
    "pines", "avon", "ajax", "farrell", "ajax-farrell", "centaur", "prize", "hillman",
    "ziff-davis", "ziff davis", "youthful", "star publications", "toby", "st. john",
    "st john", "fawcett", "quality comics", "comic media", "stanmor", "story comics",
    "trojan", "key publications", "mikeross", "aragon", "ribage", "superior",
    "magazine enterprises", "orbit", "novelty", "holyoke", "rural home", "harry a chesler",
    "chesler", "eastern color", "columbia comics", "famous funnies", "parents",
    "ned pines", "sterling", "premier", "ajax farrell", "atlas magazines",
)
# Brands and publishers still under active copyright. Any hit rejects the item.
PROTECTED_MARKERS = (
    "disney", "mickey", "donald duck", "uncle scrooge", "mad magazine", "mad",
    "marvel", "timely", "atlas comics", "captain america", "dc comics", "national comics",
    "national periodical", "superman", "batman", "wonder woman", "archie", "harvey",
    "casper", "richie rich", "dell", "gold key", "western publishing", "ec comics",
    "entertaining comics", "tales from the crypt", "classics illustrated", "gilberton",
    "popeye", "tarzan", "looney tunes", "bugs bunny", "warner", "hanna", "barbera",
    "peanuts", "tintin", "asterix", "king features", "blondie", "flash gordon",
    "prince valiant", "little lulu", "lone ranger", "roy rogers", "gene autry",
    "walt kelly", "pogo", "charlie chan", "dick tracy", "li'l abner", "lil abner",
    "the spirit", "will eisner", "plastic man", "blackhawk", "captain marvel", "shazam",
)


def identifier_ok(value):
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,99}", str(value or "")))


def first(value):
    if isinstance(value, list):
        return clean(value[0]) if value else ""
    return clean(value)


def year_of(value):
    match = re.search(r"(1[89]\d\d|20\d\d)", first(value))
    return int(match.group(1)) if match else None


def _joined(row, keys):
    parts = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            parts.extend(clean(v) for v in value)
        elif value:
            parts.append(clean(value))
    return " | ".join(parts).casefold()


def declared_public_domain(row):
    """Return the metadata field carrying an explicit public domain declaration."""
    for key in ("licenseurl", "rights", "possible-copyright-status", "license"):
        value = first(row.get(key)).casefold()
        if value and any(marker in value for marker in PD_MARKERS):
            return f"{key}: {first(row.get(key))}"
    for key in ("description", "notes"):
        value = first(row.get(key)).casefold()
        if "public domain" in value:
            return f"{key}: public domain"
    return ""


def _marker_in(text, markers):
    tokens = set(re.findall(r"[a-z0-9.&'-]+", text))
    for name in markers:
        if " " in name or "-" in name:
            if name in text:
                return name
        elif name in tokens:
            return name
    return ""


def protected_marker(row):
    text = _joined(row, ("title", "publisher", "creator", "subject", "collection", "description"))
    return _marker_in(text, PROTECTED_MARKERS)


def lapsed_publisher(row):
    text = _joined(row, ("publisher", "creator", "collection", "subject", "title"))
    # Collection slugs such as "ace-comics" become searchable words.
    return _marker_in(text.replace("-", " ") + " " + text, LAPSED_PUBLISHERS)


def public_domain_evidence(row):
    """Explicit declaration first; otherwise a lapsed Golden Age publisher.

    Both routes reject anything that names a still-protected brand.
    """
    if protected_marker(row):
        return ""
    declared = declared_public_domain(row)
    if declared:
        return "declared " + declared
    publisher = lapsed_publisher(row)
    if publisher:
        return f"inferred: publisher '{publisher}' (closed before 1964, unrenewed Golden Age catalogue)"
    return ""


def build_query(topic, settings):
    base = settings.archive_query
    date = f"date:[1920-01-01 TO {settings.archive_max_year}-12-31]"
    parts = [f"({base})", date]
    if not topic and settings.channel_theme == "superheroes" and settings.archive_theme_query.strip():
        parts.append(f"({settings.archive_theme_query})")
    if topic:
        # A free-text topic narrows the archive search instead of the open web.
        safe = re.sub(r'[^\w\s\-\'"]', " ", topic)
        parts.append(f"({safe})")
    return " AND ".join(parts)


class Archive:
    """Small Internet Archive client: search, metadata, file download."""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; ComicFactoryArchiveReader/1.0)"
        )

    def close(self):
        self.session.close()

    def search(self, query, rows=60, page=1):
        params = {
            "q": query,
            "fl[]": [
                "identifier",
                "title",
                "date",
                "year",
                "publisher",
                "creator",
                "subject",
                "licenseurl",
                "rights",
                "description",
                "downloads",
                "imagecount",
            ],
            "rows": rows,
            "page": page,
            "output": "json",
            "sort[]": "downloads desc",
        }
        response = self.session.get(SEARCH_URL, params=params, timeout=(8, 40))
        response.raise_for_status()
        payload = response.json().get("response", {})
        docs = [row for row in payload.get("docs", []) if isinstance(row, dict)]
        return docs, int(payload.get("numFound") or len(docs))

    def metadata(self, identifier):
        if not identifier_ok(identifier):
            raise ValueError("Geçersiz arşiv kimliği.")
        response = self.session.get(
            METADATA_URL + quote(identifier, safe=""), timeout=(8, 40)
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("metadata"):
            raise ValueError("Arşiv kaydı bulunamadı.")
        return payload

    def download(self, identifier, filename, destination: Path, maximum=MAX_DOWNLOAD):
        url = DOWNLOAD_URL + quote(identifier, safe="") + "/" + quote(filename, safe="/")
        public_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.session.get(url, stream=True, timeout=(8, 60)) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", "0") or 0)
            if size > maximum:
                raise ValueError("Arşiv dosyası çok büyük.")
            total = 0
            with destination.open("wb") as output:
                for chunk in response.iter_content(1 << 20):
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("Arşiv dosyası çok büyük.")
                    output.write(chunk)
        return destination


def eligible(row, settings):
    """Keep only items with a public domain declaration and an early date."""
    identifier = first(row.get("identifier"))
    if not identifier_ok(identifier):
        return None
    year = year_of(row.get("year")) or year_of(row.get("date"))
    evidence = public_domain_evidence(row)
    if not year or year > settings.archive_max_year or not evidence:
        return None
    title = first(row.get("title"))
    if not title:
        return None
    return {
        "identifier": identifier,
        "title": title,
        "year": year,
        "publisher": first(row.get("publisher")) or first(row.get("creator")),
        "description": first(row.get("description"))[:1200],
        "subjects": [clean(s) for s in (row.get("subject") or [])][:12]
        if isinstance(row.get("subject"), list)
        else [first(row.get("subject"))],
        "public_domain_evidence": evidence,
        "downloads": int(row.get("downloads") or 0),
        "imagecount": int(row.get("imagecount") or 0),
        "url": DETAILS_URL + identifier,
    }


def series_key(title):
    words = [w for w in re.findall(r"[a-z]+", str(title).casefold()) if w not in {"the", "a", "of", "and", "comics", "comic", "no", "vol", "issue"}]
    return " ".join(words[:2])


def recent_series(limit=6):
    """Series keys of the most recently produced issues (repo-tracked history)."""
    folder = ROOT / "data" / "history" / "issues"
    try:
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except OSError:
        return set()
    keys = set()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for field in ("series", "title"):
                if data.get(field):
                    keys.add(series_key(data[field]))
        except (OSError, ValueError):
            continue
    return keys


def issue_key(identifier):
    return "ia_" + hashlib.sha256(identifier.encode()).hexdigest()[:16]


def shortlist(api, topic, used, archive=None):
    """Search the archive, filter licence/date, let the model pick the best stories."""
    import random

    client = archive or Archive()
    query = build_query(topic, api.settings)
    rows, failures, visited = [], [], []
    try:
        api.check()
        try:
            docs, found = client.search(query, rows=100, page=1)
            rows.extend(docs)
            visited.append(1)
        except (requests.RequestException, ValueError) as error:
            found = 0
            failures.append({"page": 1, "error": type(error).__name__})
        # Popular items (Disney, MAD) sit at the top; sample deeper pages so a
        # daily schedule sees a different slice of the catalogue each run.
        last_page = max(1, min(40, -(-found // 100)))
        for page in random.sample(range(2, last_page + 1), min(2, max(0, last_page - 1))):
            api.check()
            try:
                docs, _ = client.search(query, rows=100, page=page)
                rows.extend(docs)
                visited.append(page)
            except (requests.RequestException, ValueError) as error:
                failures.append({"page": page, "error": type(error).__name__})
    except BaseException:
        if archive is None:
            client.close()
        raise
    items = []
    for row in rows:
        item = eligible(row, api.settings)
        if not item:
            continue
        item["id"] = issue_key(item["identifier"])
        if item["id"] in used or item["identifier"].casefold() in used:
            continue
        if item["imagecount"] and item["imagecount"] < 16:
            continue  # Single stories/ads rarely carry a full narrative.
        items.append(item)
    items = list({i["identifier"]: i for i in items}.values())
    # Variety: do not return to a series the channel covered recently.
    recent = recent_series(6)
    if recent:
        fresh = [i for i in items if series_key(i["title"]) not in recent]
        if len(fresh) >= 3:
            items = fresh
    if api.settings.channel_theme == "superheroes":
        def heroic(item):
            text = " ".join([item["title"], item.get("description", ""), " ".join(item.get("subjects", []))]).casefold()
            return any(m in text for m in HERO_MARKERS)
        heroes = [i for i in items if heroic(i)]
        if len(heroes) >= 3:
            items = heroes
        else:
            # Look deeper into the catalogue before settling for non-hero issues.
            extra_rows = []
            try:
                for page in random.sample(range(2, last_page + 1), min(4, max(0, last_page - 1))):
                    api.check()
                    docs, _ = client.search(query, rows=100, page=page)
                    extra_rows.extend(docs)
                    visited.append(page)
            except (requests.RequestException, ValueError) as error:
                failures.append({"page": "hero-search", "error": type(error).__name__})
            for row in extra_rows:
                item = eligible(row, api.settings)
                if item and item["identifier"] not in {i["identifier"] for i in items}:
                    item["id"] = issue_key(item["identifier"])
                    if item["id"] in used or item["identifier"].casefold() in used:
                        continue
                    if item["imagecount"] and item["imagecount"] < 16:
                        continue
                    items.append(item)
            heroes = [i for i in items if heroic(i)]
            items = heroes if heroes else items
    if archive is None:
        client.close()
    save_json(
        api.directory / "research" / "archive_search.json",
        {"query": query, "pages": visited, "raw_rows": len(rows), "eligible": items, "failures": failures},
    )
    if not items:
        raise SourceUnavailable(
            "Arşivde kamu malı olarak işaretlenmiş, kullanılmamış bir sayı bulunamadı; archive_search.json kaydedildi."
        )
    random.shuffle(items)
    allowed = {item["identifier"]: item for item in items[:60]}
    payload = api.json(
        "Arşiv sayı adayları",
        f"""Below are Internet Archive comic issues whose metadata declares them public domain. Pick up to {api.settings.max_events} issues most likely to contain ONE strong, surprising, self-contained Golden Age story. Channel focus: {api.settings.channel_theme} — strongly prefer SUPERHERO issues (costumed heroes, villains, origins, deaths, betrayals, shocking twists, bizarre powers); crime/horror only if no hero issue fits. Judge by title/series/description which issue promises the most attention-grabbing event for a global YouTube Shorts audience. Use ONLY the listed identifiers; invent nothing. Requested topic: {topic or "automatic"}.
{json.dumps(list(allowed.values()), ensure_ascii=False)}
Return {{"events":[{{"identifier":"","title":"{language_name(api)} hook title (curiosity-driven, no clickbait lies)","series":"exact original series title","issue":"issue number or empty","summary":"Turkish one-sentence expectation based only on metadata","importance":0,"popularity":0,"niche":0}}]}}. Scores 0..100 are editorial judgments.""",
        list_key="events",
    )
    candidates = []
    for row in payload.get("events", []):
        if not isinstance(row, dict):
            continue
        item = allowed.get(first(row.get("identifier")))
        if not item or not clean(row.get("title")):
            continue
        try:
            score = sum(
                max(0, min(100, float(row.get(k, 0)))) * w
                for k, w in (("importance", 0.4), ("popularity", 0.2), ("niche", 0.4))
            )
        except (ValueError, TypeError):
            continue
        candidates.append(
            {
                **item,
                "title": clean(row["title"]),
                "series": clean(row.get("series")) or item["title"],
                "issue": clean(row.get("issue")) or "?",
                "summary": clean(row.get("summary")),
                "universe": "Golden Age (public domain)",
                "source_urls": [item["url"]],
                "selection_score": score,
            }
        )
    candidates.sort(key=lambda r: r["selection_score"], reverse=True)
    save_json(
        api.directory / "research" / "candidates.json",
        {"requested_topic": topic, "events": candidates},
    )
    if not candidates:
        raise SourceUnavailable("Model listedeki arşiv sayılarından seçim yapamadı.")
    return candidates[: api.settings.max_events]


def ranked_files(files):
    """Page packages in preference order: raw image zips, cbz, jp2, pdf, cbr."""
    ranked = []
    for row in files:
        name = first(row.get("name"))
        lower = name.casefold()
        size = int(row.get("size") or 0)
        if lower.endswith("_images.zip") or lower.endswith(".cbz"):
            rank = 0
        elif lower.endswith("_jp2.zip"):
            rank = 1
        elif lower.endswith(".zip"):
            rank = 2
        elif lower.endswith(".pdf"):
            rank = 3
        elif lower.endswith(".cbr") and shutil.which("7z"):
            rank = 4
        else:
            continue
        if size > MAX_DOWNLOAD or "_meta" in lower:
            continue
        ranked.append((rank, -size, name))
    if not ranked:
        raise SourceUnavailable("Arşiv kaydında okunabilir sayfa paketi (zip/cbz/jp2/pdf) yok.")
    ranked.sort()
    return [name for _, _, name in ranked]


def choose_file(files):
    return ranked_files(files)[0]


def natural_key(name):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


def extract_pages(package: Path, output: Path, limit=80):
    """Turn a cbz/zip/pdf/cbr into ordered RGB JPEG pages."""
    output.mkdir(parents=True, exist_ok=True)
    lower = package.name.casefold()
    pictures = []
    if lower.endswith((".cbz", ".zip")):
        with zipfile.ZipFile(package) as archive:
            names = sorted(
                (n for n in archive.namelist() if n.casefold().endswith(PAGE_EXTENSIONS)
                 and not Path(n).name.startswith(".")),
                key=natural_key,
            )
            for name in names[:limit]:
                pictures.append(("zip", archive.read(name)))
    elif lower.endswith(".pdf"):
        if not shutil.which("pdftoppm"):
            raise SourceUnavailable("PDF sayfaları için pdftoppm (poppler-utils) gerekli.")
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                ["pdftoppm", "-r", "120", "-jpeg", "-l", str(limit), str(package), f"{temporary}/p"],
                check=True, capture_output=True, timeout=900,
            )
            for path in sorted(Path(temporary).glob("p*.jpg"), key=lambda p: natural_key(p.name)):
                pictures.append(("pdf", path.read_bytes()))
    elif lower.endswith(".cbr"):
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                ["7z", "x", "-y", f"-o{temporary}", str(package)],
                check=True, capture_output=True, timeout=900,
            )
            files = sorted(
                (p for p in Path(temporary).rglob("*") if p.suffix.casefold() in PAGE_EXTENSIONS),
                key=lambda p: natural_key(p.name),
            )
            for path in files[:limit]:
                pictures.append(("rar", path.read_bytes()))
    else:
        raise SourceUnavailable("Desteklenmeyen sayfa paketi.")
    pages = []
    for index, (_, body) in enumerate(pictures):
        try:
            with Image.open(io.BytesIO(body)) as opened:
                opened.load()
                picture = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception:
            continue
        if min(picture.size) < 500:
            continue
        if picture.width > 3200:
            scale = 3200 / picture.width
            picture = picture.resize((3200, round(picture.height * scale)), Image.Resampling.LANCZOS)
        path = output / f"scan_{index:03}.jpg"
        picture.save(path, "JPEG", quality=96, subsampling=0)
        pages.append(path)
    if len(pages) < 8:
        raise SourceUnavailable(f"Paketten yalnız {len(pages)} okunabilir sayfa çıktı.")
    return pages


def collect_pages(api, event, archive=None):
    """Download the issue, keep the story pages, return pages + evidence article."""
    client = archive or Archive()
    root = api.directory / "events" / event["id"]
    try:
        api.check()
        payload = client.metadata(event["identifier"])
        metadata = payload.get("metadata", {})
        evidence = public_domain_evidence(metadata) or event.get("public_domain_evidence", "")
        year = year_of(metadata.get("year")) or year_of(metadata.get("date")) or event["year"]
        if not evidence or year > api.settings.archive_max_year:
            raise SourceUnavailable("Arşiv kaydı kamu malı beyanı taşımıyor; sayı atlandı.")
        files = [f for f in payload.get("files", []) if isinstance(f, dict)]
        scans, filename, errors = [], "", []
        for candidate in ranked_files(files)[:3]:
            api.check()
            package = root / "download" / Path(candidate).name
            try:
                client.download(event["identifier"], candidate, package)
                scans = extract_pages(package, root / "scans", limit=api.settings.archive_scan_limit)
                filename = candidate
                break
            except (requests.RequestException, ValueError, OSError, SourceUnavailable,
                    zipfile.BadZipFile, subprocess.SubprocessError) as error:
                errors.append({"file": candidate, "error": f"{type(error).__name__}: {error}"[:300]})
            finally:
                try:
                    package.unlink()
                except OSError:
                    pass
        if not scans:
            save_json(root / "download_errors.json", errors)
            raise SourceUnavailable("Sayfa paketi indirilemedi veya açılamadı; download_errors.json kaydedildi.")
        ocr = ""
        text_file = next((f["name"] for f in files if first(f.get("name")).endswith("_djvu.txt")), "")
        if text_file:
            try:
                target = root / "download" / "ocr.txt"
                client.download(event["identifier"], text_file, target, maximum=6_000_000)
                ocr = clean(target.read_text(encoding="utf-8", errors="replace"))[:60000]
            except Exception:
                ocr = ""
    finally:
        if archive is None:
            client.close()
    article = {
        "id": "source_000",
        "url": event["url"],
        "title": first(metadata.get("title")) or event["title"],
        "text": " ".join(
            filter(
                None,
                [
                    f"Internet Archive item {event['identifier']}.",
                    f"Title: {first(metadata.get('title'))}.",
                    f"Publisher: {first(metadata.get('publisher'))}.",
                    f"Date: {first(metadata.get('date'))}.",
                    f"Licence: {evidence}.",
                    f"Description: {first(metadata.get('description'))[:4000]}",
                    ("OCR: " + ocr) if ocr else "",
                ],
            )
        ),
        "images": [],
        "public_domain_evidence": evidence,
        "year": year,
    }
    save_json(root / "sources" / "source_000.json", article)
    pages = []
    for index, path in enumerate(scans):
        with Image.open(path) as picture:
            width, height = picture.size
        pages.append(
            {
                "id": f"page_{index:03}",
                "file": str(path.relative_to(api.directory)),
                "scan_index": index,
                "source_id": "source_000",
                "source_url": event["url"],
                "source_title": article["title"],
                "image_url": DOWNLOAD_URL + event["identifier"] + "/" + filename,
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    save_json(root / "pages.json", pages)
    return pages, {"source_000": article}


def remember_issue(event, run_id):
    """Repository-tracked history so a scheduled run never repeats an issue."""
    path = ROOT / "data" / "history" / "issues" / (event["id"] + ".json")
    save_json(
        path,
        {
            "id": event["id"],
            "identifier": event["identifier"],
            "title": event["title"],
            "series": event.get("series"),
            "year": event.get("year"),
            "url": event.get("url"),
            "run_id": run_id,
        },
    )
    return path
