import hashlib
import io
import ipaddress
import json
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from PIL import Image, ImageOps
from .api import SourceUnavailable
from .core import save_json


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def public_url(url):
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if (
        p.scheme not in {"http", "https"}
        or not host
        or p.username
        or p.password
        or p.port not in (None, 80, 443)
    ):
        raise ValueError("Geçersiz kaynak adresi.")
    if (
        host == "localhost"
        or host.endswith((".local", ".internal", ".localhost"))
        or "." not in host
        or "%" in host
    ):
        raise ValueError("Yerel ağ kaynak olamaz.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Özel ağ kaynak olamaz.")
    return url


def event_key(event):
    return hashlib.sha256(
        "|".join(
            clean(event.get(k)).casefold()
            for k in ("publisher", "series", "issue", "year")
        ).encode()
    ).hexdigest()[:20]


def search(queries, images=False, each=8):
    output, seen = [], set()
    for query in queries:
        try:
            client = DDGS(timeout=12)
            rows = (
                client.images(query, max_results=each)
                if images
                else client.text(query, max_results=each)
            )
            for row in rows:
                url = row.get("url") if images else row.get("href", row.get("url"))
                if not url or url in seen:
                    continue
                public_url(url)
                seen.add(url)
                output.append(
                    {
                        "url": url,
                        "title": clean(row.get("title")),
                        "image_url": row.get("image", ""),
                    }
                )
        except Exception as error:
            print(
                f"Bir arama kaynağı yanıt vermedi: {type(error).__name__}", flush=True
            )
    return output


class Fetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; ComicFactorySourceReader/2.0)"
        )

    def get(self, url, maximum=16_000_000):
        for _ in range(5):
            public_url(url)
            with self.session.get(
                url, stream=True, allow_redirects=False, timeout=(8, 18)
            ) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                if int(response.headers.get("Content-Length", "0") or 0) > maximum:
                    raise ValueError("Kaynak çok büyük.")
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > maximum:
                        raise ValueError("Kaynak çok büyük.")
                return bytes(body), url
        raise ValueError("Çok fazla yönlendirme.")

    def article(self, url):
        body, final = self.get(url, 3_000_000)
        soup = BeautifulSoup(body, "html.parser")
        title = clean(soup.title.get_text()) if soup.title else ""
        article = soup.find("article") or soup.find("main") or soup
        images = []
        for element in article.find_all("img"):
            candidates = [
                element.get("data-src"),
                element.get("data-lazy-src"),
                element.get("src"),
            ]
            srcset = element.get("srcset") or element.get("data-srcset")
            if srcset and srcset.split(",")[-1].strip():
                candidates.insert(0, srcset.split(",")[-1].strip().split()[0])
            for source in candidates:
                if source and not source.startswith("data:"):
                    address = urljoin(final, source)
                    if address not in images:
                        images.append(address)
                    break
        for item in article.select("script,style,nav,footer,header,form,aside"):
            item.decompose()
        return {
            "url": final,
            "title": title,
            "text": clean(article.get_text(" "))[:22000],
            "images": images[:40],
        }


def discover_evidence(api, topic):
    """Read independent search results before asking the model for candidates."""
    import random
    heroes = [
        "Spider-Man", "Batman", "Superman", "Wolverine", "Hulk", "Thanos", "Joker", "Deadpool",
        "Venom", "Thor", "Iron Man", "Captain America", "Doctor Doom", "Flash", "Wonder Woman",
        "Green Lantern", "Daredevil", "Punisher", "Doctor Strange", "Magneto", "Darkseid",
        "Ghost Rider", "Black Panther", "Silver Surfer", "Galactus", "Harley Quinn", "Moon Knight",
    ]
    if topic:
        queries = [f'{topic} comic issue review panels', f'{topic} comic publisher preview']
    elif getattr(api.settings, "channel_theme", "") == "superheroes":
        picks = random.sample(heroes, 3)
        queries = [f'{picks[0]} comic shocking moment issue preview pages',
                   f'{picks[1]} comic bizarre moment issue review panels',
                   f'{picks[2]} comic preview interior pages site:marvel.com OR site:dc.com OR site:cbr.com OR site:bleedingcool.com OR site:aiptcomics.com']
    else:
        queries = random.sample([
                   'Marvel comics historic turning point issue review panels',
                   'DC comics shocking transformation issue review panels',
                   'independent comics landmark issue illustrated review',
                   'comic book first appearance origin issue retrospective panels',
                   'manga historic story arc chapter illustrated review',
                   'European comics classic album illustrated review',
               ], 3)
    rows = search(queries, each=6)
    fetcher, articles, failures = Fetcher(), [], []
    try:
        for row in rows[:18]:
            api.check()
            try:
                article = fetcher.article(row['url'])
                if len(article['text']) < 180 or not article['images']:
                    continue
                articles.append({**article, 'text': article['text'][:9000]})
                if len(articles) == 8:
                    break
            except (requests.RequestException, ValueError, OSError, KeyError) as error:
                failures.append({'url': row['url'], 'error': type(error).__name__})
    finally:
        fetcher.session.close()
    evidence = {'provider': 'independent_web_search', 'queries': queries,
                'articles': articles, 'failures': failures}
    save_json(api.directory / 'research' / 'discovery.json', evidence)
    if not articles:
        raise SourceUnavailable('Web aramasında okunabilir resimli kaynak bulunamadı; discovery.json kaydedildi. Modelden kaynak uydurması istenmedi.')
    return evidence


def shortlist(api, topic, used):
    evidence = discover_evidence(api, topic)
    allowed_urls = {article['url'] for article in evidence['articles']}
    payload = api.json(
        "Konu adayları",
        f"""Choose up to {api.settings.max_events} specific SHOCKING superhero moments (a death, a betrayal, a bizarre transformation, a villain's cruelest act, an impossible feat) from ONLY the fetched articles below — the kind of moment a global YouTube Shorts audience stops scrolling for. No new facts or links. Exclude used IDs/titles {json.dumps(used, ensure_ascii=False)}. Require exact series, issue/chapter, year, publisher and universe supported by articles. Prefer public publisher previews and illustrated reviews with actual interior panels. Requested topic {topic or "automatic"}; if explicit, ALL candidates must be that exact event.
{json.dumps(evidence, ensure_ascii=False)}
Return {{"events":[{{"title":"Turkish","publisher":"","series":"exact original title","issue":"","year":2000,"universe":"","characters":[],"summary":"Turkish","importance":0,"popularity":0,"niche":0,"source_urls":[]}}]}}. Scores 0..100 are editorial judgments.""",
        list_key="events",
    )
    candidates = []
    for row in payload.get("events", []):
        if not isinstance(row, dict) or not all(
            clean(row.get(k))
            for k in ("title", "publisher", "series", "issue", "universe", "summary")
        ):
            continue
        urls = row.get('source_urls')
        if not isinstance(urls, list):
            continue
        row['source_urls'] = list(dict.fromkeys(u for u in urls if isinstance(u, str) and u in allowed_urls))
        if not row['source_urls']:
            continue
        if type(row.get("year")) is not int or not 1930 <= row["year"] <= 2100:
            continue
        row["id"] = event_key(row)
        if row["id"] in used or clean(row["title"]).casefold() in used:
            continue
        try:
            row["selection_score"] = sum(
                max(0, min(100, float(row.get(k, 0)))) * weight
                for k, weight in (
                    ("importance", 0.45),
                    ("popularity", 0.25),
                    ("niche", 0.30),
                )
            )
        except (ValueError, TypeError):
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    save_json(
        api.directory / "research" / "candidates.json",
        {"requested_topic": topic, "events": candidates},
    )
    if not candidates:
        raise SourceUnavailable(
            "Kaynaklı, kullanılmamış bir olay bulunamadı; araştırma kaydedildi."
        )
    return candidates[: 1 if topic else api.settings.max_events]


def collect_pages(api, event, fetcher):
    query = f'"{event["series"]}" "{event["issue"]}"'
    results = search(
        [f"{query} {event['year']} preview", f"{query} review comic panels"]
    )
    image_results = search([f"{query} comic interior pages"], True, 14)
    sources = {url: [] for url in event.get("source_urls", []) if isinstance(url, str)}
    for row in results + image_results:
        sources.setdefault(row["url"], []).append(row)
    root = api.directory / "events" / event["id"]
    queues, articles, pages, seen = [], {}, [], set()
    for url, rows in list(sources.items())[:14]:
        api.check()
        try:
            article = fetcher.article(url)
        except Exception:
            continue
        if len(article["text"]) < 180:
            continue
        identifier = f"source_{len(articles):03}"
        article["id"] = identifier
        articles[identifier] = article
        save_json(root / "sources" / (identifier + ".json"), article)
        images = list(
            dict.fromkeys(
                article["images"] + [r["image_url"] for r in rows if r.get("image_url")]
            )
        )
        queues.append((article, images[:18]))
    # Round-robin prevents one site's covers from filling the image budget.
    for position in range(18):
        for article, images in queues:
            if position >= len(images):
                continue
            api.check()
            if len(pages) >= api.settings.max_pages:
                break
            try:
                body, url = fetcher.get(images[position])
                digest = hashlib.sha256(body).hexdigest()
                if digest in seen:
                    continue
                with Image.open(io.BytesIO(body)) as opened:
                    if (
                        min(opened.size) < 420
                        or opened.width * opened.height > 35_000_000
                    ):
                        continue
                    picture = ImageOps.exif_transpose(opened).convert("RGB")
                if not 0.2 <= picture.width / picture.height <= 3:
                    continue
                identifier = f"page_{len(pages):03}"
                path = root / "pages" / (identifier + ".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                picture.save(path, "JPEG", quality=96)
                pages.append(
                    {
                        "id": identifier,
                        "file": str(path.relative_to(api.directory)),
                        "source_id": article["id"],
                        "source_url": article["url"],
                        "source_title": article["title"],
                        "image_url": url,
                        "width": picture.width,
                        "height": picture.height,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                seen.add(digest)
            except Exception:
                continue
        if len(pages) >= api.settings.max_pages:
            break
    save_json(root / "pages.json", pages)
    if not pages:
        raise SourceUnavailable(
            "Bu olay için okunabilir gerçek çizgi roman sayfası bulunamadı."
        )
    return pages, articles
