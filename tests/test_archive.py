import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image, ImageDraw
from factory import archive, panels
from factory.api import SourceUnavailable
from factory.config import Settings
from tests.test_studio import TemporaryTest


def synthetic_page(seed=1, width=1200, height=1800, layout=((2,), (3,), (1,), (2,))):
    import random
    random.seed(seed)
    picture = Image.new("RGB", (width, height), (228, 214, 170))
    draw = ImageDraw.Draw(picture)
    margin, gap = 60, 24
    rows = len(layout)
    row_height = (height - 2 * margin) / rows
    for row, (columns,) in enumerate(layout):
        top = round(margin + row * row_height + (gap if row else 0))
        bottom = round(margin + (row + 1) * row_height - (gap if row < rows - 1 else 0))
        col_width = (width - 2 * margin) / columns
        for column in range(columns):
            left = round(margin + column * col_width + (gap if column else 0))
            right = round(margin + (column + 1) * col_width - (gap if column < columns - 1 else 0))
            draw.rectangle((left, top, right, bottom), outline=(20, 20, 20), width=6)
            for _ in range(60):
                x, y = random.randint(left + 8, right - 30), random.randint(top + 8, bottom - 30)
                draw.rectangle(
                    (x, y, min(right - 8, x + random.randint(10, 120)), min(bottom - 8, y + random.randint(10, 120))),
                    fill=(random.randint(0, 255), random.randint(0, 200), random.randint(0, 200)),
                )
    return picture


def ad_page(width=1200, height=1800):
    picture = Image.new("RGB", (width, height), (228, 214, 170))
    draw = ImageDraw.Draw(picture)
    for line in range(40):
        draw.rectangle((100, 120 + line * 40, 1100, 130 + line * 40), fill=(40, 40, 40))
    return picture


class SegmentationTests(TemporaryTest):
    def test_grid_page_yields_panels_in_reading_order(self):
        boxes = panels.segment_page(synthetic_page())
        self.assertEqual(len(boxes), 8)
        tops = [round(b[1], 2) for b in boxes]
        self.assertEqual(tops, sorted(tops))
        self.assertLess(boxes[0][0], boxes[1][0])
        self.assertTrue(all(0 <= l < r <= 1 and 0 <= t < b <= 1 for l, t, r, b in boxes))

    def test_text_page_is_not_story_like(self):
        boxes = panels.segment_page(ad_page())
        self.assertLessEqual(len(boxes), 1)

    def test_story_pages_skip_cover_and_ads(self):
        pages = []
        for index, kind in enumerate(["cover", "ad", "story", "story", "story", "story", "story", "ad"]):
            picture = ad_page() if kind in {"cover", "ad"} else synthetic_page(seed=index)
            path = self.root / f"page_{index:03}.jpg"
            picture.save(path, "JPEG")
            pages.append({"id": f"page_{index:03}", "file": path.name, "width": 1200, "height": 1800,
                          "source_url": "https://archive.org/details/x", "source_title": "X"})
        chosen = panels.select_story_pages(pages, self.root, maximum=3)
        self.assertEqual([p["id"] for p in chosen], ["page_002", "page_003", "page_004"])
        with self.assertRaises(SourceUnavailable):
            panels.select_story_pages(pages[:4], self.root, maximum=3)


class ArchiveFilterTests(TemporaryTest):
    def settings(self):
        return Settings()

    def test_copyrighted_or_undeclared_items_are_rejected(self):
        settings = self.settings()
        disney = {"identifier": "WaltDisneysComics-1959", "title": "Walt Disney's Comics", "date": "1959-07-01"}
        self.assertIsNone(archive.eligible(disney, settings))
        late = {"identifier": "x-1970", "title": "X", "date": "1970", "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"}
        self.assertIsNone(archive.eligible(late, settings))
        ok = {"identifier": "hand-of-fate-23", "title": "Hand of Fate 23", "date": "1954-08-01",
              "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/", "publisher": "Ace"}
        item = archive.eligible(ok, settings)
        self.assertEqual(item["year"], 1954)
        self.assertIn("declared licenseurl", item["public_domain_evidence"])
        # Real archive records often carry no licence field: a lapsed publisher
        # named in the collection slug is accepted as inferred evidence.
        inferred = {"identifier": "hand-of-fate-23", "title": "Hand of Fate 23", "date": "1954",
                    "collection": ["ace-comics", "comics", "folkscanomy"]}
        self.assertIn("inferred", archive.eligible(inferred, settings)["public_domain_evidence"])
        # A protected brand is rejected even with a licence claim or lapsed-looking words.
        claimed = {"identifier": "donald-1950", "title": "Donald Duck Four Color", "date": "1950",
                   "publisher": "Dell", "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"}
        self.assertIsNone(archive.eligible(claimed, settings))
        nomad = {"identifier": "nomad-1950", "title": "Nomad Adventures", "date": "1950", "publisher": "Fox Feature Syndicate"}
        self.assertIsNotNone(archive.eligible(nomad, settings))

    def test_model_cannot_invent_identifier(self):
        api = SimpleNamespace(directory=self.root, check=Mock(), settings=self.settings(), json=Mock())
        client = Mock()
        client.search.return_value = ([
            {"identifier": "hand-of-fate-23", "title": "Hand of Fate 23", "date": "1954", "imagecount": 36,
             "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"},
            {"identifier": "mad_magazine_029", "title": "MAD 29", "date": "1956", "imagecount": 36,
             "publisher": "Ace"},
        ], 2)
        api.json.return_value = {"events": [
            {"identifier": "mad_magazine_029", "title": "MAD", "importance": 99},
            {"identifier": "invented-item", "title": "Hayali", "importance": 99},
            {"identifier": "hand-of-fate-23", "title": "Kaderin Eli", "series": "Hand of Fate", "issue": "23", "importance": 50},
        ]}
        events = archive.shortlist(api, "", [], archive=client)
        self.assertEqual([e["identifier"] for e in events], ["hand-of-fate-23"])
        self.assertEqual(events[0]["source_urls"], ["https://archive.org/details/hand-of-fate-23"])
        sent = api.json.call_args.args[1]
        self.assertNotIn("mad_magazine_029", sent)

    def test_used_issue_is_skipped(self):
        api = SimpleNamespace(directory=self.root, check=Mock(), settings=self.settings(), json=Mock())
        client = Mock()
        client.search.return_value = ([
            {"identifier": "hand-of-fate-23", "title": "Hand of Fate 23", "date": "1954", "imagecount": 36,
             "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"}], 1)
        with self.assertRaises(SourceUnavailable):
            archive.shortlist(api, "", [archive.issue_key("hand-of-fate-23")], archive=client)
        api.json.assert_not_called()

    def test_topic_narrows_archive_query(self):
        query = archive.build_query("jungle horror", self.settings())
        self.assertIn("jungle horror", query)
        self.assertIn("1963-12-31", query)


class ExtractionTests(TemporaryTest):
    def test_cbz_pages_are_extracted_in_natural_order(self):
        package = self.root / "issue.cbz"
        with zipfile.ZipFile(package, "w") as bundle:
            for number in (10, 2, 1, 3, 4, 5, 6, 7, 8, 9):
                buffer = io.BytesIO()
                synthetic_page(seed=number, width=700, height=1000).save(buffer, "JPEG")
                bundle.writestr(f"page{number}.jpg", buffer.getvalue())
            bundle.writestr("thumbs/.hidden.jpg", b"not an image")
        pages = archive.extract_pages(package, self.root / "scans")
        self.assertEqual(len(pages), 10)
        self.assertEqual(pages[0].name, "scan_000.jpg")

    def test_prefers_raw_image_zip_then_jp2_then_pdf(self):
        files = [{"name": "a.pdf", "size": 10}, {"name": "a_images.zip", "size": 20}, {"name": "a_jp2.zip", "size": 5},
                 {"name": "a_images.zip_meta.txt", "size": 1}, {"name": "a_djvu.txt", "size": 1}]
        self.assertEqual(archive.ranked_files(files), ["a_images.zip", "a_jp2.zip", "a.pdf"])
        self.assertEqual(archive.choose_file(files[:1]), "a.pdf")
        with self.assertRaises(SourceUnavailable):
            archive.choose_file([{"name": "a.txt"}])

    def test_collect_pages_rejects_missing_licence(self):
        api = SimpleNamespace(directory=self.root, check=Mock(), settings=Settings())
        client = Mock()
        client.metadata.return_value = {"metadata": {"title": "X", "date": "1950", "publisher": "Unknown Press"}, "files": [{"name": "x.cbz"}]}
        event = {"id": "ia_x", "identifier": "x", "title": "X", "year": 1950, "url": "https://archive.org/details/x"}
        with self.assertRaises(SourceUnavailable):
            archive.collect_pages(api, event, archive=client)
        client.download.assert_not_called()


class CatalogArchiveTests(TemporaryTest):
    def test_catalog_uses_numbered_panels_and_visible_quotes(self):
        pages = []
        for index in range(6):
            picture = ad_page() if index == 0 else synthetic_page(seed=index)
            path = self.root / "events" / "ia_x" / "scans" / f"scan_{index:03}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            picture.save(path, "JPEG")
            pages.append({"id": f"page_{index:03}", "file": str(path.relative_to(self.root)), "width": 1200,
                          "height": 1800, "source_url": "https://archive.org/details/x", "source_title": "X"})
        articles = {"source_000": {"text": "Internet Archive item x. Licence: publicdomain mark.",
                                   "public_domain_evidence": "licenseurl: publicdomain"}}
        quote = "YOU WILL NEVER LEAVE THIS HOUSE ALIVE"

        def answer(label, prompt, images=(), **kwargs):
            return {"pages": [
                {"page_id": page_id, "page_role": "interior",
                 "panels": [{"number": n, "keep": True, "characters": ["Dedektif"], "action": "Adam kapıyı açar",
                             "ocr": quote if n == 1 else "", "narrative_fact": "Kapı açılır", "confidence": 95}
                            for n in range(1, 9)] + [{"number": 99, "keep": True, "action": "x", "confidence": 99}],
                 "facts": [{"text": "Tehdit edilir", "quote": quote}, {"text": "Uydurma", "quote": "NOT ON THE PAGE AT ALL"}]}
                for page_id, _ in images]}

        api = SimpleNamespace(directory=self.root, check=Mock(), settings=Settings(max_pages=4), json=Mock(side_effect=answer))
        inventory, facts = panels.catalog_archive(api, {"id": "ia_x", "title": "X", "year": 1950}, pages, articles)
        self.assertEqual(len(inventory), 32)
        self.assertEqual(len(facts), 4)
        self.assertTrue(all(f["quote"] == quote for f in facts))
        self.assertTrue((self.root / inventory[0]["file"]).is_file())
        self.assertEqual(inventory[0]["reading_order"], 1)
        catalog = json.loads((self.root / "events" / "ia_x" / "catalog.json").read_text())
        self.assertEqual(catalog["story_pages"], ["page_001", "page_002", "page_003", "page_004"])


    def test_collect_pages_falls_back_to_next_package(self):
        api = SimpleNamespace(directory=self.root, check=Mock(), settings=Settings())
        client = Mock()
        client.metadata.return_value = {
            "metadata": {"title": "Hand of Fate 23", "date": "1954", "collection": ["ace-comics", "comics"]},
            "files": [{"name": "h_images.zip", "size": 10}, {"name": "h_jp2.zip", "size": 5}, {"name": "h_djvu.txt", "size": 3}],
        }

        def download(identifier, filename, destination, maximum=None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if filename == "h_images.zip":
                destination.write_bytes(b"broken zip")
            elif filename == "h_djvu.txt":
                destination.write_text("THE HAND OF FATE REACHES OUT TONIGHT", encoding="utf-8")
            else:
                with zipfile.ZipFile(destination, "w") as bundle:
                    for number in range(10):
                        buffer = io.BytesIO()
                        synthetic_page(seed=number, width=700, height=1000).save(buffer, "JPEG")
                        bundle.writestr(f"h_jp2/h_{number:04}.jpg", buffer.getvalue())
            return destination

        client.download.side_effect = download
        event = {"id": "ia_h", "identifier": "hand-of-fate-23", "title": "Kaderin Eli", "year": 1954,
                 "url": "https://archive.org/details/hand-of-fate-23"}
        pages, articles = archive.collect_pages(api, event, archive=client)
        self.assertEqual(len(pages), 10)
        self.assertIn("inferred", articles["source_000"]["public_domain_evidence"])
        self.assertIn("HAND OF FATE REACHES", articles["source_000"]["text"])
        self.assertTrue((self.root / pages[0]["file"]).is_file())
        self.assertFalse(list((self.root / "events" / "ia_h" / "download").glob("*.zip")))
