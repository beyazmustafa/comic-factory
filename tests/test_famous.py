import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile

from factory import famous, render
from factory.story import validate_story


class FakeApi:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.settings = SimpleNamespace(max_pages=16, language="en")


class FamousTests(unittest.TestCase):
    def test_shortlist_skips_used_and_keeps_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeApi(tmp)
            used = {famous.moment_id(famous.MOMENTS[0][0]), famous.MOMENTS[1][1].casefold()}
            events = famous.shortlist(api, "", used)
            self.assertTrue(1 <= len(events) <= 3)
            for event in events:
                self.assertEqual(event["_source"], "famous")
                self.assertNotIn(event["id"], used)
                self.assertNotIn(event["title"].casefold(), used)
                for key in ("publisher", "series", "issue", "year", "characters", "famous_line"):
                    self.assertTrue(event[key])

    def test_shortlist_topic_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = famous.shortlist(FakeApi(tmp), "Gwen Stacy", set())
            self.assertTrue(all("Gwen" in e["title"] or "Gwen Stacy" in e["characters"] for e in events))

    def test_all_moments_unique_and_wellformed(self):
        keys = [m[0] for m in famous.MOMENTS]
        self.assertEqual(len(keys), len(set(keys)))
        for key, event, series, issue, year, publisher, characters, line in famous.MOMENTS:
            self.assertIn(publisher, {"Marvel", "DC"})
            self.assertTrue(1930 <= year <= 2100)
            self.assertTrue(characters and line)

    def test_hook_card_text_fallback(self):
        event = {"famous_line": "Bane breaks Batman's back over his knee"}
        self.assertEqual(famous.hook_card_text({"hook_card": "BANE BROKE THE BAT"}, event), "BANE BROKE THE BAT")
        self.assertEqual(famous.hook_card_text({"hook_card": "x"}, event), "Bane breaks Batman's back over his")
        self.assertEqual(famous.hook_card_text({"hook_card": "one two three four five six seven eight"}, event),
                         "Bane breaks Batman's back over his")

    def test_validate_story_keeps_hook_card(self):
        panels = [{"id": f"p{i}", "page_id": f"g{i}"} for i in range(6)]
        facts = [{"id": "f0", "page_id": "g0"}]
        shots = [{"panel_id": f"p{i}", "narration": "Spider-Man watches the bridge fall apart", "fact_ids": ["f0"],
                  "motion": "push", "emphasis": "normal"} for i in range(6)]
        value = validate_story({"title": "A title", "hook_card": "SPIDER-MAN KILLED HER?", "shots": shots}, panels, facts)
        self.assertEqual(value["hook_card"], "SPIDER-MAN KILLED HER?")

    def test_hook_lines_fit_and_balance(self):
        style = {"font_style": "condensed_heavy"}
        lines, size = render.hook_lines("BANE BROKE THE BAT", style)
        self.assertLessEqual(len(lines), 2)
        self.assertEqual(" ".join(lines), "BANE BROKE THE BAT")
        self.assertGreaterEqual(size, 84)
        lines, size = render.hook_lines("THE JOKER SHOT BARBARA GORDON TONIGHT", style)
        self.assertEqual(len(lines), 2)
        self.assertGreaterEqual(size, 56)
        events = render.hook_events("Bane broke the Bat", style, "en")
        self.assertEqual(len(events), 2)
        self.assertIn("BANE BROKE", events[1])
        self.assertIn("THE BAT", events[1])
        self.assertEqual(render.hook_events("", style, "en"), [])


if __name__ == "__main__":
    unittest.main()
