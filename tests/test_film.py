import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory import cut, film, footage, topics_film
from factory.api import FactoryError


def _clip(identifier, windows, kind="footage", asset="nasa:a"):
    return {"id": identifier, "asset": asset, "title": "t", "date": "2021-02-22", "credit": "NASA", "page_url": "", "file": "x.mp4",
            "source_start": 0, "duration": 12.0, "width": 1920, "height": 1080, "fps": 30, "windows": windows, "focus": [0.5, 0.5],
            "shows": "stuff", "quality": 8, "kind": kind, "what": "w"}


class TopicTests(unittest.TestCase):
    def test_topics_are_well_formed(self):
        keys = set()
        for topic in topics_film.TOPICS:
            self.assertNotIn(topic["key"], keys)
            keys.add(topic["key"])
            self.assertGreaterEqual(len(topic["facts"]), 5)
            self.assertTrue(topic["nasa"])
            self.assertLessEqual(len(topic["title"].split()), 8)
            for callout in topic.get("callouts", []):
                self.assertTrue(callout["value"] and callout["label"])

    def test_requested_topic_comes_first_then_least_recently_used(self):
        with patch.object(film, "used_topics", return_value=["juno-jupiter", "pluto"]):
            order = film.pick_topics("pluto")
            self.assertEqual(order[0], "pluto")
            self.assertEqual(order[-1], "juno-jupiter")  # pluto requested, juno is the only other used one
            order = film.pick_topics("")
            self.assertNotIn(order[0], ("juno-jupiter", "pluto"))


class ScriptTests(unittest.TestCase):
    def test_validation_keeps_shape_and_limits_callouts(self):
        topic = topics_film.by_key("perseverance-landing")
        clips = [_clip("c00", [(0, 8)]), _clip("c01", [(0, 8)], asset="nasa:b")]
        beats = [{"clip": "c00" if i % 2 == 0 else "c01", "narration": "This is real footage of the rover arriving at Mars after a journey of many months.",
                  "label": "NASA · Perseverance · 2021", "callout": 0 if i < 4 else None} for i in range(9)]
        script = film.validate_script({"hook": "Seven minutes of terror.", "youtube_title": "x" * 90, "description": "d", "beats": beats, "closing": "Every frame crossed two hundred million kilometres."}, topic, clips)
        self.assertEqual(len(script["beats"]), 9)
        self.assertEqual(sum(1 for b in script["beats"] if b["callout"]), 2)
        self.assertFalse(script["beats"][0]["callout"] and script["beats"][1]["callout"])
        self.assertEqual(len(script["youtube_title"]), 70)

    def test_too_short_script_is_rejected(self):
        topic = topics_film.by_key("perseverance-landing")
        with self.assertRaises(FactoryError):
            film.validate_script({"beats": [{"clip": "c00", "narration": "Too short."}]}, topic, [_clip("c00", [(0, 8)])])


class TimelineTests(unittest.TestCase):
    def test_every_beat_gets_footage_and_shots_tile_the_duration(self):
        clips = [_clip("c00", [(0.3, 5.8), (6.0, 11.7)]), _clip("c01", [(0.0, 20.0)], kind="visualization", asset="nasa:b"),
                 _clip("c02", [(0.5, 3.5)], asset="svs:c")]
        beats = [{"clip": c, "narration": "x", "label": "L", "callout": None} for c in ("c00", "c01", "c02", "c00", "c01", "c02", "c00")]
        starts = [0, 8, 17, 25, 33, 42, 50]
        shots = film.build_timeline({"beats": beats}, clips, starts, 60.0)
        self.assertAlmostEqual(shots[0]["t0"], 0.0)
        self.assertAlmostEqual(shots[-1]["t1"], 60.0)
        for a, b in zip(shots, shots[1:]):
            self.assertAlmostEqual(a["t1"], b["t0"])
            self.assertGreater(a["t1"], a["t0"])
        for shot in shots:
            self.assertGreaterEqual(shot["speed"], 0.72)
            self.assertLessEqual(shot["src_out"] - shot["src_in"], 20.0 + 0.001)
        self.assertEqual(sorted({s["beat"] for s in shots}), list(range(7)))


class FootageTests(unittest.TestCase):
    def test_duration_parsing_and_rendition_tags(self):
        self.assertEqual(footage._parse_duration("0:02:34"), 154)
        self.assertEqual(footage._parse_duration("2:34.5"), 154.5)
        self.assertEqual(footage._parse_duration("93 s"), 93)
        self.assertIsNone(footage._parse_duration("n/a"))
        self.assertEqual(footage._rendition_tag("https://x/y~orig.mp4"), "orig")
        self.assertEqual(footage._rendition_tag("https://x/y~small.mp4"), "small")

    def test_sample_times_stay_inside_the_video(self):
        times = footage.sample_times(120.0, 12)
        self.assertEqual(len(times), 12)
        self.assertGreater(times[0], 1.0)
        self.assertLess(times[-1], 119.0)

    def test_net_options_only_for_urls(self):
        self.assertTrue(footage.net_options("https://a/b.mp4"))
        self.assertEqual(footage.net_options("/tmp/x.mp4"), [])


class CutTests(unittest.TestCase):
    def test_crop_window_is_9_16_and_follows_focus(self):
        cw, ch, cx, cy = cut.crop_window(1920, 1080, (0.9, 0.5))
        self.assertEqual((cw, ch), (606, 1080))
        self.assertEqual(cx, 1920 - 606)
        cw, ch, cx, cy = cut.crop_window(3840, 2160, (0.5, 0.5))
        self.assertAlmostEqual(cw / ch, 9 / 16, places=2)

    def test_callout_counts_up_to_the_exact_value(self):
        values, sequence, label = cut.callout_layers({"value": "20,000 km/h", "label": "speed"})
        self.assertGreater(len(sequence), 10)
        self.assertEqual(sequence[-1], len(values) - 1)
        values, sequence, _ = cut.callout_layers({"value": "+120 °C / -150 °C", "label": "sun and shade"})
        self.assertEqual(len(values), 1)

    def test_captions_file_has_one_event_per_word(self):
        words = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate("This is a six word line".split())]
        with tempfile.TemporaryDirectory() as temporary:
            path = film.captions(words, Path(temporary) / "c.ass", 0.9, "DejaVu Sans")
            text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("Dialogue:"), 6)
        self.assertIn("\\pos(540,1344)", text)


if __name__ == "__main__":
    unittest.main()
