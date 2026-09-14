import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from factory import (
    core,
    panels,
    quality,
    style as editing,
    render,
    research,
    state,
    story,
    studio,
    voice,
)
from factory.api import Api, FactoryError, SpeechFailure
from factory.config import VERSION, Settings


def timestamps(text, step=0.4):
    return [
        {"word": word, "start": i * step, "end": i * step + step * 0.875}
        for i, word in enumerate(text.split())
    ]


def style():
    return {
        "video_observed": True,
        "duration_seconds": 164,
        "observations": [
            {"second": i, "visual_detail": "Test observation"} for i in (0, 80, 160)
        ],
        "font_style": "condensed_heavy",
        "uppercase": True,
        "font_size": 72,
        "stroke_width": 4,
        "caption_x": 0.48,
        "caption_y": 0.69,
        "caption_mode": "current_next",
        "caption_words": 2,
        "text_color": "#FFFFFF",
        "active_color": "#FFD13D",
        "stroke_color": "#000000",
        "background_color": "#10151D",
        "background": "blurred_page",
        "transition": "cut",
        "transition_frames": 4,
        "mean_shot_seconds": 4,
        "zoom_amount": 0.065,
        "panel_framing": "contain",
        "panel_max_width": 0.91,
        "panel_max_height": 0.86,
        "panel_center_y": 0.48,
        "music_present": False,
        "narrator_delivery": "Clear energetic Turkish",
        "story_structure": "Event, cause, consequence",
        "reference_url": "https://www.youtube.com/watch?v=yIZLrxqUUbg",
    }


def script_fixture():
    inventory = [
        {
            "id": f"panel_{i:03}",
            "file": f"events/test/panels/panel_{i:03}.jpg",
            "page_file": "events/test/pages/page_000.jpg",
            "source_url": "https://example.com/issue",
            "action": "A visible figure stands beside a door.",
        }
        for i in range(6)
    ]
    facts = [
        {
            "id": "fact_000",
            "text": "Karakter kapıya gelir",
            "quote": "The character reaches the door.",
            "source_url": "https://example.com/issue",
        }
    ]
    script = story.validate_story(
        {
            "title": "Türkçe test",
            "description": "Deneme",
            "shots": [
                {
                    "panel_id": p["id"],
                    "narration": "Thor kapıya geldi.",
                    "fact_ids": ["fact_000"],
                    "motion": ["push", "pull", "left", "right", "hold", "push"][i],
                }
                for i, p in enumerate(inventory)
            ],
        },
        inventory,
        facts,
    )
    return script, inventory, facts


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))


class SourceTests(TemporaryTest):
    def test_literal_source_quote(self):
        self.assertTrue(
            panels.quote_present(
                "The character reaches the door.",
                "Intro. The character  reaches the door. End.",
            )
        )
        self.assertFalse(
            panels.quote_present(
                "The character kills the villain.", "The character reaches the door."
            )
        )

    def test_bad_boxes_and_confidence(self):
        for value in (
            [0, 0, 2, 1],
            [0.7, 0, 0.1, 1],
            [0, float("nan"), 1, 1],
            [0, 1, 2],
        ):
            with self.assertRaises(ValueError):
                panels.box(value)
        self.assertFalse(panels.confident(float("nan"), 90))
        self.assertFalse(panels.confident(101, 90))

    def test_unknown_panel_or_fact(self):
        for field, value in (("panel_id", "invented"), ("fact_ids", ["invented"])):
            script, inventory, facts = script_fixture()
            script["shots"][0][field] = value
            with self.assertRaises(ValueError):
                story.validate_story(script, inventory, facts)

    def test_one_reused_picture_cannot_pass(self):
        script, inventory, facts = script_fixture()
        for shot in script["shots"]:
            shot["panel_id"] = inventory[0]["id"]
        with self.assertRaises(ValueError):
            story.validate_story(script, inventory, facts)

    def test_private_url_rejected(self):
        for url in (
            "file:///tmp/file",
            "http://127.0.0.1/",
            "http://169.254.169.254/",
            "https://x.local/",
            "https://user:pass@example.com",
        ):
            with self.assertRaises(ValueError):
                research.public_url(url)

    def test_fixed_style_validated_offline(self):
        editing.validate_style(editing.load_style())
        with self.assertRaises(FactoryError):
            editing.validate_style({**editing.load_style(), "caption_y": float("nan")})

    def test_quality_rejects_unobserved_video_and_nan(self):
        good = {
            "candidate_observed": True,

            "turkish_narration": True,
            "no_critical_errors": True,
            "subtitle_sync": 95,
            "scene_match": 95,
            "delivery": 90,
            "visual_readability": 90,
            "observations": [{"second": i, "detail": "visible"} for i in (1, 40, 80)],
        }
        self.assertTrue(quality.validate_review(dict(good))["passed"])
        for values in (
            {"candidate_observed": False},
            {"subtitle_sync": float("nan")},
            {"observations": []},
        ):
            self.assertFalse(quality.validate_review({**good, **values})["passed"])

    def test_layout_repair_does_not_hide_speech_problem(self):
        self.assertIsNone(
            quality.adjusted_style(
                style(),
                {
                    "failed_checks": ["subtitle_sync"],
                    "style_adjustments": {"font_size": 80},
                },
            )
        )


class AudioTests(TemporaryTest):
    def api(self, **settings):
        return SimpleNamespace(
            directory=self.root, settings=replace(Settings(), **settings), check=Mock()
        )

    def make_audio(self, api, text, selected, profile, path):
        voice.write_wave(
            path,
            (2500 * np.sin(np.arange(len(text.split()) * 9600) * 0.08)).astype("<i2"),
        )
        return path

    def align(self, api, path, text, directory):
        duration = len(voice.read_wave(path)) / voice.RATE
        words, metrics = core.align_words(text, timestamps(text), duration)
        return {
            "words": words,
            "metrics": metrics,
            "duration": duration,
            "model": "fixture",
        }

    def test_silent_and_empty_audio(self):
        for samples in (np.array([], dtype="<i2"), np.zeros(24000, dtype="<i2")):
            with self.assertRaises(SpeechFailure):
                voice.trim_edges(samples)

    def test_internal_pause_is_preserved(self):
        data = np.concatenate(
            [
                np.zeros(12000),
                np.ones(12000) * 3000,
                np.zeros(24000),
                np.ones(12000) * 3000,
                np.zeros(12000),
            ]
        ).astype("<i2")
        trimmed = voice.trim_edges(data)
        self.assertGreater(len(trimmed), 48000)
        self.assertLess(len(trimmed), len(data))
        self.assertTrue(np.all(trimmed[18000:30000] == 0))

    def test_extra_speech_lowers_score_without_moving_later_word(self):
        words, metrics = core.align_words(
            "Thor sonunda kazandı", timestamps("Thor bir sonunda kazandı")
        )
        self.assertAlmostEqual(words[-1]["start"], 1.2)
        self.assertLess(metrics["score"], 95)
        self.assertIn("bir", metrics["extra_asr_words"])

    def test_bad_asr_retries_same_recording(self):
        api = self.api()
        text = "Thor geri döndü"
        path = self.make_audio(api, text, "Orus", style(), self.root / "a.wav")
        with patch.object(
            voice,
            "transcribe",
            side_effect=[[{"word": "Thor", "start": 1, "end": 0}], timestamps(text)],
        ) as asr:
            result = voice.align_clip(api, path, text, self.root / "attempt")
        self.assertEqual(result["metrics"]["score"], 100)
        self.assertEqual(asr.call_args_list[0].args[0], asr.call_args_list[1].args[0])

    def test_failed_chunk_resumes_from_artifact_without_shared_cache(self):
        api = self.api(chunk_words=3, repair_attempts=1)
        script, _, _ = script_fixture()
        script["shots"] = script["shots"][:2]
        script["shots"][1]["narration"] = "Kapı tekrar açıldı."
        script["narration"] = " ".join(s["narration"] for s in script["shots"])

        def flaky(api, path, text, directory):
            if text.startswith("Kapı"):
                raise SpeechFailure("fixture outage")
            return self.align(api, path, text, directory)

        with (
            patch.object(voice, "synthesize", side_effect=self.make_audio),
            patch.object(voice, "align_clip", side_effect=flaky),
        ):
            with self.assertRaises(SpeechFailure):
                voice.build_audio(api, script, "Orus", style(), self.root / "cache")
        import shutil

        shutil.rmtree(self.root / "cache")
        with (
            patch.object(voice, "synthesize", side_effect=self.make_audio) as synth,
            patch.object(voice, "align_clip", side_effect=self.align),
        ):
            output, words, duration = voice.build_audio(
                api, script, "Orus", style(), self.root / "cache"
            )
        self.assertEqual(
            [c.args[1] for c in synth.call_args_list], ["Kapı tekrar açıldı."]
        )
        self.assertEqual(len(voice.read_wave(output)), 57600)
        self.assertEqual(duration, 2.4)
        self.assertAlmostEqual(words[3]["start"], 1.2)

    def test_voice_change_invalidates_cache(self):
        api = self.api()
        script, _, _ = script_fixture()
        with (
            patch.object(voice, "synthesize", side_effect=self.make_audio) as synth,
            patch.object(voice, "align_clip", side_effect=self.align),
        ):
            voice.build_audio(api, script, "Orus", style(), self.root / "cache")
            before = synth.call_count
            voice.build_audio(api, script, "Gacrux", style(), self.root / "cache")
        self.assertGreater(synth.call_count, before)

    def test_current_next_and_turkish_uppercase(self):
        path = render.captions(
            timestamps("Thor şimdi geri döndü"), style(), self.root / "captions.ass"
        )
        rows = [r for r in path.read_text().splitlines() if r.startswith("Dialogue")]
        self.assertEqual(len(rows), 4)
        self.assertIn("ŞİMDİ", rows[0])
        self.assertNotIn("THOR", rows[1])
        self.assertIn("GERİ", rows[1])

    def test_total_frame_count_has_no_accumulating_error(self):
        shots = [{"narration": "bir"} for _ in range(14)]
        words = [
            {"word": "bir", "start": i * 0.343, "end": (i + 1) * 0.343 - 0.01}
            for i in range(14)
        ]
        timeline = core.scene_timeline(shots, words, 14 * 0.343)
        self.assertEqual(
            sum(round((b - a) * 30) for a, b in timeline), round(14 * 0.343 * 30)
        )
        self.assertTrue(all(a[1] == b[0] for a, b in zip(timeline, timeline[1:])))


class StateTests(TemporaryTest):
    def test_changed_checkpoint_is_rejected(self):
        core.save_json(self.root / "run.json", {"schema": 2, "version": VERSION})
        file = self.root / "story.json"
        file.write_text("original")
        checkpoint = state.Checkpoints(self.root)
        checkpoint.save("story", "same", {"title": "test"}, [file])
        file.write_text("changed")
        with self.assertRaises(FactoryError):
            state.resume(self.root, self.root / "after")

    def test_changed_settings_invalidate_checkpoint(self):
        file = self.root / "data.json"
        file.write_text("{}")
        checkpoint = state.Checkpoints(self.root)
        checkpoint.save("voice", "Orus", {"result": 1}, [file])
        self.assertIsNone(checkpoint.read("voice", "Gacrux"))

    def test_artifact_paths_cannot_escape(self):
        for path in ("../outside.json", "/etc/passwd"):
            with self.assertRaises(FactoryError):
                state.inside(self.root, path)

    def test_failure_before_api_still_has_review_and_output(self):
        output = self.root / "outputs.txt"
        with (
            patch.object(studio, "ROOT", self.root),
            patch.dict(
                os.environ,
                {"GITHUB_OUTPUT": str(output), "GROQ_API_KEY": ""},
                clear=True,
            ),
        ):
            with self.assertRaises(FactoryError):
                studio.generate(Settings())
        directory = next((self.root / "data/runs").iterdir())
        self.assertEqual(
            json.loads((directory / "run.json").read_text())["status"], "failed"
        )
        self.assertTrue((directory / "review.html").exists())
        self.assertIn(str(directory), output.read_text())

    def test_api_budget_does_not_block_local_render(self):
        api = Api(Settings(max_api_calls=1), self.root, client=Mock())
        api.request("first", lambda: 123)
        api.check()
        with self.assertRaises(FactoryError):
            api.request("second", lambda: 456)

    def test_cli_voice_overrides_repository_variable(self):
        with patch.dict(os.environ, {"GEMINI_TTS_VOICE": "Gacrux"}, clear=True):
            value = Settings.load(self.root / "none.json", voice="auto")
        self.assertEqual(value.voice, "auto")

    def test_invalid_settings_fail_early(self):
        for value in (
            {"target_seconds": 400},
            {"caption_offset": float("nan")},
            {"surprise": 1},
        ):
            core.save_json(self.root / "config.json", value)
            with self.assertRaises(ValueError):
                Settings.load(self.root / "config.json")
