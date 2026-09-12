import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from factory import core, publishing, studio
from factory.config import Settings


def timestamps(text):
    return [
        {"word": word, "start": i * 0.4, "end": i * 0.4 + 0.35}
        for i, word in enumerate(text.split())
    ]


class TimingTests(unittest.TestCase):
    def test_repeated_words_preserve_sequence(self):
        text = "Thor geri geldi ve Thor yeniden savaştı"
        words, metrics = core.align_words(text, timestamps(text))
        self.assertEqual(
            [word["start"] for word in words],
            [word["start"] for word in timestamps(text)],
        )
        self.assertEqual(metrics["score"], 100)

    def test_inserted_asr_word_does_not_shift_later_words(self):
        words, metrics = core.align_words(
            "Thor sonunda kazandı", timestamps("Thor bir sonunda kazandı")
        )
        self.assertAlmostEqual(words[-1]["start"], 1.2)
        self.assertEqual(metrics["score"], 100)

    def test_split_turkish_suffix_is_merged(self):
        words, metrics = core.align_words("Thor'un gücü", timestamps("Thor un gücü"))
        self.assertEqual(words[0]["end"], 0.75)
        self.assertEqual(metrics["score"], 100)

    def test_missing_words_never_extend_past_audio(self):
        words, metrics = core.align_words(
            "Thor kazandı sonra yeniden", timestamps("Thor kazandı")
        )
        self.assertLessEqual(words[-1]["end"], 0.75)
        self.assertLess(metrics["score"], 95)

    def test_empty_and_invalid_alignment_are_rejected(self):
        for words in (
            [],
            [{"word": "Thor", "start": 1, "end": 0}],
            [{"word": "Thor", "start": 0, "end": float("nan")}],
        ):
            with self.assertRaises(ValueError):
                core.align_words("Thor", words)

    def test_turkish_case_comparison(self):
        _, metrics = core.align_words("İstanbul ışık", timestamps("istanbul IŞIK"))
        self.assertEqual(metrics["score"], 100)

    def test_frame_boundaries_do_not_accumulate_rounding_error(self):
        scenes = [{"narration": "bir"} for _ in range(14)]
        words = [
            {"word": "bir", "start": i * 0.343, "end": (i + 1) * 0.343 - 0.01}
            for i in range(14)
        ]
        duration = 14 * 0.343
        timeline = core.scene_timeline(scenes, words, duration)
        for left, right in zip(timeline, timeline[1:]):
            self.assertEqual(left[1], right[0])
        self.assertEqual(
            sum(round(end * 30) - round(start * 30) for start, end in timeline),
            round(duration * 30),
        )

    def test_subtitles_are_current_and_next_not_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = core.create_captions(
                timestamps("Thor şimdi geri döndü"), Path(directory) / "captions.ass"
            )
            dialogues = [
                line
                for line in path.read_text().splitlines()
                if line.startswith("Dialogue:")
            ]
            self.assertEqual(len(dialogues), 4)
            self.assertIn("THOR", dialogues[0])
            self.assertIn("ŞİMDİ", dialogues[0])
            self.assertNotIn("THOR", dialogues[1])
            self.assertIn("GERİ", dialogues[1])
            self.assertNotIn("ŞİMDİ", dialogues[2])
            self.assertIn("DÖNDÜ", dialogues[3])

    def test_subtitle_centisecond_carry(self):
        self.assertEqual(core.ass_time(59.999), "0:01:00.00")


class OrchestrationTests(unittest.TestCase):
    def test_empty_research_has_finite_limit(self):
        engine = SimpleNamespace(
            check_budget=Mock(), research_events=Mock(return_value=[])
        )
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            studio.select_and_build(engine, None, Settings(max_research_rounds=3), "")
        self.assertEqual(engine.research_events.call_count, 3)

    def test_requested_topic_never_switches_after_rejection(self):
        class Rejected(Exception):
            pass

        engine = SimpleNamespace(
            check_budget=Mock(),
            research_events=Mock(return_value=[{"topic": "Thor"}, {"topic": "Batman"}]),
            event_key=lambda item: item["topic"],
            EventRejectedError=Rejected,
            build_single_event_video=Mock(side_effect=Rejected("weak visuals")),
            record_rejected_event=Mock(),
        )
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(Rejected):
            studio.select_and_build(engine, None, Settings(), "Thor")
        self.assertEqual(engine.build_single_event_video.call_count, 1)

    def test_invalid_config_is_rejected_before_api_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factory.json"
            for payload in (
                {"target_seconds": 400},
                {"ai_reconstruction": "false"},
                {"surprise_setting": 2},
            ):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    Settings.load(path)

    def test_decode_failure_cannot_pass_technical_gate(self):
        video_metadata = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "10"},
        }
        with (
            patch.object(core, "inspect_media", return_value=video_metadata),
            patch.object(core, "ffmpeg_binary", return_value="ffmpeg"),
            patch.object(
                core.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)
            ),
        ):
            self.assertFalse(
                core.check_video(Path("video.mp4"), Path("audio.wav"))["passed"]
            )


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root / "preview"
        self.directory.mkdir()
        (self.directory / "video.mp4").write_bytes(b"test-video")
        core.save_json(
            self.directory / "metadata.json",
            {"script": {"title": "Thor", "narration": "Thor geri geldi"}},
        )
        self.manifest = {
            "status": "ready",
            "technical_passed": True,
            "video_sha256": core.file_hash(self.directory / "video.mp4"),
            "metadata_sha256": core.file_hash(self.directory / "metadata.json"),
        }
        core.save_json(self.directory / "run.json", self.manifest)
        self.root_patch = patch.object(publishing, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def test_changed_preview_is_blocked(self):
        (self.directory / "video.mp4").write_bytes(b"another-video")
        with self.assertRaises(ValueError):
            publishing.validate_run(self.directory)

    def test_failed_run_is_never_published(self):
        self.manifest["status"] = "failed"
        core.save_json(self.directory / "run.json", self.manifest)
        with self.assertRaises(ValueError):
            publishing.publish_run(self.directory, "both")

    def test_successful_uploads_are_not_repeated(self):
        history = (
            self.root / "data/publishing" / (self.manifest["video_sha256"] + ".json")
        )
        core.save_json(
            history,
            {
                "youtube": {"status": "success", "id": "yt"},
                "instagram": {"status": "success", "id": "ig"},
            },
        )
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(publishing, "prepare_youtube_credentials") as credentials,
        ):
            results = publishing.publish_run(self.directory, "both")
        credentials.assert_not_called()
        self.assertEqual(results["instagram"]["id"], "ig")

    def test_uncertain_upload_is_not_silently_retried(self):
        history = (
            self.root / "data/publishing" / (self.manifest["video_sha256"] + ".json")
        )
        core.save_json(history, {"youtube": {"status": "sending"}})
        with self.assertRaises(RuntimeError):
            publishing.publish_run(self.directory, "youtube")

    def test_base64_and_existing_credentials_share_one_path(self):
        import base64

        payload = {
            "client_id": "fixture-client",
            "client_secret": "fixture-secret",
            "refresh_token": "fixture-refresh",
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        with patch.dict(
            publishing.os.environ, {"YOUTUBE_TOKEN_B64": encoded}, clear=True
        ):
            publishing.prepare_youtube_credentials()
        self.assertEqual(json.loads((self.root / "token.json").read_text()), payload)
        self.assertTrue((self.root / "client_secret.json").exists())


if __name__ == "__main__":
    unittest.main()
