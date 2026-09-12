import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from factory import core, engine, studio
from factory.config import Settings
from tests.test_pipeline import timestamps


class TimestampRegressionTests(unittest.TestCase):
    def test_zero_duration_word_is_estimated_without_claiming_perfect_timing(self):
        raw = timestamps("Thor geri döndü")
        raw[1]["end"] = raw[1]["start"]
        aligned, metrics = core.align_words("Thor geri döndü", raw)
        self.assertEqual(raw[1]["start"], raw[1]["end"])  # Input is preserved.
        self.assertEqual(metrics["coverage"], 100)
        self.assertEqual(metrics["zero_duration_asr_words"], 1)
        self.assertEqual(aligned[1]["timing_source"], "estimated")
        self.assertGreater(aligned[1]["end"], aligned[1]["start"])
        self.assertLess(metrics["score"], 95)

    def test_all_zero_duration_times_cannot_pass(self):
        raw = [
            {"word": word, "start": 0, "end": 0} for word in "Thor geri döndü".split()
        ]
        _, metrics = core.align_words("Thor geri döndü", raw)
        self.assertEqual(metrics["score"], 0)

    def test_unordered_timestamps_are_rejected_instead_of_reordering_speech(self):
        raw = timestamps("Thor geri döndü")
        raw[1]["start"] = 1.0
        raw[1]["end"] = 1.2
        with self.assertRaisesRegex(ValueError, "ASR kelimesi 3"):
            core.align_words("Thor geri döndü", raw)

    def test_large_overlaps_cannot_pass_after_clamping(self):
        raw = [
            {"word": word, "start": 0, "end": 1} for word in "Thor geri döndü".split()
        ]
        aligned, metrics = core.align_words("Thor geri döndü", raw)
        self.assertLess(metrics["score"], 95)
        self.assertTrue(all(item["end"] <= 1 for item in aligned))

    def test_joined_asr_token_keeps_both_script_words_and_marks_split_timing(self):
        aligned, metrics = core.align_words(
            "geri döndü Thor", timestamps("geridöndü Thor")
        )
        self.assertEqual([item["word"] for item in aligned], ["geri", "döndü", "Thor"])
        self.assertEqual(metrics["coverage"], 100)
        self.assertEqual(metrics["similarity"], 100)
        self.assertAlmostEqual(aligned[1]["end"], 0.35)
        self.assertEqual(aligned[0]["end"], aligned[1]["start"])
        self.assertEqual(aligned[1]["timing_source"], "estimated")
        self.assertEqual(aligned[2]["timing_source"], "asr")
        self.assertLess(metrics["score"], 95)

    def test_punctuation_only_asr_record_does_not_require_a_spoken_timestamp(self):
        raw = timestamps("Thor döndü")
        raw.insert(1, {"word": ",", "start": None, "end": None})
        _, metrics = core.align_words("Thor döndü", raw)
        self.assertEqual(metrics["ignored_asr_punctuation"], 1)
        self.assertEqual(metrics["score"], 100)

    def test_out_of_recording_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "kayıt süresinin dışında"):
            core.align_words("Thor", [{"word": "Thor", "start": 0, "end": 5}], 1)

    def test_twenty_millisecond_rounding_is_clamped_to_recording_end(self):
        aligned, metrics = core.align_words(
            "Thor", [{"word": "Thor", "start": 0, "end": 1.01}], 1
        )
        self.assertEqual(aligned[0]["end"], 1)
        self.assertEqual(metrics["score"], 100)

    def test_unrelated_speech_stays_below_gate(self):
        _, metrics = core.align_words(
            "Thor yeniden dünyaya döndü", timestamps("bugün hava oldukça sıcak")
        )
        self.assertLess(metrics["score"], 95)


class AudioRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "voice.wav"
        self.audio.write_bytes(b"fixture; media probing is mocked in these unit tests")
        self.narration = "Thor geri döndü ve Thor yeniden savaştı"
        self.stack = self.enterContext(contextlib.ExitStack())
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for name, value in {
            "AUDIO_DIAGNOSTICS_DIR": self.root / "diagnostics",
            "LATEST_AUDIO_FILE": self.root / "latest.wav",
            "LATEST_WORDS_FILE": self.root / "words.json",
            "ALIGNED_WORDS_FILE": self.root / "aligned.json",
            "CHECKPOINT_FILE": self.root / "checkpoints.json",
            "CHECKPOINTS": engine.CheckpointManager(),
            "GROQ_MODEL": "whisper-large-v3-turbo",
            "AUDIO_REPAIR_CYCLES": 3,
            "AUDIO_ALIGNMENT_THRESHOLD": 95.0,
        }.items():
            self.stack.enter_context(patch.object(engine, name, value))
        self.stack.enter_context(patch.object(engine, "check_budget"))
        self.stack.enter_context(
            patch.object(engine, "media_duration", return_value=3.0)
        )
        self.tts = self.stack.enter_context(
            patch.object(engine, "generate_gacrux_voice", return_value=self.audio)
        )

    def attempts(self):
        return json.loads(
            (self.root / "diagnostics/narration/attempts.json").read_text()
        )

    def test_invalid_first_asr_uses_fallback_on_same_audio(self):
        invalid = [{"word": "Thor", "start": 1, "end": 0}]
        with patch.object(
            engine,
            "transcribe_words",
            side_effect=[invalid, timestamps(self.narration)],
        ) as asr:
            output, _, aligned = engine.build_quality_audio(None, self.narration)
        self.assertTrue(output.exists())
        self.assertEqual(len(aligned), len(self.narration.split()))
        self.assertEqual(self.tts.call_count, 1)
        self.assertEqual(
            [call.args[0] for call in asr.call_args_list], [self.audio, self.audio]
        )
        self.assertEqual(
            [call.kwargs["model"] for call in asr.call_args_list],
            ["whisper-large-v3-turbo", "whisper-large-v3"],
        )
        self.assertEqual(
            [item["status"] for item in self.attempts()], ["error", "passed"]
        )
        self.assertEqual(
            json.loads((self.root / "words.json").read_text())["model"],
            "whisper-large-v3",
        )
        self.assertTrue(
            (self.root / "diagnostics/narration/cycle_01/narration.wav").exists()
        )

    def test_low_score_then_invalid_times_does_not_skip_next_voice_attempt(self):
        # Reproduces the screenshot's low score followed by a timestamp exception.
        low = timestamps("Thor geri")
        invalid = [{"word": "Thor", "start": 1, "end": 0}]
        with patch.object(
            engine,
            "transcribe_words",
            side_effect=[low, invalid, timestamps(self.narration)],
        ) as asr:
            engine.build_quality_audio(None, self.narration)
        self.assertEqual(self.tts.call_count, 2)
        self.assertEqual(asr.call_count, 3)
        self.assertEqual(
            [item["status"] for item in self.attempts()],
            ["below_threshold", "error", "passed"],
        )

    def test_all_invalid_attempts_stop_at_limit_and_keep_diagnostics(self):
        invalid = [{"word": "Thor", "start": 0, "end": float("nan")}]
        with patch.object(engine, "transcribe_words", return_value=invalid) as asr:
            with self.assertRaises(engine.AudioAlignmentError) as caught:
                engine.build_quality_audio(None, self.narration)
        self.assertNotIsInstance(caught.exception, engine.EventRejectedError)
        self.assertEqual(self.tts.call_count, 3)
        self.assertEqual(asr.call_count, 6)
        self.assertEqual(len(self.attempts()), 6)
        self.assertFalse((self.root / "latest.wav").exists())
        self.assertEqual(
            len(list((self.root / "diagnostics").rglob("narration.wav"))), 3
        )

    def test_low_coverage_never_passes_at_default_threshold(self):
        with patch.object(
            engine, "transcribe_words", return_value=timestamps("Thor geri")
        ):
            with self.assertRaises(engine.AudioAlignmentError):
                engine.build_quality_audio(None, self.narration)
        self.assertTrue(all(item["metrics"]["score"] < 95 for item in self.attempts()))
        self.assertTrue(all(not item.passed for item in engine.CHECKPOINTS.results))

    def test_full_model_override_is_not_called_twice_per_voice(self):
        with (
            patch.object(engine, "GROQ_MODEL", "whisper-large-v3"),
            patch.object(engine, "transcribe_words", return_value=[]) as asr,
        ):
            with self.assertRaises(engine.AudioAlignmentError):
                engine.build_quality_audio(None, self.narration)
        self.assertEqual(asr.call_count, 3)

    def test_tts_failure_consumes_one_attempt_and_can_recover(self):
        self.tts.side_effect = [engine.ComicFactoryError("empty audio"), self.audio]
        with patch.object(
            engine, "transcribe_words", return_value=timestamps(self.narration)
        ) as asr:
            engine.build_quality_audio(None, self.narration)
        self.assertEqual(self.tts.call_count, 2)
        self.assertEqual(asr.call_count, 1)
        self.assertEqual(self.attempts()[0]["stage"], "tts")

    def test_audio_failure_does_not_blacklist_topic_or_start_new_research(self):
        fake = SimpleNamespace(
            check_budget=Mock(),
            research_events=Mock(return_value=[{"topic": "Thor"}, {"topic": "Batman"}]),
            event_key=lambda event: event["topic"],
            EventRejectedError=engine.EventRejectedError,
            build_single_event_video=Mock(
                side_effect=engine.AudioAlignmentError("audio failed")
            ),
            record_rejected_event=Mock(),
        )
        with self.assertRaises(engine.AudioAlignmentError):
            studio.select_and_build(fake, None, Settings(), "")
        self.assertEqual(fake.research_events.call_count, 1)
        self.assertEqual(fake.build_single_event_video.call_count, 1)
        fake.record_rejected_event.assert_not_called()


class TranscriptionDiagnosticsTests(unittest.TestCase):
    def test_legacy_audio_rejections_do_not_block_research_but_visual_rejections_do(
        self,
    ):
        payload = {
            "events": [
                {
                    "event_key": "audio-times",
                    "reason": "Geçersiz veya sırasız ses zamanları.",
                },
                {
                    "event_key": "audio-score",
                    "reason": "Bu eventin narration/audio kombinasyonu istenen alignment seviyesine ulaşamadı. Best=50.0.",
                },
                {"event_key": "visual", "reason": "Görsel kalite yetersiz."},
            ]
        }
        with patch.object(engine, "load_json", return_value=payload):
            self.assertEqual(engine.rejected_event_keys(), {"visual"})
        self.assertEqual(len(payload["events"]), 3)

    def test_short_response_and_bad_times_are_saved_before_validation(self):
        raw = {
            "text": "Thor döndü",
            "duration": 2.0,
            "words": [{"word": "Thor", "start": 1.0, "end": 0.0}],
            "segments": [{"text": "Thor döndü", "avg_logprob": -0.2}],
        }
        client = MagicMock()
        client.__enter__.return_value = client
        client.audio.transcriptions.create.return_value = raw
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "audio.wav").write_bytes(b"fixture")
            with (
                patch.object(engine, "Groq", return_value=client),
                patch.object(engine, "require_env", return_value="fixture-key"),
            ):
                words = engine.transcribe_words(
                    path / "audio.wav",
                    "Thor döndü",
                    model="whisper-large-v3",
                    diagnostics_path=path / "raw.json",
                )
            self.assertEqual(
                json.loads((path / "raw.json").read_text())["response"], raw
            )
            with self.assertRaises(engine.AudioAlignmentError):
                engine.align_narration_to_audio("Thor döndü", words)
        call = client.audio.transcriptions.create.call_args
        self.assertEqual(call.kwargs["timestamp_granularities"], ["word", "segment"])
        self.assertNotIn("prompt", call.kwargs)
        self.assertEqual(call.kwargs["language"], "tr")


if __name__ == "__main__":
    unittest.main()
