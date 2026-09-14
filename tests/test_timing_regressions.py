import unittest
from factory import core
from tests.test_studio import timestamps


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
