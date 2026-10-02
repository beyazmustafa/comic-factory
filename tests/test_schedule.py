import os
import unittest
from unittest.mock import patch
from factory.config import ROOT
from factory.publishers import youtube

WORKFLOW = ROOT / ".github" / "workflows" / "comic-factory.yml"


class ScheduleTests(unittest.TestCase):
    def test_workflow_runs_twice_daily_with_safe_defaults(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("cron: '23 5 * * *'", text)  # 08:23 Türkiye
        self.assertIn("cron: '23 14 * * *'", text)  # 17:23 Türkiye
        self.assertIn("inputs.task || 'create_and_publish'", text)
        self.assertIn("vars.SCHEDULED_PLATFORMS || 'both'", text)
        self.assertIn("vars.YOUTUBE_VISIBILITY || 'public'", text)
        self.assertIn("contents: write", text)
        self.assertIn("git add -A data/history", text)
        self.assertIn("poppler-utils", text)

    def test_history_folder_is_tracked_in_git(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("data/*", ignore)
        self.assertIn("!data/history/", ignore)
        self.assertNotIn("data/", ignore)


class VisibilityTests(unittest.TestCase):
    def test_default_is_unlisted_and_values_are_validated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(youtube.visibility(), "unlisted")
        with patch.dict(os.environ, {"YOUTUBE_VISIBILITY": " Public "}, clear=True):
            self.assertEqual(youtube.visibility(), "public")
        with patch.dict(os.environ, {"YOUTUBE_VISIBILITY": "scheduled"}, clear=True):
            with self.assertRaises(youtube.YouTubeUploaderError):
                youtube.visibility()


class ProviderChainTests(unittest.TestCase):
    def make_api(self, temporary, **overrides):
        from pathlib import Path
        from unittest.mock import Mock
        from factory.api import Api
        from factory.config import Settings
        settings = Settings(gemini_model="primary", gemini_fallback_models="backup,stable", **overrides)
        return Api(settings, Path(temporary), client=Mock())

    def test_overloaded_models_are_skipped_for_the_rest_of_the_run(self):
        import tempfile
        from factory.api import FactoryError

        class Overloaded(Exception):
            code = 503

        with tempfile.TemporaryDirectory() as temporary, patch("factory.api.time.sleep"):
            api = self.make_api(temporary)
            seen = []

            class Retired(Exception):
                code = 404

            def operation(model):
                seen.append(model)
                if model == "primary":
                    raise Overloaded("high demand")
                if model == "backup":
                    raise Retired("This model is no longer available to new users.")
                return "ok"

            self.assertEqual(api.gemini("Deneme", operation), "ok")
            self.assertEqual(seen, ["primary"] * 3 + ["backup"] + ["stable"])
            self.assertEqual(api.dead, {"primary", "backup"})
            seen.clear()
            self.assertEqual(api.gemini("Deneme", operation), "ok")
            self.assertEqual(seen, ["stable"])
            with self.assertRaises(FactoryError):
                api.gemini("Deneme", lambda model: (_ for _ in ()).throw(ValueError("bad request")))

    def test_json_moves_to_groq_when_gemini_is_down_but_not_for_audio(self):
        import tempfile
        from unittest.mock import Mock
        from factory.api import FactoryError

        class Overloaded(Exception):
            code = 503

        with tempfile.TemporaryDirectory() as temporary, patch("factory.api.time.sleep"), \
                patch.dict("os.environ", {"GROQ_API_KEY": "x"}):
            api = self.make_api(temporary)
            api.client.models.generate_content.side_effect = Overloaded("high demand")
            groq = Mock()
            groq.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content='{"events": [1]}'))])
            api.groq_client = groq
            from PIL import Image
            from pathlib import Path
            picture = Path(temporary) / "p.jpg"
            Image.new("RGB", (64, 64), "red").save(picture)
            value = api.json("Deneme", "Return JSON", images=[("page", picture)], list_key="events")
            self.assertEqual(value, {"events": [1]})
            sent = groq.chat.completions.create.call_args.kwargs
            self.assertEqual(sent["response_format"], {"type": "json_object"})
            self.assertEqual(sent["messages"][0]["content"][2]["type"], "image_url")
            (Path(temporary) / "a.wav").write_bytes(b"RIFF")
            with self.assertRaises(FactoryError):
                api.json("Deneme", "Return JSON", audio=[("a", Path(temporary) / "a.wav")])


class VoiceFallbackTests(unittest.TestCase):
    def test_gemini_tts_overload_switches_to_edge_voice(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        from factory import voice
        from factory.api import ProviderOverloaded

        with tempfile.TemporaryDirectory() as temporary:
            api = SimpleNamespace(settings=Mock(tts_model="t", language="tr"), note=Mock(), directory=Path(temporary))
            edge = patch.object(voice, "edge_synthesize", side_effect=lambda text, name, path: (name, path))
            gemini = patch.object(voice, "gemini_synthesize", side_effect=ProviderOverloaded("503"))
            with edge as edge_mock, gemini:
                result = voice.synthesize(api, "Merhaba", "Gacrux", {}, Path(temporary) / "a.wav")
                self.assertEqual(result[0], "Emel")
                self.assertTrue(api.tts_fallback)
                # Later chunks go straight to edge-tts without touching Gemini again.
                voice.synthesize(api, "Merhaba", "Orus", {}, Path(temporary) / "b.wav")
                self.assertEqual(edge_mock.call_count, 2)


class DiscoveryTests(unittest.TestCase):
    def test_chain_is_rebuilt_from_what_the_providers_list(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        from factory.api import Api
        from factory.config import Settings

        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"GROQ_API_KEY": "x"}):
            settings = Settings(
                gemini_model="gemini-3.7-flash",
                gemini_fallback_models="gemini-2.5-flash,gemini-3.8-flash",
                groq_model="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            api = Api(settings, Path(temporary), client=Mock())
            api.client.models.list.return_value = [
                SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.9-pro", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.9-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.1-flash-tts", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/embedding-001", supported_actions=["embedContent"]),
                SimpleNamespace(name="models/gemini-3.7-flash", supported_actions=["generateContent"]),
            ]
            groq = Mock()
            groq.models.list.return_value = SimpleNamespace(data=[
                SimpleNamespace(id="whisper-large-v3"),
                SimpleNamespace(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
                SimpleNamespace(id="llama-3.3-70b-versatile"),
                SimpleNamespace(id="qwen/qwen3-vl-32b"),
            ])
            api.groq_client = groq
            found = api.discover()
            self.assertEqual(api.models, ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.9-flash", "gemini-3.9-pro"])
            self.assertEqual(api.groq_models, ["meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-vl-32b"])
            self.assertEqual(found["errors"], [])
            self.assertTrue((Path(temporary) / "diagnostics" / "providers.json").is_file())

    def test_discovery_failure_keeps_configured_names(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        from factory.api import Api
        from factory.config import Settings

        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {}, clear=True):
            api = Api(Settings(gemini_model="a", gemini_fallback_models="b"), Path(temporary), client=Mock())
            api.client.models.list.side_effect = RuntimeError("offline")
            api.discover()
            self.assertEqual(api.models, ["a", "b"])


class JsonRobustnessTests(unittest.TestCase):
    def test_prose_wrapped_object_is_parsed(self):
        from factory.api import parse_object
        self.assertEqual(parse_object('Here you go:\n```json\n{"a": 1}\n```\nHope this helps'), {"a": 1})
        with self.assertRaises(ValueError):
            parse_object("no object here")

    def test_broken_json_is_retried_once_with_strict_prompt(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        from factory.api import Api, FactoryError
        from factory.config import Settings

        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {}, clear=True):
            api = Api(Settings(gemini_model="m", gemini_fallback_models=""), Path(temporary), client=Mock())
            api.client.models.list.side_effect = RuntimeError("offline")
            api.client.models.generate_content.side_effect = [Mock(text="not json {"), Mock(text='{"ok": true}')]
            self.assertEqual(api.json("Deneme", "Return JSON"), {"ok": True})
            second = api.client.models.generate_content.call_args_list[1].kwargs["contents"][1]
            self.assertIn("STRICT OUTPUT", second)
            api.client.models.generate_content.side_effect = [Mock(text="bad"), Mock(text="still bad")]
            with self.assertRaises(FactoryError):
                api.json("Deneme", "Return JSON")

    def test_text_only_request_can_use_groq_text_models(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        from factory.api import Api
        from factory.config import Settings

        class Overloaded(Exception):
            code = 503

        with tempfile.TemporaryDirectory() as temporary, patch("factory.api.time.sleep"), \
                patch.dict("os.environ", {"GROQ_API_KEY": "x"}):
            api = Api(Settings(gemini_model="m", gemini_fallback_models="", groq_model=""), Path(temporary), client=Mock())
            api.client.models.list.return_value = []
            api.client.models.generate_content.side_effect = Overloaded("503")
            groq = Mock()
            groq.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b"), SimpleNamespace(id="whisper-large-v3")])
            groq.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content='{"events": []}'))])
            api.groq_client = groq
            self.assertEqual(api.json("Deneme", "Return JSON"), {"events": []})
            self.assertEqual(groq.chat.completions.create.call_args.kwargs["model"], "openai/gpt-oss-120b")
            from PIL import Image
            picture = Path(temporary) / "p.jpg"
            Image.new("RGB", (32, 32), "red").save(picture)
            from factory.api import FactoryError
            with self.assertRaises(FactoryError):
                api.json("Deneme", "Return JSON", images=[("p", picture)])


class JsonRepairTests(unittest.TestCase):
    def test_orphan_ids_are_restored(self):
        from factory.api import parse_object
        broken = '[{"shot_id": "shot_000", "match_score": 90}, {"shot_036", "match_score": 95, "supported": true}]'
        value = parse_object(broken, list_key="shots")
        self.assertEqual(value["shots"][1]["shot_id"], "shot_036")
        story = '{"shots": [{"panel_003", "narration": "x", "fact_ids": ["fact_001"]}]}'
        self.assertEqual(parse_object(story)["shots"][0]["panel_id"], "panel_003")

    def test_lite_models_sort_last(self):
        from factory.api import version_key
        names = ["gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-3.1-flash-lite"]
        self.assertEqual(sorted(names, key=version_key, reverse=True)[-2:], ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])


class LearningTests(unittest.TestCase):
    def test_playbook_and_experiment_are_written_and_fed_to_writer(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        from factory import learning
        from factory.config import Settings

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(learning, "ROOT", root):
                api = SimpleNamespace(settings=Settings(), directory=root / "run", note=Mock(), run_id="42", json=Mock(return_value={
                    "playbook": "- Hook in 8 words.\n- End on the strongest panel.\n- Red emphasis only for mortal danger, max 3 per video.",
                    "experiment": {"name": "short hook", "change": "First sentence under 8 words.", "rationale": "Retention."},
                    "verdict_on_last_experiment": "unknown",
                }))
                extras = learning.evolve(api, {})
                self.assertIn("Hook in 8 words", extras["playbook"])
                self.assertEqual(extras["experiment"]["name"], "short hook")
                self.assertTrue((root / "data" / "history" / "playbook.md").is_file())
                experiments = json.loads((root / "data" / "history" / "experiments.json").read_text())
                self.assertEqual(experiments[0]["run_id"], "42")
                # A publication is recorded with the run's profile and the experiment it carried.
                run = root / "run"
                run.mkdir(parents=True, exist_ok=True)
                (run / "run.json").write_text(json.dumps({"run_id": "42", "duration": 100.0, "voice": "Christopher",
                                                           "event": {"title": "T", "series": "S", "issue": "1", "year": 1950, "publisher": "Fox"}}))
                (run / "metadata.json").write_text(json.dumps({"script": {"title": "Hero Falls", "shots": [
                    {"narration": "The hero falls.", "emphasis": "danger"}, {"narration": "Nobody saw it coming at all.", "emphasis": "normal"}]}}))
                (run / "quality_review.json").write_text(json.dumps({"review_mode": "frames", "subtitle_sync": 97, "issues": ["x"]}))
                learning.record_publication(run, "youtube", "abc123")
                learning.record_publication(run, "youtube", "abc123")
                rows = learning.load_performance()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["profile"]["shots"], 2)
                self.assertEqual(rows[0]["experiment"]["name"], "short hook")

    def test_model_failure_keeps_existing_playbook(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock
        from factory import learning
        from factory.config import Settings

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(learning, "ROOT", Path(temporary)):
                api = SimpleNamespace(settings=Settings(), directory=Path(temporary) / "run", note=Mock(), run_id="1",
                                      json=Mock(side_effect=RuntimeError("down")))
                extras = learning.evolve(api, {})
                self.assertIn("Hook", extras["playbook"])
                self.assertIsNone(extras["experiment"])


class LearningNumbersTests(unittest.TestCase):
    def test_views_at_fixed_age_and_numeric_verdicts(self):
        from factory import learning
        history = [{"h": 6, "views": 10}, {"h": 30, "views": 100}]
        self.assertEqual(learning.views_at(history, 24), 78)
        self.assertIsNone(learning.views_at(history, 72))
        rows = [
            {"run_id": "1", "platform": "youtube", "profile": {}, "stats": {"views_per_hour": 1.0, "hours_live": 40}},
            {"run_id": "2", "platform": "youtube", "profile": {}, "stats": {"views_per_hour": 1.2, "hours_live": 30}},
            {"run_id": "3", "platform": "youtube", "profile": {}, "stats": {"views_per_hour": 0.9, "hours_live": 20}},
            {"run_id": "4", "platform": "youtube", "profile": {}, "stats": {"views_per_hour": 5.0, "hours_live": 14}},
            {"run_id": "5", "platform": "youtube", "profile": {}, "stats": {"views_per_hour": 0.2, "hours_live": 2}},
        ]
        experiments = [{"run_id": "4", "experiment": {"name": "x"}}, {"run_id": "5", "experiment": {"name": "y"}}]
        self.assertTrue(learning.judge_experiments(rows, experiments))
        self.assertEqual(experiments[0]["verdict"], "kept")
        self.assertEqual(experiments[0]["judged_by"], "numbers")
        self.assertNotIn("verdict", experiments[1])  # too young to judge
