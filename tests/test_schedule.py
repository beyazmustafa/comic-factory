import os
import unittest
from unittest.mock import patch
from factory.config import ROOT
from factory.publishers import youtube

WORKFLOW = ROOT / ".github" / "workflows" / "comic-factory.yml"


class ScheduleTests(unittest.TestCase):
    def test_workflow_runs_twice_daily_with_safe_defaults(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("cron: '0 6,15 * * *'", text)  # 09:00 / 18:00 Türkiye
        self.assertIn("inputs.task || 'create_and_publish'", text)
        self.assertIn("vars.SCHEDULED_PLATFORMS || 'youtube'", text)
        self.assertIn("vars.YOUTUBE_VISIBILITY || 'unlisted'", text)
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
            api = SimpleNamespace(settings=Mock(tts_model="t"), note=Mock(), directory=Path(temporary))
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
            self.assertEqual(api.groq_models, ["qwen/qwen3-vl-32b", "meta-llama/llama-4-maverick-17b-128e-instruct"])
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
