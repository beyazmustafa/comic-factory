import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from factory import archive, core, panels, research, story, studio, voice
from factory.api import SpeechFailure
from factory.config import Settings
from tests.test_studio import TemporaryTest, script_fixture, style


class PipelineResumeTests(TemporaryTest):
    def test_speech_failure_preserves_story_and_resume_skips_research(self):
        script, inventory, facts = script_fixture()
        event = {
            "id": "test",
            "title": "Test event",
            "publisher": "Test",
            "series": "Test Series",
            "issue": "1",
            "year": 2000,
        }
        self.enterContext(patch.object(studio, "ROOT", self.root))
        self.enterContext(
            patch.dict(os.environ, {"GROQ_API_KEY": "fixture"}, clear=True)
        )

        def api_factory(settings, directory):
            return SimpleNamespace(
                directory=directory,
                settings=settings,
                check=Mock(),
                close=Mock(),
                calls=0,
            )

        def catalog(api, event, pages, articles):
            for panel in inventory:
                for field in ("file", "page_file"):
                    path = api.directory / panel[field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (500, 500), "#1c3649").save(path)
            return inventory, facts

        def create(api, event, inventory, facts, profile):
            core.save_json(api.directory / "story.json", script)
            return script

        def select(api, profile, cache):
            result = {"voice": "Orus"}
            core.save_json(api.directory / "voice_selection.json", result)
            return result

        # The production profile is loaded locally, including on resume.
        shortlist = self.enterContext(
            patch.object(
                research, "shortlist", return_value=[event, {**event, "id": "other"}]
            )
        )
        self.enterContext(
            patch.object(
                research, "Fetcher", return_value=SimpleNamespace(session=Mock())
            )
        )
        collected = self.enterContext(
            patch.object(research, "collect_pages", return_value=([], {}))
        )
        self.enterContext(patch.object(panels, "catalog", side_effect=catalog))
        self.enterContext(patch.object(story, "create", side_effect=create))
        self.enterContext(patch.object(voice, "select_voice", side_effect=select))
        self.enterContext(
            patch.object(
                voice,
                "build_audio",
                side_effect=SpeechFailure("simulated speech outage"),
            )
        )
        with self.assertRaises(SpeechFailure):
            studio.generate(Settings(source="web"), api_factory=api_factory)
        before = next((self.root / "data/runs").iterdir())
        self.assertTrue((before / "story.json").exists())
        self.assertEqual(collected.call_count, 1)
        self.assertEqual(
            json.loads((before / "run.json").read_text())["stage"],
            "Konuşma ve kelime zamanları",
        )
        with self.assertRaises(SpeechFailure):
            studio.generate(Settings(source="web"), resume_run=before, api_factory=api_factory)
        self.assertEqual(shortlist.call_count, 1)
        self.assertEqual(collected.call_count, 1)


    def test_archive_source_uses_archive_modules_and_records_history(self):
        script, inventory, facts = script_fixture()
        event = {
            "id": "ia_test",
            "identifier": "hand-of-fate-23",
            "title": "Kaderin Eli",
            "publisher": "Ace",
            "series": "Hand of Fate",
            "issue": "23",
            "year": 1954,
            "url": "https://archive.org/details/hand-of-fate-23",
        }
        self.enterContext(patch.object(studio, "ROOT", self.root))
        self.enterContext(patch.object(archive, "ROOT", self.root))
        self.enterContext(
            patch.dict(os.environ, {"GROQ_API_KEY": "fixture"}, clear=True)
        )

        def api_factory(settings, directory):
            return SimpleNamespace(
                directory=directory, settings=settings, check=Mock(), close=Mock(), calls=0
            )

        def catalog(api, event, pages, articles):
            for panel in inventory:
                for field in ("file", "page_file"):
                    path = api.directory / panel[field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (500, 500), "#1c3649").save(path)
            return inventory, facts

        def create(api, event, inventory, facts, profile):
            core.save_json(api.directory / "story.json", script)
            return script

        web_shortlist = self.enterContext(patch.object(research, "shortlist"))
        shortlist = self.enterContext(
            patch.object(archive, "shortlist", return_value=[event])
        )
        collected = self.enterContext(
            patch.object(archive, "collect_pages", return_value=([], {}))
        )
        web_catalog = self.enterContext(patch.object(panels, "catalog"))
        self.enterContext(patch.object(panels, "catalog_archive", side_effect=catalog))
        self.enterContext(patch.object(story, "create", side_effect=create))
        self.enterContext(
            patch.object(
                voice, "build_audio", side_effect=SpeechFailure("simulated speech outage")
            )
        )
        def select(api, profile, cache):
            core.save_json(api.directory / "voice_selection.json", {"voice": "Orus"})
            return {"voice": "Orus"}

        self.enterContext(patch.object(voice, "select_voice", side_effect=select))
        with self.assertRaises(SpeechFailure):
            studio.generate(Settings(), api_factory=api_factory)
        self.assertEqual(shortlist.call_count, 1)
        self.assertEqual(collected.call_count, 1)
        web_shortlist.assert_not_called()
        web_catalog.assert_not_called()
        # History is written only when a video is ready, never on failure.
        self.assertFalse((self.root / "data" / "history" / "issues").exists())
        path = archive.remember_issue(event, "123")
        self.assertIn("ia_test", studio.load_used())
        self.assertIn("hand-of-fate-23", studio.load_used())
        self.assertTrue(path.is_file())
