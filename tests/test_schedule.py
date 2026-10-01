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
