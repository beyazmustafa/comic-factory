import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from factory import core, publishing


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
        core.save_json(self.directory / "quality_review.json", {"passed": True})
        self.manifest = {
            "schema": 2,
            "quality_passed": True,
            "quality_sha256": core.file_hash(self.directory / "quality_review.json"),
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
