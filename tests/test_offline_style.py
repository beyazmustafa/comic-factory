import contextlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

from factory import quality, render, studio
from factory.api import Api
from factory.style import load_style
from tests.test_studio import TemporaryTest, timestamps


class OfflineStyleTests(TemporaryTest):
    def test_voice_only_never_reads_or_uploads_reference(self):
        import os
        api = Mock()
        api.calls = 0
        with patch.object(studio, 'ROOT', self.root), patch.dict(os.environ, {'GROQ_API_KEY': 'fixture'}), patch.object(studio.voice, 'select_voice'):
            directory = studio.generate(studio.Settings(), voice_only=True, api_factory=lambda *args: api)
        api.video_json.assert_not_called()
        api.json.assert_not_called()
        api.uploaded_video.assert_not_called()
        self.assertTrue((directory / 'editing_profile.json').is_file())
        self.assertFalse((directory / 'reference_profile.json').exists())

    def test_quality_sends_only_generated_video(self):
        report = dict(candidate_observed=True, turkish_narration=True, no_critical_errors=True,
                      subtitle_sync=96, scene_match=96, delivery=90, visual_readability=90,
                      observations=[{'second': s, 'detail': 'visible'} for s in (0, 50, 99)])
        api = SimpleNamespace(directory=self.root, video_json=Mock(return_value=report))
        video = self.root / 'output.mp4'
        with patch.object(quality, 'inspect_media', return_value={'format': {'duration': '100'}}):
            self.assertTrue(quality.review(api, video, {'shots': []}, load_style())['passed'])
        args = api.video_json.call_args.args
        self.assertEqual(len(args), 3)
        self.assertEqual(args[-1], video)
        self.assertNotIn('REFERENCE', args[1])

    def test_video_api_accepts_no_second_video(self):
        api = Api(studio.Settings(), self.root, client=Mock())
        with patch.object(api, 'uploaded_video', return_value=contextlib.nullcontext('candidate-uri')), patch.object(api, 'json', return_value={}) as request:
            api.video_json('quality', 'inspect output', self.root / 'output.mp4')
        self.assertEqual(request.call_args.kwargs['videos'], [('CANDIDATE video', 'candidate-uri')])
        self.assertNotIn('reference_uri', inspect.signature(Api.video_json).parameters)

    def test_colored_turkish_words_keep_audio_times(self):
        words = timestamps('Şimdi tehlike büyüdü')
        words[1]['emphasis'] = 'danger'
        path = render.captions(words, {**load_style(), 'language': 'tr'}, self.root / 'captions.ass')
        rows = [line for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('Dialogue')]
        self.assertIn('ŞİMDİ', rows[0])
        self.assertNotIn('TEHLİKE', rows[0])
        self.assertIn('0:00:00.40', rows[1])
        self.assertIn(render.ass_color('#FF2020'), rows[1])

    def test_vertical_motion_changes_y_without_changing_frame_budget(self):
        for motion in ('up', 'down'):
            expression = render.motion_filter(motion, 90, .12)
            self.assertIn(':d=90:', expression)
            self.assertIn('(ih-ih/zoom)*', expression)
