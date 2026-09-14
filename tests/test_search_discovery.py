from types import SimpleNamespace
from unittest.mock import Mock, patch
from factory import research
from factory.api import Api, SourceUnavailable
from factory.config import Settings
from tests.test_studio import TemporaryTest


class SearchDiscoveryTests(TemporaryTest):
    def api(self):
        return SimpleNamespace(directory=self.root, check=Mock(), settings=Settings(), json=Mock())

    def test_reads_sources_without_model_or_google_tool(self):
        api = self.api()
        fetcher = Mock()
        fetcher.article.return_value = {'url': 'https://example.com/review', 'text': 'Evidence ' * 50, 'images': ['panel.jpg']}
        with patch.object(research, 'search', return_value=[{'url': 'https://example.com/review'}]), patch.object(research, 'Fetcher', return_value=fetcher):
            result = research.discover_evidence(api, 'Thor')
        self.assertEqual(len(result['articles']), 1)
        api.json.assert_not_called()
        fetcher.session.close.assert_called_once()
        self.assertFalse(hasattr(Api, 'grounded'))

    def test_no_sources_fails_before_model(self):
        api = self.api()
        with patch.object(research, 'search', return_value=[]):
            with self.assertRaises(SourceUnavailable):
                research.shortlist(api, '', [])
        api.json.assert_not_called()

    def test_invented_source_cannot_become_candidate(self):
        api = self.api()
        row = dict(title='Test', publisher='Marvel', series='Test', issue='1', year=2000, universe='616', summary='Test', source_urls=['https://invented.example/review'])
        api.json.return_value = {'events': [row]}
        with patch.object(research, 'discover_evidence', return_value={'articles': [{'url': 'https://example.com/review'}]}):
            with self.assertRaises(SourceUnavailable):
                research.shortlist(api, '', [])

    def test_only_fetched_source_is_retained(self):
        api = self.api()
        api.json.return_value = {'events': [dict(title='Test', publisher='Marvel', series='Test', issue='1', year=2000, universe='616', summary='Test', source_urls=['https://example.com/review', 'https://invented.example'])]}
        with patch.object(research, 'discover_evidence', return_value={'articles': [{'url': 'https://example.com/review'}]}):
            events = research.shortlist(api, '', [])
        self.assertEqual(events[0]['source_urls'], ['https://example.com/review'])
