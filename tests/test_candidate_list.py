import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from factory.api import Api, parse_object
from factory.config import Settings
from factory import research
from tests.test_studio import TemporaryTest

FIXTURE = Path(__file__).parent / 'fixtures' / 'candidate_list.json'


class CandidateListTests(TemporaryTest):
    def test_actual_failed_response_becomes_candidates_without_retry(self):
        raw = FIXTURE.read_text(encoding='utf-8')
        rows = json.loads(raw)
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(text=raw)
        api = Api(Settings(), self.root, client=client)
        evidence = {'articles': [{'url': url} for row in rows for url in row['source_urls']]}
        with patch.object(research, 'discover_evidence', return_value=evidence):
            events = research.shortlist(api, '', [])
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]['series'], 'Action Comics')
        self.assertEqual(api.calls, 1)
        client.models.generate_content.assert_called_once()

    def test_unrequested_list_still_rejected(self):
        with self.assertRaises(ValueError):
            parse_object('[{"title":"x"}]')

    def test_primitive_items_rejected(self):
        with self.assertRaises(ValueError):
            parse_object('["unexpected", 7]', list_key='events')

    def test_existing_object_unchanged(self):
        self.assertEqual(parse_object('{"events":[]}', list_key='events'), {'events': []})

    def test_fenced_list_normalized(self):
        self.assertEqual(parse_object('```json\n[{"title":"İlk olay"}]\n```', list_key='events'), {'events': [{'title': 'İlk olay'}]})
