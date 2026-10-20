import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from werkzeug.security import generate_password_hash
from app import create_app
from ner_service import NERService, ModelUnavailable


class APITests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.predict.return_value = [{'word': 'John', 'entity': 'PER'}]
        self.client = create_app({'TESTING': True, 'NER_USERNAME': 'tester',
            'NER_PASSWORD_HASH': generate_password_hash('test-secret')}, self.service).test_client()
        self.headers = {'Authorization': 'Basic ' + base64.b64encode(b'tester:test-secret').decode()}

    def test_success(self):
        response = self.client.post('/api/predict', json={'text': 'John'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['entities'], self.service.predict.return_value)

    def test_invalid_inputs_never_reach_model(self):
        for data in (None, [], ['text'], 'text', 42, {}, {'text': None},
                     {'text': []}, {'text': 123}, {'text': ''}, {'text': '  '}, {'text': 'a' * 10001}):
            with self.subTest(data=str(data)[:50]):
                response = self.client.post('/api/predict', data=json.dumps(data),
                    content_type='application/json', headers=self.headers)
                self.assertEqual(response.status_code, 400)
        self.service.predict.assert_not_called()

    def test_auth_on_both_inference_routes(self):
        for route in ('/api/predict', '/predict'):
            response = self.client.post(route, json={'text': 'John'})
            self.assertEqual(response.status_code, 401)
            self.assertIn('WWW-Authenticate', response.headers)
        self.service.predict.assert_not_called()

    def test_protocol_errors(self):
        for body, content_type, expected in (('{', 'application/json', 400),
                ('John', 'text/plain', 415), ('a' * 65537, 'application/json', 413)):
            response = self.client.post('/api/predict', data=body,
                content_type=content_type, headers=self.headers)
            self.assertEqual(response.status_code, expected)
            self.assertIsInstance(response.json, dict)

    def test_inference_errors(self):
        for exception, expected in ((ModelUnavailable('private path'), 503),
                (RuntimeError('private detail'), 500), (ValueError('too many tokens'), 400)):
            self.service.predict.side_effect = exception
            response = self.client.post('/api/predict', json={'text': 'John'}, headers=self.headers)
            self.assertEqual(response.status_code, expected)
            self.assertNotIn('private', response.get_data(as_text=True))

    def test_html_escape_and_empty_results(self):
        self.service.predict.return_value = []
        response = self.client.post('/predict', data={'text': '<script>alert(1)</script>'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('<script>', response.get_data(as_text=True))

    def test_credentials_required(self):
        with self.assertRaises(RuntimeError):
            create_app({'NER_USERNAME': None, 'NER_PASSWORD_HASH': None})


class ServiceTests(unittest.TestCase):
    def test_incomplete_model(self):
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ModelUnavailable):
                NERService(path).predict('John')

    def test_token_limit_boundary_and_format(self):
        service = NERService()
        pipeline = Mock(return_value=[{'word': 'John', 'entity_group': 'PER'}])
        pipeline.tokenizer = Mock(return_value={'input_ids': [1] * 512})
        pipeline.tokenizer.model_max_length = 512
        pipeline.model.config = SimpleNamespace(max_position_embeddings=512)
        service._pipeline = pipeline
        self.assertEqual(service.predict('John'), [{'word': 'John', 'entity': 'PER'}])
        pipeline.reset_mock()
        pipeline.tokenizer.return_value = {'input_ids': [1] * 513}
        with self.assertRaises(ValueError):
            service.predict('John')
        pipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main()
