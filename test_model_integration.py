"""Real model checks: run with python -m pytest test_model_integration.py -q."""
import base64
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from ner_service import NERService


@pytest.fixture(scope='module')
def service():
    model_path = Path(__file__).resolve().parent / 'ner-bert-model'
    if not (model_path / 'model.safetensors').is_file():
        pytest.skip('Local model weights are required')
    return NERService(model_path)


def test_real_predictions(service):
    text = 'John works at Google in New York.'
    entities = service.predict(text)
    assert entities
    assert all(set(entity) == {'word', 'entity'} for entity in entities)
    assert all(entity['entity'] in {'PER', 'ORG', 'LOC', 'MISC'} for entity in entities)
    print(entities)


def test_real_token_boundary(service):
    # Each "hello" is one BERT token; [CLS] and [SEP] add two tokens.
    assert isinstance(service.predict('hello ' * 510), list)
    with pytest.raises(ValueError, match='512 tokens'):
        service.predict('hello ' * 511)


def test_real_api_and_web_form(service):
    client = create_app({'TESTING': True, 'NER_USERNAME': 'integration',
        'NER_PASSWORD_HASH': generate_password_hash('integration-secret')}, service).test_client()
    headers = {'Authorization': 'Basic ' + base64.b64encode(
        b'integration:integration-secret').decode()}
    text = 'John works at Google in New York.'
    response = client.post('/api/predict', json={'text': text}, headers=headers)
    assert response.status_code == 200
    assert response.json == {'text': text, 'entities': service.predict(text)}
    assert client.post('/predict', data={'text': text}, headers=headers).status_code == 200
    assert client.post('/api/predict', json={'text': text}).status_code == 401
    response = client.post('/api/predict', json={'text': 'hello ' * 511}, headers=headers)
    assert response.status_code == 400


def test_unicode_and_no_entities(service):
    for text in ('こんにちは世界', 'hello', '😀'):
        assert isinstance(service.predict(text), list)
