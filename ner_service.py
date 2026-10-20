"""Shared, bounded NER inference for Flask and Streamlit."""
import os
from pathlib import Path
from threading import Lock


class ModelUnavailable(RuntimeError):
    pass


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' must be a non-empty string")
    if len(text) > 10000:
        raise ValueError("'text' must not exceed 10000 characters")


class NERService:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or os.environ.get('NER_MODEL_PATH',
            str(Path(__file__).resolve().parent / 'ner-bert-model'))).expanduser().resolve()
        self._pipeline = None
        self._lock = Lock()

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        path = self.model_path
        if not (path / 'config.json').is_file() or not any((path / name).is_file() for name in (
            'model.safetensors', 'pytorch_model.bin',
            'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
            raise ModelUnavailable('Model directory must contain config and trained weights')
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(str(path), local_files_only=True)
            self._pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
        except Exception as error:
            raise ModelUnavailable('Unable to load local model and tokenizer') from error
        return self._pipeline

    def _run(self, text):
        validate_text(text)
        with self._lock:
            ner = self._load()
            limit = min(ner.tokenizer.model_max_length, ner.model.config.max_position_embeddings)
            tokens = ner.tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
            if len(tokens) > limit:
                raise ValueError(f"'text' exceeds the model limit of {limit} tokens")
            return ner(text)

    def predict(self, text):
        return [{'word': entity['word'], 'entity': entity['entity_group']} for entity in self._run(text)]

    def predict_detailed(self, text):
        """Entities with character offsets and confidence, for highlighting UIs."""
        return [{'word': entity['word'], 'entity': entity['entity_group'], 'score': float(entity['score']),
                 'start': entity['start'], 'end': entity['end']} for entity in self._run(text)]
