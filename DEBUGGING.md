# Debugging findings

The application extracts named entities using a fine-tuned BERT token
classifier, aggregates token labels, and exposes Flask and Streamlit UIs.

| Problem | Root cause | Correction |
| --- | --- | --- |
| Flask crashes during import | Model loading is eager and the repository has only config.json, without weights or tokenizer | Shared lazy loader checks artifacts, uses local files, and reports 503 |
| Documented API call fails | README sends JSON to the form endpoint /predict and omits authentication | Document authenticated /api/predict |
| Invalid JSON values cause unexpected errors or unintended batched inference | API checks only membership of text, without object or string validation | Validate JSON structure, nonblank strings, content type, and size |
| Long input can lose entities | Token-classification pipeline may truncate beyond the model context | Check token count including special tokens and reject oversized inputs |
| Known credentials permit access and the form bypasses API protection | Hard-coded admin/password and unprotected form route | Require environment credentials with a password hash; protect both prediction routes |
| Debug server runs in Docker | Docker starts app.py with debug=True | Use Waitress and disable development debugging |
| Streamlit fails on other machines | Hard-coded D drive and omitted dependency | Shared configurable model path and explicit Streamlit dependency |

Regression tests verify 9 test cases and 12 invalid-input subtests.
Dependency compatibility and all application imports pass in .venv.

The missing-artifact blocker was resolved by copying all six files from
`D:\PROJECTS\NER\ner-bert-model` and verifying matching SHA-256 hashes.
Real inference identifies John as PER, Google as ORG, and New York as LOC.
Integration tests cover authenticated API and form predictions, Unicode input,
and rejection above the 512-token limit. Dataset-level accuracy, load testing,
and container execution remain unverified.
