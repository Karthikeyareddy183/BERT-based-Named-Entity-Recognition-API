"""Flask NER interface; deploy with a WSGI server."""
import os
from flask import Flask, jsonify, render_template, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.security import check_password_hash
from ner_service import NERService, ModelUnavailable, validate_text


def create_app(config=None, service=None):
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=65536,
        NER_USERNAME=os.environ.get('NER_USERNAME'),
        NER_PASSWORD_HASH=os.environ.get('NER_PASSWORD_HASH'))
    if config:
        app.config.update(config)
    if not app.config['NER_USERNAME'] or not app.config['NER_PASSWORD_HASH']:
        raise RuntimeError('Set NER_USERNAME and NER_PASSWORD_HASH before starting the API')
    service = service if service is not None else NERService()
    auth = HTTPBasicAuth()

    @auth.verify_password
    def verify_password(username, password):
        if username == app.config['NER_USERNAME'] and check_password_hash(app.config['NER_PASSWORD_HASH'], password):
            return username
        return None

    @auth.error_handler
    def authentication_error(status):
        return jsonify(error='Authentication required'), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error='Request body too large'), 413

    def run_prediction(text):
        try:
            validate_text(text)
            return service.predict(text), None
        except ValueError as error:
            return None, (jsonify(error=str(error)), 400)
        except ModelUnavailable:
            app.logger.exception('NER model unavailable')
            return None, (jsonify(error='NER model is unavailable'), 503)
        except Exception:
            app.logger.exception('NER inference failed')
            return None, (jsonify(error='NER inference failed'), 500)

    @app.get('/')
    def home():
        return render_template('home.html')

    @app.route('/predict', methods=['GET', 'POST'])
    @auth.login_required
    def predict():
        if request.method == 'GET':
            return render_template('predict.html', text='', entities=[])
        text = request.form.get('text')
        entities, error = run_prediction(text)
        if error is not None:
            return error
        return render_template('predict.html', text=text, entities=entities)

    @app.post('/api/predict')
    @auth.login_required
    def predict_api():
        if not request.is_json:
            return jsonify(error='Content-Type must be application/json'), 415
        try:
            data = request.get_json()
        except BadRequest:
            return jsonify(error='Malformed JSON'), 400
        if not isinstance(data, dict) or 'text' not in data:
            return jsonify(error="JSON input must be an object with a 'text' field"), 400
        entities, error = run_prediction(data['text'])
        if error is not None:
            return error
        return jsonify(text=data['text'], entities=entities)

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
