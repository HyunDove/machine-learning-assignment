from flask import Flask
from flask_cors import CORS
from .config import Config
from .errors import register_error_handlers


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app)

    register_error_handlers(app)

    from .routes.prediction import bp as prediction_bp
    app.register_blueprint(prediction_bp, url_prefix="/api/predictions")

    return app
