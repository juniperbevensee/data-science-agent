from flask import Flask
import os
import logging
import config


def create_app():
    app = Flask(__name__)
    
    # Configure verbose logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    app.logger.setLevel(logging.DEBUG)
    
    # Ensure workspace directory exists
    workspace_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.WORKSPACE_DIR)
    os.makedirs(workspace_path, exist_ok=True)
    
    # Register routes
    from app.routes import bp
    app.register_blueprint(bp)
    
    return app

