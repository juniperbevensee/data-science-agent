#!/usr/bin/env python3
# Run with: python3 run.py
from app import create_app
import config

app = create_app()

if __name__ == '__main__':
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

