#!/usr/bin/env python3
# Run with: python3 run.py

# Configure logging BEFORE any imports to suppress verbose library output
import logging
import os

# Suppress matplotlib DEBUG spam
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Suppress NLTK/urllib SSL warnings that we handle with fallback
import warnings
warnings.filterwarnings('ignore', message='.*SSL.*')
warnings.filterwarnings('ignore', message='.*certificate.*')

# Set matplotlib backend before it's imported anywhere
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from app import create_app
import config

app = create_app()

if __name__ == '__main__':
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )

