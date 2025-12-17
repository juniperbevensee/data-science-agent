# Configuration settings

# LM Studio server (OpenAI-compatible API)
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "local-model"  # LM Studio ignores this, uses loaded model

# Sandboxed workspace directory
WORKSPACE_DIR = "workspace"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# AWS/Botocore logging control (defaults to suppressed)
# Set to False to see botocore DEBUG logs
SUPPRESS_AWS_LOGGING = True

