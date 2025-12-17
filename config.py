# Configuration settings
import os

# LLM Provider: "local", "openai", "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

# Local (LM Studio)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")

# API Keys (set via environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Sandboxed workspace directory
WORKSPACE_DIR = "workspace"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

