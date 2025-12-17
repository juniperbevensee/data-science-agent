#!/bin/bash

# Data Science Agent Startup Script

set -e  # Exit on error

echo "=================================="
echo "Data Science Agent - Starting..."
echo "=================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create workspace directory if it doesn't exist
if [ ! -d "workspace" ]; then
    echo "Creating workspace directory..."
    mkdir -p workspace
fi

echo "✓ Workspace directory ready"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install/update dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo "✓ All dependencies installed"

# Check if LM Studio is running (optional warning)
echo ""
echo "Checking configuration..."
if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "⚠ Warning: LM Studio doesn't appear to be running on localhost:1234"
    echo "  Please make sure LM Studio is running, or update LLM_BASE_URL in config.py"
    echo ""
fi

# Start the Flask application
echo "=================================="
echo "Starting Flask server..."
echo "Server will be available at http://0.0.0.0:5000"
echo "Press Ctrl+C to stop"
echo "=================================="
echo ""

python3 run.py
