#!/usr/bin/env bash
# One-command setup for PyCWT REST API

set -e

echo "================================================================"
echo "  PyCWT REST API - Quick Setup"
echo "================================================================"
echo ""

# Check if python3-venv is available
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "❌ python3-venv is not installed"
    echo ""
    echo "Please install it first:"
    echo "  sudo apt install python3-venv"
    echo ""
    exit 1
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate and install
echo ""
echo "📥 Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r server/requirements.txt --quiet

# Create .env
if [ ! -f "server/.env" ]; then
    cp server/.env.example server/.env
    echo "✓ Created server/.env"
fi

# Run diagnostic
echo ""
echo "🔍 Running diagnostics..."
echo ""
python diagnose-server.py

echo ""
echo "================================================================"
echo "  Setup Complete!"
echo "================================================================"
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  python -m server.main"
echo ""
