#!/usr/bin/env bash
# Quick activation script for the development environment

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: python3 -m venv .venv"
    echo "Or see SETUP.md for full instructions"
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "✓ Virtual environment activated"
echo ""
echo "Available commands:"
echo "  python -m server.main     - Start the API server"
echo "  python diagnose-server.py - Run diagnostics"
echo "  python test-server.py     - Run integration tests"
echo "  pytest server/tests/      - Run unit tests"
echo "  deactivate                - Exit virtual environment"
echo ""
