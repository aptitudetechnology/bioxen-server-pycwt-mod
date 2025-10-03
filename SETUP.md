# PyCWT REST API - Setup Guide

Complete guide to setting up the PyCWT REST API development environment.

## Quick Setup (Recommended)

### Option 1: One-Command Setup
```bash
sudo apt install python3-venv  # If not already installed
bash quick-setup.sh
```

This will:
- Create virtual environment
- Install all dependencies (FastAPI, numpy, scipy, etc.)
- Create configuration file
- Run diagnostics

### Option 2: Manual Setup

Follow the steps below if you prefer manual control.

---

## Manual Setup Instructions

### Prerequisites

Install python3-venv if not already installed:
```bash
sudo apt install python3-venv
```

## Setup Steps

### 1. Create Virtual Environment
```bash
cd /home/chris/pycwt-mod
python3 -m venv .venv
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install Server Dependencies
```bash
pip install -r server/requirements.txt
```

### 5. Create Configuration File
```bash
cp server/.env.example server/.env
```

### 6. Verify Setup
```bash
python diagnose-server.py
```

All checks should pass (except possibly some backends if dependencies aren't installed).

### 7. Start the Server
```bash
python -m server.main
```

The server will start at http://localhost:8000

### 8. Test the Server
In a new terminal (with venv activated):
```bash
source .venv/bin/activate
python test-server.py
```

Or visit http://localhost:8000/docs in your browser.

## Common Issues

### "externally-managed-environment" error
- **Cause**: Trying to install packages system-wide
- **Solution**: Always activate the virtual environment first: `source .venv/bin/activate`

### "No module named 'pycwt_mod'" error
- **Cause**: The src directory is not in Python path
- **Solution**: The server code now handles this automatically

### Virtual environment creation fails
- **Cause**: python3-venv not installed
- **Solution**: Run `sudo apt install python3-venv`

## Daily Workflow

Every time you work on the project:

```bash
# 1. Navigate to project
cd /home/chris/pycwt-mod

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Start server
python -m server.main

# When done, deactivate
deactivate
```

## Automated Setup (Alternative)

If python3-venv is installed, you can use:
```bash
bash setup-dev.sh
```

This will automatically create the venv, install dependencies, and run diagnostics.
