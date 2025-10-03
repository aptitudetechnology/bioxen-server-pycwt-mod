#!/usr/bin/env python3
"""Diagnostic script to verify server setup and dependencies."""

import sys
import os
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_header(text):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'='*60}{NC}")

def check_pass(text):
    print(f"{GREEN}✓{NC} {text}")

def check_fail(text):
    print(f"{RED}✗{NC} {text}")

def check_warn(text):
    print(f"{YELLOW}⚠{NC} {text}")

def check_python_version():
    """Check Python version."""
    print_header("Python Environment")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        check_pass(f"Python {version.major}.{version.minor} is supported")
    else:
        check_fail(f"Python {version.major}.{version.minor} is too old. Need >= 3.8")
        return False
    return True

def check_dependencies():
    """Check required dependencies."""
    print_header("Server Dependencies")
    
    dependencies = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'pydantic': 'Data validation',
        'pydantic_settings': 'Settings management',
    }
    
    all_present = True
    for module, description in dependencies.items():
        try:
            __import__(module)
            check_pass(f"{module}: {description}")
        except ImportError:
            check_fail(f"{module} not installed ({description})")
            all_present = False
    
    return all_present

def check_pycwt_mod():
    """Check pycwt_mod backend system."""
    print_header("PyCWT Backend System")
    
    try:
        from pycwt_mod.backends import list_backends, get_backend
        check_pass("pycwt_mod.backends module found")
        
        backends = list_backends()
        print(f"\nRegistered backends: {len(backends)}")
        
        for backend_name in backends:
            try:
                backend = get_backend(backend_name)
                available = backend.is_available()
                if available:
                    check_pass(f"{backend_name}: available")
                else:
                    check_warn(f"{backend_name}: registered but not available")
            except Exception as e:
                check_fail(f"{backend_name}: error - {e}")
        
        return True
    except ImportError as e:
        check_fail(f"Cannot import pycwt_mod.backends: {e}")
        return False

def check_server_structure():
    """Check server directory structure."""
    print_header("Server Directory Structure")
    
    base_path = Path(__file__).parent / "server"
    
    required_files = [
        "main.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "core/config.py",
        "api/routes/backends.py",
        "tests/test_phase1.py",
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            check_pass(f"{file_path}")
        else:
            check_fail(f"{file_path} not found")
            all_present = False
    
    return all_present

def check_configuration():
    """Check server configuration."""
    print_header("Server Configuration")
    
    env_example = Path(__file__).parent / "server" / ".env.example"
    env_file = Path(__file__).parent / "server" / ".env"
    
    if env_example.exists():
        check_pass(".env.example template found")
    else:
        check_fail(".env.example not found")
    
    if env_file.exists():
        check_pass(".env configuration file found")
    else:
        check_warn(".env not found (will use defaults)")
        print(f"   Run: cp server/.env.example server/.env")

def check_port_availability():
    """Check if default port is available."""
    print_header("Port Availability")
    
    import socket
    
    port = 8000
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            check_pass(f"Port {port} is available")
            return True
    except OSError:
        check_warn(f"Port {port} is in use (server may already be running)")
        print(f"   Check with: lsof -i :{port}")
        return False

def run_quick_test():
    """Run a quick import test of the server."""
    print_header("Quick Import Test")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from server.core.config import Settings
        check_pass("Config module imports correctly")
        
        settings = Settings()
        print(f"   Host: {settings.HOST}")
        print(f"   Port: {settings.PORT}")
        print(f"   Default backend: {settings.DEFAULT_BACKEND}")
        
        return True
    except Exception as e:
        check_fail(f"Import test failed: {e}")
        return False

def print_summary(results):
    """Print summary and next steps."""
    print_header("Summary")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n{GREEN}✓ All checks passed!{NC}\n")
        print("Next steps:")
        print("1. Install dependencies: pip install -r server/requirements.txt")
        print("2. Start server: python -m server.main")
        print("3. Visit: http://localhost:8000/docs")
        print("4. Run tests: pytest server/tests/test_phase1.py")
    else:
        print(f"\n{RED}✗ Some checks failed{NC}\n")
        print("Failed checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  - {check}")
        print("\nRefer to TESTING.md for troubleshooting guidance.")

def main():
    """Run all diagnostic checks."""
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       PyCWT REST API Server - Diagnostic Tool             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{NC}")
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'PyCWT Backend System': check_pycwt_mod(),
        'Server Structure': check_server_structure(),
        'Configuration': True,  # Always pass (warnings only)
        'Port Availability': True,  # Always pass (warnings only)
        'Import Test': run_quick_test(),
    }
    
    check_configuration()
    check_port_availability()
    
    print_summary(results)

if __name__ == "__main__":
    main()
