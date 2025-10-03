#!/usr/bin/env bash
# Remote server testing guide for PyCWT REST API

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         PyCWT REST API - Remote Server Testing               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

You're testing on a remote server via SSH. Here's how to proceed:

═══════════════════════════════════════════════════════════════
METHOD 1: Terminal Testing (Recommended for SSH)
═══════════════════════════════════════════════════════════════

Step 1: Start the server in background
────────────────────────────────────────────────
  source .venv/bin/activate
  nohup python -m server.main > server.log 2>&1 &
  
  # Note the process ID (PID) to stop it later

Step 2: Wait a moment for server to start
────────────────────────────────────────────────
  sleep 2

Step 3: Test with curl (in same terminal)
────────────────────────────────────────────────
  # Health check
  curl http://localhost:8000/health
  
  # List backends
  curl http://localhost:8000/api/v1/backends/ | jq
  
  # Get specific backend
  curl http://localhost:8000/api/v1/backends/sequential | jq

Step 4: Run automated test client
────────────────────────────────────────────────
  python test-server.py

Step 5: Check server logs
────────────────────────────────────────────────
  tail -f server.log

Step 6: Stop the server when done
────────────────────────────────────────────────
  pkill -f "python -m server.main"
  # or: kill <PID>

═══════════════════════════════════════════════════════════════
METHOD 2: SSH Port Forwarding (Access from Local Browser)
═══════════════════════════════════════════════════════════════

From your LOCAL machine, reconnect with port forwarding:
────────────────────────────────────────────────
  ssh -L 8000:localhost:8000 user@remote-server

Then on the REMOTE server:
────────────────────────────────────────────────
  source .venv/bin/activate
  python -m server.main

Now on your LOCAL machine, open browser to:
────────────────────────────────────────────────
  http://localhost:8000/docs

═══════════════════════════════════════════════════════════════
METHOD 3: Use tmux/screen (Persistent Sessions)
═══════════════════════════════════════════════════════════════

Start tmux session:
────────────────────────────────────────────────
  tmux new -s pycwt-server

Start server in tmux:
────────────────────────────────────────────────
  source .venv/bin/activate
  python -m server.main

Detach from tmux:
────────────────────────────────────────────────
  Press: Ctrl+B, then D

Run tests in main terminal:
────────────────────────────────────────────────
  source .venv/bin/activate
  python test-server.py

Reattach to see server logs:
────────────────────────────────────────────────
  tmux attach -t pycwt-server

Kill tmux session when done:
────────────────────────────────────────────────
  tmux kill-session -t pycwt-server

═══════════════════════════════════════════════════════════════
QUICK REFERENCE: Testing Commands
═══════════════════════════════════════════════════════════════

# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/

# List all backends
curl http://localhost:8000/api/v1/backends/

# Get backend details
curl http://localhost:8000/api/v1/backends/sequential

# Pretty print JSON (requires jq)
curl http://localhost:8000/api/v1/backends/ | jq

# Check if server is running
ps aux | grep "server.main"

# Check port 8000
lsof -i :8000
netstat -tlnp | grep 8000

# View server logs (if using nohup)
tail -f server.log

# Stop server
pkill -f "python -m server.main"

═══════════════════════════════════════════════════════════════
AUTOMATED TESTING (Best for Remote)
═══════════════════════════════════════════════════════════════

The test-server.py script is perfect for SSH testing:
────────────────────────────────────────────────
  # Terminal 1: Start server
  source .venv/bin/activate
  python -m server.main
  
  # Terminal 2: Run tests
  source .venv/bin/activate
  python test-server.py

Or use the all-in-one test script:
────────────────────────────────────────────────
  bash test-remote.sh

═══════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════

Server won't start?
  - Check if port is in use: lsof -i :8000
  - Check server.log for errors: tail server.log

Can't connect?
  - Verify server is running: ps aux | grep server.main
  - Check firewall: sudo ufw status
  - Test locally first: curl http://localhost:8000/health

Tests failing?
  - Make sure server is started first
  - Check both terminals are using the venv
  - Review logs: tail -f server.log

═══════════════════════════════════════════════════════════════

TIP: For quick testing on remote servers, I recommend:
  1. Use tmux to run server in background
  2. Use test-server.py for automated validation
  3. Use curl for quick manual checks

Ready to start? Run:
  bash test-remote.sh

EOF
