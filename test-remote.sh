#!/usr/bin/env bash
# All-in-one remote testing script for PyCWT REST API

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       PyCWT REST API - Remote Server Test                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Activate venv
if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Run: bash quick-setup.sh"
    exit 1
fi

source .venv/bin/activate

# Check if server is already running
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Server already running on port 8000${NC}"
    EXISTING_SERVER=true
else
    EXISTING_SERVER=false
fi

# Start server if not running
if [ "$EXISTING_SERVER" = false ]; then
    echo -e "${BLUE}Starting server in background...${NC}"
    nohup python -m server.main > server-test.log 2>&1 &
    SERVER_PID=$!
    echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
    
    # Wait for server to be ready
    echo -e "${BLUE}Waiting for server to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is ready${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo -e "${RED}✗ Server failed to start${NC}"
            cat server-test.log
            exit 1
        fi
    done
else
    echo -e "${GREEN}✓ Using existing server${NC}"
fi

# Run tests
echo ""
echo -e "${BLUE}Running test suite...${NC}"
echo ""

python test-server.py
TEST_RESULT=$?

# Cleanup
if [ "$EXISTING_SERVER" = false ]; then
    echo ""
    echo -e "${BLUE}Stopping test server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Server stopped${NC}"
    
    # Show logs if tests failed
    if [ $TEST_RESULT -ne 0 ]; then
        echo ""
        echo -e "${YELLOW}Server logs:${NC}"
        tail -20 server-test.log
    fi
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}Phase 1 is complete and working!${NC}"
    echo ""
    echo -e "${BLUE}To run the server manually:${NC}"
    echo -e "  source .venv/bin/activate"
    echo -e "  python -m server.main"
    echo ""
else
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Check the output above for details${NC}"
    echo ""
fi

exit $TEST_RESULT
