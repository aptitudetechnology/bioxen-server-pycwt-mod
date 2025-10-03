#!/usr/bin/env bash
# Quick curl-based testing for remote servers

BASE_URL="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       PyCWT REST API - Quick curl Tests                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if server is running
if ! curl -s $BASE_URL/health > /dev/null 2>&1; then
    echo -e "${RED}✗ Server is not running or not accessible${NC}"
    echo ""
    echo "Start the server first:"
    echo "  source .venv/bin/activate"
    echo "  python -m server.main"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Test 1: Health check
echo -e "${BLUE}[Test 1] Health Check${NC}"
echo "GET $BASE_URL/health"
curl -s $BASE_URL/health | jq 2>/dev/null || curl -s $BASE_URL/health
echo ""
echo ""

# Test 2: Root endpoint
echo -e "${BLUE}[Test 2] Root Endpoint${NC}"
echo "GET $BASE_URL/"
curl -s $BASE_URL/ | jq 2>/dev/null || curl -s $BASE_URL/
echo ""
echo ""

# Test 3: List backends
echo -e "${BLUE}[Test 3] List All Backends${NC}"
echo "GET $BASE_URL/api/v1/backends/"
curl -s $BASE_URL/api/v1/backends/ | jq 2>/dev/null || curl -s $BASE_URL/api/v1/backends/
echo ""
echo ""

# Test 4: Get sequential backend
echo -e "${BLUE}[Test 4] Get Sequential Backend Details${NC}"
echo "GET $BASE_URL/api/v1/backends/sequential"
curl -s $BASE_URL/api/v1/backends/sequential | jq 2>/dev/null || curl -s $BASE_URL/api/v1/backends/sequential
echo ""
echo ""

# Test 5: Get joblib backend
echo -e "${BLUE}[Test 5] Get Joblib Backend Details${NC}"
echo "GET $BASE_URL/api/v1/backends/joblib"
curl -s $BASE_URL/api/v1/backends/joblib | jq 2>/dev/null || curl -s $BASE_URL/api/v1/backends/joblib
echo ""
echo ""

# Test 6: Invalid backend (should return 404)
echo -e "${BLUE}[Test 6] Invalid Backend (expect 404)${NC}"
echo "GET $BASE_URL/api/v1/backends/invalid"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/api/v1/backends/invalid)
if [ "$HTTP_CODE" = "404" ]; then
    echo -e "${GREEN}✓ Correctly returned 404${NC}"
else
    echo -e "${RED}✗ Expected 404, got $HTTP_CODE${NC}"
fi
echo ""

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Manual tests complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Tip: Install jq for pretty JSON output:${NC}"
echo "  sudo apt install jq"
echo ""
