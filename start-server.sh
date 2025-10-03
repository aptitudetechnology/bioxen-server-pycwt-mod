#!/usr/bin/env bash
# Startup script for PyCWT REST API server

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PyCWT REST API Server...${NC}"

# Check if .env exists
if [ ! -f "server/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from template...${NC}"
    cp server/.env.example server/.env
    echo -e "${GREEN}Created .env file. Please review and update if needed.${NC}"
fi

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r server/requirements.txt
fi

# Run the server
echo -e "${GREEN}Server starting at http://localhost:8000${NC}"
echo -e "${GREEN}API Documentation: http://localhost:8000/docs${NC}"
echo -e "${GREEN}Health Check: http://localhost:8000/health${NC}"
echo ""

python -m server.main
