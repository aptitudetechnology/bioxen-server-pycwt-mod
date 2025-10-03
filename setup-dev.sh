#!/usr/bin/env bash
# Setup script for PyCWT REST API development environment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       PyCWT REST API - Development Setup                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check for virtual environment
echo -e "${BLUE}[1/5] Checking Python virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    if ! python3 -m venv .venv 2>/dev/null; then
        echo -e "${YELLOW}⚠ python3-venv not installed${NC}"
        echo -e "${YELLOW}Please run: sudo apt install python3-venv${NC}"
        echo -e "${YELLOW}Then re-run this script${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Step 2: Activate virtual environment
echo -e "\n${BLUE}[2/5] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 3: Upgrade pip
echo -e "\n${BLUE}[3/5] Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 4: Install dependencies
echo -e "\n${BLUE}[4/5] Installing dependencies...${NC}"
pip install -r server/requirements.txt
echo -e "${GREEN}✓ Server dependencies installed${NC}"

# Step 5: Create .env if it doesn't exist
echo -e "\n${BLUE}[5/5] Setting up configuration...${NC}"
if [ ! -f "server/.env" ]; then
    cp server/.env.example server/.env
    echo -e "${GREEN}✓ Created server/.env from template${NC}"
else
    echo -e "${GREEN}✓ server/.env already exists${NC}"
fi

# Run diagnostics
echo -e "\n${BLUE}Running diagnostics...${NC}"
python diagnose-server.py

echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "  2. Start the server:"
echo -e "     ${YELLOW}python -m server.main${NC}"
echo ""
echo -e "  3. Visit the API documentation:"
echo -e "     ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "  4. Run tests:"
echo -e "     ${YELLOW}python test-server.py${NC}"
echo ""
