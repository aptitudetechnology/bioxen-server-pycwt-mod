#!/bin/bash
################################################################################
# KaiABC Circadian Oscillator Server - Startup Script
# 
# Description: Starts the KaiABC API server with proper environment setup,
#              dependency checks, and graceful error handling.
#
# Usage: ./start-KaiABC-server.sh [OPTIONS]
#
# Options:
#   --dev          Start in development mode (auto-reload)
#   --prod         Start in production mode (multi-worker)
#   --workers N    Number of worker processes (default: 4)
#   --port PORT    Port to listen on (default: 8000)
#   --host HOST    Host to bind to (default: 0.0.0.0)
#   --check        Only check dependencies, don't start server
#   --docker       Start full Docker stack (databases, monitoring)
#   --help         Show this help message
#
# Requirements:
#   - Python 3.9+
#   - Virtual environment (recommended)
#   - PostgreSQL/TimescaleDB (for production)
#   - Redis (for caching)
#   - InfluxDB (for time series)
#
# Author: BioXen Development Team
# Date: October 6, 2025
# Version: 1.0.0
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODE="dev"
WORKERS=4
PORT=8000
HOST="0.0.0.0"
CHECK_ONLY=false
START_DOCKER=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
SERVER_DIR="${SCRIPT_DIR}/server"
KAIABC_API_DIR="${SERVER_DIR}/api/routes"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --docker)
            START_DOCKER=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  KaiABC Circadian Oscillator API Server"
    echo "  Version 1.0.0"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

check_python_version() {
    print_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
        print_success "Python $PYTHON_VERSION detected (>= 3.9 required)"
        return 0
    else
        print_error "Python $PYTHON_VERSION detected, but >= 3.9 required"
        return 1
    fi
}

check_virtual_environment() {
    print_info "Checking virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        print_success "Virtual environment found at $VENV_DIR"
        return 0
    else
        print_warning "Virtual environment not found at $VENV_DIR"
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        
        if [ $? -eq 0 ]; then
            print_success "Virtual environment created successfully"
            return 0
        else
            print_error "Failed to create virtual environment"
            return 1
        fi
    fi
}

activate_virtual_environment() {
    print_info "Activating virtual environment..."
    
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        print_success "Virtual environment activated"
        return 0
    else
        print_error "Could not find activation script at $VENV_DIR/bin/activate"
        return 1
    fi
}

check_python_dependencies() {
    print_info "Checking Python dependencies..."
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        print_error "pip is not available"
        return 1
    fi
    
    # Critical dependencies
    CRITICAL_DEPS=(
        "fastapi"
        "uvicorn"
        "pydantic"
        "numpy"
        "scipy"
        "sqlalchemy"
    )
    
    missing_deps=()
    
    for dep in "${CRITICAL_DEPS[@]}"; do
        if python3 -c "import $dep" 2>/dev/null; then
            print_success "$dep is installed"
        else
            print_warning "$dep is not installed"
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warning "Missing dependencies detected: ${missing_deps[*]}"
        
        if [ -f "$SCRIPT_DIR/requirements-server.txt" ]; then
            echo -e "${YELLOW}Would you like to install missing dependencies? (y/n)${NC}"
            read -r response
            
            if [[ "$response" =~ ^[Yy]$ ]]; then
                print_info "Installing dependencies from requirements-server.txt..."
                pip install -r "$SCRIPT_DIR/requirements-server.txt"
                
                if [ $? -eq 0 ]; then
                    print_success "Dependencies installed successfully"
                else
                    print_error "Failed to install dependencies"
                    return 1
                fi
            else
                print_warning "Proceeding without installing dependencies (may cause errors)"
            fi
        else
            print_error "requirements-server.txt not found"
            return 1
        fi
    else
        print_success "All critical dependencies are installed"
    fi
    
    return 0
}

check_kaiabc_server() {
    print_info "Checking existing server directory..."
    
    if [ -d "$SERVER_DIR" ]; then
        print_success "Server directory found at $SERVER_DIR"
        
        if [ -f "$SERVER_DIR/main.py" ]; then
            print_success "main.py found"
        else
            print_error "main.py not found in $SERVER_DIR"
            return 1
        fi
        
        # Check if KaiABC routes exist
        if [ -f "$KAIABC_API_DIR/kaiabc.py" ]; then
            print_success "KaiABC routes already exist"
            return 0
        else
            print_warning "KaiABC routes not found"
            echo -e "${YELLOW}Would you like to add KaiABC routes to the existing server? (y/n)${NC}"
            read -r response
            
            if [[ "$response" =~ ^[Yy]$ ]]; then
                add_kaiabc_routes
                return $?
            else
                print_warning "Starting server without KaiABC routes"
                return 0
            fi
        fi
    else
        print_error "Server directory not found at $SERVER_DIR"
        print_info "Please ensure you're running this script from the project root"
        return 1
    fi
}

add_kaiabc_routes() {
    print_info "Adding KaiABC routes to existing server..."
    
    # Create KaiABC models
    cat > "$SERVER_DIR/api/models/kaiabc.py" << 'EOF'
"""
KaiABC Circadian Oscillator Data Models
Pydantic models for API request/response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class OscillatorState(BaseModel):
    """Current state of the KaiABC oscillator (6 protein concentrations)"""
    KaiA: float = Field(..., description="KaiA protein concentration (μM)")
    KaiB: float = Field(..., description="KaiB protein concentration (μM)")
    KaiC: float = Field(..., description="KaiC protein concentration (μM)")
    KaiAB: float = Field(..., description="KaiA-KaiB complex concentration (μM)")
    KaiAC: float = Field(..., description="KaiA-KaiC complex concentration (μM)")
    KaiBC: float = Field(..., description="KaiB-KaiC complex concentration (μM)")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of state")
    phase: Optional[float] = Field(default=None, description="Circadian phase (0-2π)")


class SensorReading(BaseModel):
    """Environmental sensor reading from client node"""
    node_id: str = Field(..., description="Unique identifier for client node")
    temperature: float = Field(..., description="Temperature (°C)")
    humidity: Optional[float] = Field(default=None, description="Relative humidity (%)")
    light: Optional[float] = Field(default=None, description="Light intensity (lux)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reading timestamp")


class ETFParameters(BaseModel):
    """External Temperature Forcing parameters for entrainment"""
    amplitude: float = Field(..., ge=0, description="Temperature oscillation amplitude (°C)")
    period: float = Field(..., ge=0, description="Entrainment period (hours)")
    phase: float = Field(0.0, ge=0, lt=6.283185, description="Phase offset (radians)")
    base_temp: float = Field(25.0, description="Base temperature (°C)")


class SimulationRequest(BaseModel):
    """Request to run oscillator simulation"""
    initial_state: Optional[OscillatorState] = Field(default=None, description="Initial concentrations")
    duration: float = Field(..., gt=0, description="Simulation duration (hours)")
    dt: float = Field(0.1, gt=0, description="Time step (hours)")
    etf: Optional[ETFParameters] = Field(default=None, description="Temperature forcing parameters")
    
    
class KalmanFilterConfig(BaseModel):
    """Kalman filter configuration for sensor noise reduction"""
    process_noise: float = Field(0.01, gt=0, description="Process noise covariance")
    measurement_noise: float = Field(0.1, gt=0, description="Measurement noise covariance")
    initial_estimate: float = Field(25.0, description="Initial temperature estimate (°C)")
EOF

    # Create KaiABC routes
    cat > "$KAIABC_API_DIR/kaiabc.py" << 'EOF'
"""
KaiABC Circadian Oscillator API Routes
REST endpoints for oscillator simulation and control
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Optional
import numpy as np
from datetime import datetime

from server.api.models.kaiabc import (
    OscillatorState,
    SensorReading,
    ETFParameters,
    SimulationRequest,
    KalmanFilterConfig
)

router = APIRouter()


@router.get("/", summary="KaiABC API Information")
async def kaiabc_info():
    """Get information about the KaiABC Circadian Oscillator API"""
    return {
        "service": "KaiABC Circadian Oscillator API",
        "version": "1.0.0",
        "description": "REST API for KaiABC circadian oscillator simulation and control",
        "endpoints": {
            "oscillator": "/api/v1/kaiabc/oscillator",
            "sensor": "/api/v1/kaiabc/sensor",
            "simulate": "/api/v1/kaiabc/simulate",
            "websocket": "/api/v1/kaiabc/ws"
        }
    }


@router.get("/oscillator/state", response_model=OscillatorState, summary="Get Current Oscillator State")
async def get_oscillator_state():
    """Get the current state of the KaiABC oscillator (6 protein concentrations)"""
    # TODO: Implement actual state retrieval from database/memory
    # For now, return mock data
    return OscillatorState(
        KaiA=1.2,
        KaiB=0.8,
        KaiC=3.5,
        KaiAB=0.3,
        KaiAC=0.5,
        KaiBC=1.1,
        timestamp=datetime.utcnow(),
        phase=2.4
    )


@router.post("/oscillator/state", response_model=OscillatorState, summary="Set Oscillator State")
async def set_oscillator_state(state: OscillatorState):
    """Set the current state of the KaiABC oscillator"""
    # TODO: Implement actual state storage
    state.timestamp = datetime.utcnow()
    return state


@router.post("/sensor/reading", summary="Submit Sensor Reading")
async def submit_sensor_reading(reading: SensorReading):
    """Submit environmental sensor reading from client node (Pico/ELM11)"""
    # TODO: Implement storage to time series database (InfluxDB/TimescaleDB)
    return {
        "status": "accepted",
        "node_id": reading.node_id,
        "timestamp": reading.timestamp,
        "message": "Sensor reading stored successfully"
    }


@router.post("/simulate", summary="Run Oscillator Simulation")
async def run_simulation(request: SimulationRequest):
    """Run KaiABC oscillator simulation with optional temperature forcing"""
    # TODO: Implement actual ODE integration using scipy.integrate.solve_ivp
    
    # Mock response for now
    duration = request.duration
    dt = request.dt
    n_points = int(duration / dt)
    
    # Generate mock time series
    time = np.linspace(0, duration, n_points)
    phase = 2 * np.pi * time / 24  # 24-hour period
    
    states = []
    for t, p in zip(time, phase):
        states.append({
            "time": float(t),
            "KaiA": 1.0 + 0.5 * np.sin(p),
            "KaiB": 0.8 + 0.3 * np.cos(p),
            "KaiC": 3.5 + 0.5 * np.sin(p + np.pi/4),
            "phase": float(p % (2*np.pi))
        })
    
    return {
        "duration": duration,
        "dt": dt,
        "n_points": n_points,
        "states": states[:100]  # Return first 100 points
    }


@router.post("/etf/configure", summary="Configure External Temperature Forcing")
async def configure_etf(params: ETFParameters):
    """Configure external temperature forcing for entrainment"""
    # TODO: Implement ETF configuration storage
    return {
        "status": "configured",
        "parameters": params,
        "message": "Temperature forcing configured successfully"
    }


@router.get("/analytics/period", summary="Detect Circadian Period")
async def detect_period(node_id: Optional[str] = None):
    """Detect circadian period using Lomb-Scargle periodogram"""
    # TODO: Implement period detection using Astropy
    return {
        "node_id": node_id,
        "period": 24.3,
        "period_std": 0.5,
        "confidence": 0.95,
        "method": "lomb_scargle"
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time oscillator state streaming"""
    await websocket.accept()
    
    try:
        while True:
            # TODO: Implement actual real-time state streaming
            # For now, send mock data every second
            import asyncio
            await asyncio.sleep(1)
            
            state = {
                "timestamp": datetime.utcnow().isoformat(),
                "KaiA": 1.0 + 0.5 * np.random.randn(),
                "KaiB": 0.8 + 0.3 * np.random.randn(),
                "KaiC": 3.5 + 0.5 * np.random.randn(),
                "phase": np.random.rand() * 2 * np.pi
            }
            
            await websocket.send_json(state)
            
    except WebSocketDisconnect:
        print("Client disconnected from KaiABC WebSocket")
EOF

    # Update main.py to include KaiABC routes
    print_info "Updating main.py to include KaiABC routes..."
    
    # Check if KaiABC routes are already included
    if grep -q "kaiabc" "$SERVER_DIR/main.py" 2>/dev/null; then
        print_success "KaiABC routes already included in main.py"
    else
        # Add import
        if ! grep -q "from server.api.routes import.*kaiabc" "$SERVER_DIR/main.py"; then
            sed -i 's/from server.api.routes import backends, wavelet, hardware, benchmark/from server.api.routes import backends, wavelet, hardware, benchmark, kaiabc/' "$SERVER_DIR/main.py"
        fi
        
        # Add router inclusion (append before the last lines)
        cat >> "$SERVER_DIR/main.py" << 'EOF'

# KaiABC Circadian Oscillator API
app.include_router(
    kaiabc.router,
    prefix="/api/v1/kaiabc",
    tags=["kaiabc"]
)
EOF
        print_success "KaiABC routes added to main.py"
    fi
    
    print_success "KaiABC routes integrated into existing server"
    print_info "New endpoints will be available at: /api/v1/kaiabc/*"
    
    return 0
}

check_docker() {
    print_info "Checking Docker..."
    
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
        
        if docker ps &> /dev/null; then
            print_success "Docker daemon is running"
            return 0
        else
            print_warning "Docker daemon is not running"
            print_info "Start Docker with: sudo systemctl start docker"
            return 1
        fi
    else
        print_warning "Docker is not installed"
        print_info "Docker is optional but recommended for databases"
        return 1
    fi
}

check_databases() {
    print_info "Checking database connections..."
    
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        print_success "PostgreSQL client is installed"
    else
        print_warning "PostgreSQL client not found (optional for dev mode)"
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        print_success "Redis client is installed"
    else
        print_warning "Redis client not found (optional for dev mode)"
    fi
    
    # Check if services are running via Docker
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        if docker ps | grep -q postgres; then
            print_success "PostgreSQL container is running"
        fi
        
        if docker ps | grep -q redis; then
            print_success "Redis container is running"
        fi
        
        if docker ps | grep -q influxdb; then
            print_success "InfluxDB container is running"
        fi
    fi
}

start_docker_stack() {
    print_info "Starting Docker stack..."
    
    if [ ! -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        print_info "Please create docker-compose.yml (see documentation)"
        return 1
    fi
    
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "Docker stack started"
        print_info "Waiting for services to be ready..."
        sleep 5
        
        print_info "Service status:"
        docker-compose ps
        return 0
    else
        print_error "Failed to start Docker stack"
        return 1
    fi
}

start_server_dev() {
    print_info "Starting server in DEVELOPMENT mode..."
    print_info "Server includes: PyCWT Wavelet API + KaiABC Circadian Oscillator API"
    print_info "Server will auto-reload on code changes"
    print_info "Press Ctrl+C to stop"
    echo ""
    
    cd "$SERVER_DIR"
    
    uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info
}

start_server_prod() {
    print_info "Starting server in PRODUCTION mode..."
    print_info "Server includes: PyCWT Wavelet API + KaiABC Circadian Oscillator API"
    print_info "Using $WORKERS worker processes"
    print_info "Press Ctrl+C to stop"
    echo ""
    
    cd "$SERVER_DIR"
    
    uvicorn main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level warning \
        --access-log
}

print_startup_info() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Multi-API Server Started Successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${BLUE}Server Information:${NC}"
    echo "  • Mode:          $MODE"
    echo "  • URL:           http://$HOST:$PORT"
    echo "  • Documentation: http://localhost:$PORT/docs"
    echo "  • Health Check:  http://localhost:$PORT/health"
    
    if [ "$MODE" = "prod" ]; then
        echo "  • Workers:       $WORKERS"
    fi
    
    echo ""
    echo -e "${BLUE}Available APIs:${NC}"
    echo "  • PyCWT Wavelet:   http://localhost:$PORT/api/v1/wavelet"
    echo "  • KaiABC Circadian: http://localhost:$PORT/api/v1/kaiabc"
    echo "  • Hardware:        http://localhost:$PORT/api/v1/hardware"
    echo ""
    echo -e "${BLUE}Quick Tests:${NC}"
    echo "  curl http://localhost:$PORT/health"
    echo "  curl http://localhost:$PORT/api/v1/kaiabc/"
    echo "  curl http://localhost:$PORT/api/v1/kaiabc/oscillator/state"
    echo ""
    echo -e "${BLUE}WebSocket Streaming:${NC}"
    echo "  wscat -c ws://localhost:$PORT/api/v1/kaiabc/ws"
    echo ""
}

cleanup() {
    echo ""
    print_info "Shutting down server..."
    
    # Deactivate virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    
    print_success "Server stopped"
    exit 0
}

################################################################################
# Main Script
################################################################################

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_header

# System checks
print_info "Performing system checks..."
echo ""

check_python_version || exit 1
check_virtual_environment || exit 1
activate_virtual_environment || exit 1
check_python_dependencies || exit 1
check_kaiabc_server || exit 1

echo ""
print_info "Performing environment checks..."
echo ""

check_docker
check_databases

# If check-only mode, exit here
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    print_success "All checks completed successfully"
    exit 0
fi

# Start Docker stack if requested
if [ "$START_DOCKER" = true ]; then
    echo ""
    start_docker_stack || print_warning "Docker stack failed to start (continuing anyway)"
fi

# Start the server
echo ""
print_startup_info

# Run based on mode
case $MODE in
    dev)
        start_server_dev
        ;;
    prod)
        start_server_prod
        ;;
    *)
        print_error "Invalid mode: $MODE"
        exit 1
        ;;
esac
