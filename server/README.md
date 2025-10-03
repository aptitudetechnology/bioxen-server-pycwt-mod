# PyCWT REST API Server

REST API server providing HTTP endpoints for continuous wavelet transform analysis.

## Quick Start

### 1. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run the Server

```bash
# From the project root
python -m server.main

# Or with uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- API Documentation: http://localhost:8000/docs
- ReDoc Documentation: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## API Endpoints

### Phase 1: Foundation (✅ Complete)

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/v1/backends` - List available backends
- `GET /api/v1/backends/{name}` - Get backend details

### Phase 2: Analysis (Coming Soon)

- `POST /api/v1/wavelet/cwt` - Continuous Wavelet Transform
- `POST /api/v1/wavelet/wct` - Wavelet Coherence Transform
- `POST /api/v1/wavelet/significance` - Significance testing

### Phase 3: Job Management (Coming Soon)

- `POST /api/v1/jobs` - Submit analysis job
- `GET /api/v1/jobs/{job_id}` - Get job status
- `GET /api/v1/jobs/{job_id}/result` - Get job result

## Configuration

Environment variables (see `.env.example`):

- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `MAX_CONCURRENT_JOBS` - Maximum concurrent jobs (default: 10)
- `JOB_TIMEOUT_SECONDS` - Job timeout in seconds (default: 3600)
- `DEFAULT_BACKEND` - Default computation backend (default: sequential)
- `AVAILABLE_BACKENDS` - Comma-separated list of allowed backends

## Architecture

```
server/
├── main.py                 # FastAPI application
├── core/
│   └── config.py          # Configuration management
├── api/
│   ├── routes/
│   │   └── backends.py    # Backend discovery endpoints
│   └── models/            # Pydantic request/response models
└── tests/                 # Unit and integration tests
```

## Development

### Running Tests

```bash
pytest server/tests/
```

### Code Quality

```bash
# Format code
black server/

# Type checking
mypy server/

# Linting
ruff check server/
```

## Implementation Status

- [x] Phase 1: Foundation
  - [x] Directory structure
  - [x] Configuration management
  - [x] Backend discovery endpoints
  - [x] Health check
  - [x] OpenAPI documentation
- [ ] Phase 2: Analysis Endpoints (Week 2)
- [ ] Phase 3: Job Management (Week 3)
- [ ] Phase 4: Testing & Documentation

## License

See LICENSE.txt in the project root.
