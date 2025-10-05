# SMP (Multi-Worker) Configuration

The PyCWT REST API server now runs with **SMP (Symmetric Multiprocessing)** enabled by default for production workloads.

## What Changed

### Default Behavior
- **Production Mode**: Multi-worker SMP (default: 4 workers, auto-detects CPU cores)
- **Development Mode**: Single worker with auto-reload (set `DEV_MODE=true`)

### Configuration Options

#### 1. Environment Variable
Set the number of workers via environment variable:
```bash
export WORKERS=8  # Use 8 worker processes
./start-server.sh
```

#### 2. .env File
Edit `server/.env` (created automatically from `.env.example`):
```bash
WORKERS=4  # Adjust based on your CPU cores
```

#### 3. Auto-Detection
If `WORKERS` is not set, it auto-detects CPU cores:
```bash
./start-server.sh  # Uses all available CPU cores
```

#### 4. Development Mode
For development with auto-reload (single worker):
```bash
export DEV_MODE=true
python -m server.main
```

Or:
```bash
DEV_MODE=true ./start-server.sh
```

## Running the Server

### Production (Default - SMP Enabled)
```bash
# Auto-detect CPU cores
./start-server.sh

# Or specify workers
WORKERS=8 ./start-server.sh
```

### Development (Single Worker)
```bash
# With auto-reload
DEV_MODE=true python -m server.main
```

### Manual Uvicorn Command
```bash
# Production with 4 workers
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4

# Development with reload
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

## Benefits of SMP

1. **Concurrent Request Handling**: Multiple workers handle requests simultaneously
2. **CPU Utilization**: Utilizes all CPU cores for better performance
3. **Load Distribution**: Automatic load balancing across workers
4. **Resilience**: If one worker crashes, others continue serving requests

## Performance

### Server Level (Request Handling)
- **Single Worker**: ~100-200 requests/sec
- **4 Workers**: ~400-800 requests/sec (scales with CPU cores)

### Computation Level (Monte Carlo)
The Joblib backend provides additional parallelization:
```python
POST /api/v1/wavelet/wct
{
  "backend": "joblib",  # Multi-core Monte Carlo
  "mc_count": 100
}
```

**Combined**: SMP at server level + Joblib at computation level = maximum throughput

## Monitoring

Check running workers:
```bash
ps aux | grep uvicorn
```

Expected output (4 workers):
```
chris  1234  ... uvicorn server.main:app --workers 4  # Master
chris  1235  ... uvicorn server.main:app --workers 4  # Worker 1
chris  1236  ... uvicorn server.main:app --workers 4  # Worker 2
chris  1237  ... uvicorn server.main:app --workers 4  # Worker 3
chris  1238  ... uvicorn server.main:app --workers 4  # Worker 4
```

## Hardware Requirements

- **Minimum**: 2 CPU cores (2 workers recommended)
- **Optimal**: 4+ CPU cores (4-8 workers recommended)
- **Maximum**: Set workers = CPU cores (going beyond doesn't help)

## Troubleshooting

### "Too many workers"
If you see performance degradation:
```bash
# Reduce workers
WORKERS=2 ./start-server.sh
```

### Memory Usage
Each worker uses ~100-200 MB RAM. For 4 workers:
- Expected: ~400-800 MB total
- Consider reducing workers on low-memory systems

### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill existing server
pkill -f uvicorn
```

## Files Modified

1. **server/core/config.py**: Added `WORKERS` setting (default: 4)
2. **server/main.py**: Added DEV_MODE detection and worker support
3. **start-server.sh**: Auto-detects CPU cores, uses uvicorn with workers
4. **server/.env.example**: Added WORKERS configuration

## Backward Compatibility

Old startup methods still work:
```bash
# Old way (single worker)
python -m server.main

# New way (multi-worker)
./start-server.sh
```

To force old behavior:
```bash
DEV_MODE=true python -m server.main
```
