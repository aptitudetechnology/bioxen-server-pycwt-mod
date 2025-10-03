# Quick Testing Reference - PyCWT REST API

## 🚀 Start Server
```bash
python -m server.main
# or
./start-server.sh
```

## 🔍 Pre-Flight Check
```bash
python diagnose-server.py
```

## 🧪 Quick Manual Tests

### Test 1: Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","api_version":"1.0.0"}
```

### Test 2: List Backends
```bash
curl http://localhost:8000/api/v1/backends/ | jq
# Expected: JSON with backends array
```

### Test 3: Get Backend Details
```bash
curl http://localhost:8000/api/v1/backends/sequential | jq
# Expected: Backend info with name, available, description, type
```

### Test 4: Invalid Backend (Should Fail)
```bash
curl http://localhost:8000/api/v1/backends/invalid
# Expected: 404 error
```

## 📊 Interactive Testing
```bash
# Open browser to:
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

## ✅ Automated Tests
```bash
# Run all tests
pytest server/tests/test_phase1.py -v

# Run with coverage
pytest server/tests/ --cov=server --cov-report=term-missing

# Run specific test
pytest server/tests/test_phase1.py::test_health_check -v
```

## 🔧 Troubleshooting

### Port in use?
```bash
lsof -i :8000
# Kill process: kill -9 <PID>
```

### Dependencies missing?
```bash
pip install -r server/requirements.txt
```

### Import errors?
```bash
export PYTHONPATH=/home/chris/pycwt-mod:$PYTHONPATH
```

## 📈 Performance Testing
```bash
# Light load test (100 requests)
ab -n 100 -c 10 http://localhost:8000/health

# Heavy load test (30 seconds)
wrk -t10 -c100 -d30s http://localhost:8000/api/v1/backends/
```

## ✓ Test Checklist

Phase 1 Completion Criteria:
- [ ] Server starts on port 8000
- [ ] `/health` returns 200 OK
- [ ] `/` returns API info
- [ ] `/api/v1/backends/` lists backends
- [ ] `/api/v1/backends/{name}` works for valid backends
- [ ] `/api/v1/backends/invalid` returns 404
- [ ] `/docs` shows Swagger UI
- [ ] All pytest tests pass
- [ ] Backend availability matches direct pycwt_mod

## 📝 Test Results Template

```
Date: _____________
Tester: _____________

Phase 1 Test Results:
┌─────────────────────────────┬────────┬─────────────┐
│ Test                        │ Status │ Notes       │
├─────────────────────────────┼────────┼─────────────┤
│ Server startup              │ [ ]    │             │
│ Health endpoint             │ [ ]    │             │
│ Root endpoint               │ [ ]    │             │
│ List backends               │ [ ]    │             │
│ Get backend details         │ [ ]    │             │
│ Invalid backend (404)       │ [ ]    │             │
│ OpenAPI docs                │ [ ]    │             │
│ Automated tests (pytest)    │ [ ]    │             │
│ Backend integration         │ [ ]    │             │
│ Performance (> 100 req/s)   │ [ ]    │             │
└─────────────────────────────┴────────┴─────────────┘

Issues Found:
1. ________________________________
2. ________________________________
3. ________________________________

Overall Status: PASS / FAIL / NEEDS REVIEW

Next Steps:
_____________________________________________
```

## 🎯 Success Criteria

Phase 1 is complete when:
1. ✅ All endpoints respond correctly
2. ✅ All automated tests pass
3. ✅ Documentation is accessible
4. ✅ Performance meets targets
5. ✅ No critical issues found

Ready for Phase 2? → Analysis Endpoints (WCT/CWT)
