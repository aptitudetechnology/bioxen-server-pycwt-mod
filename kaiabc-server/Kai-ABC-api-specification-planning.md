# KaiABC Circadian Oscillator API Specification

**Version:** 1.0.0  
**Date:** October 6, 2025  
**Architecture:** Client-Server Model with RESTful and WebSocket APIs

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [API Endpoints](#api-endpoints)
4. [WebSocket Events](#websocket-events)
5. [Data Models](#data-models)
6. [Authentication & Security](#authentication--security)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

---

## Overview

### Purpose

This API specification defines the interface for the KaiABC Circadian Oscillator system, transitioning from a standalone embedded architecture to a distributed client-server model. The API enables:

- Real-time monitoring of circadian oscillator state across multiple nodes
- Remote sensor data ingestion from Raspberry Pi Pico and ELM11 nodes
- Centralized ODE computation and entrainment control
- Multi-client visualization and parameter management
- Historical data logging and analysis with time-series optimization
- Fault-tolerant operation with automatic failover

### Design Principles

1. **Real-Time Performance:** WebSocket support for sub-100ms state updates
2. **Modularity:** Separate endpoints for sensors, simulation, and control
3. **Scalability:** Support multiple sensor nodes and client applications
4. **Fault Tolerance:** Graceful degradation when network connectivity fails
5. **Scientific Fidelity:** Preserve biological accuracy of the oscillator model

---

## Architecture Design

### System Components

```
┌──────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│ RPi Pico / ELM11 │────────▶│  KaiABC Server   │◀────────│  Web Dashboard  │
│  (Sensor Node)   │  MQTT/  │  (Computation)   │  HTTP/  │   (Monitor)     │
│  - BME280        │  HTTP   │  - ODE Solver    │  WS     │  - Visualization│
│  - Local PWM     │         │  - Kalman Filter │         │  - Control      │
│  - MicroPython/  │         │  - ETF Engine    │         └─────────────────┘
│    Lua Runtime   │         │  - State DB      │
└──────────────────┘         └──────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  Time Series DB │
                            │  (InfluxDB/     │
                            │   TimescaleDB)  │
                            └─────────────────┘
```

### Communication Protocols

- **HTTP/REST:** Configuration, queries, and non-time-critical operations
- **WebSocket:** Real-time state streaming and low-latency control
- **MQTT:** Lightweight sensor data ingestion from Pico/ELM11 nodes (optional)

### Rationale for Client-Server Architecture with Pico/ELM11

The client-server architecture is particularly well-suited for Raspberry Pi Pico and ELM11 boards:

**1. Computational Offloading**
- **No Hardware FPU:** Pico's RP2040 lacks hardware floating-point unit
- **Lua Overhead:** ELM11's Lua interpreter adds computational cost
- **Server Advantage:** Centralized server performs intensive ODE integration using optimized C/C++ code
- **Node Advantage:** Nodes focus on efficient sensor reading and local PWM control

**2. Memory Constraints**
- **Pico (264KB SRAM):** Sufficient for sensor buffering, not full ODE state history
- **ELM11 (128-512KB):** Variable capacity depending on model
- **Server Advantage:** Unlimited storage for historical data, Kalman filter matrices, PRC lookup tables

**3. Multi-Node Coordination**
- Server can coordinate multiple sensor nodes across different environments
- Centralized entrainment control for synchronized experiments
- Aggregate data analysis across all nodes

**4. Development Flexibility**
- **Pico:** MicroPython enables rapid sensor integration
- **ELM11:** Lua scripting simplifies field deployment and configuration
- **Server:** Python/FastAPI or similar enables sophisticated scientific computing libraries

**5. Fault Tolerance**
- Nodes maintain basic PWM output during network outages
- Server maintains simulation continuity across node disconnections
- Automatic recovery and state synchronization on reconnection

---

## API Endpoints

### Base URL

```
Production:  https://api.kaiabc.example.com/v1
Development: http://localhost:8000/v1
```

---

## 1. System Status & Health

### `GET /health`

Health check endpoint for monitoring service availability.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "timestamp": "2025-10-06T12:00:00Z",
  "components": {
    "ode_solver": "operational",
    "database": "operational",
    "sensor_nodes": {
      "connected": 3,
      "total": 5
    }
  }
}
```

---

## 2. Oscillator State Management

### `GET /oscillator/state`

Retrieve the current state of the circadian oscillator.

**Query Parameters:**
- `node_id` (optional): Filter by specific sensor node
- `detail_level`: `minimal` | `standard` | `full` (default: `standard`)

**Response:**
```json
{
  "timestamp": "2025-10-06T12:00:00Z",
  "circadian_time": 14.5,
  "phase": 0.604,
  "period_hours": 24.1,
  "state_variables": {
    "C_U": 0.45,
    "C_S": 0.15,
    "C_T": 0.08,
    "C_ST": 0.32,
    "A_free": 0.58,
    "CABC_complex": 0.22
  },
  "koa_metric": 0.67,
  "output_pwm": 170,
  "entrainment_active": true,
  "temperature_kelvin": 298.5
}
```

### `POST /oscillator/reset`

Reset the oscillator to initial conditions.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "initial_conditions": {
    "C_U": 0.5,
    "C_S": 0.0,
    "C_T": 0.0,
    "C_ST": 0.0,
    "A_free": 1.0,
    "CABC_complex": 0.0
  },
  "reason": "manual_reset"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Oscillator reset to initial conditions",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00Z"
}
```

### `GET /oscillator/history`

Retrieve historical oscillator state data.

**Query Parameters:**
- `start_time`: ISO 8601 timestamp
- `end_time`: ISO 8601 timestamp
- `resolution`: `1s` | `10s` | `1m` | `10m` | `1h` (default: `1m`)
- `variables`: Comma-separated list (default: all)
- `node_id` (optional): Filter by node

**Response:**
```json
{
  "node_id": "pico_001",
  "hardware_type": "raspberry_pi_pico",
  "start_time": "2025-10-05T00:00:00Z",
  "end_time": "2025-10-06T00:00:00Z",
  "resolution": "1m",
  "data_points": 1440,
  "series": {
    "timestamp": ["2025-10-05T00:00:00Z", "2025-10-05T00:01:00Z", "..."],
    "C_ST": [0.32, 0.33, "..."],
    "koa_metric": [0.67, 0.68, "..."],
    "temperature_kelvin": [298.5, 298.6, "..."]
  }
}
```

---

## 3. Sensor Data Ingestion

### `POST /sensors/temperature`

Submit temperature sensor reading from Pico/ELM11 node.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00.123Z",
  "temperature_kelvin": 298.65,
  "raw_value": 25.5,
  "sensor_type": "BME280",
  "quality": "good"
}
```

**Response:**
```json
{
  "accepted": true,
  "filtered_value": 298.60,
  "kalman_variance": 0.02,
  "message": "Sensor data processed"
}
```

### `POST /sensors/batch`

Submit batch sensor readings (temperature, humidity, pressure).

**Request Body:**
```json
{
  "node_id": "pico_001",
  "readings": [
    {
      "timestamp": "2025-10-06T12:00:00.000Z",
      "temperature_kelvin": 298.65,
      "humidity_percent": 45.2,
      "pressure_pa": 101325
    },
    {
      "timestamp": "2025-10-06T12:00:01.000Z",
      "temperature_kelvin": 298.66,
      "humidity_percent": 45.3,
      "pressure_pa": 101326
    }
  ]
}
```

**Response:**
```json
{
  "accepted": 2,
  "rejected": 0,
  "latest_filtered": {
    "temperature_kelvin": 298.65,
    "humidity_percent": 45.25,
    "pressure_pa": 101325.5
  }
}
```

### `GET /sensors/nodes`

List all registered sensor nodes.

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "pico_001",
      "name": "Lab Incubator A",
      "hardware_type": "raspberry_pi_pico",
      "runtime": "micropython",
      "status": "online",
      "last_seen": "2025-10-06T12:00:00Z",
      "sensor_types": ["temperature", "humidity", "pressure"],
      "firmware_version": "1.2.3",
      "location": "Building 3, Room 201"
    },
    {
      "node_id": "elm11_002",
      "name": "Field Station B",
      "hardware_type": "elm11",
      "runtime": "lua",
      "status": "offline",
      "last_seen": "2025-10-06T10:30:00Z",
      "sensor_types": ["temperature"],
      "firmware_version": "1.2.1",
      "location": "Outdoor Site 1"
    }
  ]
}
```

### `POST /sensors/nodes`

Register a new sensor node.

**Request Body:**
```json
{
  "node_id": "pico_003",
  "name": "Growth Chamber C",
  "hardware_type": "raspberry_pi_pico" | "elm11",
  "runtime": "micropython" | "lua",
  "sensor_types": ["temperature", "humidity", "pressure", "light"],
  "location": "Building 2, Room 105",
  "calibration": {
    "temperature_offset": 0.0,
    "humidity_offset": 0.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "node_id": "pico_003",
  "hardware_type": "raspberry_pi_pico",
  "api_key": "kaiabc_node_abc123xyz789",
  "mqtt_topic": "kaiabc/sensors/pico_003"
}
```

---

## 4. Entrainment Control

### `GET /entrainment/parameters`

Retrieve current entrainment transfer function parameters.

**Query Parameters:**
- `node_id` (optional): Filter by node

**Response:**
```json
{
  "node_id": "pico_001",
  "etf_active": true,
  "reference_temperature_kelvin": 298.15,
  "parameters": {
    "arrhenius_scaling": {
      "activation_energy_kJ_mol": 40.0,
      "pre_exponential_factor": 1.5e8
    },
    "structural_coupling": {
      "baseline_d_i": 0.85,
      "modulation_function": "nonlinear_sigmoid",
      "sensitivity": 1.2
    },
    "atpase_constraint": {
      "baseline_f_hyd": 0.5,
      "temperature_dependence": "case_ii_mutant",
      "coupling_strength": 0.9
    }
  },
  "prc_model": "standard",
  "phase_lock_target": null
}
```

### `PUT /entrainment/parameters`

Update entrainment transfer function parameters.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "parameters": {
    "structural_coupling": {
      "sensitivity": 1.5
    }
  },
  "reason": "Increase entrainment responsiveness"
}
```

**Response:**
```json
{
  "success": true,
  "message": "ETF parameters updated",
  "effective_timestamp": "2025-10-06T12:00:01Z",
  "previous_parameters": { "...": "..." },
  "new_parameters": { "...": "..." }
}
```

### `POST /entrainment/phase-shift`

Trigger a manual phase shift using PRC logic.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "shift_type": "advance" | "delay",
  "magnitude_hours": 2.0,
  "method": "temperature_step",
  "temperature_delta_kelvin": 3.0,
  "duration_minutes": 30
}
```

**Response:**
```json
{
  "success": true,
  "shift_id": "shift_20251006_001",
  "expected_phase_change": 1.8,
  "current_circadian_time": 14.5,
  "optimal_ct_window": "16-22",
  "estimated_completion": "2025-10-06T12:30:00Z"
}
```

### `GET /entrainment/prc`

Retrieve the Phase Response Curve data.

**Query Parameters:**
- `model`: `standard` | `custom` (default: `standard`)
- `resolution`: Number of CT points (default: 24)

**Response:**
```json
{
  "model": "standard",
  "reference": "Rust et al. 2007",
  "resolution": 24,
  "data": [
    {"circadian_time": 0, "phase_shift_advance": 0.0, "phase_shift_delay": 0.0},
    {"circadian_time": 1, "phase_shift_advance": 0.1, "phase_shift_delay": -0.05},
    {"circadian_time": 16, "phase_shift_advance": 1.8, "phase_shift_delay": -1.2},
    {"circadian_time": 22, "phase_shift_advance": 1.5, "phase_shift_delay": -1.5}
  ]
}
```

---

## 5. Simulation Control

### `GET /simulation/config`

Retrieve current ODE solver configuration.

**Response:**
```json
{
  "solver_type": "RKF45",
  "tolerance": {
    "absolute": 1e-6,
    "relative": 1e-5
  },
  "max_step_size": 0.1,
  "min_step_size": 1e-6,
  "integration_method": "adaptive",
  "performance_metrics": {
    "average_steps_per_second": 8500,
    "average_iteration_time_ms": 45,
    "stability_index": 0.98
  }
}
```

### `PUT /simulation/config`

Update solver configuration parameters.

**Request Body:**
```json
{
  "tolerance": {
    "absolute": 1e-7,
    "relative": 1e-6
  },
  "max_step_size": 0.05
}
```

**Response:**
```json
{
  "success": true,
  "message": "Solver configuration updated",
  "restart_required": false
}
```

### `POST /simulation/pause`

Pause the oscillator simulation.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "reason": "maintenance"
}
```

**Response:**
```json
{
  "success": true,
  "state_saved": true,
  "paused_at_ct": 14.5
}
```

### `POST /simulation/resume`

Resume a paused simulation.

**Request Body:**
```json
{
  "node_id": "pico_001"
}
```

**Response:**
```json
{
  "success": true,
  "resumed_from_ct": 14.5,
  "timestamp": "2025-10-06T12:05:00Z"
}
```

---

## 6. Output Control

### `GET /output/koa`

Retrieve current KOA (Kai-complex Output Activity) metric.

**Query Parameters:**
- `node_id` (optional): Filter by node

**Response:**
```json
{
  "node_id": "pico_001",
  "koa_value": 0.67,
  "koa_normalized": 0.67,
  "koa_phase": "ascending",
  "contributing_states": {
    "C_ST": 0.32,
    "C_S": 0.15
  },
  "pwm_mapping": {
    "duty_cycle": 170,
    "duty_cycle_percent": 66.7,
    "frequency_hz": 1000
  }
}
```

### `PUT /output/mapping`

Configure the KOA to PWM output mapping.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "mapping_function": "linear" | "sigmoid" | "custom",
  "pwm_min": 0,
  "pwm_max": 255,
  "koa_threshold_low": 0.1,
  "koa_threshold_high": 0.9,
  "invert": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Output mapping updated",
  "test_values": {
    "koa_0.0": {"pwm": 0},
    "koa_0.5": {"pwm": 128},
    "koa_1.0": {"pwm": 255}
  }
}
```

### `POST /output/manual-override`

Manually override the output signal.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "override_enabled": true,
  "pwm_value": 200,
  "duration_minutes": 60,
  "reason": "Testing LED response"
}
```

**Response:**
```json
{
  "success": true,
  "override_id": "override_20251006_001",
  "expires_at": "2025-10-06T13:00:00Z"
}
```

---

## 7. Kalman Filter Management

### `GET /filter/state`

Retrieve current Kalman filter state.

**Query Parameters:**
- `node_id`: Sensor node identifier

**Response:**
```json
{
  "node_id": "pico_001",
  "filter_type": "extended_kalman",
  "state_estimate": {
    "temperature_kelvin": 298.60,
    "humidity_percent": 45.25,
    "pressure_pa": 101325.5
  },
  "covariance_matrix": [
    [0.02, 0.001, 0.0],
    [0.001, 0.05, 0.0],
    [0.0, 0.0, 0.1]
  ],
  "innovation": {
    "temperature": 0.05,
    "humidity": 0.1,
    "pressure": 0.5
  },
  "measurement_count": 8642
}
```

### `PUT /filter/parameters`

Update Kalman filter parameters.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "process_noise": {
    "temperature": 0.01,
    "humidity": 0.02,
    "pressure": 0.05
  },
  "measurement_noise": {
    "temperature": 0.05,
    "humidity": 0.1,
    "pressure": 0.2
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Kalman filter parameters updated",
  "filter_reset": false
}
```

### `POST /filter/reset`

Reset the Kalman filter to initial state.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "reason": "Sensor recalibration"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Kalman filter reset",
  "timestamp": "2025-10-06T12:00:00Z"
}
```

---

## 8. Analytics & Reporting

### `GET /analytics/period-stability`

Analyze period stability over time.

**Query Parameters:**
- `node_id`: Sensor node identifier
- `days`: Number of days to analyze (default: 7)

**Response:**
```json
{
  "node_id": "pico_001",
  "analysis_period": {
    "start": "2025-09-29T00:00:00Z",
    "end": "2025-10-06T00:00:00Z"
  },
  "statistics": {
    "mean_period_hours": 24.08,
    "std_deviation_hours": 0.15,
    "min_period_hours": 23.85,
    "max_period_hours": 24.32,
    "q10_value": 1.06,
    "coefficient_variation": 0.006
  },
  "temperature_correlation": {
    "pearson_r": 0.08,
    "p_value": 0.45
  },
  "quality_assessment": "excellent"
}
```

### `GET /analytics/entrainment-efficiency`

Evaluate entrainment response efficiency.

**Query Parameters:**
- `node_id`: Sensor node identifier
- `shift_id` (optional): Specific phase shift event

**Response:**
```json
{
  "node_id": "pico_001",
  "recent_shifts": [
    {
      "shift_id": "shift_20251005_001",
      "timestamp": "2025-10-05T08:00:00Z",
      "type": "advance",
      "target_shift_hours": 2.0,
      "actual_shift_hours": 1.8,
      "efficiency": 0.90,
      "settling_time_hours": 4.5,
      "temperature_delta_applied": 3.0
    }
  ],
  "average_efficiency": 0.88,
  "recommendations": [
    "Consider increasing structural coupling sensitivity to 1.8 for better response"
  ]
}
```

### `GET /analytics/export`

Export comprehensive data for external analysis.

**Query Parameters:**
- `node_id`: Sensor node identifier
- `start_time`: ISO 8601 timestamp
- `end_time`: ISO 8601 timestamp
- `format`: `json` | `csv` | `hdf5` (default: `json`)
- `variables`: Comma-separated list (default: all)

**Response:**
- For JSON/CSV: Direct download
- For HDF5: Pre-signed URL to download location

```json
{
  "export_id": "export_20251006_001",
  "status": "completed",
  "download_url": "https://api.kaiabc.example.com/v1/downloads/export_20251006_001.hdf5",
  "expires_at": "2025-10-07T12:00:00Z",
  "size_bytes": 15728640,
  "record_count": 86400
}
```

---

## WebSocket Events

### Connection

**Endpoint:** `wss://api.kaiabc.example.com/v1/ws`

**Authentication:** JWT token in query parameter or header

```
wss://api.kaiabc.example.com/v1/ws?token=eyJhbGc...
```

### Subscription Model

Clients subscribe to specific event streams:

```json
{
  "action": "subscribe",
  "streams": [
    "oscillator.state.pico_001",
    "sensors.temperature.pico_001",
    "output.koa.pico_001"
  ]
}
```

### Event: `oscillator.state`

Real-time oscillator state updates (typically every 1-10 seconds).

```json
{
  "event": "oscillator.state",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00.123Z",
  "data": {
    "circadian_time": 14.52,
    "phase": 0.605,
    "state_variables": {
      "C_ST": 0.32,
      "A_free": 0.58
    },
    "koa_metric": 0.67
  }
}
```

### Event: `sensors.reading`

Real-time sensor data after Kalman filtering.

```json
{
  "event": "sensors.reading",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00.123Z",
  "data": {
    "temperature_kelvin": 298.60,
    "temperature_raw": 298.65,
    "kalman_variance": 0.02,
    "quality": "good"
  }
}
```

### Event: `output.pwm`

PWM output changes.

```json
{
  "event": "output.pwm",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00.123Z",
  "data": {
    "duty_cycle": 170,
    "duty_cycle_percent": 66.7,
    "koa_source": 0.67,
    "manual_override": false
  }
}
```

### Event: `alert`

System alerts and warnings.

```json
{
  "event": "alert",
  "severity": "warning",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:00:00.123Z",
  "message": "Integration error exceeded tolerance threshold",
  "code": "ODE_STABILITY_WARNING",
  "details": {
    "error_magnitude": 1.5e-5,
    "tolerance": 1.0e-5,
    "action_taken": "step_size_reduced"
  }
}
```

### Event: `phase_shift.complete`

Phase shift operation completed.

```json
{
  "event": "phase_shift.complete",
  "shift_id": "shift_20251006_001",
  "node_id": "pico_001",
  "timestamp": "2025-10-06T12:30:00Z",
  "data": {
    "target_shift_hours": 2.0,
    "actual_shift_hours": 1.8,
    "efficiency": 0.90,
    "duration_minutes": 30
  }
}
```

---

## Data Models

### OscillatorState

```typescript
interface OscillatorState {
  timestamp: string;              // ISO 8601
  circadian_time: number;         // 0-24 hours
  phase: number;                  // 0-1 (fractional cycle)
  period_hours: number;           // Current period estimate
  state_variables: {
    C_U: number;                  // Unphosphorylated KaiC
    C_S: number;                  // Singly phosphorylated
    C_T: number;                  // Threonine phosphorylated
    C_ST: number;                 // Doubly phosphorylated
    A_free: number;               // Free KaiA
    CABC_complex: number;         // Sequestration complex
  };
  koa_metric: number;             // 0-1
  output_pwm: number;             // 0-255
  entrainment_active: boolean;
  temperature_kelvin: number;
}
```

### SensorReading

```typescript
interface SensorReading {
  node_id: string;
  timestamp: string;
  temperature_kelvin?: number;
  humidity_percent?: number;
  pressure_pa?: number;
  light_lux?: number;
  sensor_type: string;
  quality: 'good' | 'fair' | 'poor' | 'invalid';
  raw_values?: {
    temperature?: number;
    humidity?: number;
    pressure?: number;
  };
}
```

### ETFParameters

```typescript
interface ETFParameters {
  arrhenius_scaling: {
    activation_energy_kJ_mol: number;
    pre_exponential_factor: number;
  };
  structural_coupling: {
    baseline_d_i: number;
    modulation_function: 'linear' | 'nonlinear_sigmoid' | 'exponential';
    sensitivity: number;
  };
  atpase_constraint: {
    baseline_f_hyd: number;
    temperature_dependence: 'compensated' | 'case_ii_mutant';
    coupling_strength: number;
  };
}
```

### PhaseShiftRequest

```typescript
interface PhaseShiftRequest {
  node_id: string;
  shift_type: 'advance' | 'delay';
  magnitude_hours: number;
  method: 'temperature_step' | 'parameter_modulation';
  temperature_delta_kelvin?: number;
  duration_minutes?: number;
}
```

---

## Authentication & Security

### Authentication Methods

1. **API Keys** (for Raspberry Pi Pico and ELM11 nodes)
   - Header: `X-API-Key: kaiabc_node_abc123xyz789`
   - Scoped to specific node_id

2. **JWT Tokens** (for web clients)
   - Header: `Authorization: Bearer eyJhbGc...`
   - Includes user permissions and roles

3. **OAuth2** (for advanced web applications)
   - Authorization Code Grant flow
   - Client Credentials for service-to-service communication

### Authorization Roles

- **admin**: Full system access
- **researcher**: Read/write access to simulation and entrainment
- **observer**: Read-only access
- **node**: Sensor node access (data submission only)
  - `node:pico`: Raspberry Pi Pico devices
  - `node:elm11`: ELM11 Lua devices

### Rate Limiting

- **Sensor data ingestion**: 100 requests/minute per node
- **State queries**: 60 requests/minute per client
- **Configuration updates**: 10 requests/minute per client
- **WebSocket connections**: 5 concurrent per client

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Temperature value out of valid range",
    "details": {
      "parameter": "temperature_kelvin",
      "provided": 400.0,
      "valid_range": [273.15, 323.15]
    },
    "timestamp": "2025-10-06T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: State conflict (e.g., node already registered)
- `422 Unprocessable Entity`: Valid syntax but semantic errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: System overload or maintenance

### Error Codes

- `INVALID_PARAMETER`: Parameter validation failed
- `NODE_NOT_FOUND`: Sensor node does not exist
- `NODE_OFFLINE`: Sensor node not responding
- `ODE_INSTABILITY`: Numerical instability detected
- `FILTER_DIVERGENCE`: Kalman filter diverged
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `AUTHENTICATION_FAILED`: Invalid credentials
- `PERMISSION_DENIED`: Insufficient permissions

---

## Rate Limiting

Rate limits are applied per API key/token:

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1696598400
```

**Rate Limit Exceeded Response:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 100 requests per minute exceeded",
    "retry_after_seconds": 45
  }
}
```

---

## API Versioning

### Version Strategy

The API uses URL-based versioning to ensure backward compatibility:

- **Current Version:** `v1` (2025-10-06)
- **Version Format:** `/v{major}` (e.g., `/v1`, `/v2`)
- **Deprecation Policy:** 12 months notice for breaking changes
- **Sunset Policy:** 24 months support after deprecation

### Version Headers

```http
Accept: application/vnd.kaiabc.v1+json
X-API-Version: 1.0.0
```

### Breaking Changes Policy

- PATCH versions: Bug fixes only
- MINOR versions: Backward-compatible additions
- MAJOR versions: Breaking changes

---

## Pagination

List endpoints support cursor-based pagination for efficient data retrieval:

**Query Parameters:**
- `limit`: Maximum items per page (default: 50, max: 1000)
- `cursor`: Cursor for next page
- `sort`: Sort field and direction (e.g., `timestamp:desc`)

**Response Format:**
```json
{
  "data": [...],
  "pagination": {
    "has_next": true,
    "has_prev": false,
    "next_cursor": "eyJ0aW1lc3RhbXAiOiIyMDI1LTEwLTA2VDEyOjAwOjAwWiIsImlkIjoicGljb18wMDEifQ==",
    "prev_cursor": null,
    "total_count": 1250,
    "limit": 50
  }
}
```

---

## 9. Bulk Operations

### `POST /bulk/sensor-readings`

Submit multiple sensor readings in a single request for efficiency.

**Request Body:**
```json
{
  "readings": [
    {
      "node_id": "pico_001",
      "timestamp": "2025-10-06T12:00:00Z",
      "temperature_kelvin": 298.65,
      "humidity_percent": 45.2,
      "pressure_pa": 101325
    },
    {
      "node_id": "elm11_002",
      "timestamp": "2025-10-06T12:00:00Z",
      "temperature_kelvin": 299.10,
      "light_lux": 850
    }
  ]
}
```

**Response:**
```json
{
  "accepted": 2,
  "rejected": 0,
  "processing_time_ms": 45,
  "results": [
    {
      "node_id": "pico_001",
      "status": "processed",
      "filtered_temperature": 298.60
    },
    {
      "node_id": "elm11_002",
      "status": "processed",
      "filtered_temperature": 299.05
    }
  ]
}
```

### `POST /bulk/oscillator-states`

Retrieve current state for multiple nodes efficiently.

**Request Body:**
```json
{
  "node_ids": ["pico_001", "elm11_002", "pico_003"],
  "detail_level": "standard"
}
```

**Response:**
```json
{
  "states": {
    "pico_001": {
      "circadian_time": 14.5,
      "phase": 0.604,
      "koa_metric": 0.67,
      "status": "online"
    },
    "elm11_002": {
      "circadian_time": 8.2,
      "phase": 0.342,
      "koa_metric": 0.23,
      "status": "online"
    },
    "pico_003": {
      "error": "node_offline",
      "last_seen": "2025-10-06T10:30:00Z"
    }
  },
  "timestamp": "2025-10-06T12:00:00Z"
}
```

---

## 10. Monitoring & Diagnostics

### `GET /diagnostics/performance`

Retrieve system performance metrics.

**Response:**
```json
{
  "server": {
    "cpu_usage_percent": 45.2,
    "memory_usage_mb": 1024,
    "active_connections": 12,
    "uptime_seconds": 86400
  },
  "oscillator": {
    "average_integration_time_ms": 42,
    "stability_index": 0.98,
    "active_nodes": 5,
    "total_simulations": 1250
  },
  "database": {
    "connections_active": 8,
    "query_latency_ms": 12.5,
    "storage_used_gb": 25.6
  },
  "network": {
    "requests_per_second": 45.2,
    "error_rate_percent": 0.1,
    "average_response_time_ms": 85
  }
}
```

### `GET /diagnostics/logs`

Retrieve system logs with filtering.

**Query Parameters:**
- `level`: `debug` | `info` | `warning` | `error`
- `node_id`: Filter by specific node
- `start_time`: ISO 8601 timestamp
- `end_time`: ISO 8601 timestamp
- `limit`: Maximum entries (default: 100)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2025-10-06T12:00:00Z",
      "level": "info",
      "node_id": "pico_001",
      "message": "Sensor reading processed successfully",
      "details": {
        "temperature_kelvin": 298.65,
        "processing_time_ms": 15
      }
    },
    {
      "timestamp": "2025-10-06T11:59:45Z",
      "level": "warning",
      "node_id": "elm11_002",
      "message": "Network latency exceeded threshold",
      "details": {
        "latency_ms": 125,
        "threshold_ms": 100
      }
    }
  ],
  "pagination": {
    "has_next": true,
    "next_cursor": "..."
  }
}
```

### `POST /diagnostics/health-check`

Perform comprehensive health check on specific components.

**Request Body:**
```json
{
  "components": ["database", "oscillator", "network"],
  "detailed": true
}
```

**Response:**
```json
{
  "overall_status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "latency_ms": 12,
      "connections_available": 15
    },
    "oscillator": {
      "status": "healthy",
      "stability_check": "passed",
      "active_nodes": 5
    },
    "network": {
      "status": "warning",
      "latency_ms": 95,
      "packet_loss_percent": 0.05
    }
  },
  "timestamp": "2025-10-06T12:00:00Z"
}
```

---

## 11. Configuration Management

### `GET /config/templates`

Retrieve configuration templates for different node types.

**Query Parameters:**
- `hardware_type`: `raspberry_pi_pico` | `elm11`
- `runtime`: `micropython` | `lua`

**Response:**
```json
{
  "templates": {
    "sensor_config": {
      "bme280": {
        "i2c_address": "0x76",
        "mode": "normal",
        "oversampling": {
          "temperature": 2,
          "humidity": 2,
          "pressure": 2
        },
        "filter_coefficient": 2,
        "standby_time": 1000
      },
      "kalman_filter": {
        "process_noise": {
          "temperature": 0.01,
          "humidity": 0.02,
          "pressure": 0.05
        },
        "measurement_noise": {
          "temperature": 0.05,
          "humidity": 0.1,
          "pressure": 0.2
        }
      }
    },
    "network_config": {
      "wifi": {
        "ssid": "KaiABC_Network",
        "security": "wpa2",
        "reconnect_attempts": 5,
        "reconnect_delay_ms": 1000
      },
      "mqtt": {
        "broker": "mqtt.kaiabc.example.com",
        "port": 8883,
        "tls": true,
        "keepalive_seconds": 60,
        "qos": 1
      }
    },
    "oscillator_config": {
      "update_interval_ms": 5000,
      "pwm_frequency_hz": 1000,
      "buffer_size": 100,
      "fallback_mode": "local_pwm"
    }
  }
}
```

### `POST /config/validate`

Validate configuration before deployment.

**Request Body:**
```json
{
  "node_id": "pico_001",
  "config": {
    "sensor": {
      "bme280": {
        "i2c_address": "0x76",
        "oversampling": {
          "temperature": 4,
          "humidity": 4,
          "pressure": 4
        }
      }
    },
    "network": {
      "wifi": {
        "ssid": "MyNetwork",
        "password": "secret123"
      }
    }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "warnings": [
    {
      "field": "sensor.bme280.oversampling.temperature",
      "message": "High oversampling may increase power consumption",
      "suggestion": "Consider reducing to 2 for battery-powered applications"
    }
  ],
  "optimizations": [
    {
      "field": "network.wifi",
      "message": "WiFi credentials validated successfully"
    }
  ]
}
```

---

## Enhanced Error Handling

### Detailed Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Multiple validation errors found",
    "details": {
      "field_errors": [
        {
          "field": "temperature_kelvin",
          "code": "OUT_OF_RANGE",
          "message": "Temperature must be between 273.15 and 323.15 K",
          "provided": 400.0,
          "valid_range": [273.15, 323.15]
        },
        {
          "field": "node_id",
          "code": "INVALID_FORMAT",
          "message": "Node ID must match pattern: ^[a-zA-Z0-9_-]+$",
          "provided": "pico@001"
        }
      ],
      "general_errors": [
        {
          "code": "RATE_LIMIT_EXCEEDED",
          "message": "Too many requests. Please retry after 30 seconds"
        }
      ]
    },
    "timestamp": "2025-10-06T12:00:00Z",
    "request_id": "req_abc123xyz789",
    "path": "/v1/sensors/temperature",
    "method": "POST",
    "user_agent": "KaiABC-Pico/1.0.0",
    "client_ip": "192.168.1.100"
  }
}
```

### Error Recovery Suggestions

The API provides actionable recovery suggestions:

```json
{
  "error": {
    "code": "NODE_OFFLINE",
    "message": "Sensor node is not responding",
    "recovery_suggestions": [
      {
        "action": "check_power",
        "description": "Verify node has adequate power supply",
        "priority": "high"
      },
      {
        "action": "check_network",
        "description": "Ensure WiFi connectivity and signal strength",
        "priority": "high"
      },
      {
        "action": "restart_node",
        "description": "Power cycle the device",
        "priority": "medium"
      },
      {
        "action": "update_firmware",
        "description": "Check for firmware updates",
        "priority": "low"
      }
    ]
  }
}
```

---

## Testing Guidelines

### API Testing Strategy

1. **Unit Tests**: Individual endpoint functionality
2. **Integration Tests**: End-to-end workflows
3. **Load Tests**: Performance under high concurrency
4. **Chaos Tests**: Fault tolerance and recovery

### Test Data Management

Use dedicated test nodes and data isolation:

```bash
# Create test node
curl -X POST /v1/sensors/nodes \
  -H "X-API-Key: test_key" \
  -d '{"node_id": "test_pico_001", "name": "Test Node"}'

# Run test suite
npm test -- --environment=staging --node-id=test_pico_001
```

### Mock Data Endpoints

For development and testing:

- `GET /mock/sensor-data`: Generate synthetic sensor readings
- `POST /mock/simulate-oscillator`: Simulate oscillator behavior
- `GET /mock/test-scenarios`: Predefined test scenarios

---

## Deployment Considerations

### Environment Configuration

```yaml
# config/production.yaml
api:
  version: "1.0.0"
  host: "api.kaiabc.example.com"
  port: 443
  tls: true

database:
  type: "postgresql"
  host: "db.kaiabc.example.com"
  pool_size: 20
  timeout_ms: 30000

cache:
  type: "redis"
  host: "cache.kaiabc.example.com"
  ttl_seconds: 3600

monitoring:
  enabled: true
  metrics_endpoint: "/metrics"
  health_endpoint: "/health"
```

### Scaling Strategy

- **Horizontal Scaling**: Multiple API server instances behind load balancer
- **Database Sharding**: Partition data by node_id for large deployments
- **Caching**: Redis for frequently accessed oscillator states
- **CDN**: Static assets and documentation

### Security Hardening

- **API Gateway**: Rate limiting, authentication, request validation
- **WAF**: Web Application Firewall for attack prevention
- **Encryption**: TLS 1.3, encrypted database connections
- **Secrets Management**: Vault or similar for API keys and credentials

---

## Future Enhancements (v1.1.0)

### Planned Features

1. **GraphQL Support**: Flexible query interface for complex data requirements
2. **Real-time Analytics**: Streaming analytics and anomaly detection
3. **Machine Learning Integration**: Predictive maintenance and optimization
4. **Multi-Protocol Support**: MQTT, AMQP, and gRPC alternatives
5. **Federated Architecture**: Cross-region deployment support
6. **Advanced Scheduling**: Cron-like job scheduling for experiments
7. **Plugin Architecture**: Extensible sensor and actuator support

### Backward Compatibility

All v1.1.0 changes will maintain backward compatibility with v1.0.0 clients.

### Raspberry Pi Pico Client Integration

Example Raspberry Pi Pico MicroPython client code:

```python
import urequests
import ujson
from machine import Pin, I2C, PWM
import time

API_BASE = "http://192.168.1.100:8000/v1"
API_KEY = "kaiabc_node_pico_001"
NODE_ID = "pico_001"

# PWM setup for output control
pwm_pin = PWM(Pin(15))
pwm_pin.freq(1000)

def send_sensor_data(temp_k, humidity, pressure):
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "node_id": NODE_ID,
        "timestamp": get_iso_timestamp(),
        "temperature_kelvin": temp_k,
        "humidity_percent": humidity,
        "pressure_pa": pressure,
        "sensor_type": "BME280",
        "quality": "good"
    }
    response = urequests.post(
        f"{API_BASE}/sensors/batch",
        headers=headers,
        data=ujson.dumps({"node_id": NODE_ID, "readings": [payload]})
    )
    return response.json()

def get_current_state():
    headers = {"X-API-Key": API_KEY}
    response = urequests.get(
        f"{API_BASE}/oscillator/state?node_id={NODE_ID}",
        headers=headers
    )
    state = response.json()
    # Update local PWM based on KOA
    if 'output_pwm' in state:
        pwm_pin.duty_u16(state['output_pwm'] * 257)  # Convert 0-255 to 0-65535
    return state

def register_node():
    """Register this Pico with the server on startup"""
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "node_id": NODE_ID,
        "name": "Pico Lab Node 1",
        "hardware_type": "raspberry_pi_pico",
        "runtime": "micropython",
        "sensor_types": ["temperature", "humidity", "pressure"],
        "location": "Lab Bench 3"
    }
    try:
        response = urequests.post(
            f"{API_BASE}/sensors/nodes",
            headers=headers,
            data=ujson.dumps(payload)
        )
        return response.json()
    except Exception as e:
        print(f"Registration failed: {e}")
        return None
```

### ELM11 Lua Client Integration

Example ELM11 Lua client code:

```lua
-- ELM11 Lua Client for KaiABC
local http = require("socket.http")
local json = require("cjson")
local ltn12 = require("ltn12")

API_BASE = "http://192.168.1.100:8000/v1"
API_KEY = "kaiabc_node_elm11_001"
NODE_ID = "elm11_001"

-- PWM setup (ELM11 specific)
pwm = require("pwm")
pwm.setup(1, 1000, 0)  -- pin 1, 1kHz frequency, 0% duty
pwm.start(1)

function send_sensor_data(temp_k, humidity, pressure)
    local payload = json.encode({
        node_id = NODE_ID,
        readings = {{
            timestamp = os.date("!%Y-%m-%dT%H:%M:%SZ"),
            temperature_kelvin = temp_k,
            humidity_percent = humidity,
            pressure_pa = pressure,
            sensor_type = "BME280",
            quality = "good"
        }}
    })
    
    local response_body = {}
    local res, code, headers = http.request({
        url = API_BASE .. "/sensors/batch",
        method = "POST",
        headers = {
            ["X-API-Key"] = API_KEY,
            ["Content-Type"] = "application/json",
            ["Content-Length"] = #payload
        },
        source = ltn12.source.string(payload),
        sink = ltn12.sink.table(response_body)
    })
    
    if code == 200 then
        return json.decode(table.concat(response_body))
    else
        print("Error: HTTP " .. code)
        return nil
    end
end

function get_current_state()
    local response_body = {}
    local res, code = http.request({
        url = API_BASE .. "/oscillator/state?node_id=" .. NODE_ID,
        method = "GET",
        headers = {
            ["X-API-Key"] = API_KEY
        },
        sink = ltn12.sink.table(response_body)
    })
    
    if code == 200 then
        local state = json.decode(table.concat(response_body))
        -- Update PWM based on KOA
        if state.output_pwm then
            local duty = math.floor((state.output_pwm / 255) * 1023)
            pwm.setduty(1, duty)
        end
        return state
    else
        return nil
    end
end

function register_node()
    local payload = json.encode({
        node_id = NODE_ID,
        name = "ELM11 Field Node 1",
        hardware_type = "elm11",
        runtime = "lua",
        sensor_types = {"temperature", "humidity"},
        location = "Outdoor Station"
    })
    
    local response_body = {}
    local res, code = http.request({
        url = API_BASE .. "/sensors/nodes",
        method = "POST",
        headers = {
            ["X-API-Key"] = API_KEY,
            ["Content-Type"] = "application/json",
            ["Content-Length"] = #payload
        },
        source = ltn12.source.string(payload),
        sink = ltn12.sink.table(response_body)
    })
    
    return code == 200 or code == 201
end

-- Main loop
function main_loop()
    -- Register on startup
    register_node()
    
    while true do
        -- Read sensors (pseudo-code, adjust for ELM11 I2C)
        local temp, humidity, pressure = read_bme280()
        
        -- Send to server
        send_sensor_data(temp + 273.15, humidity, pressure)
        
        -- Get updated state and output
        get_current_state()
        
        -- Sleep for 5 seconds
        tmr.delay(5000000)  -- microseconds
    end
end
```

### Hardware Considerations

#### Raspberry Pi Pico
- **Processor:** RP2040 dual-core ARM Cortex-M0+ @ 133MHz
- **Memory:** 264KB SRAM, 2MB Flash
- **Floating-Point:** Software floating-point (no FPU)
- **Networking:** Requires external WiFi module (e.g., ESP8266 via UART, or Pico W variant)
- **PWM Channels:** 16 PWM channels
- **I2C Support:** 2 I2C controllers
- **MicroPython:** Full support, excellent performance

**Performance Notes:**
- Sufficient for sensor reading and data transmission
- Server-side ODE computation recommended due to no hardware FPU
- Local PWM output control is efficient
- 264KB RAM adequate for buffering ~1 hour of sensor data

#### ELM11
- **Processor:** Typically ARM Cortex-M3/M4 (varies by model)
- **Memory:** 128-512KB SRAM (model dependent)
- **Lua Runtime:** eLua or NodeMCU-style Lua interpreter
- **Networking:** Built-in WiFi (model dependent)
- **PWM Support:** Multiple channels
- **I2C Support:** Yes

**Performance Notes:**
- Lua interpreter adds overhead but simplifies development
- Server-side computation strongly recommended
- Good for rapid prototyping and scripting
- Check specific ELM11 model documentation for exact capabilities

### Server-Side Fallback

If network connectivity is lost, Raspberry Pi Pico and ELM11 nodes should:
1. Continue local PWM output based on last known state
2. Buffer sensor readings (up to 1 hour)
3. Resume data transmission when connectivity restored
4. Flag data gaps in metadata

---

## Versioning & Changelog

**Version 1.0.1** (October 6, 2025)
- Fixed all ESP32 references to use Raspberry Pi Pico and ELM11
- Added comprehensive error handling with recovery suggestions
- Added API versioning strategy with deprecation policy
- Added cursor-based pagination for list endpoints
- Added bulk operations for efficient data submission
- Added monitoring and diagnostics endpoints
- Added configuration management and validation
- Added testing guidelines and mock data endpoints
- Added deployment considerations and scaling strategy
- Enhanced security with OAuth2 support
- Added detailed performance monitoring

**Version 1.0.0** (October 6, 2025)
- Initial API specification
- REST endpoints for core functionality
- WebSocket support for real-time streaming
- Multi-node architecture support

**Future Considerations (v1.1.0):**
- GraphQL alternative endpoint for flexible queries
- MQTT broker integration for high-volume sensor networks
- Machine learning model integration for adaptive entrainment
- Redox sensor integration (LdpA pathway)
- Multi-oscillator synchronization endpoints
- Advanced analytics and predictive capabilities

---

## Support & Documentation

- **API Reference:** https://docs.kaiabc.example.com/api
- **Client Libraries:** Python, JavaScript, MicroPython
- **Issue Tracker:** https://github.com/aptitudetechnology/KaiABC/issues
- **Discussion Forum:** https://forum.kaiabc.example.com

---

**End of API Specification v1.0.0**
