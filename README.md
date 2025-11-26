# 🚂 Panoptique Ferroviaire - HNPS v6.0

**Hybrid Neuro-Physical System for Railway Surveillance**

A state-of-the-art, autonomous railway surveillance system that discovers, ingests, fuses, and simulates all trains operating in France in real-time.

## 🎯 Overview

Panoptique Ferroviaire is an industrial-grade system implementing advanced concepts:

- **Omniscience Engine**: Auto-discovery of GTFS-RT data sources from transport.data.gouv.fr
- **High-Performance Harvester**: Asynchronous ingestion of Protocol Buffer feeds
- **Hybrid Neuro-Physics Engine**: Combines Davis equation physics with Unscented Kalman Filtering
- **Moving Block System**: Cantonnement collision prevention logic
- **Rail-Lock Engine (v6.0)**: Absolute spatial awareness with track geometry projection
- **Holographic Positioning (v5.0)**: GPS-less positioning from station arrival times

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PANOPTIQUE FERROVIAIRE                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Omniscience  │───▶│  Harvester   │───▶│    Kafka     │  │
│  │   Engine     │    │  (GTFS-RT)   │    │   Message    │  │
│  │ (Discovery)  │    │              │    │     Bus      │  │
│  └──────────────┘    └──────────────┘    └───────┬──────┘  │
│                                                    │          │
│                                                    ▼          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Hybrid Neuro-Physics Engine                 │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │    │
│  │  │   Davis     │  │   Unscented  │  │  Moving   │ │    │
│  │  │  Physics    │◀▶│    Kalman    │◀▶│   Block   │ │    │
│  │  │  Equation   │  │    Filter    │  │Cantonnement│ │    │
│  │  └─────────────┘  └──────────────┘  └───────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PostGIS + pgrouting                     │    │
│  │         (Spatial Database & Network Analysis)        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)

### 1. Prepare Static Data (v6.0 - Required for Rail-Lock)

```bash
# Download and prepare GTFS static data
python -m src.tools.prepare_data

# Or use a local GTFS file
python -m src.tools.prepare_data --local /path/to/gtfs.zip

# Or specify a custom URL
python -m src.tools.prepare_data --url https://example.com/gtfs.zip
```

This generates:
- `data/stops.txt` - Station locations for Holographic Positioning
- `data/topology.json` - Route geometries for Rail-Lock spatial awareness

### 2. Launch the System

```bash
# Clone the repository
git clone <repository-url>
cd PositionMetro

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f neural-engine
```

The system will:
1. Auto-discover all French GTFS-RT feeds
2. Start harvesting real-time data
3. Simulate train physics and maintain state
4. Apply Rail-Lock projection for absolute spatial awareness
5. Store data in PostGIS database

### Access Points

- **Kafka UI**: http://localhost:8080 (Monitor message streams)
- **PostgreSQL**: localhost:5432 (Database: panoptique, User: panoptique)

## 📦 Components

### Module 0: Data Preparation Tool (v6.0)
**File**: `src/tools/prepare_data.py`

Automated static data acquisition:
- Downloads GTFS ZIP from URL or reads local file
- Extracts `stops.txt` for Holographic Positioning
- Generates `topology.json` for Rail-Lock engine
- Crosses `shapes.txt` and `trips.txt` to map routes to geometries
- Production-ready with comprehensive error handling

**Usage**:
```bash
# Default (IDFM)
python -m src.tools.prepare_data

# Local file
python -m src.tools.prepare_data --local gtfs.zip

# Custom URL
python -m src.tools.prepare_data --url https://example.com/gtfs.zip

# Custom output directory
python -m src.tools.prepare_data --output-dir /custom/data
```

### Module A: Omniscience Engine
**File**: `src/core/omniscience.py`

Auto-discovery system that:
- Queries transport.data.gouv.fr API
- Recursively iterates through all pages
- Filters GTFS-RT resources (TripUpdates & VehiclePositions)
- Builds dynamic registry of transport operators

**Key Features**:
- No hardcoded URLs
- Discovers SNCF, RATP, Keolis, Transdev, and more
- Handles pagination automatically

### Module B: High-Performance Harvester
**File**: `src/ingestion/harvester.py`

Asynchronous data ingestion system:
- Uses `aiohttp` for concurrent requests
- Parses Protocol Buffer GTFS-RT data
- Normalizes to standard JSON format
- Publishes to Kafka topic `raw_telemetry`

**Resilience**:
- Handles timeouts gracefully
- Detects dead APIs
- Adapts to format changes
- Concurrent harvesting of multiple feeds

### Module C: Hybrid Neuro-Physics Engine
**File**: `src/engine/fusion.py`

The core intelligence combining:

#### 1. Davis Equation Physics
Models train resistance: `R = A + B·v + C·v² + mg·sin(θ)`
- A: Rolling resistance
- B: Mechanical resistance  
- C: Aerodynamic resistance
- θ: Track gradient

#### 2. Unscented Kalman Filter
State estimation for non-linear dynamics:
- State vector: [latitude, longitude, velocity, acceleration, bearing]
- Fuses GTFS-RT measurements with physics predictions
- Handles sensor noise and model uncertainty

#### 3. Moving Block (Cantonnement)
Collision prevention:
- Tracks train spacing on same route
- Enforces safe distance (default 500m)
- Applies emergency braking when needed
- Considers relative velocities

#### 4. Rail-Lock Engine (v6.0)
Absolute spatial awareness:
- Projects GPS coordinates onto track geometry
- Calculates curvilinear distance (PK - Point Kilométrique)
- Extracts track gradient for physics simulation
- Enables accurate train ordering for Cantonnement
- Supports loops and U-turns without confusion

#### 5. Holographic Positioning (v5.0)
GPS-less positioning:
- Infers train position from next stop arrival times
- Works when VehiclePosition data is unavailable
- Uses `stops.txt` to convert stop_id to coordinates
- Creates "ghost trains" from TripUpdate alone

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  topic_raw_telemetry: "raw_telemetry"

harvester:
  harvest_interval: 30.0  # seconds
  max_concurrent_harvests: 20

fusion:
  simulation_rate: 1.0  # Hz
  moving_block:
    safe_distance: 500.0  # meters
```

## 📊 Database Schema

PostGIS database with spatial extensions:

- `railway.trains`: Current train states
- `railway.train_history`: Historical positions
- `railway.tracks`: Track network geometry
- `railway.operators`: Transport operators
- `railway.data_sources`: GTFS-RT feed URLs

## 🧪 Development

### Local Setup

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally (requires Kafka & PostGIS running)
python main.py
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

## 📐 Technical Specifications

### Performance
- **Language**: Python 3.12+
- **Async Runtime**: asyncio with uvloop optimization
- **Concurrency**: Up to 20 concurrent harvests
- **Simulation Rate**: 1 Hz (configurable)

### Type Safety
- Full type hints throughout codebase
- Pydantic models for data validation
- Dataclasses for structured data

### Scalability
- Horizontal scaling via Kafka partitions
- Stateless harvester workers
- Database connection pooling
- Docker-based microservices

## 🔬 Mathematical Models

### Davis Equation
```
F_resistance = A + B·v + C·v²
F_gravity = m·g·sin(gradient)
F_net = F_traction - F_resistance - F_gravity
acceleration = F_net / mass
```

### Unscented Kalman Filter
```
Prediction:
  σ_points = generate_sigma_points(x, P)
  x̂ = f(σ_points)
  P̂ = Q + Σ[W_c · (x̂ - μ)(x̂ - μ)ᵀ]

Update:
  K = P̂·H·(H·P̂·Hᵀ + R)⁻¹
  x = x̂ + K·(z - H·x̂)
  P = P̂ - K·H·P̂
```

## 🛡️ Security

- Environment-based configuration
- No credentials in code
- Docker network isolation
- PostgreSQL password authentication

## 📝 Logging

Structured JSON logging with multiple levels:
- Console output for monitoring
- File rotation (100MB, 10 backups)
- Metrics export for Prometheus

## 🚦 Status Indicators

The system reports:
- Number of discovered operators
- Number of active feeds
- Number of tracked trains
- Harvest success/failure rates
- Physics simulation status

## 🤝 Contributing

This is an industrial R&D project. Code follows strict standards:
- Type hints mandatory
- Async-first design
- Comprehensive error handling
- Structured logging
- Docker containerization

## 📄 License

Confidential - National Railway Surveillance Program

## 🎓 References

- GTFS-RT Specification: https://gtfs.org/realtime/
- transport.data.gouv.fr: https://transport.data.gouv.fr
- Davis Equation: Railway vehicle dynamics
- Unscented Kalman Filter: Julier & Uhlmann (1997)
- PostGIS: https://postgis.net
- pgrouting: https://pgrouting.org

---

**Version**: 6.0.0  
**Status**: ✅ Production Ready  
**Architecture**: État de l'Art (State of the Art)  
**New in v6.0**: Rail-Lock spatial awareness with automated data preparation

🚂 *"From Theory to Industrial Reality"*
