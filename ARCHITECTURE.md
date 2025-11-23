# Panoptique Ferroviaire - Architecture Documentation

## System Overview

The Panoptique Ferroviaire is an industrial-grade Hybrid Neuro-Physical System (HNPS v5.0) designed for real-time railway surveillance across France. It combines advanced physics simulation, sensor fusion, and autonomous discovery capabilities.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              main.py - Orchestrator                      │  │
│  │  • Lifecycle management                                   │  │
│  │  • Signal handling                                        │  │
│  │  • Configuration loading                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Omniscience  │  │  Harvester   │  │  Fusion Engine       │ │
│  │   Engine     │  │              │  │  ┌────────────────┐  │ │
│  │              │  │ • aiohttp    │  │  │ Davis Physics  │  │ │
│  │ • API Query  │─▶│ • Protobuf   │─▶│  │ UKF Filtering  │  │ │
│  │ • Pagination │  │ • Kafka Pub  │  │  │ Moving Block   │  │ │
│  │ • Discovery  │  │              │  │  └────────────────┘  │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Kafka      │  │   PostGIS    │  │    Monitoring        │ │
│  │              │  │              │  │                      │ │
│  │ • Topics     │  │ • Spatial DB │  │ • Prometheus         │ │
│  │ • Streams    │  │ • pgrouting  │  │ • Structlog          │ │
│  │ • Zookeeper  │  │ • Indexes    │  │ • Kafka UI           │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Omniscience Engine (`src/core/omniscience.py`)

**Purpose**: Autonomous discovery of GTFS-RT data sources

**Key Classes**:
- `OmniscienceEngine`: Main discovery orchestrator
- `GTFSRTResource`: Represents a discovered feed
- `TransportOperator`: Represents a transport company

**Algorithm**:
1. Query transport.data.gouv.fr API
2. Iterate through paginated results
3. Filter for GTFS-RT resources
4. Build operator registry
5. Return discovered feeds

**Performance**:
- Concurrent requests: 10 (configurable)
- Async/await pattern
- Connection pooling
- Error recovery

### 2. High-Performance Harvester (`src/ingestion/harvester.py`)

**Purpose**: Real-time data ingestion from GTFS-RT feeds

**Key Classes**:
- `GTFSRTHarvester`: Main harvesting engine
- `NormalizedVehiclePosition`: Standardized position data
- `NormalizedTripUpdate`: Standardized trip data
- `HarvestMetrics`: Performance tracking

**Data Flow**:
```
GTFS-RT Feed (Protobuf)
    ↓
Parse with gtfs_realtime_pb2
    ↓
Normalize to JSON
    ↓
Publish to Kafka
    ↓
Fusion Engine consumes
```

**Features**:
- Concurrent harvesting of multiple feeds
- Timeout handling (10s default)
- Dead feed detection
- Automatic retry (3 attempts)
- Metrics collection

### 3. Hybrid Neuro-Physics Engine (`src/engine/fusion.py`)

**Purpose**: State estimation and collision prevention

**Key Classes**:
- `HybridFusionEngine`: Main orchestrator
- `TrainEntity`: Individual train state
- `UnscentedKalmanFilter`: State estimator
- `Position2D`: Geographic calculations

#### 3.1 Davis Equation

Models train resistance:

```
R_total = A + B·v + C·v² + m·g·sin(gradient)

where:
  A = Rolling resistance (5.0 kN)
  B = Mechanical resistance (0.03 kN/(m/s))
  C = Aerodynamic resistance (0.0015 kN/(m/s)²)
  m = Train mass (400,000 kg)
  g = Gravity (9.81 m/s²)
  v = Velocity (m/s)
```

**Acceleration Calculation**:
```
F_net = F_traction - R_total
a = F_net / m

Constrained to:
  max_braking ≤ a ≤ max_acceleration
  -1.5 m/s² ≤ a ≤ 1.2 m/s²
```

#### 3.2 Unscented Kalman Filter

**State Vector**:
```
x = [latitude, longitude, velocity, acceleration, bearing]ᵀ
```

**Process**:
1. **Prediction**:
   - Generate sigma points
   - Propagate through motion model
   - Compute predicted mean and covariance

2. **Update**:
   - Transform sigma points to measurement space
   - Calculate Kalman gain
   - Update state with measurement
   - Update covariance

**Motion Model**:
```python
def motion_model(state, dt, control_acceleration):
    # Update acceleration with damping
    new_acceleration = 0.7 * acceleration + 0.3 * control_acceleration
    
    # Update velocity
    new_velocity = max(0, velocity + new_acceleration * dt)
    
    # Update position (convert to lat/lon)
    displacement = new_velocity * dt
    delta_lat = (displacement * cos(bearing)) / 111000
    delta_lon = (displacement * sin(bearing)) / (111000 * cos(lat))
    
    return [new_lat, new_lon, new_velocity, new_acceleration, bearing]
```

#### 3.3 Moving Block (Cantonnement)

**Purpose**: Prevent train collisions

**Algorithm**:
```python
if distance_to_preceding_train < safe_distance:
    # Emergency braking proportional to danger
    braking_factor = (safe_distance - distance) / safe_distance
    emergency_braking = max_braking * braking_factor
    return min(desired_acceleration, emergency_braking)

# Preventive braking if closing too fast
time_to_collision = distance / relative_velocity
if time_to_collision < 30s:
    preventive_braking = -relative_velocity / 10
    return min(desired_acceleration, preventive_braking)
```

**Safety Parameters**:
- Safe distance: 500m
- Warning time: 30s
- Maximum braking: -1.5 m/s²

## Data Models

### Vehicle Position Message

```json
{
  "type": "vehicle_position",
  "source_url": "https://...",
  "organization": "SNCF",
  "harvested_at": "2024-01-01T12:00:00Z",
  "data": {
    "vehicle_id": "TRAIN_001",
    "trip_id": "TRIP_001",
    "route_id": "RER_A",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "bearing": 45.0,
    "speed": 20.0,
    "timestamp": 1704110400,
    "current_stop_sequence": 5,
    "current_status": "IN_TRANSIT_TO",
    "congestion_level": "RUNNING_SMOOTHLY",
    "occupancy_status": "FEW_SEATS_AVAILABLE"
  }
}
```

### Trip Update Message

```json
{
  "type": "trip_update",
  "source_url": "https://...",
  "organization": "RATP",
  "harvested_at": "2024-01-01T12:00:00Z",
  "data": {
    "trip_id": "TRIP_002",
    "route_id": "METRO_1",
    "vehicle_id": "M1_042",
    "start_date": "20240101",
    "start_time": "12:00:00",
    "schedule_relationship": "SCHEDULED",
    "stop_time_updates": [
      {
        "stop_sequence": 1,
        "stop_id": "STOP_001",
        "arrival_delay": 30,
        "departure_delay": 45
      }
    ],
    "timestamp": 1704110400
  }
}
```

## Kafka Topics

### raw_telemetry
- **Producer**: Harvester
- **Consumer**: Fusion Engine
- **Partitions**: 8
- **Retention**: 24 hours
- **Format**: JSON

### fused_train_states
- **Producer**: Fusion Engine
- **Consumer**: Analytics / Dashboard
- **Partitions**: 8
- **Retention**: 7 days
- **Format**: JSON

## Database Schema

### railway.trains
Current state of all tracked trains.

| Column       | Type                 | Description                |
|-------------|----------------------|----------------------------|
| train_id    | VARCHAR(100)         | Primary key                |
| trip_id     | VARCHAR(100)         | Current trip               |
| route_id    | VARCHAR(100)         | Route identifier           |
| position    | GEOMETRY(POINT)      | Current geographic position|
| velocity    | DOUBLE PRECISION     | Speed in m/s               |
| acceleration| DOUBLE PRECISION     | Acceleration in m/s²       |
| bearing     | DOUBLE PRECISION     | Direction in degrees       |
| state       | VARCHAR(50)          | Operational state          |
| last_update | TIMESTAMP            | Last update time           |

**Indexes**:
- Spatial index on `position` (GIST)
- Index on `route_id`
- Index on `last_update`

### railway.train_history
Historical positions for analysis.

| Column       | Type                 | Description                |
|-------------|----------------------|----------------------------|
| id          | BIGSERIAL            | Primary key                |
| train_id    | VARCHAR(100)         | Train reference            |
| position    | GEOMETRY(POINT)      | Historic position          |
| velocity    | DOUBLE PRECISION     | Speed at time              |
| timestamp   | TIMESTAMP            | Record time                |

**Indexes**:
- Index on `train_id`
- Spatial index on `position`
- Index on `timestamp` DESC

## Performance Characteristics

### Throughput
- Concurrent harvests: 20 feeds
- Harvest interval: 30 seconds
- Messages/second: ~1000 (estimated)
- Trains tracked: Unlimited (memory-bound)

### Latency
- Discovery: ~5s per API page
- Harvest cycle: 30s (configurable)
- Fusion update: <1ms per train
- End-to-end: ~31s (discovery to state update)

### Resource Usage
- CPU: 2-4 cores (depending on train count)
- Memory: ~500MB base + ~1KB per train
- Disk: ~100GB/month (with 24h Kafka retention)
- Network: ~10Mbps sustained

## Scalability

### Horizontal Scaling
1. **Harvester**: Multiple instances with different feed subsets
2. **Fusion Engine**: Partition by route_id
3. **Kafka**: Add partitions and brokers
4. **PostGIS**: Read replicas for queries

### Vertical Scaling
1. Increase Kafka partition count
2. Add more CPU cores for fusion engine
3. Increase PostGIS connection pool
4. Add more RAM for train state cache

## Monitoring

### Metrics (Prometheus)
- `harvester_requests_total`: Total harvest attempts
- `harvester_success_rate`: Success percentage
- `fusion_trains_active`: Active trains
- `fusion_predictions_per_second`: Prediction rate
- `kafka_lag`: Consumer lag

### Logs (Structlog)
- JSON format for easy parsing
- Log levels: DEBUG, INFO, WARNING, ERROR
- Structured fields for filtering
- Rotation: 100MB files, 10 backups

### Alerts
- Dead feed detection (no data for 5 minutes)
- High consumer lag (>1000 messages)
- Low harvest success rate (<80%)
- Moving block activations (potential collisions)

## Deployment

### Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Production
```bash
docker-compose up -d
docker-compose logs -f neural-engine
```

### Configuration
Environment variables override config.yaml:
- `KAFKA_BOOTSTRAP_SERVERS`
- `POSTGRES_HOST`
- `LOG_LEVEL`
- `SIMULATION_RATE`
- `HARVEST_INTERVAL`

## Security

### Authentication
- PostgreSQL: Username/password
- Kafka: SASL/PLAIN (optional)
- APIs: No authentication (public data)

### Network
- Internal Docker network isolation
- Exposed ports: 8080 (Kafka UI), 5432 (PostgreSQL), 9092 (Kafka)
- TLS: Can be enabled via configuration

### Data
- No PII collected
- Public transit data only
- GDPR compliant
- Data retention: Configurable

## Future Enhancements

1. **Neural Network Integration**: Deep learning for anomaly detection
2. **Predictive Analytics**: ETA predictions for passengers
3. **Real-time Dashboard**: Web UI for visualization
4. **Mobile API**: REST API for mobile apps
5. **Historical Analysis**: Big data analytics on patterns
6. **Multi-country Support**: Expand beyond France

## References

- GTFS-RT Specification: https://gtfs.org/realtime/
- Davis Equation: Railway vehicle dynamics literature
- Unscented Kalman Filter: Julier & Uhlmann (1997)
- PostGIS: https://postgis.net
- Kafka: https://kafka.apache.org

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Maintained By**: Supreme Architecture Team
