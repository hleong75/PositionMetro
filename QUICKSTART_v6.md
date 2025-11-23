# HNPS v6.0 Quick Start Guide

## New in v6.0: Automated Data Preparation

HNPS v6.0 introduces **Rail-Lock** spatial awareness and **automated data preparation**. Follow these steps to get started.

## Step 1: Prepare Static Data

Before running the main system, you need to prepare static GTFS data files:

```bash
# Use default IDFM data source (recommended for France)
python -m src.tools.prepare_data

# Or use a local GTFS file
python -m src.tools.prepare_data --local /path/to/gtfs.zip

# Or use a custom URL
python -m src.tools.prepare_data --url https://example.com/gtfs.zip

# For large files on slow connections, increase timeout
python -m src.tools.prepare_data --timeout 600
```

This generates:
- **`data/stops.txt`** - Station locations for Holographic Positioning
- **`data/topology.json`** - Route geometries for Rail-Lock engine

## Step 2: Verify Data Files

```bash
ls -lh data/
# Should show:
#   stops.txt      - GTFS stops file
#   topology.json  - Route geometries
```

## Step 3: Launch the System

```bash
# Start all services
docker-compose up -d

# Or run locally
python main.py
```

## What's New in v6.0

### 🎯 Rail-Lock Engine
- Projects GPS coordinates onto track geometry
- Calculates precise track distance (PK - Point Kilométrique)
- Extracts track gradient for accurate physics simulation
- Enables safe Cantonnement (Moving Block) train ordering

### 🔮 Holographic Positioning (v5.0+)
- Creates trains from TripUpdate alone (no GPS needed)
- Infers position from next stop arrival times
- Works when VehiclePosition data is unavailable

### 🚀 Automated Data Preparation
- One-command data acquisition
- Supports URLs and local files
- Automatic validation and error handling
- Progress tracking for large downloads

## Configuration

You can customize data paths in `config/config.yaml`:

```yaml
static_data:
  topology_path: "data/topology.json"
  stops_path: "data/stops.txt"
```

## Degraded Mode

If static data files are missing, the system still works but with reduced capabilities:

**Without `topology.json`:**
- ❌ No Rail-Lock spatial awareness
- ❌ Cantonnement uses GPS-based ordering (unsafe for loops)
- ❌ No gradient data for physics

**Without `stops.txt`:**
- ❌ No Holographic Positioning
- ❌ Cannot create trains from TripUpdate alone

**Warning messages will be logged when files are missing.**

## Troubleshooting

### "GTFS file not found"
Check the file path is correct. Use absolute paths.

### Download timeout
Increase timeout: `--timeout 600` (10 minutes)

### "stops.txt not found in GTFS ZIP"
Your GTFS file may be incomplete. Download from official source.

### "shapes.txt not found in GTFS ZIP"
Not all GTFS feeds include shape data. Rail-Lock requires shapes.

## Data Sources

### France (Recommended)
- **IDFM** (Île-de-France): Default in the tool
- **transport.data.gouv.fr**: Complete French transport data

### Other Countries
Use your local transport authority's GTFS feed:
```bash
python -m src.tools.prepare_data --url <YOUR_GTFS_URL>
```

## Performance

Typical processing times:

| Dataset Size | Download | Processing | Total |
|-------------|----------|------------|-------|
| IDFM (~50MB) | ~30s | ~5s | ~35s |
| France (~500MB) | ~5min | ~30s | ~6min |

## Updates

Re-run the data preparation tool periodically to get:
- Updated schedules
- New routes
- Route modifications
- New stations

```bash
# Monthly update recommended
python -m src.tools.prepare_data
```

## Help

```bash
# Full help
python -m src.tools.prepare_data --help

# Verbose output
python -m src.tools.prepare_data --verbose

# Version
python -m src.tools.prepare_data --version
```

## See Also

- [Full Documentation](docs/DATA_PREPARATION.md)
- [Rail-Lock Details](docs/RAIL_LOCK_v6.md)
- [Main README](README.md)

---

**HNPS v6.0** - État de l'Art Railway Surveillance
