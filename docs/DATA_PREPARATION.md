# Data Preparation Tool - HNPS v6.0

## Overview

The Data Preparation Tool (`src/tools/prepare_data.py`) automates the acquisition and processing of static GTFS data required for HNPS v6.0 features:

- **Rail-Lock**: Requires `topology.json` for track geometry projection
- **Holographic Positioning**: Requires `stops.txt` for GPS-less positioning

## Features

✅ **Automated Download**: Downloads GTFS ZIP files from URLs  
✅ **Local File Support**: Can process local GTFS ZIP files  
✅ **Stops Extraction**: Extracts `stops.txt` for station location mapping  
✅ **Topology Generation**: Generates `topology.json` by crossing `shapes.txt` and `trips.txt`  
✅ **Production Ready**: Comprehensive error handling, logging, and cleanup  

## Usage

### Basic Usage (Default IDFM)

```bash
python -m src.tools.prepare_data
```

This downloads from the default IDFM (Île-de-France Mobilités) URL and generates:
- `data/stops.txt`
- `data/topology.json`

### Using a Local GTFS File

```bash
python -m src.tools.prepare_data --local /path/to/gtfs.zip
```

### Using a Custom URL

```bash
python -m src.tools.prepare_data --url https://example.com/gtfs.zip
```

### Custom Output Directory

```bash
python -m src.tools.prepare_data --output-dir /custom/data
```

### Custom Output Paths

```bash
python -m src.tools.prepare_data \
  --stops-output /custom/stops.txt \
  --topology-output /custom/topology.json
```

### Verbose Logging

```bash
python -m src.tools.prepare_data --verbose
```

## Output Files

### 1. `data/stops.txt`

GTFS stops file containing station locations.

**Format**: Standard GTFS `stops.txt` format
```csv
stop_id,stop_name,stop_lat,stop_lon
STOP1,Gare du Nord,48.8809,2.3553
STOP2,Chatelet,48.8586,2.3475
```

**Used By**: 
- Holographic Positioning (v5.0) - converts stop_id to GPS coordinates
- Station proximity detection

### 2. `data/topology.json`

Route geometries with shape-to-route mapping.

**Format**: JSON array of topology entries
```json
[
  {
    "shape_id": "SHAPE_123",
    "route_id": "ROUTE_A",
    "points": [
      [2.3553, 48.8809],
      [2.3500, 48.8700],
      [2.3475, 48.8586]
    ]
  }
]
```

**Used By**:
- Rail-Lock Engine (v6.0) - projects GPS onto track geometry
- Track distance (PK) calculation
- Gradient extraction for Davis physics
- Cantonnement (Moving Block) train ordering

## Requirements

- Python 3.12+
- Dependencies:
  - `pandas` - GTFS data processing
  - `tqdm` - Progress bars
  - `requests` - HTTP downloads
  - `zipfile` (built-in) - ZIP extraction

## Integration with main.py

The `main.py` file automatically checks for these files when starting the Fusion Engine:

```python
async def _start_fusion_engine(self) -> None:
    topology_path = "data/topology.json"
    stops_path = "data/stops.txt"
    
    # Warns if files are missing
    if not os.path.exists(topology_path):
        logger.warning("Topology file missing - Rail-Lock degraded")
    
    if not os.path.exists(stops_path):
        logger.warning("Stops file missing - Holographic Positioning degraded")
    
    # Injects paths into Fusion Engine
    self.fusion_engine = HybridFusionEngine(
        topology_path=topology_path if os.path.exists(topology_path) else None,
        stops_path=stops_path if os.path.exists(stops_path) else None
    )
```

## Degraded Mode

If static data files are missing:

**Without `topology.json`**:
- ❌ Rail-Lock disabled - no track projection
- ❌ Cantonnement uses GPS-based ordering (unsafe for loops/U-turns)
- ❌ No gradient data for Davis physics

**Without `stops.txt`**:
- ❌ Holographic Positioning disabled
- ❌ Cannot create trains from TripUpdate alone
- ❌ Must rely on VehiclePosition data only

## Architecture

The tool reuses existing infrastructure:

```
prepare_data.py
    ├─> _load_gtfs()
    │   ├─> _download_gtfs() [if URL]
    │   └─> [local path]
    │
    ├─> extract_stops()
    │   └─> Extract stops.txt from ZIP
    │
    └─> generate_topology()
        └─> GTFSTopologyConverter.convert()
            ├─> load_gtfs_data()
            ├─> associate_shapes_to_routes()
            └─> generate_topology()
```

## Error Handling

The tool handles common errors gracefully:

- **Network Errors**: Reports download failures with clear messages
- **Missing Files**: Validates ZIP contents before processing
- **Invalid Data**: Validates GTFS structure (required columns, etc.)
- **Disk Space**: Creates output directories automatically
- **Cleanup**: Removes temporary downloaded files

## Example Output

```
================================================================================
🚂 HNPS v6.0 - Data Preparation Tool
   Automated Static Data Acquisition
================================================================================

📥 Downloading GTFS from https://...
  Progress: 50.0% (25.5 MB)
  ✓ Downloaded 51.2 MB

📍 Extracting stops.txt to data/stops.txt...
  ✓ Extracted stops.txt (45.23 KB)

🌐 Generating topology.json to data/topology.json...
📦 Loading GTFS data from data/gtfs_temp.zip...
  ✓ Loaded 125,432 shape points
  ✓ Loaded 8,234 trips with shapes
🌐 Generating topology structure...
🔗 Associating shapes to routes...
  ✓ Mapped 1,245 unique shapes to routes
Building topology: 100%|████████████████| 1245/1245 [00:02<00:00, 512.34it/s]
  ✓ Generated 1,245 topology entries
💾 Writing output to data/topology.json...
  ✓ Wrote 12.45 MB to data/topology.json
✅ Conversion completed successfully!
  ✓ Topology generated successfully

✅ Data preparation completed successfully!
   📍 stops.txt: data/stops.txt
   🌐 topology.json: data/topology.json

🧹 Cleaning up temporary files...
  ✓ Temporary files removed

================================================================================
✅ SUCCESS - Data ready for HNPS v6.0
================================================================================
```

## Testing

The tool includes comprehensive test coverage:

```bash
# Run all data preparation tests
pytest tests/test_prepare_data.py -v

# Test with local file
pytest tests/test_prepare_data.py::TestDataPreparationTool::test_prepare_complete_workflow -v
```

## Performance

Typical processing times (IDFM dataset ~50MB):

- Download: ~30 seconds (depends on connection)
- Stops extraction: <1 second
- Topology generation: ~5 seconds
- **Total**: ~35-40 seconds

For larger datasets (e.g., France entière ~500MB):

- Download: ~5 minutes
- Topology generation: ~30 seconds
- **Total**: ~6 minutes

## Troubleshooting

### "GTFS file not found"
- Check that the file path is correct
- Use absolute paths for clarity

### "stops.txt not found in GTFS ZIP"
- GTFS file may be incomplete or corrupted
- Download from official source
- Verify ZIP file integrity

### "shapes.txt not found in GTFS ZIP"
- Not all GTFS feeds include shape data
- Rail-Lock will be disabled without shapes
- Use alternative data source with shapes

### Download timeout
- Increase timeout in `requests.get()` call
- Check network connection
- Try local file instead

## Best Practices

1. **Run on Production Setup**: Always prepare data before launching the main system
2. **Version Control**: Keep `data/` directory in `.gitignore` (data files are large)
3. **Regular Updates**: Re-run tool monthly to get updated schedules and routes
4. **Backup**: Keep backup copies of generated files for disaster recovery
5. **Validation**: Check that both files are generated successfully before starting main system

## See Also

- [Rail-Lock Documentation](RAIL_LOCK_v6.md) - Details on topology format and usage
- [GTFS Specification](https://gtfs.org/schedule/) - GTFS static format reference
- [src/tools/gtfs_to_topology.py](../src/tools/gtfs_to_topology.py) - Underlying topology converter
