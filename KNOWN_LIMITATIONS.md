# Known Limitations and Future Improvements

This document outlines known limitations in the HNPS v5.0 system and recommended improvements for future versions.

## 🟡 Pending Improvements

### 1. Snap-to-Rail (Map Matching)

**Current State**: The Unscented Kalman Filter (UKF) operates in free 2D space (latitude/longitude), which means train positions can "float" beside the actual rail tracks on the map.

**Impact**: 
- Visual inaccuracy on maps
- Positions may show trains slightly off the track geometry
- Affects precision of distance calculations between trains

**Recommended Solution**:
Implement Map Matching constraint to force the state vector `[lat, lon]` to snap to the PostGIS rail network geometry:

```sql
-- Example PostGIS query for rail snapping
SELECT ST_ClosestPoint(
    railway_geometry,
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
) FROM railway_tracks
WHERE route_id = ?
```

**Implementation Requirements**:
- PostGIS database with rail network geometry (tracks table)
- pgrouting for network topology
- Modify `UnscentedKalmanFilter.update()` to project corrected position onto rail
- See reference document: `rail_lock_map_matching.tex` (if available)

**Effort**: Medium (requires PostGIS schema and geometric constraints)

### 2. UKF Geodesic Calculations

**Current State**: The motion model in `_motion_model()` uses a simplified "Flat Earth" approximation for position updates:

```python
# Current approximation (lines 323-324 in fusion.py)
delta_lat = (displacement * math.cos(bearing_rad)) / 111000
delta_lon = (displacement * math.sin(bearing_rad)) / (111000 * math.cos(math.radians(lat)))
```

**Impact**:
- Accumulates error over large distances (national scale)
- France spans ~1000km, where flat Earth approximation introduces measurable errors
- Less accurate for high-speed trains (TGV) traveling long distances

**Recommended Solution**:
Use proper geodesic calculations with the `pyproj` library:

```python
from pyproj import Geod

geod = Geod(ellps='WGS84')
new_lon, new_lat, _ = geod.fwd(
    lon, lat, 
    bearing, 
    displacement
)
```

**Implementation Requirements**:
- Add `pyproj` to requirements.txt
- Replace flat Earth approximation in `_motion_model()`
- Update distance calculations in `Position2D.distance_to()` if needed

**Effort**: Low (straightforward library integration)

**References**:
- pyproj documentation: https://pyproj4.github.io/pyproj/
- Vincenty's formulae for geodesic calculations

## ✅ Recently Fixed Issues

### 1. Moving Block Persistence (FIXED)
- **Issue**: Trains sorted by `lat + lon` caused incorrect ordering on loops/U-turns
- **Solution**: Added `track_distance` attribute (PK - Point Kilométrique) with fallback to lat/lon
- **Status**: ✅ Fixed in current version

### 2. Harvester Temporal Drift (FIXED)
- **Issue**: Sleep after processing caused data point loss over 24h
- **Solution**: Dynamic sleep calculation: `sleep = max(0, interval - processing_time)`
- **Status**: ✅ Fixed in current version

### 3. Fragile GTFS-RT Detection (FIXED)
- **Issue**: String-based detection ("gtfs-rt" in title) was brittle
- **Solution**: Multi-layer detection including Content-Type header inspection
- **Status**: ✅ Fixed in current version

### 4. Hardcoded Davis Physics (FIXED)
- **Issue**: Same coefficients for all vehicle types (TGV vs Metro)
- **Solution**: Dynamic coefficients based on GTFS route_type
- **Status**: ✅ Fixed in current version

## 🔮 Future Enhancements

### 3. Track Gradient Integration

**Current State**: Track gradient is a parameter but not automatically populated from real track data.

**Recommendation**: Integrate with PostGIS to query actual track gradients from digital elevation models (DEM).

### 4. Multi-Track Route Support

**Current State**: Moving block assumes single track per route.

**Recommendation**: Support multiple tracks (e.g., northbound/southbound) with proper direction detection.

### 5. Station Stop Prediction

**Current State**: Physics model treats all track segments equally.

**Recommendation**: Integrate GTFS stop times to predict station stops and adjust deceleration profiles accordingly.

## 📊 Performance Characteristics

Current system performance metrics:

- **Simulation Rate**: 1 Hz (configurable)
- **Harvest Interval**: 30s (configurable, now with accurate timing)
- **Position Accuracy**: ±10m (GPS accuracy, before snap-to-rail)
- **Velocity Accuracy**: ±2 m/s (Kalman filter estimation)
- **Moving Block Safety Distance**: 500m (configurable)

## 🛠️ Implementation Priority

1. **High Priority**: Snap-to-Rail (improves visual accuracy and distance calculations)
2. **Medium Priority**: UKF Geodesic (improves long-distance accuracy)
3. **Low Priority**: Track gradient integration (nice-to-have for physics accuracy)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Related Files**: 
- `src/engine/fusion.py` (UKF and Moving Block)
- `src/ingestion/harvester.py` (Harvester timing)
- `src/core/omniscience.py` (GTFS-RT detection)
