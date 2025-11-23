"""
Module E: Stop Registry for Holographic Positioning
====================================================
This module implements the Stop Registry system for HNPS v5.0, enabling
positional inference when VehiclePosition data is unavailable.

The Stop Registry loads GTFS static stops.txt files and provides fast lookup
of stop coordinates. This enables the system to infer train positions from
TripUpdate messages by mapping stop_id to geographic coordinates.

Key Features:
- Load stops.txt from GTFS static data
- Fast O(1) lookup of stop coordinates by stop_id
- Graceful degradation when stops.txt is unavailable
- Memory-efficient storage of stop data

HNPS v5.0 Component: Holographic Positioning (Positional Inference)
"""

import csv
import structlog
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = structlog.get_logger(__name__)


class StopRegistry:
    """
    Registry of transit stops loaded from GTFS static stops.txt.
    
    This class enables Holographic Positioning by converting stop IDs
    to geographic coordinates. When VehiclePosition data is unavailable,
    the system can infer train positions from TripUpdate stop_ids.
    
    Usage:
        registry = StopRegistry("data/stops.txt")
        lat, lon = registry.get_stop_location("STOP_001")
        if lat and lon:
            # Use coordinates for positional inference
    """
    
    def __init__(self, stops_file_path: Optional[str] = None) -> None:
        """
        Initialize the Stop Registry.
        
        Args:
            stops_file_path: Path to GTFS stops.txt file. If None or file doesn't exist,
                           operates in degraded mode (no positional inference available).
        """
        self.stops_file_path = stops_file_path
        self._stops: Dict[str, Tuple[float, float]] = {}  # stop_id -> (lat, lon)
        self._available = False
        
        if stops_file_path:
            self._load_stops(stops_file_path)
    
    def _load_stops(self, file_path: str) -> None:
        """
        Load stops from GTFS stops.txt file.
        
        The stops.txt file follows the GTFS static specification:
        https://gtfs.org/schedule/reference/#stopstxt
        
        Required columns:
        - stop_id: Unique identifier for the stop
        - stop_lat: Latitude of the stop
        - stop_lon: Longitude of the stop
        
        Args:
            file_path: Path to stops.txt file.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(
                "stops_file_not_found",
                path=file_path,
                message="Stop Registry operating in degraded mode - positional inference disabled"
            )
            return
        
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Verify required columns
                if not all(col in reader.fieldnames for col in ['stop_id', 'stop_lat', 'stop_lon']):
                    logger.error(
                        "stops_file_invalid_format",
                        path=file_path,
                        fieldnames=reader.fieldnames,
                        message="stops.txt missing required columns (stop_id, stop_lat, stop_lon)"
                    )
                    return
                
                stops_loaded = 0
                for row in reader:
                    stop_id = row['stop_id'].strip()
                    
                    try:
                        stop_lat = float(row['stop_lat'])
                        stop_lon = float(row['stop_lon'])
                        
                        # Basic validation of coordinates
                        if not (-90 <= stop_lat <= 90) or not (-180 <= stop_lon <= 180):
                            logger.warning(
                                "invalid_stop_coordinates",
                                stop_id=stop_id,
                                lat=stop_lat,
                                lon=stop_lon,
                                message="Stop coordinates out of valid range"
                            )
                            continue
                        
                        self._stops[stop_id] = (stop_lat, stop_lon)
                        stops_loaded += 1
                        
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "stop_parse_error",
                            stop_id=stop_id,
                            error=str(e)
                        )
                        continue
                
                self._available = stops_loaded > 0
                
                logger.info(
                    "stops_registry_loaded",
                    file_path=file_path,
                    stops_count=stops_loaded,
                    status="available" if self._available else "empty"
                )
                
        except Exception as e:
            logger.error(
                "stops_registry_load_error",
                path=file_path,
                error=str(e),
                message="Failed to load stops.txt - positional inference disabled"
            )
    
    def get_stop_location(self, stop_id: str) -> Optional[Tuple[float, float]]:
        """
        Get geographic coordinates for a stop.
        
        This is the core method for Holographic Positioning, converting
        a stop_id from a TripUpdate into geographic coordinates that can
        be used to infer train position.
        
        Args:
            stop_id: Stop identifier from GTFS data.
            
        Returns:
            Tuple of (latitude, longitude) if stop exists, None otherwise.
        """
        return self._stops.get(stop_id)
    
    def is_available(self) -> bool:
        """
        Check if Stop Registry is available.
        
        Returns:
            True if stops have been loaded and registry is operational.
        """
        return self._available
    
    def get_stops_count(self) -> int:
        """
        Get the number of stops loaded in the registry.
        
        Returns:
            Number of stops.
        """
        return len(self._stops)
