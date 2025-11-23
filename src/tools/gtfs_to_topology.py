#!/usr/bin/env python3
"""
GTFS to Topology Converter - HNPS v6.0 Rail-Lock Engine
========================================================
Converts static GTFS ZIP files to optimized JSON topology format for the
Rail-Lock engine. This script processes shapes.txt and trips.txt from GTFS
data to create a topology file containing route geometries.

Technical Specifications:
- Input: GTFS static ZIP file (standard format)
- Output: JSON topology file with shape_id, route_id, and coordinate points
- Performance: Handles millions of points using pandas for efficiency
- Memory: Optimized for large datasets (e.g., GTFS France entière)
- Progress: Real-time progress bar using tqdm

Author: HNPS Engineering Team
Version: 6.0.0
Python: 3.12+
"""

import argparse
import json
import sys
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import pandas as pd
from tqdm import tqdm


class GTFSTopologyConverter:
    """
    Converts GTFS static data to topology JSON format.
    
    This class handles the complete conversion process:
    1. Loading GTFS data from ZIP file
    2. Associating shapes with routes via trips
    3. Ordering and cleaning shape points
    4. Generating optimized JSON output
    """
    
    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize the converter.
        
        Args:
            input_path: Path to input GTFS ZIP file
            output_path: Path to output JSON file
        """
        self.input_path = input_path
        self.output_path = output_path
        self.shapes_df: Optional[pd.DataFrame] = None
        self.trips_df: Optional[pd.DataFrame] = None
    
    def load_gtfs_data(self) -> None:
        """
        Load shapes.txt and trips.txt from GTFS ZIP file.
        
        Only loads required columns to minimize memory usage.
        Uses efficient pandas data types for better memory efficiency.
        
        Raises:
            FileNotFoundError: If input ZIP file doesn't exist
            KeyError: If required GTFS files are missing from ZIP
            ValueError: If required columns are missing
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        print(f"📦 Loading GTFS data from {self.input_path}...")
        
        try:
            with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
                # Load shapes.txt - required columns
                if 'shapes.txt' not in zip_ref.namelist():
                    raise KeyError("shapes.txt not found in GTFS ZIP file")
                
                with zip_ref.open('shapes.txt') as shapes_file:
                    # Use TextIOWrapper to handle encoding
                    shapes_text = TextIOWrapper(shapes_file, encoding='utf-8-sig')
                    self.shapes_df = pd.read_csv(
                        shapes_text,
                        usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'],
                        dtype={
                            'shape_id': str,
                            'shape_pt_lat': float,
                            'shape_pt_lon': float,
                            'shape_pt_sequence': int
                        }
                    )
                
                print(f"  ✓ Loaded {len(self.shapes_df):,} shape points")
                
                # Load trips.txt - only required columns for shape-to-route mapping
                if 'trips.txt' not in zip_ref.namelist():
                    raise KeyError("trips.txt not found in GTFS ZIP file")
                
                with zip_ref.open('trips.txt') as trips_file:
                    trips_text = TextIOWrapper(trips_file, encoding='utf-8-sig')
                    # Only load trips that have shape_id
                    self.trips_df = pd.read_csv(
                        trips_text,
                        usecols=['route_id', 'shape_id'],
                        dtype={
                            'route_id': str,
                            'shape_id': str
                        }
                    )
                    # Remove trips without shape_id
                    self.trips_df = self.trips_df.dropna(subset=['shape_id'])
                
                print(f"  ✓ Loaded {len(self.trips_df):,} trips with shapes")
                
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {self.input_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading GTFS data: {e}")
    
    def associate_shapes_to_routes(self) -> Dict[str, str]:
        """
        Associate each shape_id to a route_id.
        
        Handles N:1 relationship (multiple shapes per route) by:
        - Counting which route_id uses each shape_id most frequently
        - Selecting the most common route_id for shapes used by multiple routes
        
        Uses efficient pandas groupby for better performance on large datasets.
        
        Returns:
            Dictionary mapping shape_id to route_id
        """
        print("🔗 Associating shapes to routes...")
        
        # Use groupby for efficient processing of all shapes at once
        # For each shape_id, count route_id occurrences and select most common
        shape_route_mapping: Dict[str, str] = {}
        
        grouped = self.trips_df.groupby('shape_id')['route_id']
        
        for shape_id, routes in tqdm(grouped, desc="Processing shape-route associations"):
            # Count occurrences and select most frequent route
            route_counter = Counter(routes)
            most_common_route = route_counter.most_common(1)[0][0]
            shape_route_mapping[shape_id] = most_common_route
        
        print(f"  ✓ Mapped {len(shape_route_mapping):,} unique shapes to routes")
        return shape_route_mapping
    
    def clean_and_order_points(self, shape_points: pd.DataFrame) -> List[List[float]]:
        """
        Clean and order shape points.
        
        Performs:
        1. Sort by shape_pt_sequence (strict ordering)
        2. Remove duplicate consecutive points
        3. Basic validation (lat/lon bounds)
        
        Args:
            shape_points: DataFrame with shape points for a single shape_id
        
        Returns:
            List of [longitude, latitude] coordinate pairs
        """
        # Create a copy and sort by sequence number to avoid modifying the input
        shape_points = shape_points.copy().sort_values('shape_pt_sequence')
        
        # Extract coordinates as [lon, lat] pairs (GeoJSON format)
        points = shape_points[['shape_pt_lon', 'shape_pt_lat']].values.tolist()
        
        # Remove consecutive duplicate points
        cleaned_points: List[List[float]] = []
        prev_point = None
        
        for point in points:
            lon, lat = point
            
            # Basic validation: check if coordinates are within reasonable bounds
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                continue  # Skip invalid points
            
            # Skip consecutive duplicates
            if prev_point is None or prev_point != point:
                cleaned_points.append([lon, lat])
                prev_point = point
        
        return cleaned_points
    
    def generate_topology(self) -> List[Dict]:
        """
        Generate the topology JSON structure.
        
        Creates the final data structure with:
        - shape_id: Unique identifier for the geometry
        - route_id: Associated route
        - points: Array of [lon, lat] coordinates
        
        Returns:
            List of topology dictionaries
        """
        print("🌐 Generating topology structure...")
        
        # Get shape to route mapping
        shape_route_mapping = self.associate_shapes_to_routes()
        
        # Get unique shape_ids
        unique_shapes = self.shapes_df['shape_id'].unique()
        
        topology_list: List[Dict] = []
        
        # Process each shape
        for shape_id in tqdm(unique_shapes, desc="Building topology"):
            # Get route_id for this shape
            route_id = shape_route_mapping.get(shape_id)
            
            if route_id is None:
                # Skip shapes without associated route (orphaned shapes)
                continue
            
            # Get all points for this shape
            shape_points = self.shapes_df[self.shapes_df['shape_id'] == shape_id]
            
            # Clean and order points
            points = self.clean_and_order_points(shape_points)
            
            # Skip shapes with too few points (invalid geometries)
            if len(points) < 2:
                continue
            
            # Create topology entry
            topology_entry = {
                "shape_id": shape_id,
                "route_id": route_id,
                "points": points
            }
            
            topology_list.append(topology_entry)
        
        print(f"  ✓ Generated {len(topology_list):,} topology entries")
        return topology_list
    
    def write_output(self, topology: List[Dict]) -> None:
        """
        Write topology to JSON file.
        
        Uses compact JSON format for optimal file size while maintaining readability.
        
        Args:
            topology: List of topology dictionaries to write
        """
        print(f"💾 Writing output to {self.output_path}...")
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON with minimal whitespace for optimal size
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(
                topology,
                f,
                ensure_ascii=False,
                separators=(',', ':')  # Compact format
            )
        
        # Calculate file size
        file_size = self.output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"  ✓ Wrote {file_size_mb:.2f} MB to {self.output_path}")
    
    def convert(self) -> None:
        """
        Execute the complete conversion process.
        
        This is the main entry point that orchestrates:
        1. Loading GTFS data
        2. Generating topology
        3. Writing output
        """
        try:
            # Load GTFS data
            self.load_gtfs_data()
            
            # Generate topology
            topology = self.generate_topology()
            
            if not topology:
                print("⚠️  Warning: No valid topology entries generated")
                sys.exit(1)
            
            # Write output
            self.write_output(topology)
            
            print("✅ Conversion completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during conversion: {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """
    CLI entry point for GTFS to Topology converter.
    
    Parses command-line arguments and executes the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert GTFS static ZIP to topology JSON for HNPS v6.0 Rail-Lock Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input gtfs.zip --output topology.json
  %(prog)s --input /path/to/france_gtfs.zip --output /output/topology.json

Output Format:
  [
    {
      "shape_id": "SHAPE_123",
      "route_id": "ROUTE_A",
      "points": [[lon1, lat1], [lon2, lat2], ...]
    }
  ]

Requirements:
  - Input must be a valid GTFS static ZIP file
  - Must contain shapes.txt and trips.txt
  - Python 3.12+ with pandas and tqdm
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input GTFS ZIP file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output JSON topology file'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 6.0.0 (HNPS Rail-Lock Engine)'
    )
    
    args = parser.parse_args()
    
    # Create converter and execute
    converter = GTFSTopologyConverter(
        input_path=args.input,
        output_path=args.output
    )
    
    converter.convert()


if __name__ == '__main__':
    main()
