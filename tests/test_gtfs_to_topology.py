"""
Test Suite for GTFS to Topology Converter
==========================================
Tests the conversion of GTFS static data to JSON topology format.

This test suite creates minimal GTFS ZIP files for testing and validates
the conversion logic.
"""

import json
import zipfile
from pathlib import Path

import pytest
import pandas as pd

from src.tools.gtfs_to_topology import GTFSTopologyConverter


@pytest.fixture
def sample_gtfs_zip(tmp_path: Path) -> Path:
    """
    Create a minimal valid GTFS ZIP file for testing.
    
    Returns:
        Path to the created ZIP file
    """
    zip_path = tmp_path / "sample_gtfs.zip"
    
    # Create sample shapes.txt data
    shapes_data = """shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
SHAPE_A,48.8566,2.3522,1
SHAPE_A,48.8570,2.3530,2
SHAPE_A,48.8575,2.3540,3
SHAPE_B,48.8600,2.3600,1
SHAPE_B,48.8605,2.3610,2
SHAPE_C,48.8700,2.3700,1
SHAPE_C,48.8705,2.3710,2
SHAPE_C,48.8710,2.3720,3
SHAPE_C,48.8715,2.3730,4
"""
    
    # Create sample trips.txt data
    trips_data = """route_id,trip_id,shape_id
ROUTE_1,TRIP_A1,SHAPE_A
ROUTE_1,TRIP_A2,SHAPE_A
ROUTE_2,TRIP_B1,SHAPE_B
ROUTE_3,TRIP_C1,SHAPE_C
ROUTE_3,TRIP_C2,SHAPE_C
"""
    
    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('shapes.txt', shapes_data)
        zf.writestr('trips.txt', trips_data)
    
    return zip_path


@pytest.fixture
def sample_gtfs_with_duplicates(tmp_path: Path) -> Path:
    """
    Create a GTFS ZIP with duplicate consecutive points for testing cleaning.
    
    Returns:
        Path to the created ZIP file
    """
    zip_path = tmp_path / "gtfs_with_duplicates.zip"
    
    # Create shapes with duplicate consecutive points
    shapes_data = """shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
SHAPE_DUP,48.8566,2.3522,1
SHAPE_DUP,48.8566,2.3522,2
SHAPE_DUP,48.8570,2.3530,3
SHAPE_DUP,48.8570,2.3530,4
SHAPE_DUP,48.8575,2.3540,5
"""
    
    trips_data = """route_id,trip_id,shape_id
ROUTE_DUP,TRIP_DUP,SHAPE_DUP
"""
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('shapes.txt', shapes_data)
        zf.writestr('trips.txt', trips_data)
    
    return zip_path


@pytest.fixture
def sample_gtfs_multi_route(tmp_path: Path) -> Path:
    """
    Create a GTFS ZIP where one shape is used by multiple routes.
    Tests the most-frequent route selection logic.
    
    Returns:
        Path to the created ZIP file
    """
    zip_path = tmp_path / "gtfs_multi_route.zip"
    
    shapes_data = """shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
SHAPE_MULTI,48.8566,2.3522,1
SHAPE_MULTI,48.8570,2.3530,2
"""
    
    # SHAPE_MULTI is used more by ROUTE_X (3 times) than ROUTE_Y (1 time)
    trips_data = """route_id,trip_id,shape_id
ROUTE_X,TRIP_1,SHAPE_MULTI
ROUTE_X,TRIP_2,SHAPE_MULTI
ROUTE_X,TRIP_3,SHAPE_MULTI
ROUTE_Y,TRIP_4,SHAPE_MULTI
"""
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('shapes.txt', shapes_data)
        zf.writestr('trips.txt', trips_data)
    
    return zip_path


class TestGTFSTopologyConverter:
    """Test suite for GTFSTopologyConverter class."""
    
    def test_load_gtfs_data(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test loading GTFS data from ZIP file."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        
        converter.load_gtfs_data()
        
        # Verify shapes loaded
        assert converter.shapes_df is not None
        assert len(converter.shapes_df) == 9  # Total shape points
        assert 'shape_id' in converter.shapes_df.columns
        assert 'shape_pt_lat' in converter.shapes_df.columns
        assert 'shape_pt_lon' in converter.shapes_df.columns
        assert 'shape_pt_sequence' in converter.shapes_df.columns
        
        # Verify trips loaded
        assert converter.trips_df is not None
        assert len(converter.trips_df) == 5  # Total trips
        assert 'route_id' in converter.trips_df.columns
        assert 'shape_id' in converter.trips_df.columns
    
    def test_load_gtfs_missing_file(self, tmp_path: Path):
        """Test error handling when input file doesn't exist."""
        non_existent = tmp_path / "non_existent.zip"
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(non_existent, output_path)
        
        with pytest.raises(FileNotFoundError):
            converter.load_gtfs_data()
    
    def test_load_gtfs_invalid_zip(self, tmp_path: Path):
        """Test error handling for invalid ZIP file."""
        invalid_zip = tmp_path / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(invalid_zip, output_path)
        
        with pytest.raises(ValueError, match="Invalid ZIP file"):
            converter.load_gtfs_data()
    
    def test_associate_shapes_to_routes(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test shape to route association."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        converter.load_gtfs_data()
        
        mapping = converter.associate_shapes_to_routes()
        
        assert len(mapping) == 3  # Three unique shapes
        assert mapping['SHAPE_A'] == 'ROUTE_1'  # Used twice by ROUTE_1
        assert mapping['SHAPE_B'] == 'ROUTE_2'
        assert mapping['SHAPE_C'] == 'ROUTE_3'  # Used twice by ROUTE_3
    
    def test_associate_shapes_multi_route_selection(
        self, 
        sample_gtfs_multi_route: Path, 
        tmp_path: Path
    ):
        """Test that the most frequent route is selected for shapes used by multiple routes."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_multi_route, output_path)
        converter.load_gtfs_data()
        
        mapping = converter.associate_shapes_to_routes()
        
        # SHAPE_MULTI should be associated with ROUTE_X (3 uses vs 1)
        assert mapping['SHAPE_MULTI'] == 'ROUTE_X'
    
    def test_clean_and_order_points(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test point cleaning and ordering."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        converter.load_gtfs_data()
        
        # Get points for SHAPE_A
        shape_points = converter.shapes_df[converter.shapes_df['shape_id'] == 'SHAPE_A']
        points = converter.clean_and_order_points(shape_points)
        
        # Should have 3 points in correct order
        assert len(points) == 3
        assert points[0] == [2.3522, 48.8566]  # [lon, lat] format
        assert points[1] == [2.3530, 48.8570]
        assert points[2] == [2.3540, 48.8575]
    
    def test_clean_and_order_removes_duplicates(
        self, 
        sample_gtfs_with_duplicates: Path, 
        tmp_path: Path
    ):
        """Test that consecutive duplicate points are removed."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_with_duplicates, output_path)
        converter.load_gtfs_data()
        
        shape_points = converter.shapes_df[converter.shapes_df['shape_id'] == 'SHAPE_DUP']
        points = converter.clean_and_order_points(shape_points)
        
        # Should have 3 unique points (duplicates removed)
        assert len(points) == 3
        assert points[0] == [2.3522, 48.8566]
        assert points[1] == [2.3530, 48.8570]
        assert points[2] == [2.3540, 48.8575]
    
    def test_generate_topology(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test complete topology generation."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        converter.load_gtfs_data()
        
        topology = converter.generate_topology()
        
        # Should have 3 topology entries (one per shape)
        assert len(topology) == 3
        
        # Verify structure
        for entry in topology:
            assert 'shape_id' in entry
            assert 'route_id' in entry
            assert 'points' in entry
            assert isinstance(entry['points'], list)
            assert len(entry['points']) >= 2
            
            # Verify point format [lon, lat]
            for point in entry['points']:
                assert len(point) == 2
                assert isinstance(point[0], float)  # longitude
                assert isinstance(point[1], float)  # latitude
    
    def test_write_output(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test JSON output writing."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        converter.load_gtfs_data()
        
        topology = converter.generate_topology()
        converter.write_output(topology)
        
        # Verify file exists
        assert output_path.exists()
        
        # Verify JSON is valid and has correct structure
        with open(output_path, 'r') as f:
            loaded_topology = json.load(f)
        
        assert isinstance(loaded_topology, list)
        assert len(loaded_topology) == 3
        
        # Verify first entry
        entry = loaded_topology[0]
        assert 'shape_id' in entry
        assert 'route_id' in entry
        assert 'points' in entry
    
    def test_full_conversion(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test the complete end-to-end conversion process."""
        output_path = tmp_path / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, output_path)
        
        converter.convert()
        
        # Verify output file exists and is valid
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            topology = json.load(f)
        
        # Basic validation
        assert len(topology) == 3
        
        # Find SHAPE_A entry and verify it
        shape_a = next((e for e in topology if e['shape_id'] == 'SHAPE_A'), None)
        assert shape_a is not None
        assert shape_a['route_id'] == 'ROUTE_1'
        assert len(shape_a['points']) == 3
        
        # Verify point format
        assert shape_a['points'][0][0] == 2.3522  # lon
        assert shape_a['points'][0][1] == 48.8566  # lat
    
    def test_output_directory_creation(self, sample_gtfs_zip: Path, tmp_path: Path):
        """Test that output directory is created if it doesn't exist."""
        nested_output = tmp_path / "nested" / "path" / "output.json"
        converter = GTFSTopologyConverter(sample_gtfs_zip, nested_output)
        converter.load_gtfs_data()
        
        topology = converter.generate_topology()
        converter.write_output(topology)
        
        # Verify nested directory was created
        assert nested_output.exists()
        assert nested_output.parent.is_dir()
