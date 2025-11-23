"""
Tests for Data Preparation Tool - HNPS v6.0
============================================
Tests for src/tools/prepare_data.py
"""

import json
import pytest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import io

from src.tools.prepare_data import (
    DataPreparationTool,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_GTFS_URL
)


@pytest.fixture
def sample_gtfs_zip(tmp_path):
    """
    Create a sample GTFS ZIP file for testing.
    
    Contains:
    - stops.txt with valid GTFS data
    - shapes.txt with sample shape points
    - trips.txt with sample trips
    """
    zip_path = tmp_path / "sample_gtfs.zip"
    
    # Sample stops.txt content
    stops_content = """stop_id,stop_name,stop_lat,stop_lon
STOP1,Station A,48.8566,2.3522
STOP2,Station B,48.8584,2.2945
STOP3,Station C,48.8738,2.2950
"""
    
    # Sample shapes.txt content
    shapes_content = """shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
SHAPE1,48.8566,2.3522,1
SHAPE1,48.8575,2.3400,2
SHAPE1,48.8584,2.2945,3
SHAPE2,48.8584,2.2945,1
SHAPE2,48.8650,2.2947,2
SHAPE2,48.8738,2.2950,3
"""
    
    # Sample trips.txt content
    trips_content = """route_id,trip_id,shape_id
ROUTE_A,TRIP1,SHAPE1
ROUTE_A,TRIP2,SHAPE1
ROUTE_B,TRIP3,SHAPE2
"""
    
    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.writestr('stops.txt', stops_content)
        zf.writestr('shapes.txt', shapes_content)
        zf.writestr('trips.txt', trips_content)
    
    return zip_path


class TestDataPreparationTool:
    """Test suite for DataPreparationTool."""
    
    def test_initialization(self, tmp_path):
        """Test tool initialization with default parameters."""
        tool = DataPreparationTool(
            source="test.zip",
            output_dir=tmp_path
        )
        
        assert tool.source == "test.zip"
        assert tool.output_dir == tmp_path
        assert tool.stops_output == tmp_path / "stops.txt"
        assert tool.topology_output == tmp_path / "topology.json"
        assert tool.is_temporary_file is False
    
    def test_initialization_custom_paths(self, tmp_path):
        """Test tool initialization with custom output paths."""
        custom_stops = tmp_path / "custom_stops.txt"
        custom_topology = tmp_path / "custom_topology.json"
        
        tool = DataPreparationTool(
            source="test.zip",
            output_dir=tmp_path,
            stops_output=custom_stops,
            topology_output=custom_topology
        )
        
        assert tool.stops_output == custom_stops
        assert tool.topology_output == custom_topology
    
    def test_is_url_detection(self, tmp_path):
        """Test URL detection."""
        tool = DataPreparationTool(source="http://example.com/gtfs.zip", output_dir=tmp_path)
        assert tool._is_url("http://example.com/gtfs.zip") is True
        assert tool._is_url("https://example.com/gtfs.zip") is True
        assert tool._is_url("/local/path/gtfs.zip") is False
        assert tool._is_url("gtfs.zip") is False
    
    def test_load_gtfs_local(self, tmp_path, sample_gtfs_zip):
        """Test loading GTFS from local path."""
        tool = DataPreparationTool(
            source=str(sample_gtfs_zip),
            output_dir=tmp_path
        )
        
        loaded_path = tool._load_gtfs()
        
        assert loaded_path == sample_gtfs_zip
        assert tool.is_temporary_file is False
    
    def test_load_gtfs_local_missing_file(self, tmp_path):
        """Test loading GTFS from non-existent local path."""
        tool = DataPreparationTool(
            source="/nonexistent/gtfs.zip",
            output_dir=tmp_path
        )
        
        with pytest.raises(FileNotFoundError):
            tool._load_gtfs()
    
    @patch('src.tools.prepare_data.requests.get')
    def test_download_gtfs_success(self, mock_get, tmp_path):
        """Test downloading GTFS from URL."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test data chunk']
        mock_get.return_value = mock_response
        
        tool = DataPreparationTool(
            source="http://example.com/gtfs.zip",
            output_dir=tmp_path
        )
        
        downloaded_path = tool._download_gtfs()
        
        assert downloaded_path.exists()
        assert tool.is_temporary_file is True
        assert downloaded_path.name == "gtfs_temp.zip"
    
    @patch('src.tools.prepare_data.requests.get')
    def test_download_gtfs_failure(self, mock_get, tmp_path):
        """Test downloading GTFS with network error."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")
        
        tool = DataPreparationTool(
            source="http://example.com/gtfs.zip",
            output_dir=tmp_path
        )
        
        with pytest.raises(requests.RequestException):
            tool._download_gtfs()
    
    def test_extract_stops_success(self, tmp_path, sample_gtfs_zip):
        """Test extracting stops.txt from GTFS ZIP."""
        output_dir = tmp_path / "output"
        tool = DataPreparationTool(
            source=str(sample_gtfs_zip),
            output_dir=output_dir
        )
        
        # Load GTFS first
        tool.gtfs_zip_path = tool._load_gtfs()
        
        # Extract stops
        tool.extract_stops()
        
        # Verify extraction
        assert tool.stops_output.exists()
        content = tool.stops_output.read_text()
        assert "stop_id" in content
        assert "STOP1" in content
        assert "Station A" in content
    
    def test_extract_stops_missing_file(self, tmp_path):
        """Test extracting stops.txt when GTFS not loaded."""
        tool = DataPreparationTool(
            source="test.zip",
            output_dir=tmp_path
        )
        
        with pytest.raises(FileNotFoundError):
            tool.extract_stops()
    
    def test_extract_stops_missing_in_zip(self, tmp_path):
        """Test extracting stops.txt when not in ZIP."""
        # Create ZIP without stops.txt
        zip_path = tmp_path / "no_stops.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('trips.txt', 'trip_id,route_id\n')
        
        tool = DataPreparationTool(
            source=str(zip_path),
            output_dir=tmp_path / "output"
        )
        tool.gtfs_zip_path = tool._load_gtfs()
        
        with pytest.raises(KeyError):
            tool.extract_stops()
    
    def test_generate_topology_success(self, tmp_path, sample_gtfs_zip):
        """Test generating topology.json from GTFS ZIP."""
        output_dir = tmp_path / "output"
        tool = DataPreparationTool(
            source=str(sample_gtfs_zip),
            output_dir=output_dir
        )
        
        # Load GTFS first
        tool.gtfs_zip_path = tool._load_gtfs()
        
        # Generate topology
        tool.generate_topology()
        
        # Verify generation
        assert tool.topology_output.exists()
        
        # Verify topology structure
        with open(tool.topology_output, 'r') as f:
            topology = json.load(f)
        
        assert isinstance(topology, list)
        assert len(topology) > 0
        
        # Check first entry structure
        entry = topology[0]
        assert "shape_id" in entry
        assert "route_id" in entry
        assert "points" in entry
        assert isinstance(entry["points"], list)
    
    def test_generate_topology_missing_file(self, tmp_path):
        """Test generating topology when GTFS not loaded."""
        tool = DataPreparationTool(
            source="test.zip",
            output_dir=tmp_path
        )
        
        with pytest.raises(FileNotFoundError):
            tool.generate_topology()
    
    def test_cleanup_temporary_file(self, tmp_path):
        """Test cleanup of temporary downloaded file."""
        temp_file = tmp_path / "gtfs_temp.zip"
        temp_file.write_text("test")
        
        tool = DataPreparationTool(
            source="http://example.com/gtfs.zip",
            output_dir=tmp_path
        )
        tool.gtfs_zip_path = temp_file
        tool.is_temporary_file = True
        
        tool.cleanup()
        
        assert not temp_file.exists()
    
    def test_cleanup_no_temporary_file(self, tmp_path, sample_gtfs_zip):
        """Test cleanup when using local file (no cleanup needed)."""
        tool = DataPreparationTool(
            source=str(sample_gtfs_zip),
            output_dir=tmp_path
        )
        tool.gtfs_zip_path = sample_gtfs_zip
        tool.is_temporary_file = False
        
        tool.cleanup()
        
        # Original file should still exist
        assert sample_gtfs_zip.exists()
    
    def test_prepare_complete_workflow(self, tmp_path, sample_gtfs_zip):
        """Test complete data preparation workflow."""
        output_dir = tmp_path / "output"
        
        tool = DataPreparationTool(
            source=str(sample_gtfs_zip),
            output_dir=output_dir
        )
        
        # Run complete preparation
        tool.prepare()
        
        # Verify both outputs exist
        assert tool.stops_output.exists()
        assert tool.topology_output.exists()
        
        # Verify stops.txt content
        stops_content = tool.stops_output.read_text()
        assert "stop_id" in stops_content
        assert "STOP1" in stops_content
        
        # Verify topology.json content
        with open(tool.topology_output, 'r') as f:
            topology = json.load(f)
        assert isinstance(topology, list)
        assert len(topology) > 0
    
    @patch('src.tools.prepare_data.requests.get')
    def test_prepare_with_download(self, mock_get, tmp_path, sample_gtfs_zip):
        """Test complete workflow with download."""
        # Mock download to return local file content
        with open(sample_gtfs_zip, 'rb') as f:
            file_content = f.read()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': str(len(file_content))}
        mock_response.iter_content.return_value = [file_content]
        mock_get.return_value = mock_response
        
        output_dir = tmp_path / "output"
        tool = DataPreparationTool(
            source="http://example.com/gtfs.zip",
            output_dir=output_dir
        )
        
        # Run complete preparation
        tool.prepare()
        
        # Verify both outputs exist
        assert tool.stops_output.exists()
        assert tool.topology_output.exists()
        
        # Verify temporary file was cleaned up
        temp_file = output_dir / "gtfs_temp.zip"
        assert not temp_file.exists()


class TestMainCLI:
    """Test suite for CLI functionality."""
    
    def test_default_gtfs_url(self):
        """Test that default GTFS URL is defined."""
        assert DEFAULT_GTFS_URL.startswith("http")
    
    def test_default_output_dir(self):
        """Test that default output directory is 'data'."""
        assert DEFAULT_OUTPUT_DIR == Path("data")
