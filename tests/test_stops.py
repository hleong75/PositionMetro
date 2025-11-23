"""
Tests for the Stop Registry and Holographic Positioning
"""

import pytest
import tempfile
from pathlib import Path
from src.engine.stops import StopRegistry


def test_stop_registry_initialization_no_file():
    """Test Stop Registry initialization without file."""
    registry = StopRegistry()
    
    assert not registry.is_available()
    assert registry.get_stops_count() == 0
    assert registry.get_stop_location("STOP_001") is None


def test_stop_registry_initialization_missing_file():
    """Test Stop Registry initialization with non-existent file."""
    registry = StopRegistry("/non/existent/path/stops.txt")
    
    assert not registry.is_available()
    assert registry.get_stops_count() == 0
    assert registry.get_stop_location("STOP_001") is None


def test_stop_registry_load_valid_stops():
    """Test loading valid stops.txt file."""
    # Create temporary stops.txt file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write("STOP_001,Central Station,48.8566,2.3522\n")
        f.write("STOP_002,North Station,48.8800,2.3550\n")
        f.write("STOP_003,South Station,48.8300,2.3500\n")
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        assert registry.is_available()
        assert registry.get_stops_count() == 3
        
        # Test stop lookups
        loc1 = registry.get_stop_location("STOP_001")
        assert loc1 is not None
        assert loc1[0] == pytest.approx(48.8566, abs=0.0001)
        assert loc1[1] == pytest.approx(2.3522, abs=0.0001)
        
        loc2 = registry.get_stop_location("STOP_002")
        assert loc2 is not None
        assert loc2[0] == pytest.approx(48.8800, abs=0.0001)
        assert loc2[1] == pytest.approx(2.3550, abs=0.0001)
        
        loc3 = registry.get_stop_location("STOP_003")
        assert loc3 is not None
        
        # Test non-existent stop
        assert registry.get_stop_location("STOP_999") is None
        
    finally:
        Path(temp_path).unlink()


def test_stop_registry_handles_invalid_coordinates():
    """Test that registry skips stops with invalid coordinates."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write("STOP_001,Valid Station,48.8566,2.3522\n")
        f.write("STOP_002,Invalid Lat,999.0,2.3550\n")  # Invalid latitude
        f.write("STOP_003,Invalid Lon,48.8300,999.0\n")  # Invalid longitude
        f.write("STOP_004,Valid Station 2,45.7640,4.8357\n")
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        # Should only load valid stops
        assert registry.get_stops_count() == 2
        assert registry.get_stop_location("STOP_001") is not None
        assert registry.get_stop_location("STOP_002") is None  # Skipped
        assert registry.get_stop_location("STOP_003") is None  # Skipped
        assert registry.get_stop_location("STOP_004") is not None
        
    finally:
        Path(temp_path).unlink()


def test_stop_registry_handles_malformed_data():
    """Test that registry handles malformed CSV rows gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write("STOP_001,Valid Station,48.8566,2.3522\n")
        f.write("STOP_002,Bad Data,not_a_number,2.3550\n")  # Invalid float
        f.write("STOP_003,Valid Station 2,45.7640,4.8357\n")
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        # Should load valid stops and skip malformed ones
        assert registry.get_stops_count() == 2
        assert registry.get_stop_location("STOP_001") is not None
        assert registry.get_stop_location("STOP_002") is None  # Skipped
        assert registry.get_stop_location("STOP_003") is not None
        
    finally:
        Path(temp_path).unlink()


def test_stop_registry_handles_missing_columns():
    """Test that registry handles missing required columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name\n")  # Missing stop_lat and stop_lon
        f.write("STOP_001,Central Station\n")
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        # Should fail to load due to missing columns
        assert not registry.is_available()
        assert registry.get_stops_count() == 0
        
    finally:
        Path(temp_path).unlink()


def test_stop_registry_utf8_bom_handling():
    """Test that registry handles UTF-8 BOM (Byte Order Mark)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8-sig') as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write("STOP_001,Gare du Nord,48.8800,2.3550\n")
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        assert registry.is_available()
        assert registry.get_stops_count() == 1
        
        loc = registry.get_stop_location("STOP_001")
        assert loc is not None
        assert loc[0] == pytest.approx(48.8800, abs=0.0001)
        
    finally:
        Path(temp_path).unlink()


def test_stop_registry_whitespace_handling():
    """Test that registry handles whitespace in stop_id."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write(" STOP_001 ,Central Station,48.8566,2.3522\n")  # Whitespace
        temp_path = f.name
    
    try:
        registry = StopRegistry(temp_path)
        
        assert registry.is_available()
        assert registry.get_stops_count() == 1
        
        # Should be able to lookup with trimmed stop_id
        loc = registry.get_stop_location("STOP_001")
        assert loc is not None
        
    finally:
        Path(temp_path).unlink()
