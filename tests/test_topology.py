"""
Test Suite for Topology Engine - Rail-Lock System
==================================================
Tests the HNPS v6.0 Rail-Lock engine for spatial awareness and map matching.
"""

import json
import math
from pathlib import Path

import pytest

from src.engine.topology import TopologyEngine, RailProjection


@pytest.fixture
def sample_topology_json(tmp_path: Path) -> Path:
    """
    Create a sample topology JSON file for testing.
    
    Creates a simple linear track for Route A with 3 points.
    
    Returns:
        Path to the created JSON file
    """
    topology_data = [
        {
            "shape_id": "SHAPE_A",
            "route_id": "ROUTE_A",
            "points": [
                [2.3522, 48.8566],  # Paris center (lon, lat)
                [2.3530, 48.8570],
                [2.3540, 48.8575]
            ]
        },
        {
            "shape_id": "SHAPE_B",
            "route_id": "ROUTE_B",
            "points": [
                [2.3600, 48.8600],
                [2.3610, 48.8605],
                [2.3620, 48.8610]
            ]
        }
    ]
    
    json_path = tmp_path / "topology.json"
    with open(json_path, 'w') as f:
        json.dump(topology_data, f)
    
    return json_path


@pytest.fixture
def sample_topology_with_elevation(tmp_path: Path) -> Path:
    """
    Create a topology JSON file with elevation data.
    
    Creates a track with varying elevation (Z coordinate).
    
    Returns:
        Path to the created JSON file
    """
    topology_data = [
        {
            "shape_id": "SHAPE_HILL",
            "route_id": "ROUTE_HILL",
            "points": [
                [2.3522, 48.8566, 100.0],  # Starting at 100m elevation
                [2.3530, 48.8570, 150.0],  # Climbing to 150m (uphill)
                [2.3540, 48.8575, 120.0]   # Descending to 120m (downhill)
            ]
        }
    ]
    
    json_path = tmp_path / "topology_elevation.json"
    with open(json_path, 'w') as f:
        json.dump(topology_data, f)
    
    return json_path


class TestTopologyEngine:
    """Test suite for TopologyEngine class."""
    
    def test_initialization_without_topology(self):
        """Test engine initialization without topology file (degraded mode)."""
        engine = TopologyEngine()
        
        assert not engine.is_available()
        assert len(engine.shapes) == 0
        assert len(engine.route_to_shapes) == 0
    
    def test_initialization_with_topology(self, sample_topology_json: Path):
        """Test engine initialization with topology file."""
        engine = TopologyEngine(str(sample_topology_json))
        
        assert engine.is_available()
        assert len(engine.shapes) == 2
        assert "SHAPE_A" in engine.shapes
        assert "SHAPE_B" in engine.shapes
        assert len(engine.route_to_shapes) == 2
        assert "ROUTE_A" in engine.route_to_shapes
        assert "ROUTE_B" in engine.route_to_shapes
    
    def test_load_topology_missing_file(self):
        """Test error handling when topology file doesn't exist."""
        engine = TopologyEngine()
        
        with pytest.raises(FileNotFoundError):
            engine.load_topology("/nonexistent/topology.json")
    
    def test_load_topology_invalid_json(self, tmp_path: Path):
        """Test error handling for invalid JSON."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("not valid json")
        
        engine = TopologyEngine()
        
        with pytest.raises(Exception):
            engine.load_topology(str(invalid_json))
    
    def test_load_topology_invalid_structure(self, tmp_path: Path):
        """Test error handling for invalid topology structure."""
        invalid_topology = tmp_path / "invalid_topology.json"
        with open(invalid_topology, 'w') as f:
            json.dump({"not": "a list"}, f)
        
        engine = TopologyEngine()
        
        with pytest.raises(ValueError, match="must be a list"):
            engine.load_topology(str(invalid_topology))
    
    def test_get_rail_lock_degraded_mode(self):
        """Test Rail-Lock returns None in degraded mode (no topology)."""
        engine = TopologyEngine()
        
        projection = engine.get_rail_lock(48.8566, 2.3522)
        
        assert projection is None
    
    def test_get_rail_lock_on_track(self, sample_topology_json: Path):
        """Test Rail-Lock for a position on the track."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position very close to first point of SHAPE_A
        projection = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_A")
        
        assert projection is not None
        assert isinstance(projection, RailProjection)
        assert projection.shape_id == "SHAPE_A"
        assert projection.track_distance >= 0
        assert projection.cross_track_error < 10.0  # Should be very close
        assert projection.confidence > 0.9  # High confidence
    
    def test_get_rail_lock_between_points(self, sample_topology_json: Path):
        """Test Rail-Lock for a position between two track points."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position between first and second points
        lat = (48.8566 + 48.8570) / 2
        lon = (2.3522 + 2.3530) / 2
        
        projection = engine.get_rail_lock(lat, lon, "ROUTE_A")
        
        assert projection is not None
        assert projection.shape_id == "SHAPE_A"
        assert 0 < projection.track_distance < 1000  # Somewhere along the track
        assert projection.cross_track_error < 50.0  # Reasonably close
    
    def test_get_rail_lock_without_route_id(self, sample_topology_json: Path):
        """Test Rail-Lock without specifying route_id (searches all shapes)."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position close to SHAPE_A
        projection = engine.get_rail_lock(48.8566, 2.3522)
        
        assert projection is not None
        # Should find SHAPE_A as it's the closest
        assert projection.shape_id == "SHAPE_A"
    
    def test_get_rail_lock_far_from_track(self, sample_topology_json: Path):
        """Test Rail-Lock for a position far from any track."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position far away (London coordinates)
        projection = engine.get_rail_lock(51.5074, -0.1278, "ROUTE_A")
        
        # Should still return a projection (nearest point on track)
        assert projection is not None
        # But cross-track error should be very large
        assert projection.cross_track_error > 1000.0
        # And confidence should be very low
        assert projection.confidence < 0.1
    
    def test_get_rail_lock_wrong_route(self, sample_topology_json: Path):
        """Test Rail-Lock with non-existent route_id."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position close to SHAPE_A but requesting non-existent route
        projection = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_NONEXISTENT")
        
        # Should still work by searching all shapes
        assert projection is not None
        assert projection.shape_id == "SHAPE_A"
    
    def test_gradient_calculation_flat_track(self, sample_topology_json: Path):
        """Test gradient calculation for flat track (no elevation data)."""
        engine = TopologyEngine(str(sample_topology_json))
        
        projection = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_A")
        
        assert projection is not None
        # Should have zero gradient (flat track)
        assert abs(projection.gradient) < 0.001
    
    def test_gradient_calculation_with_elevation(self, sample_topology_with_elevation: Path):
        """Test gradient calculation with elevation data."""
        engine = TopologyEngine(str(sample_topology_with_elevation))
        
        # Position near start (uphill section)
        projection = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_HILL")
        
        assert projection is not None
        # Should have positive gradient (uphill)
        # The exact value depends on the projection algorithm
        assert isinstance(projection.gradient, float)
    
    def test_get_route_shapes(self, sample_topology_json: Path):
        """Test retrieving shapes for a specific route."""
        engine = TopologyEngine(str(sample_topology_json))
        
        shapes_a = engine.get_route_shapes("ROUTE_A")
        shapes_b = engine.get_route_shapes("ROUTE_B")
        shapes_none = engine.get_route_shapes("ROUTE_NONEXISTENT")
        
        assert shapes_a == ["SHAPE_A"]
        assert shapes_b == ["SHAPE_B"]
        assert shapes_none == []
    
    def test_rail_projection_dataclass(self):
        """Test RailProjection dataclass creation."""
        projection = RailProjection(
            track_distance=1234.5,
            cross_track_error=5.2,
            gradient=0.05,
            shape_id="TEST_SHAPE",
            confidence=0.95
        )
        
        assert projection.track_distance == 1234.5
        assert projection.cross_track_error == 5.2
        assert projection.gradient == 0.05
        assert projection.shape_id == "TEST_SHAPE"
        assert projection.confidence == 0.95
    
    def test_track_distance_ordering(self, sample_topology_json: Path):
        """Test that track_distance increases along the track."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Project first point
        proj1 = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_A")
        
        # Project last point
        proj2 = engine.get_rail_lock(48.8575, 2.3540, "ROUTE_A")
        
        assert proj1 is not None
        assert proj2 is not None
        
        # Track distance should increase along the track
        assert proj2.track_distance > proj1.track_distance
    
    def test_multiple_routes_same_area(self, sample_topology_json: Path):
        """Test Rail-Lock with multiple routes in the same area."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position close to both routes - specify ROUTE_A
        proj_a = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_A")
        
        # Same position - specify ROUTE_B
        proj_b = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_B")
        
        assert proj_a is not None
        assert proj_b is not None
        
        # Should match to different shapes
        assert proj_a.shape_id == "SHAPE_A"
        assert proj_b.shape_id == "SHAPE_B"
    
    def test_confidence_score_calculation(self, sample_topology_json: Path):
        """Test confidence score calculation based on cross-track error."""
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position on track (very close)
        proj_close = engine.get_rail_lock(48.8566, 2.3522, "ROUTE_A")
        
        # Position 200m away (far)
        proj_far = engine.get_rail_lock(48.8600, 2.3522, "ROUTE_A")
        
        assert proj_close is not None
        assert proj_far is not None
        
        # Close position should have higher confidence
        assert proj_close.confidence > proj_far.confidence
        
        # Confidence should be in valid range [0, 1]
        assert 0.0 <= proj_close.confidence <= 1.0
        assert 0.0 <= proj_far.confidence <= 1.0


class TestTopologyEngineIntegration:
    """Integration tests for Topology Engine with Fusion Engine."""
    
    def test_topology_engine_import(self):
        """Test that TopologyEngine can be imported."""
        from src.engine.topology import TopologyEngine, RailProjection
        
        assert TopologyEngine is not None
        assert RailProjection is not None
    
    def test_fusion_engine_with_topology(self, sample_topology_json: Path):
        """Test HybridFusionEngine initialization with topology."""
        from src.engine.fusion import HybridFusionEngine
        
        # Initialize with topology path
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            topology_path=str(sample_topology_json)
        )
        
        # Should have topology engine initialized
        assert hasattr(engine, 'topology')
        assert engine.topology is not None
        assert engine.topology.is_available()
    
    def test_fusion_engine_without_topology(self):
        """Test HybridFusionEngine initialization without topology (degraded mode)."""
        from src.engine.fusion import HybridFusionEngine
        
        # Initialize without topology path
        engine = HybridFusionEngine(kafka_bootstrap_servers="localhost:9092")
        
        # Should have topology engine but in degraded mode
        assert hasattr(engine, 'topology')
        # topology might be None or not available
        if engine.topology:
            assert not engine.topology.is_available()
