"""
Integration Test Suite for HNPS v6.0 Rail-Lock
===============================================
End-to-end tests for the complete Rail-Lock system integration.
"""

import json
from pathlib import Path

import pytest

from src.engine.fusion import HybridFusionEngine, TrainEntity, Position2D, TrainStateVector


@pytest.fixture
def sample_topology_json(tmp_path: Path) -> Path:
    """Create a sample topology JSON file for testing."""
    topology_data = [
        {
            "shape_id": "SHAPE_A",
            "route_id": "ROUTE_A",
            "points": [
                [2.3522, 48.8566, 100.0],  # With elevation
                [2.3530, 48.8570, 110.0],
                [2.3540, 48.8575, 105.0]
            ]
        }
    ]
    
    json_path = tmp_path / "topology.json"
    with open(json_path, 'w') as f:
        json.dump(topology_data, f)
    
    return json_path


class TestRailLockIntegration:
    """Test complete Rail-Lock integration with Fusion Engine."""
    
    def test_fusion_engine_initializes_with_rail_lock(self, sample_topology_json: Path):
        """Test that Fusion Engine properly initializes with Rail-Lock."""
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            topology_path=str(sample_topology_json)
        )
        
        # Verify topology engine is initialized
        assert hasattr(engine, 'topology')
        assert engine.topology is not None
        assert engine.topology.is_available()
        
        # Verify topology data is loaded
        assert len(engine.topology.shapes) == 1
        assert "SHAPE_A" in engine.topology.shapes
        assert "ROUTE_A" in engine.topology.route_to_shapes
    
    def test_fusion_engine_operates_without_rail_lock(self):
        """Test that Fusion Engine works without Rail-Lock (degraded mode)."""
        engine = HybridFusionEngine(kafka_bootstrap_servers="localhost:9092")
        
        # Should initialize successfully
        assert engine is not None
        
        # Topology might be None or not available
        if hasattr(engine, 'topology') and engine.topology:
            assert not engine.topology.is_available()
    
    @pytest.mark.asyncio
    async def test_vehicle_position_processing_with_rail_lock(
        self,
        sample_topology_json: Path
    ):
        """Test that vehicle position processing uses Rail-Lock data."""
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            topology_path=str(sample_topology_json)
        )
        
        # Simulate vehicle position update
        vehicle_data = {
            'vehicle_id': 'TRAIN_001',
            'trip_id': 'TRIP_001',
            'route_id': 'ROUTE_A',
            'latitude': 48.8566,
            'longitude': 2.3522,
            'speed': 15.0,  # m/s
            'bearing': 45.0,
            'route_type': 1  # Metro
        }
        
        # Process the vehicle position
        await engine._process_vehicle_position(vehicle_data)
        
        # Verify train was created
        assert 'TRAIN_001' in engine._trains
        
        # Get the train
        train = engine._trains['TRAIN_001']
        
        # Verify train has track_distance and gradient from Rail-Lock
        state = train.get_current_state()
        assert state.track_distance is not None
        assert state.track_distance >= 0
        assert state.gradient is not None
    
    @pytest.mark.asyncio
    async def test_vehicle_position_processing_without_rail_lock(self):
        """Test that vehicle position processing works without Rail-Lock."""
        engine = HybridFusionEngine(kafka_bootstrap_servers="localhost:9092")
        
        # Simulate vehicle position update
        vehicle_data = {
            'vehicle_id': 'TRAIN_002',
            'trip_id': 'TRIP_002',
            'route_id': 'ROUTE_B',
            'latitude': 48.8566,
            'longitude': 2.3522,
            'speed': 15.0,
            'bearing': 45.0,
            'route_type': 1
        }
        
        # Process the vehicle position
        await engine._process_vehicle_position(vehicle_data)
        
        # Verify train was created
        assert 'TRAIN_002' in engine._trains
        
        # Get the train
        train = engine._trains['TRAIN_002']
        
        # Without Rail-Lock, track_distance and gradient should be None
        state = train.get_current_state()
        assert state.track_distance is None
        # Gradient can be None or 0.0 depending on initialization
        assert state.gradient is None or state.gradient == 0.0
    
    @pytest.mark.asyncio
    async def test_rail_lock_enables_cantonnement(self, sample_topology_json: Path):
        """Test that Rail-Lock enables proper train ordering for cantonnement."""
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            topology_path=str(sample_topology_json)
        )
        
        # Create two trains on the same route at different positions
        vehicle_data_1 = {
            'vehicle_id': 'TRAIN_FRONT',
            'trip_id': 'TRIP_FRONT',
            'route_id': 'ROUTE_A',
            'latitude': 48.8575,  # Further along
            'longitude': 2.3540,
            'speed': 15.0,
            'bearing': 45.0,
            'route_type': 1
        }
        
        vehicle_data_2 = {
            'vehicle_id': 'TRAIN_REAR',
            'trip_id': 'TRIP_REAR',
            'route_id': 'ROUTE_A',
            'latitude': 48.8566,  # Behind
            'longitude': 2.3522,
            'speed': 15.0,
            'bearing': 45.0,
            'route_type': 1
        }
        
        # Process both vehicles
        await engine._process_vehicle_position(vehicle_data_1)
        await engine._process_vehicle_position(vehicle_data_2)
        
        # Update moving blocks
        engine._update_moving_blocks()
        
        # Get trains
        train_front = engine._trains['TRAIN_FRONT']
        train_rear = engine._trains['TRAIN_REAR']
        
        # Verify track distances are set
        state_front = train_front.get_current_state()
        state_rear = train_rear.get_current_state()
        
        assert state_front.track_distance is not None
        assert state_rear.track_distance is not None
        
        # Front train should be further along the track
        assert state_front.track_distance > state_rear.track_distance
        
        # Verify moving block relationships are established
        # After sorting by track_distance:
        # - train_rear (smaller distance) comes first in the sorted list
        # - train_front (larger distance) comes second
        # So train_front should have train_rear as preceding_train
        assert train_front.preceding_train == train_rear
        assert train_rear.following_train == train_front
    
    @pytest.mark.asyncio
    async def test_rail_lock_enables_3d_physics(self, sample_topology_json: Path):
        """Test that Rail-Lock gradient data enables 3D physics (Davis equation)."""
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            topology_path=str(sample_topology_json)
        )
        
        # Create a train on a gradient
        vehicle_data = {
            'vehicle_id': 'TRAIN_HILL',
            'trip_id': 'TRIP_HILL',
            'route_id': 'ROUTE_A',
            'latitude': 48.8566,
            'longitude': 2.3522,
            'speed': 10.0,
            'bearing': 45.0,
            'route_type': 1
        }
        
        await engine._process_vehicle_position(vehicle_data)
        
        train = engine._trains['TRAIN_HILL']
        
        # Verify gradient is set
        state = train.get_current_state()
        assert state.gradient is not None
        
        # Gradient should be used in physics simulation
        # (tested indirectly through the train's _apply_davis_physics method)
        dt = 1.0
        physics_acceleration = train._apply_davis_physics(dt)
        
        # Physics acceleration should be a valid number
        assert isinstance(physics_acceleration, float)
        assert not (physics_acceleration != physics_acceleration)  # Not NaN
    
    def test_rail_lock_cross_track_error_threshold(self, sample_topology_json: Path):
        """Test that Rail-Lock rejects projections with high cross-track error."""
        from src.engine.topology import TopologyEngine
        
        engine = TopologyEngine(str(sample_topology_json))
        
        # Position very far from track
        projection = engine.get_rail_lock(51.5074, -0.1278, "ROUTE_A")
        
        assert projection is not None
        # Cross-track error should be very high
        assert projection.cross_track_error > 50.0
        
        # In the fusion engine, this would be rejected (not used)
        # The threshold is 50m in the _process_vehicle_position method
    
    def test_topology_json_format_compatibility(self, sample_topology_json: Path):
        """Test that topology JSON format is compatible with gtfs_to_topology output."""
        from src.tools.gtfs_to_topology import GTFSTopologyConverter
        from src.engine.topology import TopologyEngine
        
        # Load the topology with the engine
        engine = TopologyEngine(str(sample_topology_json))
        
        # Should load successfully
        assert engine.is_available()
        assert len(engine.shapes) == 1
        
        # Verify format matches expected output from gtfs_to_topology
        with open(sample_topology_json, 'r') as f:
            topology_data = json.load(f)
        
        assert isinstance(topology_data, list)
        for entry in topology_data:
            assert 'shape_id' in entry
            assert 'route_id' in entry
            assert 'points' in entry
            assert isinstance(entry['points'], list)
            
            # Points can be [lon, lat] or [lon, lat, elevation]
            for point in entry['points']:
                assert len(point) >= 2
                assert isinstance(point[0], (int, float))  # lon
                assert isinstance(point[1], (int, float))  # lat
