"""
Tests for Holographic Positioning (Positional Inference from TripUpdates)
"""

import pytest
import tempfile
from pathlib import Path
from src.engine.fusion import HybridFusionEngine, Position2D


@pytest.fixture
def stops_file():
    """Create a temporary stops.txt file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("stop_id,stop_name,stop_lat,stop_lon\n")
        f.write("STOP_CENTRAL,Central Station,48.8566,2.3522\n")
        f.write("STOP_NORTH,North Station,48.8800,2.3550\n")
        f.write("STOP_SOUTH,South Station,48.8300,2.3500\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_trip_update_creates_train(stops_file):
    """Test that TripUpdate creates a train when none exists."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    # Simulate TripUpdate message
    trip_update_data = {
        'trip_id': 'TRIP_001',
        'vehicle_id': 'TRAIN_001',
        'route_id': 'LINE_A',
        'stop_time_updates': [
            {'stop_id': 'STOP_CENTRAL', 'arrival': {'time': 1234567890}}
        ]
    }
    
    # Process the TripUpdate
    await engine._process_trip_update(trip_update_data)
    
    # Verify train was created
    assert 'TRAIN_001' in engine._trains
    train = engine.get_train('TRAIN_001')
    
    assert train is not None
    assert train.train_id == 'TRAIN_001'
    assert train.trip_id == 'TRIP_001'
    assert train.route_id == 'LINE_A'
    
    # Verify position matches the stop
    state = train.get_current_state()
    assert state.position.latitude == pytest.approx(48.8566, abs=0.0001)
    assert state.position.longitude == pytest.approx(2.3522, abs=0.0001)
    
    # Verify train is assumed stopped at station
    assert state.velocity == 0.0


@pytest.mark.asyncio
async def test_trip_update_updates_existing_train(stops_file):
    """Test that TripUpdate updates an existing train."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    # First TripUpdate - create train
    trip_update_1 = {
        'trip_id': 'TRIP_002',
        'vehicle_id': 'TRAIN_002',
        'route_id': 'LINE_B',
        'stop_time_updates': [
            {'stop_id': 'STOP_CENTRAL'}
        ]
    }
    await engine._process_trip_update(trip_update_1)
    
    train = engine.get_train('TRAIN_002')
    initial_state = train.get_current_state()
    
    # Second TripUpdate - update train position to new stop
    trip_update_2 = {
        'trip_id': 'TRIP_002',
        'vehicle_id': 'TRAIN_002',
        'route_id': 'LINE_B',
        'stop_time_updates': [
            {'stop_id': 'STOP_NORTH'}
        ]
    }
    await engine._process_trip_update(trip_update_2)
    
    # Verify train was updated
    updated_state = train.get_current_state()
    
    # Position should be different (moved to STOP_NORTH)
    # Note: Using larger tolerance (0.01) here because the Kalman filter blends
    # the measurement with the previous state, so the position won't jump exactly
    # to the new stop coordinates but will converge toward them
    assert updated_state.position.latitude != initial_state.position.latitude
    assert updated_state.position.latitude == pytest.approx(48.8800, abs=0.01)
    assert updated_state.position.longitude == pytest.approx(2.3550, abs=0.01)


@pytest.mark.asyncio
async def test_trip_update_fallback_to_trip_id(stops_file):
    """Test that TripUpdate uses trip_id as identifier when vehicle_id is missing."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    # TripUpdate without vehicle_id
    trip_update_data = {
        'trip_id': 'TRIP_003',
        # No vehicle_id!
        'route_id': 'LINE_C',
        'stop_time_updates': [
            {'stop_id': 'STOP_SOUTH'}
        ]
    }
    
    await engine._process_trip_update(trip_update_data)
    
    # Train should be created with trip_id as identifier
    assert 'TRIP_003' in engine._trains
    train = engine.get_train('TRIP_003')
    
    assert train is not None
    assert train.train_id == 'TRIP_003'
    assert train.trip_id == 'TRIP_003'


@pytest.mark.asyncio
async def test_trip_update_without_stops_txt():
    """Test graceful degradation when stops.txt is not available."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=None  # No stops file
    )
    
    trip_update_data = {
        'trip_id': 'TRIP_004',
        'vehicle_id': 'TRAIN_004',
        'route_id': 'LINE_D',
        'stop_time_updates': [
            {'stop_id': 'STOP_UNKNOWN'}
        ]
    }
    
    await engine._process_trip_update(trip_update_data)
    
    # Train should NOT be created (no position inference possible)
    assert 'TRAIN_004' not in engine._trains


@pytest.mark.asyncio
async def test_trip_update_unknown_stop_id(stops_file):
    """Test behavior when stop_id is not in registry."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    trip_update_data = {
        'trip_id': 'TRIP_005',
        'vehicle_id': 'TRAIN_005',
        'route_id': 'LINE_E',
        'stop_time_updates': [
            {'stop_id': 'STOP_DOES_NOT_EXIST'}
        ]
    }
    
    await engine._process_trip_update(trip_update_data)
    
    # Train should NOT be created (stop not found)
    assert 'TRAIN_005' not in engine._trains


@pytest.mark.asyncio
async def test_trip_update_no_stop_time_updates(stops_file):
    """Test behavior when TripUpdate has no stop_time_updates."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    trip_update_data = {
        'trip_id': 'TRIP_006',
        'vehicle_id': 'TRAIN_006',
        'route_id': 'LINE_F',
        'stop_time_updates': []  # Empty
    }
    
    await engine._process_trip_update(trip_update_data)
    
    # Train should NOT be created
    assert 'TRAIN_006' not in engine._trains


@pytest.mark.asyncio
async def test_trip_update_maintains_trip_mapping(stops_file):
    """Test that trip_id to train_id mapping is maintained."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    trip_update_data = {
        'trip_id': 'TRIP_007',
        'vehicle_id': 'TRAIN_007',
        'route_id': 'LINE_G',
        'stop_time_updates': [
            {'stop_id': 'STOP_CENTRAL'}
        ]
    }
    
    await engine._process_trip_update(trip_update_data)
    
    # Verify mapping
    assert 'TRIP_007' in engine._trip_to_train
    assert engine._trip_to_train['TRIP_007'] == 'TRAIN_007'


@pytest.mark.asyncio
async def test_trip_update_multiple_stops_uses_first(stops_file):
    """Test that when multiple stops are present, first one is used."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    trip_update_data = {
        'trip_id': 'TRIP_008',
        'vehicle_id': 'TRAIN_008',
        'route_id': 'LINE_H',
        'stop_time_updates': [
            {'stop_id': 'STOP_NORTH'},   # This should be used
            {'stop_id': 'STOP_SOUTH'},
            {'stop_id': 'STOP_CENTRAL'}
        ]
    }
    
    await engine._process_trip_update(trip_update_data)
    
    train = engine.get_train('TRAIN_008')
    assert train is not None
    
    # Position should match STOP_NORTH (first stop)
    state = train.get_current_state()
    assert state.position.latitude == pytest.approx(48.8800, abs=0.0001)
    assert state.position.longitude == pytest.approx(2.3550, abs=0.0001)


@pytest.mark.asyncio
async def test_holographic_positioning_with_multiple_trains(stops_file):
    """Test Holographic Positioning with multiple trains."""
    engine = HybridFusionEngine(
        kafka_bootstrap_servers="localhost:9092",
        stops_path=stops_file
    )
    
    # Create three trains from TripUpdates
    trip_updates = [
        {
            'trip_id': 'TRIP_A',
            'vehicle_id': 'TRAIN_A',
            'route_id': 'LINE_1',
            'stop_time_updates': [{'stop_id': 'STOP_CENTRAL'}]
        },
        {
            'trip_id': 'TRIP_B',
            'vehicle_id': 'TRAIN_B',
            'route_id': 'LINE_1',
            'stop_time_updates': [{'stop_id': 'STOP_NORTH'}]
        },
        {
            'trip_id': 'TRIP_C',
            'vehicle_id': 'TRAIN_C',
            'route_id': 'LINE_1',
            'stop_time_updates': [{'stop_id': 'STOP_SOUTH'}]
        }
    ]
    
    for trip_update in trip_updates:
        await engine._process_trip_update(trip_update)
    
    # Verify all trains created
    assert len(engine.get_all_trains()) == 3
    
    # Verify positions
    train_a = engine.get_train('TRAIN_A')
    assert train_a.get_current_state().position.latitude == pytest.approx(48.8566, abs=0.0001)
    
    train_b = engine.get_train('TRAIN_B')
    assert train_b.get_current_state().position.latitude == pytest.approx(48.8800, abs=0.0001)
    
    train_c = engine.get_train('TRAIN_C')
    assert train_c.get_current_state().position.latitude == pytest.approx(48.8300, abs=0.0001)
