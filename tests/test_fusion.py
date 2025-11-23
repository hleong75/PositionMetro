"""
Tests for the Hybrid Fusion Engine
"""

import pytest
import numpy as np
from src.engine.fusion import (
    Position2D,
    TrainStateVector,
    TrainPhysicalProperties,
    DavisCoefficients,
    TrainEntity,
    UnscentedKalmanFilter
)


def test_position2d_distance():
    """Test Haversine distance calculation."""
    # Paris coordinates
    paris = Position2D(latitude=48.8566, longitude=2.3522)
    # Lyon coordinates  
    lyon = Position2D(latitude=45.7640, longitude=4.8357)
    
    # Distance Paris-Lyon is approximately 392 km
    distance = paris.distance_to(lyon)
    assert 390000 < distance < 395000  # Allow some margin


def test_position2d_bearing():
    """Test bearing calculation."""
    pos1 = Position2D(latitude=48.8566, longitude=2.3522)
    pos2 = Position2D(latitude=48.8600, longitude=2.3522)
    
    # Moving north should give bearing close to 0 or 360
    bearing = pos1.bearing_to(pos2)
    assert 0 <= bearing <= 5 or 355 <= bearing <= 360


def test_davis_coefficients_default():
    """Test Davis coefficients default values."""
    davis = DavisCoefficients()
    
    assert davis.A == 5.0
    assert davis.B == 0.03
    assert davis.C == 0.0015


def test_train_physical_properties_default():
    """Test train physical properties defaults."""
    props = TrainPhysicalProperties()
    
    assert props.mass == 400000.0
    assert props.length == 100.0
    assert props.max_power == 2000.0
    assert props.max_speed == 33.33
    assert isinstance(props.davis, DavisCoefficients)


def test_train_state_vector_creation():
    """Test TrainStateVector creation."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    state = TrainStateVector(
        position=position,
        velocity=20.0,
        acceleration=0.5,
        bearing=90.0
    )
    
    assert state.position.latitude == 48.8566
    assert state.position.longitude == 2.3522
    assert state.velocity == 20.0
    assert state.acceleration == 0.5
    assert state.bearing == 90.0
    assert state.gradient == 0.0


def test_train_entity_creation():
    """Test TrainEntity initialization."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    train = TrainEntity(
        train_id="TRAIN_001",
        trip_id="TRIP_001",
        route_id="RER_A",
        initial_state=initial_state
    )
    
    assert train.train_id == "TRAIN_001"
    assert train.trip_id == "TRIP_001"
    assert train.route_id == "RER_A"
    assert train.kalman is not None
    assert len(train.state_history) == 1


def test_train_entity_predict():
    """Test train state prediction."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    train = TrainEntity(
        train_id="TRAIN_002",
        trip_id="TRIP_002",
        route_id="RER_B",
        initial_state=initial_state
    )
    
    # Predict forward 1 second
    predicted = train.predict(dt=1.0)
    
    assert predicted is not None
    assert isinstance(predicted, TrainStateVector)
    # Position should have changed slightly
    assert predicted.position.latitude != initial_state.position.latitude or \
           predicted.position.longitude != initial_state.position.longitude


def test_train_entity_update_from_measurement():
    """Test updating train state from measurement."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    train = TrainEntity(
        train_id="TRAIN_003",
        trip_id="TRIP_003",
        route_id="RER_C",
        initial_state=initial_state
    )
    
    # Update with new measurement
    new_position = Position2D(latitude=48.8600, longitude=2.3522)
    train.update_from_measurement(
        position=new_position,
        velocity=15.0,
        bearing=0.0
    )
    
    # State should be updated
    current_state = train.get_current_state()
    assert current_state.position.latitude == pytest.approx(48.8600, abs=0.01)
    assert current_state.velocity == pytest.approx(15.0, abs=1.0)


def test_unscented_kalman_filter_initialization():
    """Test UKF initialization."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    ukf = UnscentedKalmanFilter(initial_state)
    
    assert ukf.state.shape == (5,)
    assert ukf.P.shape == (5, 5)
    assert ukf.Q.shape == (5, 5)
    assert ukf.R.shape == (5, 5)


def test_unscented_kalman_filter_predict():
    """Test UKF prediction step."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    ukf = UnscentedKalmanFilter(initial_state)
    initial_position = ukf.state[0:2].copy()
    
    # Predict forward
    ukf.predict(dt=1.0, control_acceleration=0.5)
    
    # State should have changed
    assert not np.array_equal(ukf.state[0:2], initial_position)


def test_unscented_kalman_filter_update():
    """Test UKF update step."""
    position = Position2D(latitude=48.8566, longitude=2.3522)
    initial_state = TrainStateVector(
        position=position,
        velocity=10.0,
        acceleration=0.0,
        bearing=0.0
    )
    
    ukf = UnscentedKalmanFilter(initial_state)
    
    # Create a measurement
    measurement = np.array([48.8567, 2.3523, 12.0, 0.5, 0.0])
    
    # Update with measurement
    ukf.update(measurement)
    
    # State should be closer to measurement
    assert ukf.state[0] == pytest.approx(48.8567, abs=0.01)
    assert ukf.state[2] == pytest.approx(12.0, abs=1.0)
