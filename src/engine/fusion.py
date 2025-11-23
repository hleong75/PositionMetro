"""
Module C: Hybrid Neuro-Physics Engine (The Core)
=================================================
This module implements the heart of the HNPS system, combining:
- Davis Equation for train physics simulation
- Unscented Kalman Filter for sensor fusion
- Moving Block (Cantonnement) collision prevention
- Rail-Lock spatial awareness (v6.0)

HNPS v5.0 Component: Physics Simulation & State Estimation
HNPS v6.0 Enhancement: Cognitive Rail-Lock Integration
"""

import asyncio
import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import structlog
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = structlog.get_logger(__name__)

# HNPS v6.0: Rail-Lock configuration
RAIL_LOCK_MAX_CROSS_TRACK_ERROR = 50.0  # meters - maximum acceptable error for Rail-Lock


class TrainState(Enum):
    """Train operational states."""
    STOPPED = "stopped"
    ACCELERATING = "accelerating"
    CRUISING = "cruising"
    BRAKING = "braking"
    UNKNOWN = "unknown"


@dataclass
class DavisCoefficients:
    """
    Davis Equation coefficients for train resistance.
    
    The Davis equation models train resistance as:
    R = A + B*v + C*v²
    
    Where:
    - A: Rolling resistance (constant)
    - B: Mechanical resistance (linear with speed)
    - C: Aerodynamic resistance (quadratic with speed)
    """
    A: float = 5.0      # Rolling resistance (kN)
    B: float = 0.03     # Mechanical resistance (kN/(m/s))
    C: float = 0.0015   # Aerodynamic resistance (kN/(m/s)²)


def get_davis_coefficients_for_route_type(route_type: Optional[int]) -> DavisCoefficients:
    """
    Get Davis coefficients based on GTFS route_type.
    
    Different vehicle types have different aerodynamic and mechanical properties.
    This function returns appropriate coefficients based on the GTFS route_type:
    
    - 0: Tram/Streetcar/Light rail (low speed, urban)
    - 1: Subway/Metro (medium speed, enclosed stations)
    - 2: Rail/Intercity rail (high speed, open track)
    - 3: Bus (road vehicle, not applicable)
    - 4: Ferry (water vehicle, not applicable)
    - 5-7: Cable/Gondola/Funicular (special cases)
    
    Args:
        route_type: GTFS route_type integer, or None for default.
        
    Returns:
        DavisCoefficients appropriate for the vehicle type.
    """
    if route_type is None:
        # Default: Subway/Metro
        return DavisCoefficients(A=5.0, B=0.03, C=0.0015)
    
    if route_type == 0:
        # Tram/Light Rail: Lower speeds, frequent stops
        return DavisCoefficients(A=4.0, B=0.025, C=0.0012)
    
    elif route_type == 1:
        # Subway/Metro: Medium speeds, aerodynamic but in tunnels
        return DavisCoefficients(A=5.0, B=0.03, C=0.0015)
    
    elif route_type == 2:
        # Rail/Intercity: High speeds, significant aerodynamic drag
        # TGV-style coefficients: lower rolling resistance but higher aero drag
        return DavisCoefficients(A=3.5, B=0.02, C=0.0025)
    
    elif route_type in [5, 6, 7]:
        # Cable/Gondola/Funicular: Very different physics
        # Lower resistance due to cable-driven motion (no wheels on rails)
        # Minimal aerodynamic drag due to low speeds
        return DavisCoefficients(A=2.0, B=0.01, C=0.0005)
    
    else:
        # Bus, Ferry, or unknown: Use default metro coefficients
        return DavisCoefficients(A=5.0, B=0.03, C=0.0015)


@dataclass
class TrainPhysicalProperties:
    """Physical properties of a train."""
    mass: float = 400000.0          # kg (typical metro train)
    length: float = 100.0           # meters
    max_power: float = 2000.0       # kW
    max_speed: float = 33.33        # m/s (120 km/h)
    max_acceleration: float = 1.2   # m/s²
    max_braking: float = -1.5       # m/s²
    davis: DavisCoefficients = field(default_factory=DavisCoefficients)


@dataclass
class Position2D:
    """2D geographical position."""
    latitude: float
    longitude: float
    
    def distance_to(self, other: "Position2D") -> float:
        """
        Calculate Haversine distance to another position in meters.
        
        Args:
            other: Another position.
            
        Returns:
            Distance in meters.
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
        
    def bearing_to(self, other: "Position2D") -> float:
        """
        Calculate bearing to another position in degrees.
        
        Args:
            other: Another position.
            
        Returns:
            Bearing in degrees (0-360).
        """
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360


@dataclass
class TrainStateVector:
    """Complete state vector for a train."""
    position: Position2D
    velocity: float             # m/s
    acceleration: float         # m/s²
    bearing: float             # degrees
    gradient: float = 0.0      # Track gradient in radians
    track_distance: Optional[float] = None  # Distance along track (PK - Point Kilométrique) in meters
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class UnscentedKalmanFilter:
    """
    Simplified Unscented Kalman Filter for train state estimation.
    
    The UKF handles non-linear motion models better than standard Kalman filters
    by using a deterministic sampling technique (sigma points) to approximate
    the probability distribution.
    
    State vector: [position_lat, position_lon, velocity, acceleration, bearing]
    """
    
    def __init__(
        self,
        initial_state: TrainStateVector,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0
    ) -> None:
        """
        Initialize the Unscented Kalman Filter.
        
        Args:
            initial_state: Initial state estimate.
            process_noise: Process noise covariance.
            measurement_noise: Measurement noise covariance.
        """
        # State vector: [lat, lon, velocity, acceleration, bearing]
        self.state = np.array([
            initial_state.position.latitude,
            initial_state.position.longitude,
            initial_state.velocity,
            initial_state.acceleration,
            initial_state.bearing
        ])
        
        # Covariance matrix
        self.P = np.eye(5) * 10.0
        
        # Process noise covariance
        self.Q = np.eye(5) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(5) * measurement_noise
        
        # UKF parameters
        self.alpha = 0.001
        self.beta = 2.0
        self.kappa = 0.0
        
        n = 5  # State dimension
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Weights
        self.Wm = np.zeros(2 * n + 1)
        self.Wc = np.zeros(2 * n + 1)
        
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * n + 1):
            self.Wm[i] = 1 / (2 * (n + self.lambda_))
            self.Wc[i] = 1 / (2 * (n + self.lambda_))
            
    def predict(self, dt: float, control_acceleration: float = 0.0) -> None:
        """
        Prediction step: propagate state forward using motion model.
        
        Args:
            dt: Time step in seconds.
            control_acceleration: Known control input acceleration.
        """
        n = len(self.state)
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through motion model
        predicted_sigma_points = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            predicted_sigma_points[:, i] = self._motion_model(
                sigma_points[:, i],
                dt,
                control_acceleration
            )
            
        # Compute predicted mean and covariance
        self.state = np.dot(self.Wm, predicted_sigma_points.T)
        
        # Predicted covariance
        self.P = self.Q.copy()
        for i in range(predicted_sigma_points.shape[1]):
            diff = predicted_sigma_points[:, i] - self.state
            self.P += self.Wc[i] * np.outer(diff, diff)
            
    def update(self, measurement: Optional[np.ndarray], apply_zupt: bool = False, 
               measured_acceleration: Optional[float] = None) -> None:
        """
        Update step: correct prediction with measurement.
        
        Args:
            measurement: Measurement vector [lat, lon, velocity, acceleration, bearing].
                        Can be None when apply_zupt is True.
            apply_zupt: Apply Zero Velocity Update (ZUPT) constraint for stopped trains.
                       When True, measurement is not used.
            measured_acceleration: Measured acceleration (m/s²) from sensors, used to
                                  dynamically adjust ZUPT covariance when train restarts.
        """
        n = len(self.state)
        
        # Apply ZUPT if requested (train is stopped)
        if apply_zupt:
            self._apply_zupt(measured_acceleration=measured_acceleration)
            return
        
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points()
        
        # Transform sigma points through measurement model (identity in this case)
        measurement_sigma_points = sigma_points.copy()
        
        # Compute measurement mean
        z_mean = np.dot(self.Wm, measurement_sigma_points.T)
        
        # Compute innovation covariance
        Pz = self.R.copy()
        Pxz = np.zeros((n, n))
        
        for i in range(measurement_sigma_points.shape[1]):
            z_diff = measurement_sigma_points[:, i] - z_mean
            Pz += self.Wc[i] * np.outer(z_diff, z_diff)
            
            x_diff = sigma_points[:, i] - self.state
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
            
        # Kalman gain
        K = np.dot(Pxz, np.linalg.inv(Pz))
        
        # Update state
        innovation = measurement - z_mean
        self.state += np.dot(K, innovation)
        
        # Update covariance
        self.P -= np.dot(K, np.dot(Pz, K.T))
    
    def _apply_zupt(self, measured_acceleration: Optional[float] = None) -> None:
        """
        Apply Zero Velocity Update (ZUPT) constraint.
        
        When a train is stopped at a station, force velocity and acceleration to zero
        and reduce their covariance to prevent the filter from drifting due to inertia.
        This improves accuracy during station stops.
        
        V2 optimization: If the train is restarting (measured acceleration exceeds threshold),
        dynamically increase velocity variance so the filter can quickly adapt to the new velocity.
        
        Args:
            measured_acceleration: Measured acceleration from sensors (m/s²). If provided
                                  and exceeds threshold, increases velocity variance for faster
                                  adaptation when train restarts.
        """
        # Force velocity and acceleration to zero
        self.state[2] = 0.0  # velocity
        self.state[3] = 0.0  # acceleration
        
        # Dynamic covariance adjustment based on measured acceleration
        # If train is restarting (high acceleration while stopped), increase velocity variance
        # to help filter adapt faster to the new velocity
        acceleration_threshold = 0.5  # m/s² - threshold for detecting train restart
        base_velocity_variance = 0.001
        
        if measured_acceleration is not None and abs(measured_acceleration) > acceleration_threshold:
            # Train is restarting - increase velocity variance for faster adaptation
            # Scale variance proportionally to measured acceleration
            scale_factor = min(10.0, abs(measured_acceleration) / acceleration_threshold)
            velocity_variance = base_velocity_variance * scale_factor
            logger.debug(
                "zupt_dynamic_covariance_adjustment",
                measured_acceleration=measured_acceleration,
                scale_factor=scale_factor,
                velocity_variance=velocity_variance
            )
        else:
            # Normal ZUPT - strong confidence that velocity is zero
            velocity_variance = base_velocity_variance
        
        # Also reduce cross-covariances involving velocity and acceleration
        # (but preserve diagonal values)
        self.P[2, :] *= 0.1
        self.P[:, 2] *= 0.1
        self.P[3, :] *= 0.1
        self.P[:, 3] *= 0.1
        
        # Set covariance for velocity and acceleration (after cross-covariance reduction)
        self.P[2, 2] = velocity_variance  # velocity variance (dynamic)
        self.P[3, 3] = 0.001  # acceleration variance (fixed)
        
    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points for UKF."""
        n = len(self.state)
        sigma_points = np.zeros((n, 2 * n + 1))
        
        # Mean
        sigma_points[:, 0] = self.state
        
        # Covariance square root
        try:
            U = np.linalg.cholesky((n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            U = np.sqrt((n + self.lambda_)) * np.eye(n)
            
        # Positive sigma points
        for i in range(n):
            sigma_points[:, i + 1] = self.state + U[:, i]
            
        # Negative sigma points
        for i in range(n):
            sigma_points[:, n + i + 1] = self.state - U[:, i]
            
        return sigma_points
        
    def _motion_model(
        self,
        state: np.ndarray,
        dt: float,
        control_acceleration: float
    ) -> np.ndarray:
        """
        Non-linear motion model for train dynamics.
        
        Args:
            state: Current state vector.
            dt: Time step.
            control_acceleration: Control input.
            
        Returns:
            Next state vector.
        """
        lat, lon, velocity, acceleration, bearing = state
        
        # Update acceleration (with some damping)
        new_acceleration = 0.7 * acceleration + 0.3 * control_acceleration
        
        # Update velocity
        new_velocity = max(0, velocity + new_acceleration * dt)
        
        # Update position (simplified - assumes local flat Earth)
        # Convert bearing to radians
        bearing_rad = math.radians(bearing)
        
        # Displacement in meters
        displacement = new_velocity * dt
        
        # Convert to lat/lon change (approximate)
        delta_lat = (displacement * math.cos(bearing_rad)) / 111000
        delta_lon = (displacement * math.sin(bearing_rad)) / (111000 * math.cos(math.radians(lat)))
        
        new_lat = lat + delta_lat
        new_lon = lon + delta_lon
        
        return np.array([new_lat, new_lon, new_velocity, new_acceleration, bearing])
        
    def get_state_vector(self) -> TrainStateVector:
        """Get current state as TrainStateVector."""
        return TrainStateVector(
            position=Position2D(
                latitude=self.state[0],
                longitude=self.state[1]
            ),
            velocity=self.state[2],
            acceleration=self.state[3],
            bearing=self.state[4],
            timestamp=datetime.now().timestamp()
        )


class TrainEntity:
    """
    Represents a single train with physics simulation and state estimation.
    
    This class maintains the complete state of a train, including:
    - Current position and velocity
    - Physics-based motion prediction (Davis equation)
    - Kalman filter for sensor fusion
    - Moving block constraints
    """
    
    def __init__(
        self,
        train_id: str,
        trip_id: Optional[str],
        route_id: Optional[str],
        initial_state: TrainStateVector,
        properties: Optional[TrainPhysicalProperties] = None,
        route_type: Optional[int] = None
    ) -> None:
        """
        Initialize a train entity.
        
        Args:
            train_id: Unique train identifier.
            trip_id: Current trip identifier.
            route_id: Route identifier.
            initial_state: Initial state vector.
            properties: Physical properties of the train.
            route_type: GTFS route_type for dynamic physics (0=Tram, 1=Subway, 2=Rail, etc.).
        """
        self.train_id = train_id
        self.trip_id = trip_id
        self.route_id = route_id
        self.route_type = route_type
        
        # Use route_type-specific physics if not provided explicitly
        if properties is None:
            davis_coeffs = get_davis_coefficients_for_route_type(route_type)
            self.properties = TrainPhysicalProperties(davis=davis_coeffs)
        else:
            self.properties = properties
        
        # Initialize Kalman filter
        self.kalman = UnscentedKalmanFilter(initial_state)
        
        # State tracking
        self.last_update = datetime.now()
        self.state_history: List[TrainStateVector] = [initial_state]
        self.current_state = TrainState.UNKNOWN
        
        # Track distance and gradient (not part of Kalman filter state)
        self.track_distance = initial_state.track_distance
        self.gradient = initial_state.gradient
        
        # Moving block tracking
        self.preceding_train: Optional["TrainEntity"] = None
        self.following_train: Optional["TrainEntity"] = None
        self.safe_distance = 500.0  # meters (minimum safe distance)
        
        logger.info(
            "train_entity_created",
            train_id=train_id,
            trip_id=trip_id,
            route_id=route_id
        )
        
    def _apply_davis_physics(self, dt: float) -> float:
        """
        Apply Davis equation to calculate resistance and acceleration.
        
        The Davis equation models total resistance as:
        R = A + B*v + C*v² + m*g*sin(θ)
        
        Where θ is the track gradient (slope).
        
        NOTE: Track gradient (state.gradient) should be provided by an external Map Matching
        module using PostGIS to query the actual track gradient from digital elevation models.
        Without gradient data, this defaults to 0.0 (flat track), which ignores uphill/downhill
        effects in the physics simulation.
        
        Args:
            dt: Time step in seconds.
            
        Returns:
            Physics-based acceleration in m/s².
        """
        state = self.kalman.get_state_vector()
        v = state.velocity
        
        # Davis resistance components
        A = self.properties.davis.A * 1000  # Convert kN to N
        B = self.properties.davis.B * 1000
        C = self.properties.davis.C * 1000
        
        # Total rolling and air resistance
        R_total = A + B * v + C * (v ** 2)
        
        # Gravitational component (gradient resistance)
        # state.gradient should come from external Map Matching with PostGIS
        g = 9.81  # m/s²
        R_gravity = self.properties.mass * g * math.sin(state.gradient)
        
        # Total resistance
        R_total += R_gravity
        
        # Available power (simplified - assumes constant power)
        # In reality, this would depend on throttle position
        if self.current_state == TrainState.ACCELERATING:
            power_ratio = 0.8
        elif self.current_state == TrainState.BRAKING:
            power_ratio = -1.0
        elif self.current_state == TrainState.CRUISING:
            power_ratio = 0.2
        else:
            power_ratio = 0.0
            
        # Tractive/braking force
        if v > 0.1:  # Avoid division by zero
            F_traction = (self.properties.max_power * 1000 * power_ratio) / v
        else:
            F_traction = self.properties.max_power * 1000 * power_ratio / 0.1
            
        # Net force
        F_net = F_traction - R_total
        
        # Acceleration (F = ma)
        acceleration = F_net / self.properties.mass
        
        # Clamp to physical limits
        acceleration = max(
            self.properties.max_braking,
            min(self.properties.max_acceleration, acceleration)
        )
        
        return acceleration
        
    def predict(self, dt: float) -> TrainStateVector:
        """
        Predict next state using physics model and Kalman filter.
        
        Args:
            dt: Time step in seconds.
            
        Returns:
            Predicted state vector.
        """
        # Calculate physics-based acceleration
        physics_acceleration = self._apply_davis_physics(dt)
        
        # Check moving block constraint
        if self.preceding_train:
            physics_acceleration = self._apply_moving_block_constraint(
                physics_acceleration
            )
            
        # Predict with Kalman filter
        self.kalman.predict(dt, physics_acceleration)
        
        # Get predicted state
        predicted_state = self.kalman.get_state_vector()
        
        # Update state history
        self.state_history.append(predicted_state)
        if len(self.state_history) > 100:  # Keep last 100 states
            self.state_history.pop(0)
            
        return predicted_state
        
    def update_from_measurement(
        self,
        position: Position2D,
        velocity: Optional[float],
        bearing: Optional[float],
        track_distance: Optional[float] = None,
        gradient: Optional[float] = None,
        acceleration: Optional[float] = None
    ) -> None:
        """
        Update train state from real-time measurement (GTFS-RT data).
        
        Args:
            position: Measured position.
            velocity: Measured velocity (m/s).
            bearing: Measured bearing (degrees).
            track_distance: Distance along track (PK) in meters.
            gradient: Track gradient in radians. Should be provided by external Map Matching
                     module with PostGIS to enable accurate gradient resistance in Davis equation.
                     Without gradient data, the physics model will assume flat track (gradient=0).
            acceleration: Measured acceleration from IMU sensors (m/s²). Used for dynamic
                         ZUPT covariance adjustment when train restarts.
        """
        current_state = self.kalman.get_state_vector()
        
        # Determine if train is stopped (for ZUPT application)
        measured_velocity = velocity if velocity is not None else current_state.velocity
        is_stopped = abs(measured_velocity) < 0.1  # Threshold for stopped train
        
        # Always update position measurement (even when stopped)
        measurement = np.array([
            position.latitude,
            position.longitude,
            measured_velocity,
            acceleration if acceleration is not None else 0.0,
            bearing if bearing is not None else current_state.bearing
        ])
        self.kalman.update(measurement)
        
        if is_stopped:
            # Apply Zero Velocity Update (ZUPT) constraint AFTER position update
            # This prevents filter drift when train is stopped at stations
            # Pass measured acceleration for dynamic covariance adjustment
            self.kalman.update(None, apply_zupt=True, measured_acceleration=acceleration)
            logger.debug(
                "train_zupt_applied",
                train_id=self.train_id,
                reason="measured_velocity near zero"
            )
        
        # Update track distance and gradient if provided
        if track_distance is not None:
            self.track_distance = track_distance
        if gradient is not None:
            self.gradient = gradient
        else:
            # Log warning if gradient is not provided (once per train to avoid spam)
            if not hasattr(self, '_gradient_warning_logged'):
                logger.warning(
                    "gradient_not_provided",
                    train_id=self.train_id,
                    message="Track gradient not provided. Davis physics will assume flat track. "
                            "Consider integrating Map Matching module with PostGIS to provide "
                            "accurate gradient data for improved physics simulation."
                )
                self._gradient_warning_logged = True
        
        # Update timestamp
        self.last_update = datetime.now()
        
        # Infer train state from acceleration
        state = self.kalman.get_state_vector()
        if abs(state.velocity) < 0.1:
            self.current_state = TrainState.STOPPED
        elif state.acceleration > 0.3:
            self.current_state = TrainState.ACCELERATING
        elif state.acceleration < -0.3:
            self.current_state = TrainState.BRAKING
        else:
            self.current_state = TrainState.CRUISING
            
        logger.debug(
            "train_state_updated",
            train_id=self.train_id,
            velocity=state.velocity,
            state=self.current_state.value
        )
        
    def _apply_moving_block_constraint(self, desired_acceleration: float) -> float:
        """
        Apply moving block (Cantonnement) constraint.
        
        A train cannot pass the train ahead of it. If too close, apply braking.
        
        Args:
            desired_acceleration: Desired acceleration from physics.
            
        Returns:
            Safe acceleration respecting moving block.
        """
        if not self.preceding_train:
            return desired_acceleration
            
        # Get current states
        my_state = self.kalman.get_state_vector()
        preceding_state = self.preceding_train.kalman.get_state_vector()
        
        # Calculate distance to preceding train
        distance = my_state.position.distance_to(preceding_state.position)
        
        # Relative velocity
        relative_velocity = my_state.velocity - preceding_state.velocity
        
        # If we're getting too close, apply emergency braking
        if distance < self.safe_distance:
            # Emergency braking proportional to how close we are
            braking_factor = (self.safe_distance - distance) / self.safe_distance
            emergency_braking = self.properties.max_braking * braking_factor
            
            logger.warning(
                "moving_block_constraint_active",
                train_id=self.train_id,
                preceding_train=self.preceding_train.train_id,
                distance=distance,
                emergency_braking=emergency_braking
            )
            
            return min(desired_acceleration, emergency_braking)
            
        # If closing in too fast, apply preventive braking
        if relative_velocity > 0:
            # Time to collision
            ttc = distance / relative_velocity if relative_velocity > 0 else float('inf')
            
            if ttc < 30.0:  # Less than 30 seconds to collision
                # Apply proportional braking
                braking = -relative_velocity / 10.0  # Smooth braking
                return min(desired_acceleration, braking)
                
        return desired_acceleration
        
    def get_current_state(self) -> TrainStateVector:
        """Get current estimated state."""
        state = self.kalman.get_state_vector()
        # Include track_distance and gradient which are tracked separately
        state.track_distance = self.track_distance
        state.gradient = self.gradient
        return state


class HybridFusionEngine:
    """
    Main fusion engine that manages all trains and processes Kafka messages.
    
    This is the orchestrator that:
    - Maintains all train entities
    - Processes real-time updates from Kafka
    - Runs physics simulation loop
    - Manages moving block relationships
    """
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "raw_telemetry",
        kafka_group_id: str = "fusion_engine",
        topology_path: Optional[str] = None,
        stops_path: Optional[str] = None
    ) -> None:
        """
        Initialize the fusion engine.
        
        Args:
            kafka_bootstrap_servers: Kafka broker addresses.
            kafka_topic: Kafka topic to consume from.
            kafka_group_id: Consumer group ID.
            topology_path: Path to topology JSON file for Rail-Lock (v6.0).
                          If None, operates without Rail-Lock (degraded mode).
            stops_path: Path to GTFS static stops.txt file for Holographic Positioning (v5.0).
                       If None, operates without positional inference (degraded mode).
        """
        self._kafka_bootstrap_servers = kafka_bootstrap_servers
        self._kafka_topic = kafka_topic
        self._kafka_group_id = kafka_group_id
        
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._trains: Dict[str, TrainEntity] = {}
        self._trip_to_train: Dict[str, str] = {}  # trip_id -> train_id mapping
        self._warned_routes: Set[str] = set()  # Track which routes we've warned about
        
        self._active = False
        self._simulation_rate = 1.0  # Hz
        
        # HNPS v5.0: Initialize Stop Registry for Holographic Positioning
        # Import here to avoid circular dependency and allow graceful degradation
        try:
            from src.engine.stops import StopRegistry
            self.stop_registry = StopRegistry(stops_path)
            if self.stop_registry.is_available():
                logger.info(
                    "fusion_engine_holographic_positioning_enabled",
                    stops_count=self.stop_registry.get_stops_count()
                )
            else:
                logger.warning(
                    "fusion_engine_holographic_positioning_degraded",
                    reason="No stops.txt loaded"
                )
        except Exception as e:
            logger.warning(
                "fusion_engine_holographic_positioning_disabled",
                error=str(e),
                reason="Stop Registry initialization failed"
            )
            self.stop_registry = None
        
        # HNPS v6.0: Initialize Topology Engine for Rail-Lock
        # Import here to avoid circular dependency and allow graceful degradation
        try:
            from src.engine.topology import TopologyEngine
            self.topology = TopologyEngine(topology_path)
            if self.topology.is_available():
                logger.info(
                    "fusion_engine_rail_lock_enabled",
                    shapes_count=len(self.topology.shapes),
                    routes_count=len(self.topology.route_to_shapes)
                )
            else:
                logger.warning(
                    "fusion_engine_rail_lock_degraded",
                    reason="No topology loaded"
                )
        except Exception as e:
            logger.warning(
                "fusion_engine_rail_lock_disabled",
                error=str(e),
                reason="Topology engine initialization failed"
            )
            self.topology = None
        
    async def start(self) -> None:
        """Start the fusion engine."""
        # Initialize Kafka consumer
        self._consumer = AIOKafkaConsumer(
            self._kafka_topic,
            bootstrap_servers=self._kafka_bootstrap_servers,
            group_id=self._kafka_group_id,
            value_deserializer=lambda m: m.decode('utf-8'),
            auto_offset_reset='latest'
        )
        
        try:
            await self._consumer.start()
            logger.info(
                "fusion_engine_started",
                kafka_servers=self._kafka_bootstrap_servers,
                topic=self._kafka_topic
            )
        except Exception as e:
            logger.error("fusion_engine_kafka_error", error=str(e))
            # Continue without Kafka for testing
            self._consumer = None
            
        self._active = True
        
        # Start background tasks
        asyncio.create_task(self._simulation_loop())
        asyncio.create_task(self._consume_loop())
        
    async def stop(self) -> None:
        """Stop the fusion engine."""
        self._active = False
        
        if self._consumer:
            await self._consumer.stop()
            
        logger.info("fusion_engine_stopped")
        
    async def _consume_loop(self) -> None:
        """Consume messages from Kafka and update train states."""
        if not self._consumer:
            logger.warning("fusion_engine_no_consumer")
            return
            
        logger.info("fusion_engine_consume_loop_started")
        
        try:
            async for message in self._consumer:
                if not self._active:
                    break
                    
                try:
                    data = json.loads(message.value)
                    await self._process_message(data)
                except Exception as e:
                    logger.error("fusion_engine_message_error", error=str(e))
                    
        except Exception as e:
            logger.error("fusion_engine_consume_error", error=str(e))
            
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """Process a single Kafka message."""
        message_type = data.get('type')
        
        if message_type == 'vehicle_position':
            await self._process_vehicle_position(data.get('data', {}))
        elif message_type == 'trip_update':
            await self._process_trip_update(data.get('data', {}))
            
    async def _process_vehicle_position(self, data: Dict[str, Any]) -> None:
        """
        Process vehicle position update.
        
        HNPS v6.0: Enhanced with Rail-Lock integration for absolute spatial awareness.
        """
        vehicle_id = data.get('vehicle_id')
        trip_id = data.get('trip_id')
        
        if not vehicle_id:
            return
        
        # Extract position data
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)
        route_id = data.get('route_id')
        
        # HNPS v6.0: Attempt Rail-Lock projection for spatial awareness
        track_distance = None
        gradient = None
        
        if self.topology and self.topology.is_available():
            try:
                rail_lock = self.topology.get_rail_lock(latitude, longitude, route_id)
                
                if rail_lock and rail_lock.cross_track_error < RAIL_LOCK_MAX_CROSS_TRACK_ERROR:
                    # Rail-Lock successful with acceptable error
                    track_distance = rail_lock.track_distance
                    gradient = rail_lock.gradient
                    
                    logger.debug(
                        "rail_lock_applied",
                        vehicle_id=vehicle_id,
                        route_id=route_id,
                        track_distance=track_distance,
                        cross_track_error=rail_lock.cross_track_error,
                        gradient_degrees=math.degrees(gradient),
                        confidence=rail_lock.confidence
                    )
                elif rail_lock:
                    # Rail-Lock found but error too high - log warning
                    logger.debug(
                        "rail_lock_rejected",
                        vehicle_id=vehicle_id,
                        route_id=route_id,
                        cross_track_error=rail_lock.cross_track_error,
                        reason=f"cross_track_error exceeds {RAIL_LOCK_MAX_CROSS_TRACK_ERROR}m threshold"
                    )
            except Exception as e:
                # Silently handle Rail-Lock errors - don't crash the fusion loop
                logger.debug(
                    "rail_lock_error",
                    vehicle_id=vehicle_id,
                    error=str(e)
                )
            
        # Get or create train entity
        if vehicle_id not in self._trains:
            initial_state = TrainStateVector(
                position=Position2D(
                    latitude=latitude,
                    longitude=longitude
                ),
                velocity=data.get('speed', 0.0) or 0.0,
                acceleration=0.0,
                bearing=data.get('bearing', 0.0) or 0.0,
                track_distance=track_distance,
                gradient=gradient
            )
            
            self._trains[vehicle_id] = TrainEntity(
                train_id=vehicle_id,
                trip_id=trip_id,
                route_id=route_id,
                initial_state=initial_state,
                route_type=data.get('route_type')  # Pass route_type for dynamic physics
            )
            
            if trip_id:
                self._trip_to_train[trip_id] = vehicle_id
        else:
            # Update existing train with Rail-Lock data
            train = self._trains[vehicle_id]
            position = Position2D(
                latitude=latitude,
                longitude=longitude
            )
            
            # HNPS v6.0: Inject Rail-Lock data into measurement update
            # This automatically enables:
            # 1. Cantonnement (Moving Block) - track_distance used for train ordering
            # 2. 3D Physics (Gravity) - gradient used in Davis equation
            train.update_from_measurement(
                position=position,
                velocity=data.get('speed'),
                bearing=data.get('bearing'),
                track_distance=track_distance,
                gradient=gradient
            )
            
    async def _process_trip_update(self, data: Dict[str, Any]) -> None:
        """
        Process trip update message with Holographic Positioning.
        
        HNPS v5.0 CRITICAL ENHANCEMENT: This method is now the CORE of the system
        when VehiclePosition data is unavailable. It implements Positional Inference
        by converting stop_id to GPS coordinates using the Stop Registry.
        
        Key Features:
        - Creates trains from TripUpdate alone (no VehiclePosition needed)
        - Infers position from next stop_id via Stop Registry
        - Injects virtual position into Kalman filter
        - Uses topology_engine to obtain PK (track_distance) if available
        - Falls back to trip_id as identifier if vehicle_id is missing
        
        This is Holographic Positioning: reconstructing full spatial awareness
        from minimal information (wait times only).
        """
        trip_id = data.get('trip_id')
        vehicle_id = data.get('vehicle_id')
        route_id = data.get('route_id')
        
        # CRITICAL: Use vehicle_id if available, otherwise fall back to trip_id
        # This handles cases where TripUpdate doesn't include vehicle_id
        train_id = vehicle_id if vehicle_id else trip_id
        
        if not train_id:
            logger.debug(
                "trip_update_skipped",
                reason="No vehicle_id or trip_id available"
            )
            return
        
        # Maintain trip_id to train_id mapping
        if trip_id and vehicle_id:
            self._trip_to_train[trip_id] = vehicle_id
        
        # Extract stop_time_updates - these contain the upcoming stops
        stop_time_updates = data.get('stop_time_updates', [])
        
        if not stop_time_updates:
            logger.debug(
                "trip_update_no_stops",
                train_id=train_id,
                reason="No stop_time_updates in TripUpdate"
            )
            return
        
        # Find the next stop (first stop in the future or with arrival/departure times)
        # GTFS-RT spec: stops are typically ordered by sequence
        next_stop = None
        for stop_update in stop_time_updates:
            stop_id = stop_update.get('stop_id')
            if stop_id:
                next_stop = stop_id
                break  # Use first stop with valid stop_id
        
        if not next_stop:
            logger.debug(
                "trip_update_no_valid_stop",
                train_id=train_id,
                reason="No valid stop_id found in stop_time_updates"
            )
            return
        
        # HOLOGRAPHIC POSITIONING: Convert stop_id to geographic coordinates
        if not self.stop_registry or not self.stop_registry.is_available():
            logger.debug(
                "holographic_positioning_unavailable",
                train_id=train_id,
                stop_id=next_stop,
                reason="Stop Registry not available - cannot infer position"
            )
            return
        
        stop_location = self.stop_registry.get_stop_location(next_stop)
        
        if not stop_location:
            logger.debug(
                "stop_not_found_in_registry",
                train_id=train_id,
                stop_id=next_stop,
                reason="Stop ID not found in Stop Registry"
            )
            return
        
        latitude, longitude = stop_location
        
        logger.debug(
            "holographic_positioning_success",
            train_id=train_id,
            stop_id=next_stop,
            latitude=latitude,
            longitude=longitude
        )
        
        # Attempt Rail-Lock projection for track_distance (PK) and gradient
        track_distance = None
        gradient = None
        
        if self.topology and self.topology.is_available():
            try:
                rail_lock = self.topology.get_rail_lock(latitude, longitude, route_id)
                
                if rail_lock and rail_lock.cross_track_error < RAIL_LOCK_MAX_CROSS_TRACK_ERROR:
                    track_distance = rail_lock.track_distance
                    gradient = rail_lock.gradient
                    
                    logger.debug(
                        "rail_lock_applied_from_trip_update",
                        train_id=train_id,
                        route_id=route_id,
                        track_distance=track_distance,
                        cross_track_error=rail_lock.cross_track_error,
                        gradient_degrees=math.degrees(gradient) if gradient else None,
                        confidence=rail_lock.confidence
                    )
            except Exception as e:
                logger.debug(
                    "rail_lock_error_from_trip_update",
                    train_id=train_id,
                    error=str(e)
                )
        
        # CRITICAL: Create train if it doesn't exist
        if train_id not in self._trains:
            # Train doesn't exist - CREATE IT NOW (this is the whole point!)
            initial_state = TrainStateVector(
                position=Position2D(
                    latitude=latitude,
                    longitude=longitude
                ),
                velocity=0.0,  # Assume stopped at station
                acceleration=0.0,
                bearing=0.0,  # Unknown bearing - will be updated later
                track_distance=track_distance,
                gradient=gradient
            )
            
            self._trains[train_id] = TrainEntity(
                train_id=train_id,
                trip_id=trip_id,
                route_id=route_id,
                initial_state=initial_state,
                route_type=data.get('route_type')
            )
            
            if trip_id:
                self._trip_to_train[trip_id] = train_id
            
            logger.info(
                "train_created_from_trip_update",
                train_id=train_id,
                trip_id=trip_id,
                route_id=route_id,
                stop_id=next_stop,
                method="holographic_positioning",
                latitude=latitude,
                longitude=longitude,
                track_distance=track_distance
            )
        else:
            # Train exists - update it with inferred position as virtual measurement
            train = self._trains[train_id]
            position = Position2D(
                latitude=latitude,
                longitude=longitude
            )
            
            # Inject inferred position as "virtual measurement" into Kalman filter
            # Since train is at a station, assume velocity is 0 (stopped)
            train.update_from_measurement(
                position=position,
                velocity=0.0,  # Assume stopped at station
                bearing=None,  # Keep existing bearing
                track_distance=track_distance,
                gradient=gradient
            )
            
            logger.debug(
                "train_updated_from_trip_update",
                train_id=train_id,
                stop_id=next_stop,
                method="holographic_positioning",
                latitude=latitude,
                longitude=longitude,
                track_distance=track_distance
            )
            
    async def _simulation_loop(self) -> None:
        """Physics simulation loop."""
        logger.info("fusion_engine_simulation_loop_started")
        
        dt = 1.0 / self._simulation_rate
        
        while self._active:
            try:
                # Predict all trains forward
                for train in self._trains.values():
                    train.predict(dt)
                    
                # Update moving block relationships (simplified)
                self._update_moving_blocks()
                
                await asyncio.sleep(dt)
                
            except Exception as e:
                logger.error("fusion_engine_simulation_error", error=str(e))
                await asyncio.sleep(1.0)
                
    def _update_moving_blocks(self) -> None:
        """Update moving block relationships between trains on same route."""
        # Group trains by route
        routes: Dict[str, List[TrainEntity]] = {}
        
        for train in self._trains.values():
            if train.route_id:
                if train.route_id not in routes:
                    routes[train.route_id] = []
                routes[train.route_id].append(train)
                
        # For each route, establish preceding/following relationships
        for route_id, route_trains in routes.items():
            if len(route_trains) < 2:
                continue
                
            # Check if all trains have track_distance
            all_have_track_distance = all(
                t.get_current_state().track_distance is not None
                for t in route_trains
            )
            
            if all_have_track_distance:
                # Sort by track distance (curvilinear distance along track)
                # This is the correct method that handles loops and U-turns
                route_trains.sort(
                    key=lambda t: t.get_current_state().track_distance
                )
                
                # Set relationships
                for i in range(len(route_trains)):
                    if i > 0:
                        route_trains[i].preceding_train = route_trains[i - 1]
                    else:
                        route_trains[i].preceding_train = None
                        
                    if i < len(route_trains) - 1:
                        route_trains[i].following_train = route_trains[i + 1]
                    else:
                        route_trains[i].following_train = None
            else:
                # CRITICAL FIX: Do NOT sort by lat/lon as it's unsafe for routes with loops/U-turns
                # Instead, disable cantonnement (degraded mode) for this route
                # Better to have no safety than false safety that causes incorrect braking
                if route_id not in self._warned_routes:
                    logger.warning(
                        "moving_block_disabled_degraded_mode",
                        route_id=route_id,
                        reason="track_distance not available - cantonnement disabled for safety"
                    )
                    self._warned_routes.add(route_id)
                
                # Clear all relationships - trains operate independently
                for train in route_trains:
                    train.preceding_train = None
                    train.following_train = None
                    
    def get_train(self, train_id: str) -> Optional[TrainEntity]:
        """Get a train by ID."""
        return self._trains.get(train_id)
        
    def get_all_trains(self) -> List[TrainEntity]:
        """Get all active trains."""
        return list(self._trains.values())


async def main() -> None:
    """Demo/test function for the Fusion Engine."""
    logging.basicConfig(level=logging.INFO)
    
    engine = HybridFusionEngine(kafka_bootstrap_servers="localhost:9092")
    
    try:
        await engine.start()
        
        print(f"\n{'='*80}")
        print(f"HYBRID FUSION ENGINE - STARTED")
        print(f"{'='*80}\n")
        print("Monitoring trains... (Press Ctrl+C to stop)")
        
        # Run for a while
        while True:
            await asyncio.sleep(10)
            
            trains = engine.get_all_trains()
            print(f"\nActive trains: {len(trains)}")
            
            for train in trains[:5]:  # Show first 5
                state = train.get_current_state()
                print(f"  {train.train_id}: v={state.velocity:.2f}m/s, "
                      f"pos=({state.position.latitude:.6f}, {state.position.longitude:.6f})")
                      
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
