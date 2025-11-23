"""
Tests for critical bug fixes in HNPS v5.0
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from src.engine.fusion import (
    Position2D,
    TrainStateVector,
    TrainEntity,
    TrainState,
    HybridFusionEngine,
    get_davis_coefficients_for_route_type,
    DavisCoefficients
)
from src.ingestion.harvester import GTFSRTHarvester, HarvestMetrics, FeedStatus
from src.core.omniscience import OmniscienceEngine, GTFSRTResource


class TestMovingBlockFix:
    """Test moving block sorting with track_distance."""
    
    def test_track_distance_attribute_exists(self):
        """Test that TrainStateVector has track_distance attribute."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        state = TrainStateVector(
            position=position,
            velocity=20.0,
            acceleration=0.5,
            bearing=90.0,
            track_distance=1500.0
        )
        
        assert state.track_distance == 1500.0
    
    def test_track_distance_optional(self):
        """Test that track_distance is optional."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        state = TrainStateVector(
            position=position,
            velocity=20.0,
            acceleration=0.5,
            bearing=90.0
        )
        
        assert state.track_distance is None
    
    @pytest.mark.asyncio
    async def test_moving_block_sorts_by_track_distance(self):
        """Test that moving blocks are sorted by track_distance when available."""
        engine = HybridFusionEngine()
        await engine.start()
        
        try:
            # Create trains with track_distance on same route
            train1_state = TrainStateVector(
                position=Position2D(latitude=48.8566, longitude=2.3522),
                velocity=10.0,
                acceleration=0.0,
                bearing=0.0,
                track_distance=2000.0  # Further along track
            )
            train1 = TrainEntity(
                train_id="TRAIN_001",
                trip_id="TRIP_001",
                route_id="RER_A",
                initial_state=train1_state
            )
            
            train2_state = TrainStateVector(
                position=Position2D(latitude=48.8500, longitude=2.3500),
                velocity=10.0,
                acceleration=0.0,
                bearing=0.0,
                track_distance=1000.0  # Earlier on track
            )
            train2 = TrainEntity(
                train_id="TRAIN_002",
                trip_id="TRIP_002",
                route_id="RER_A",
                initial_state=train2_state
            )
            
            # Add trains to engine
            engine._trains["TRAIN_001"] = train1
            engine._trains["TRAIN_002"] = train2
            
            # Update moving blocks
            engine._update_moving_blocks()
            
            # Train 1 (further along) should have train 2 as preceding
            assert train1.preceding_train == train2
            assert train2.following_train == train1
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_moving_block_disabled_without_track_distance(self):
        """Test that moving blocks are disabled (not using lat/lon) when track_distance unavailable."""
        engine = HybridFusionEngine()
        await engine.start()
        
        try:
            # Create trains WITHOUT track_distance on same route
            train1_state = TrainStateVector(
                position=Position2D(latitude=48.8566, longitude=2.3522),
                velocity=10.0,
                acceleration=0.0,
                bearing=0.0,
                track_distance=None  # No track distance
            )
            train1 = TrainEntity(
                train_id="TRAIN_001",
                trip_id="TRIP_001",
                route_id="RER_A",
                initial_state=train1_state
            )
            
            train2_state = TrainStateVector(
                position=Position2D(latitude=48.8500, longitude=2.3500),
                velocity=10.0,
                acceleration=0.0,
                bearing=0.0,
                track_distance=None  # No track distance
            )
            train2 = TrainEntity(
                train_id="TRAIN_002",
                trip_id="TRIP_002",
                route_id="RER_A",
                initial_state=train2_state
            )
            
            # Add trains to engine
            engine._trains["TRAIN_001"] = train1
            engine._trains["TRAIN_002"] = train2
            
            # Update moving blocks
            engine._update_moving_blocks()
            
            # Both trains should have NO relationships (degraded mode)
            assert train1.preceding_train is None
            assert train1.following_train is None
            assert train2.preceding_train is None
            assert train2.following_train is None
            
        finally:
            await engine.stop()


class TestHarvesterTimingFix:
    """Test harvester timing to prevent temporal drift."""
    
    @pytest.mark.asyncio
    async def test_harvest_continuously_calculates_dynamic_sleep(self):
        """Test that harvest_continuously uses dynamic sleep time."""
        
        # Mock resource
        test_resource = GTFSRTResource(
            url="https://example.com/gtfs-rt",
            dataset_id="test",
            organization="Test Operator",
            resource_type="vehicle-positions",
            title="Test Feed"
        )
        
        # Track sleep times
        sleep_times = []
        original_sleep = asyncio.sleep
        
        async def mock_sleep(duration):
            sleep_times.append(duration)
            # Use original sleep for actual waiting but stop after first cycle
            await original_sleep(0.01)
            if len(sleep_times) >= 1:
                # Deactivate harvester after first cycle to stop loop
                harvester._active = False
        
        # Create harvester with mocked methods
        # Don't connect to Kafka to avoid delays from reconnection attempts
        harvester = GTFSRTHarvester()
        harvester._session = Mock()  # Mock session to avoid real HTTP connection
        harvester._producer = None  # No Kafka producer
        harvester._own_producer = False  # Don't try to reconnect
        harvester._active = True
        
        try:
            # Mock harvest_resource to simulate processing time
            processing_time = 0.05
            async def mock_harvest(resource, timeout=10):
                await original_sleep(processing_time)  # Simulate 50ms processing
                return HarvestMetrics(
                    url=resource.url,
                    timestamp=datetime.now(),
                    status=FeedStatus.ACTIVE,
                    entities_count=10,
                    response_time_ms=processing_time * 1000
                )
            
            harvester.harvest_resource = mock_harvest
            
            # Patch asyncio.sleep to capture sleep times
            with patch('asyncio.sleep', side_effect=mock_sleep):
                # Run one cycle with 1 second interval
                task = asyncio.create_task(
                    harvester.harvest_continuously([test_resource], interval=1.0)
                )
                
                # Wait for task to complete (it should stop after one cycle due to mock)
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
            
            # Check that sleep time was calculated (should be close to interval - processing_time)
            # With 1s interval and ~50ms processing, sleep should be ~950ms
            assert len(sleep_times) > 0
            # Sleep time should be less than full interval (accounting for processing time)
            assert sleep_times[0] < 1.0
            # Sleep time should be approximately interval - processing_time
            assert 0.9 < sleep_times[0] < 1.0  # Should be close to 0.95
                
        finally:
            harvester._active = False  # Stop harvester


class TestKafkaReconnection:
    """Test Kafka reconnection with exponential backoff."""
    
    @pytest.mark.asyncio
    async def test_kafka_reconnection_on_startup_failure(self):
        """Test that harvester retries Kafka connection with exponential backoff on startup failure."""
        from aiokafka.errors import KafkaConnectionError
        
        # Track connection attempts
        connection_attempts = []
        
        # Mock AIOKafkaProducer to fail first 2 attempts, succeed on 3rd
        original_producer = None
        
        class MockProducer:
            def __init__(self, *args, **kwargs):
                connection_attempts.append(len(connection_attempts))
                self.started = False
            
            async def start(self):
                attempt = len(connection_attempts)
                if attempt < 3:  # Fail first 2 attempts
                    raise KafkaConnectionError("Connection failed")
                self.started = True
            
            async def stop(self):
                pass
        
        # Patch AIOKafkaProducer and asyncio.sleep (to speed up test)
        with patch('src.ingestion.harvester.AIOKafkaProducer', MockProducer):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                harvester = GTFSRTHarvester(
                    kafka_bootstrap_servers="localhost:9092"
                )
                
                # Start should retry and eventually succeed
                await harvester.start()
                
                # Should have made 3 attempts (initial + 2 retries)
                assert len(connection_attempts) == 3
                assert harvester._producer is not None
                assert harvester._producer.started is True
                
                await harvester.stop()
    
    @pytest.mark.asyncio
    async def test_kafka_reconnection_exhausts_retries(self):
        """Test that harvester gives up after max retries."""
        from aiokafka.errors import KafkaConnectionError
        
        # Track connection attempts
        connection_attempts = []
        
        class MockProducer:
            def __init__(self, *args, **kwargs):
                connection_attempts.append(len(connection_attempts))
            
            async def start(self):
                # Always fail
                raise KafkaConnectionError("Connection failed")
            
            async def stop(self):
                pass
        
        # Patch AIOKafkaProducer and asyncio.sleep (to speed up test)
        with patch('src.ingestion.harvester.AIOKafkaProducer', MockProducer):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                harvester = GTFSRTHarvester(
                    kafka_bootstrap_servers="localhost:9092"
                )
                
                # Start should try MAX_RETRIES times then give up
                await harvester.start()
                
                # Should have made MAX_RETRIES + 1 attempts (initial + MAX_RETRIES retries)
                assert len(connection_attempts) == harvester.MAX_RETRIES + 1
                # Producer should be None after exhausting retries
                assert harvester._producer is None
                
                await harvester.stop()
    
    @pytest.mark.asyncio
    async def test_kafka_reconnection_during_harvest_loop(self):
        """Test that _connect_kafka is called when producer is None during harvest loop."""
        
        # Track if _connect_kafka was called
        connect_called = []
        
        harvester = GTFSRTHarvester()
        
        # Mock _connect_kafka to track calls
        async def mock_connect(retry_attempt=0):
            connect_called.append(True)
            return False  # Return False to keep producer None
        
        harvester._connect_kafka = mock_connect
        
        # Set up harvester state
        harvester._producer = None
        harvester._own_producer = True
        harvester._active = True
        
        # Mock the harvest_resource to do nothing
        harvester.harvest_resource = AsyncMock(return_value=HarvestMetrics(
            url="test",
            timestamp=datetime.now(),
            status=FeedStatus.ACTIVE
        ))
        
        # Create test resource
        test_resource = GTFSRTResource(
            url="https://example.com/gtfs-rt",
            dataset_id="test",
            organization="Test Operator",
            resource_type="vehicle-positions",
            title="Test Feed"
        )
        
        # Mock asyncio.sleep to avoid delays and stop after first cycle
        call_count = [0]
        async def mock_sleep(duration):
            call_count[0] += 1
            if call_count[0] >= 1:
                harvester._active = False
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            await harvester.harvest_continuously([test_resource], interval=1.0)
        
        # Should have called _connect_kafka at least once
        assert len(connect_called) > 0
        
        await harvester.stop()


class TestGTFSRTDetectionFix:
    """Test improved GTFS-RT detection with content-type checking."""
    
    @pytest.mark.asyncio
    async def test_is_gtfs_rt_by_content_detects_protobuf(self):
        """Test that content-type checking detects protobuf."""
        async with OmniscienceEngine() as engine:
            # Mock HTTP response with protobuf content-type
            with patch('aiohttp.ClientSession.head') as mock_head:
                mock_response = AsyncMock()
                mock_response.headers = {'Content-Type': 'application/x-protobuf'}
                mock_head.return_value.__aenter__.return_value = mock_response
                
                result = await engine._is_gtfs_rt_by_content("https://example.com/feed")
                assert result is True
    
    @pytest.mark.asyncio
    async def test_is_gtfs_rt_by_content_detects_octet_stream(self):
        """Test that content-type checking detects octet-stream."""
        async with OmniscienceEngine() as engine:
            # Mock HTTP response with octet-stream content-type
            with patch('aiohttp.ClientSession.head') as mock_head:
                mock_response = AsyncMock()
                mock_response.headers = {'Content-Type': 'application/octet-stream'}
                mock_head.return_value.__aenter__.return_value = mock_response
                
                result = await engine._is_gtfs_rt_by_content("https://example.com/feed")
                assert result is True
    
    @pytest.mark.asyncio
    async def test_is_gtfs_rt_by_content_handles_failures(self):
        """Test that content-type checking handles failures gracefully."""
        async with OmniscienceEngine() as engine:
            # Mock HTTP request that raises exception
            with patch('aiohttp.ClientSession.head', side_effect=Exception("Network error")):
                result = await engine._is_gtfs_rt_by_content("https://example.com/feed")
                assert result is False


class TestDynamicDavisCoefficients:
    """Test route_type-based Davis coefficients."""
    
    def test_get_davis_coefficients_for_tram(self):
        """Test Davis coefficients for tram (route_type=0)."""
        coeffs = get_davis_coefficients_for_route_type(0)
        
        assert isinstance(coeffs, DavisCoefficients)
        # Tram should have lower coefficients than metro
        assert coeffs.A == 4.0
        assert coeffs.B == 0.025
        assert coeffs.C == 0.0012
    
    def test_get_davis_coefficients_for_subway(self):
        """Test Davis coefficients for subway (route_type=1)."""
        coeffs = get_davis_coefficients_for_route_type(1)
        
        assert isinstance(coeffs, DavisCoefficients)
        # Subway should have default metro coefficients
        assert coeffs.A == 5.0
        assert coeffs.B == 0.03
        assert coeffs.C == 0.0015
    
    def test_get_davis_coefficients_for_rail(self):
        """Test Davis coefficients for intercity rail (route_type=2)."""
        coeffs = get_davis_coefficients_for_route_type(2)
        
        assert isinstance(coeffs, DavisCoefficients)
        # Rail should have higher aerodynamic resistance
        assert coeffs.A == 3.5
        assert coeffs.B == 0.02
        assert coeffs.C == 0.0025
    
    def test_get_davis_coefficients_for_none(self):
        """Test Davis coefficients for None (default)."""
        coeffs = get_davis_coefficients_for_route_type(None)
        
        assert isinstance(coeffs, DavisCoefficients)
        # Should return default metro coefficients
        assert coeffs.A == 5.0
        assert coeffs.B == 0.03
        assert coeffs.C == 0.0015
    
    def test_train_entity_uses_route_type_coefficients(self):
        """Test that TrainEntity uses route_type-specific coefficients."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        initial_state = TrainStateVector(
            position=position,
            velocity=10.0,
            acceleration=0.0,
            bearing=0.0
        )
        
        # Create train with rail route_type
        train = TrainEntity(
            train_id="TRAIN_TGV",
            trip_id="TRIP_TGV",
            route_id="INTERCITY",
            initial_state=initial_state,
            route_type=2  # Rail
        )
        
        # Check that it has rail-specific coefficients
        assert train.properties.davis.A == 3.5
        assert train.properties.davis.B == 0.02
        assert train.properties.davis.C == 0.0025


class TestZeroVelocityUpdate:
    """Test Zero Velocity Update (ZUPT) for stopped trains."""
    
    def test_zupt_applied_when_velocity_zero(self):
        """Test that ZUPT is applied when measured velocity is zero."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        initial_state = TrainStateVector(
            position=position,
            velocity=5.0,  # Initially moving
            acceleration=0.0,
            bearing=90.0
        )
        
        train = TrainEntity(
            train_id="TRAIN_001",
            trip_id="TRIP_001",
            route_id="METRO_1",
            initial_state=initial_state
        )
        
        # Update with zero velocity (train stopped)
        train.update_from_measurement(
            position=position,
            velocity=0.0,  # Stopped
            bearing=90.0
        )
        
        # Check that velocity and acceleration are constrained to zero
        state = train.get_current_state()
        assert abs(state.velocity) < 0.01  # Should be very close to zero
        assert abs(state.acceleration) < 0.01  # Should be very close to zero
    
    def test_zupt_not_applied_when_moving(self):
        """Test that ZUPT is not applied when train is moving."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        initial_state = TrainStateVector(
            position=position,
            velocity=5.0,
            acceleration=0.0,
            bearing=90.0
        )
        
        train = TrainEntity(
            train_id="TRAIN_001",
            trip_id="TRIP_001",
            route_id="METRO_1",
            initial_state=initial_state
        )
        
        # Update with non-zero velocity (train moving)
        train.update_from_measurement(
            position=Position2D(latitude=48.8570, longitude=2.3525),
            velocity=10.0,  # Moving
            bearing=90.0
        )
        
        # Check that velocity is updated (not forced to zero)
        state = train.get_current_state()
        assert abs(state.velocity) > 5.0  # Should be non-zero and updated
    
    def test_zupt_reduces_velocity_covariance(self):
        """Test that ZUPT reduces velocity covariance in the Kalman filter."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        initial_state = TrainStateVector(
            position=position,
            velocity=5.0,
            acceleration=0.0,
            bearing=90.0
        )
        
        train = TrainEntity(
            train_id="TRAIN_001",
            trip_id="TRIP_001",
            route_id="METRO_1",
            initial_state=initial_state
        )
        
        # Get initial velocity covariance
        initial_velocity_variance = train.kalman.P[2, 2]
        
        # Apply ZUPT by updating with zero velocity
        train.update_from_measurement(
            position=position,
            velocity=0.0,
            bearing=90.0
        )
        
        # Check that velocity covariance is reduced
        final_velocity_variance = train.kalman.P[2, 2]
        assert final_velocity_variance < initial_velocity_variance
        assert final_velocity_variance < 0.01  # Should be drastically reduced
    
    def test_zupt_prevents_drift_during_stop(self):
        """Test that ZUPT prevents the UKF from drifting when train is stopped."""
        position = Position2D(latitude=48.8566, longitude=2.3522)
        initial_state = TrainStateVector(
            position=position,
            velocity=10.0,  # Moving fast
            acceleration=0.0,
            bearing=90.0
        )
        
        train = TrainEntity(
            train_id="TRAIN_001",
            trip_id="TRIP_001",
            route_id="METRO_1",
            initial_state=initial_state
        )
        
        # Suddenly stop (like at a station)
        train.update_from_measurement(
            position=position,
            velocity=0.0,  # Stopped abruptly
            bearing=90.0
        )
        
        # Without ZUPT, the filter would maintain some velocity due to inertia
        # With ZUPT, velocity should be forced to zero
        state = train.get_current_state()
        assert abs(state.velocity) < 0.01
        assert train.current_state == TrainState.STOPPED


class TestGhostTrainFix:
    """Test ghost train duplicate prevention fix."""
    
    @pytest.mark.asyncio
    async def test_trip_update_without_vehicle_id_then_with_vehicle_id_no_duplicate(self, tmp_path):
        """
        Test that when vehicle_id appears later, it updates the existing train
        instead of creating a duplicate.
        
        Scenario:
        1. First TripUpdate: trip_id='TRIP_X', vehicle_id=None → creates train 'TRIP_X'
        2. Second TripUpdate: trip_id='TRIP_X', vehicle_id='VEH_X' → should update 'TRIP_X', not create 'VEH_X'
        """
        # Create stops file
        stops_file = tmp_path / "stops.txt"
        stops_file.write_text(
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "STOP_A,Station A,48.8566,2.3522\n"
            "STOP_B,Station B,48.8800,2.3550\n"
        )
        
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            stops_path=str(stops_file)
        )
        
        # First TripUpdate: no vehicle_id
        trip_update_1 = {
            'trip_id': 'TRIP_X',
            # No vehicle_id!
            'route_id': 'LINE_1',
            'stop_time_updates': [
                {'stop_id': 'STOP_A'}
            ]
        }
        await engine._process_trip_update(trip_update_1)
        
        # Verify train was created with trip_id as identifier
        assert 'TRIP_X' in engine._trains
        assert len(engine.get_all_trains()) == 1
        train_by_trip = engine.get_train('TRIP_X')
        assert train_by_trip is not None
        assert train_by_trip.train_id == 'TRIP_X'
        
        # Second TripUpdate: vehicle_id appears
        trip_update_2 = {
            'trip_id': 'TRIP_X',
            'vehicle_id': 'VEH_X',
            'route_id': 'LINE_1',
            'stop_time_updates': [
                {'stop_id': 'STOP_B'}
            ]
        }
        await engine._process_trip_update(trip_update_2)
        
        # CRITICAL: Should still have only 1 train (no duplicate)
        assert len(engine.get_all_trains()) == 1, "Ghost train detected! Duplicate train created."
        
        # The train should now be accessible by vehicle_id
        assert 'VEH_X' in engine._trains
        train_by_vehicle = engine.get_train('VEH_X')
        assert train_by_vehicle is not None
        
        # Should be the SAME train object (not a duplicate)
        assert train_by_vehicle.train_id == 'VEH_X'
        
        # Mapping should be updated
        assert 'TRIP_X' in engine._trip_to_train
        assert engine._trip_to_train['TRIP_X'] == 'VEH_X'
        
        # Old identifier (TRIP_X) should no longer exist in _trains dict
        # (it has been migrated to VEH_X)
        assert 'TRIP_X' not in engine._trains
    
    @pytest.mark.asyncio
    async def test_trip_update_with_vehicle_id_from_start(self, tmp_path):
        """
        Test that when vehicle_id is present from the start, it works normally.
        """
        # Create stops file
        stops_file = tmp_path / "stops.txt"
        stops_file.write_text(
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "STOP_A,Station A,48.8566,2.3522\n"
        )
        
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            stops_path=str(stops_file)
        )
        
        # TripUpdate with vehicle_id from start
        trip_update = {
            'trip_id': 'TRIP_Y',
            'vehicle_id': 'VEH_Y',
            'route_id': 'LINE_2',
            'stop_time_updates': [
                {'stop_id': 'STOP_A'}
            ]
        }
        await engine._process_trip_update(trip_update)
        
        # Verify train was created with vehicle_id as identifier
        assert 'VEH_Y' in engine._trains
        assert len(engine.get_all_trains()) == 1
        train = engine.get_train('VEH_Y')
        assert train is not None
        assert train.train_id == 'VEH_Y'
        assert train.trip_id == 'TRIP_Y'
        
        # Mapping should be established
        assert 'TRIP_Y' in engine._trip_to_train
        assert engine._trip_to_train['TRIP_Y'] == 'VEH_Y'
    
    @pytest.mark.asyncio
    async def test_trip_update_resolves_from_mapping(self, tmp_path):
        """
        Test that when vehicle_id is absent, but we have a mapping from a previous update,
        it uses the mapped vehicle_id instead of creating a duplicate.
        """
        # Create stops file
        stops_file = tmp_path / "stops.txt"
        stops_file.write_text(
            "stop_id,stop_name,stop_lat,stop_lon\n"
            "STOP_A,Station A,48.8566,2.3522\n"
            "STOP_B,Station B,48.8800,2.3550\n"
        )
        
        engine = HybridFusionEngine(
            kafka_bootstrap_servers="localhost:9092",
            stops_path=str(stops_file)
        )
        
        # First TripUpdate: with vehicle_id
        trip_update_1 = {
            'trip_id': 'TRIP_Z',
            'vehicle_id': 'VEH_Z',
            'route_id': 'LINE_3',
            'stop_time_updates': [
                {'stop_id': 'STOP_A'}
            ]
        }
        await engine._process_trip_update(trip_update_1)
        
        assert 'VEH_Z' in engine._trains
        assert len(engine.get_all_trains()) == 1
        
        # Second TripUpdate: vehicle_id is missing this time
        trip_update_2 = {
            'trip_id': 'TRIP_Z',
            # No vehicle_id!
            'route_id': 'LINE_3',
            'stop_time_updates': [
                {'stop_id': 'STOP_B'}
            ]
        }
        await engine._process_trip_update(trip_update_2)
        
        # Should still have only 1 train (should resolve via mapping)
        assert len(engine.get_all_trains()) == 1, "Duplicate train created when resolving via mapping"
        
        # Train should still be accessible by vehicle_id
        train = engine.get_train('VEH_Z')
        assert train is not None
        assert train.train_id == 'VEH_Z'
