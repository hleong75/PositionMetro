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
        harvester = GTFSRTHarvester()
        await harvester.start()
        
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
