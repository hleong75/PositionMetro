"""
Module B: Ingestion Vorace (High-Performance Harvester)
========================================================
This module implements high-performance asynchronous harvesting of GTFS-RT feeds.
It parses Protocol Buffer data and publishes to Kafka for downstream processing.

HNPS v5.0 Component: Real-Time Data Ingestion Pipeline
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import aiohttp
import structlog
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

try:
    from google.transit import gtfs_realtime_pb2
except ImportError:
    # Fallback if gtfs-realtime-bindings is not installed
    gtfs_realtime_pb2 = None

from src.core.omniscience import GTFSRTResource

logger = structlog.get_logger(__name__)


class FeedStatus(Enum):
    """Status of a GTFS-RT feed."""
    ACTIVE = "active"
    TIMEOUT = "timeout"
    DEAD = "dead"
    PARSE_ERROR = "parse_error"
    HTTP_ERROR = "http_error"


@dataclass
class HarvestMetrics:
    """Metrics for a single harvest operation."""
    
    url: str
    timestamp: datetime
    status: FeedStatus
    entities_count: int = 0
    response_time_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class NormalizedVehiclePosition:
    """Normalized vehicle position data."""
    
    vehicle_id: str
    trip_id: Optional[str]
    route_id: Optional[str]
    latitude: float
    longitude: float
    bearing: Optional[float]
    speed: Optional[float]  # m/s
    timestamp: int
    current_stop_sequence: Optional[int]
    current_status: Optional[str]
    congestion_level: Optional[str]
    occupancy_status: Optional[str]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))


@dataclass
class NormalizedTripUpdate:
    """Normalized trip update data."""
    
    trip_id: str
    route_id: Optional[str]
    vehicle_id: Optional[str]
    start_date: Optional[str]
    start_time: Optional[str]
    schedule_relationship: Optional[str]
    stop_time_updates: List[Dict[str, Any]]
    timestamp: int
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))


class GTFSRTHarvester:
    """
    High-performance asynchronous harvester for GTFS-RT feeds.
    
    This harvester connects to discovered GTFS-RT URLs, parses Protocol Buffer
    data, normalizes it to a standard JSON format, and publishes to Kafka.
    It handles timeouts, dead feeds, and format changes gracefully.
    """
    
    DEFAULT_TIMEOUT = 10  # seconds
    RETRY_DELAY = 5  # seconds between retries
    MAX_RETRIES = 3
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "raw_telemetry",
        session: Optional[aiohttp.ClientSession] = None,
        producer: Optional[AIOKafkaProducer] = None,
        on_metrics: Optional[Callable[[HarvestMetrics], None]] = None
    ) -> None:
        """
        Initialize the GTFS-RT Harvester.
        
        Args:
            kafka_bootstrap_servers: Kafka broker addresses.
            kafka_topic: Kafka topic to publish to.
            session: Optional aiohttp session.
            producer: Optional Kafka producer.
            on_metrics: Optional callback for harvest metrics.
        """
        self._kafka_bootstrap_servers = kafka_bootstrap_servers
        self._kafka_topic = kafka_topic
        self._session = session
        self._own_session = session is None
        self._producer = producer
        self._own_producer = producer is None
        self._on_metrics = on_metrics
        self._active = False
        
    async def __aenter__(self) -> "GTFSRTHarvester":
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
        
    async def start(self) -> None:
        """Start the harvester (initialize connections)."""
        if self._own_session:
            self._session = aiohttp.ClientSession()
            
        if self._own_producer:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip'
            )
            try:
                await self._producer.start()
                logger.info(
                    "harvester_kafka_connected",
                    bootstrap_servers=self._kafka_bootstrap_servers
                )
            except Exception as e:
                logger.warning(
                    "harvester_kafka_connection_failed",
                    error=str(e),
                    note="Will continue without Kafka publishing"
                )
                # Continue without Kafka - useful for testing
                self._producer = None
                
        self._active = True
        logger.info("harvester_started")
        
    async def stop(self) -> None:
        """Stop the harvester (close connections)."""
        self._active = False
        
        if self._own_session and self._session:
            await self._session.close()
            
        if self._own_producer and self._producer:
            await self._producer.stop()
            
        logger.info("harvester_stopped")
        
    async def harvest_resource(
        self,
        resource: GTFSRTResource,
        timeout: float = DEFAULT_TIMEOUT
    ) -> HarvestMetrics:
        """
        Harvest data from a single GTFS-RT resource.
        
        Args:
            resource: GTFS-RT resource to harvest.
            timeout: Request timeout in seconds.
            
        Returns:
            Metrics for this harvest operation.
        """
        if not self._session:
            raise RuntimeError("Harvester not started. Use async context manager.")
            
        start_time = datetime.now()
        metrics = HarvestMetrics(
            url=resource.url,
            timestamp=start_time,
            status=FeedStatus.ACTIVE
        )
        
        try:
            # Fetch the GTFS-RT data
            async with self._session.get(
                resource.url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                metrics.response_time_ms = elapsed
                
                if response.status != 200:
                    metrics.status = FeedStatus.HTTP_ERROR
                    metrics.error_message = f"HTTP {response.status}"
                    logger.warning(
                        "harvester_http_error",
                        url=resource.url,
                        status=response.status
                    )
                    return metrics
                    
                # Read the Protocol Buffer data
                pb_data = await response.read()
                
        except asyncio.TimeoutError:
            metrics.status = FeedStatus.TIMEOUT
            metrics.error_message = "Request timeout"
            logger.warning("harvester_timeout", url=resource.url)
            return metrics
        except Exception as e:
            metrics.status = FeedStatus.DEAD
            metrics.error_message = str(e)
            logger.error("harvester_fetch_error", url=resource.url, error=str(e))
            return metrics
            
        # Parse the Protocol Buffer
        try:
            entities = await self._parse_gtfs_rt(pb_data, resource)
            metrics.entities_count = len(entities)
            
            # Publish to Kafka
            if self._producer and entities:
                await self._publish_to_kafka(entities, resource)
                
            logger.debug(
                "harvester_success",
                url=resource.url,
                entities=len(entities),
                response_time_ms=metrics.response_time_ms
            )
            
        except Exception as e:
            metrics.status = FeedStatus.PARSE_ERROR
            metrics.error_message = f"Parse error: {str(e)}"
            logger.error("harvester_parse_error", url=resource.url, error=str(e))
            
        # Call metrics callback if provided
        if self._on_metrics:
            try:
                self._on_metrics(metrics)
            except Exception as e:
                logger.error("harvester_metrics_callback_error", error=str(e))
                
        return metrics
        
    async def _parse_gtfs_rt(
        self,
        pb_data: bytes,
        resource: GTFSRTResource
    ) -> List[Dict[str, Any]]:
        """
        Parse GTFS-RT Protocol Buffer data.
        
        Args:
            pb_data: Raw Protocol Buffer bytes.
            resource: Resource metadata.
            
        Returns:
            List of normalized entity dictionaries.
        """
        if gtfs_realtime_pb2 is None:
            logger.warning("gtfs_realtime_pb2_not_available")
            return []
            
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(pb_data)
        
        entities = []
        
        for entity in feed.entity:
            # Process vehicle positions
            if entity.HasField('vehicle'):
                vehicle = entity.vehicle
                
                if not vehicle.HasField('position'):
                    continue
                    
                position = vehicle.position
                
                normalized = NormalizedVehiclePosition(
                    vehicle_id=vehicle.vehicle.id if vehicle.HasField('vehicle') else entity.id,
                    trip_id=vehicle.trip.trip_id if vehicle.HasField('trip') else None,
                    route_id=vehicle.trip.route_id if vehicle.HasField('trip') else None,
                    latitude=position.latitude,
                    longitude=position.longitude,
                    bearing=position.bearing if position.HasField('bearing') else None,
                    speed=position.speed if position.HasField('speed') else None,
                    timestamp=vehicle.timestamp if vehicle.HasField('timestamp') else int(datetime.now().timestamp()),
                    current_stop_sequence=vehicle.current_stop_sequence if vehicle.HasField('current_stop_sequence') else None,
                    current_status=self._status_to_string(vehicle.current_status) if vehicle.HasField('current_status') else None,
                    congestion_level=self._congestion_to_string(vehicle.congestion_level) if vehicle.HasField('congestion_level') else None,
                    occupancy_status=self._occupancy_to_string(vehicle.occupancy_status) if vehicle.HasField('occupancy_status') else None,
                )
                
                entities.append({
                    'type': 'vehicle_position',
                    'source_url': resource.url,
                    'organization': resource.organization,
                    'data': asdict(normalized)
                })
                
            # Process trip updates
            elif entity.HasField('trip_update'):
                trip_update = entity.trip_update
                
                if not trip_update.HasField('trip'):
                    continue
                    
                trip = trip_update.trip
                
                stop_time_updates = []
                for stu in trip_update.stop_time_update:
                    stop_time_updates.append({
                        'stop_sequence': stu.stop_sequence if stu.HasField('stop_sequence') else None,
                        'stop_id': stu.stop_id if stu.HasField('stop_id') else None,
                        'arrival_delay': stu.arrival.delay if stu.HasField('arrival') and stu.arrival.HasField('delay') else None,
                        'arrival_time': stu.arrival.time if stu.HasField('arrival') and stu.arrival.HasField('time') else None,
                        'departure_delay': stu.departure.delay if stu.HasField('departure') and stu.departure.HasField('delay') else None,
                        'departure_time': stu.departure.time if stu.HasField('departure') and stu.departure.HasField('time') else None,
                    })
                    
                normalized = NormalizedTripUpdate(
                    trip_id=trip.trip_id,
                    route_id=trip.route_id if trip.HasField('route_id') else None,
                    vehicle_id=trip_update.vehicle.id if trip_update.HasField('vehicle') else None,
                    start_date=trip.start_date if trip.HasField('start_date') else None,
                    start_time=trip.start_time if trip.HasField('start_time') else None,
                    schedule_relationship=self._schedule_to_string(trip.schedule_relationship) if trip.HasField('schedule_relationship') else None,
                    stop_time_updates=stop_time_updates,
                    timestamp=trip_update.timestamp if trip_update.HasField('timestamp') else int(datetime.now().timestamp()),
                )
                
                entities.append({
                    'type': 'trip_update',
                    'source_url': resource.url,
                    'organization': resource.organization,
                    'data': asdict(normalized)
                })
                
        return entities
        
    async def _publish_to_kafka(
        self,
        entities: List[Dict[str, Any]],
        resource: GTFSRTResource
    ) -> None:
        """
        Publish normalized entities to Kafka.
        
        Args:
            entities: List of normalized entity dictionaries.
            resource: Source resource metadata.
        """
        if not self._producer:
            return
            
        for entity in entities:
            try:
                # Add metadata
                message = {
                    'harvested_at': datetime.now().isoformat(),
                    'resource_type': resource.resource_type,
                    **entity
                }
                
                await self._producer.send_and_wait(
                    self._kafka_topic,
                    value=message
                )
                
            except KafkaError as e:
                logger.error(
                    "harvester_kafka_publish_error",
                    error=str(e),
                    entity_type=entity.get('type')
                )
            except Exception as e:
                logger.error(
                    "harvester_publish_error",
                    error=str(e),
                    entity_type=entity.get('type')
                )
                
    @staticmethod
    def _status_to_string(status: int) -> str:
        """Convert vehicle status enum to string."""
        statuses = {
            0: "INCOMING_AT",
            1: "STOPPED_AT",
            2: "IN_TRANSIT_TO"
        }
        return statuses.get(status, "UNKNOWN")
        
    @staticmethod
    def _congestion_to_string(level: int) -> str:
        """Convert congestion level enum to string."""
        levels = {
            0: "UNKNOWN_CONGESTION_LEVEL",
            1: "RUNNING_SMOOTHLY",
            2: "STOP_AND_GO",
            3: "CONGESTION",
            4: "SEVERE_CONGESTION"
        }
        return levels.get(level, "UNKNOWN")
        
    @staticmethod
    def _occupancy_to_string(occupancy: int) -> str:
        """Convert occupancy status enum to string."""
        statuses = {
            0: "EMPTY",
            1: "MANY_SEATS_AVAILABLE",
            2: "FEW_SEATS_AVAILABLE",
            3: "STANDING_ROOM_ONLY",
            4: "CRUSHED_STANDING_ROOM_ONLY",
            5: "FULL",
            6: "NOT_ACCEPTING_PASSENGERS"
        }
        return statuses.get(occupancy, "UNKNOWN")
        
    @staticmethod
    def _schedule_to_string(relationship: int) -> str:
        """Convert schedule relationship enum to string."""
        relationships = {
            0: "SCHEDULED",
            1: "ADDED",
            2: "UNSCHEDULED",
            3: "CANCELED"
        }
        return relationships.get(relationship, "UNKNOWN")
        
    async def harvest_continuously(
        self,
        resources: List[GTFSRTResource],
        interval: float = 30.0
    ) -> None:
        """
        Continuously harvest from multiple resources.
        
        Args:
            resources: List of GTFS-RT resources to harvest.
            interval: Seconds between harvest cycles.
        """
        logger.info(
            "harvester_continuous_started",
            resources_count=len(resources),
            interval=interval
        )
        
        while self._active:
            # Harvest all resources concurrently
            tasks = [
                self.harvest_resource(resource)
                for resource in resources
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log summary
            successful = sum(1 for r in results if isinstance(r, HarvestMetrics) and r.status == FeedStatus.ACTIVE)
            logger.info(
                "harvester_cycle_completed",
                total=len(resources),
                successful=successful,
                failed=len(resources) - successful
            )
            
            # Wait before next cycle
            await asyncio.sleep(interval)


async def main() -> None:
    """Demo/test function for the Harvester."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy resource for testing
    test_resource = GTFSRTResource(
        url="https://example.com/gtfs-rt",  # Replace with a real URL for testing
        dataset_id="test",
        organization="Test Operator",
        resource_type="vehicle-positions",
        title="Test Feed"
    )
    
    async with GTFSRTHarvester(kafka_bootstrap_servers="localhost:9092") as harvester:
        metrics = await harvester.harvest_resource(test_resource)
        
        print(f"\n{'='*80}")
        print(f"HARVESTER - TEST REPORT")
        print(f"{'='*80}\n")
        print(f"URL: {metrics.url}")
        print(f"Status: {metrics.status.value}")
        print(f"Entities: {metrics.entities_count}")
        print(f"Response Time: {metrics.response_time_ms:.2f}ms")
        if metrics.error_message:
            print(f"Error: {metrics.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
