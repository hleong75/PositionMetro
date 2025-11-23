"""
Tests for the Omniscience Engine (Auto-Discovery)
"""

import pytest
from src.core.omniscience import (
    OmniscienceEngine,
    GTFSRTResource,
    TransportOperator
)


def test_gtfs_rt_resource_creation():
    """Test creation of GTFSRTResource."""
    resource = GTFSRTResource(
        url="https://example.com/gtfs-rt",
        dataset_id="test_dataset",
        organization="Test Operator",
        resource_type="vehicle-positions",
        title="Test Feed"
    )
    
    assert resource.url == "https://example.com/gtfs-rt"
    assert resource.dataset_id == "test_dataset"
    assert resource.organization == "Test Operator"
    assert resource.resource_type == "vehicle-positions"
    assert resource.is_active is True


def test_transport_operator_creation():
    """Test creation of TransportOperator."""
    operator = TransportOperator(
        name="SNCF",
        organization_id="sncf"
    )
    
    assert operator.name == "SNCF"
    assert operator.organization_id == "sncf"
    assert len(operator.resources) == 0


def test_transport_operator_add_resource():
    """Test adding resources to an operator."""
    operator = TransportOperator(
        name="RATP",
        organization_id="ratp"
    )
    
    resource1 = GTFSRTResource(
        url="https://example.com/feed1",
        dataset_id="dataset1",
        organization="RATP",
        resource_type="vehicle-positions",
        title="Feed 1"
    )
    
    resource2 = GTFSRTResource(
        url="https://example.com/feed2",
        dataset_id="dataset2",
        organization="RATP",
        resource_type="trip-updates",
        title="Feed 2"
    )
    
    operator.add_resource(resource1)
    operator.add_resource(resource2)
    
    assert len(operator.resources) == 2
    assert resource1 in operator.resources
    assert resource2 in operator.resources
    
    # Adding same resource again should not duplicate
    operator.add_resource(resource1)
    assert len(operator.resources) == 2


@pytest.mark.asyncio
async def test_omniscience_engine_initialization():
    """Test Omniscience Engine initialization."""
    async with OmniscienceEngine() as engine:
        assert engine is not None
        assert len(engine.get_all_operators()) == 0
        assert len(engine.get_all_resources()) == 0
