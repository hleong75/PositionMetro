"""
Module A: Omniscience (Auto-Discovery Engine)
===============================================
This module implements the auto-discovery system for GTFS-RT data sources.
It queries the transport.data.gouv.fr API to discover all available real-time
transit feeds dynamically without hardcoded URLs.

HNPS v5.0 Component: Data Source Discovery & Registry Management
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GTFSRTResource:
    """Represents a GTFS-RT data source discovered from the API."""
    
    url: str
    dataset_id: str
    organization: str
    resource_type: str  # 'trip-updates' or 'vehicle-positions'
    title: str
    last_updated: Optional[datetime] = None
    is_active: bool = True
    
    def __hash__(self) -> int:
        return hash(self.url)


@dataclass
class TransportOperator:
    """Represents a transport operator (SNCF, RATP, Keolis, etc.)."""
    
    name: str
    organization_id: str
    resources: List[GTFSRTResource] = field(default_factory=list)
    
    def add_resource(self, resource: GTFSRTResource) -> None:
        """Add a GTFS-RT resource to this operator."""
        if resource not in self.resources:
            self.resources.append(resource)


class OmniscienceEngine:
    """
    The Omniscience Engine discovers all available GTFS-RT feeds dynamically.
    
    This system queries the French national transport data platform and builds
    a comprehensive registry of all active real-time transit feeds. It operates
    recursively across all pages of results and maintains a dynamic catalog.
    """
    
    BASE_API_URL = "https://transport.data.gouv.fr/api/datasets"
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        max_concurrent_requests: int = 10
    ) -> None:
        """
        Initialize the Omniscience Engine.
        
        Args:
            session: Optional aiohttp session. If None, one will be created.
            max_concurrent_requests: Maximum number of concurrent API requests.
        """
        self._session = session
        self._own_session = session is None
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._operators: Dict[str, TransportOperator] = {}
        self._discovered_urls: Set[str] = set()
        
    async def __aenter__(self) -> "OmniscienceEngine":
        """Async context manager entry."""
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._own_session and self._session:
            await self._session.close()
            
    async def discover_all_feeds(self) -> Dict[str, TransportOperator]:
        """
        Main discovery method that recursively queries all pages of the API.
        
        Returns:
            Dictionary mapping organization IDs to TransportOperator objects.
        """
        logger.info("omniscience_discovery_started", base_url=self.BASE_API_URL)
        
        page = 1
        total_resources = 0
        
        while True:
            try:
                resources_found = await self._fetch_page(page)
                
                if resources_found == 0:
                    # No more results, we've reached the end
                    break
                    
                total_resources += resources_found
                logger.debug(
                    "omniscience_page_processed",
                    page=page,
                    resources_on_page=resources_found,
                    total_resources=total_resources
                )
                
                page += 1
                
            except Exception as e:
                logger.error(
                    "omniscience_page_error",
                    page=page,
                    error=str(e)
                )
                # Continue to next page even if one fails
                page += 1
                continue
                
        logger.info(
            "omniscience_discovery_completed",
            total_operators=len(self._operators),
            total_resources=total_resources,
            unique_urls=len(self._discovered_urls)
        )
        
        return self._operators
        
    async def _fetch_page(self, page: int) -> int:
        """
        Fetch and process a single page of datasets.
        
        Args:
            page: Page number to fetch.
            
        Returns:
            Number of GTFS-RT resources found on this page.
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        params = {
            "page": page,
            "type": "public-transit",  # Filter for public transit datasets
        }
        
        async with self._semaphore:
            try:
                async with self._session.get(
                    self.BASE_API_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            "omniscience_api_error",
                            status=response.status,
                            page=page
                        )
                        return 0
                        
                    data = await response.json()
                    
            except asyncio.TimeoutError:
                logger.warning("omniscience_timeout", page=page)
                return 0
            except Exception as e:
                logger.error("omniscience_fetch_error", page=page, error=str(e))
                return 0
                
        # Process the datasets from this page
        datasets = data if isinstance(data, list) else []
        resources_found = 0
        
        for dataset in datasets:
            resources_found += await self._process_dataset(dataset)
            
        return resources_found
        
    async def _process_dataset(self, dataset: Dict[str, Any]) -> int:
        """
        Process a single dataset and extract GTFS-RT resources.
        
        Args:
            dataset: Dataset dictionary from the API.
            
        Returns:
            Number of GTFS-RT resources found in this dataset.
        """
        resources_found = 0
        
        # Extract basic dataset info
        dataset_id = dataset.get("id", "unknown")
        title = dataset.get("title", "")
        organization = dataset.get("organization", {})
        org_name = organization.get("name", "Unknown")
        org_id = organization.get("id", org_name.lower().replace(" ", "_"))
        
        # Get or create operator
        if org_id not in self._operators:
            self._operators[org_id] = TransportOperator(
                name=org_name,
                organization_id=org_id
            )
            
        operator = self._operators[org_id]
        
        # Check for resources
        resources = dataset.get("resources", [])
        
        for resource in resources:
            # Check if this is a GTFS-RT resource
            resource_format = resource.get("format", "").upper()
            resource_title = resource.get("title", "").lower()
            url = resource.get("url", "")
            
            # Detect GTFS-RT resources
            is_gtfs_rt = (
                resource_format == "GTFS-RT" or
                "gtfs-rt" in resource_title or
                "gtfs_rt" in resource_title or
                "realtime" in resource_title or
                "real-time" in resource_title
            )
            
            if is_gtfs_rt and url and url not in self._discovered_urls:
                # Determine resource type
                resource_type = "vehicle-positions"
                if any(keyword in resource_title for keyword in ["trip", "update", "tripupdate"]):
                    resource_type = "trip-updates"
                elif any(keyword in resource_title for keyword in ["position", "vehicle"]):
                    resource_type = "vehicle-positions"
                    
                gtfs_resource = GTFSRTResource(
                    url=url,
                    dataset_id=dataset_id,
                    organization=org_name,
                    resource_type=resource_type,
                    title=resource.get("title", ""),
                    last_updated=None,
                    is_active=True
                )
                
                operator.add_resource(gtfs_resource)
                self._discovered_urls.add(url)
                resources_found += 1
                
                logger.debug(
                    "omniscience_resource_discovered",
                    url=url,
                    operator=org_name,
                    resource_type=resource_type
                )
                
        return resources_found
        
    def get_operator(self, org_id: str) -> Optional[TransportOperator]:
        """Get a specific operator by ID."""
        return self._operators.get(org_id)
        
    def get_all_operators(self) -> List[TransportOperator]:
        """Get all discovered operators."""
        return list(self._operators.values())
        
    def get_all_resources(self) -> List[GTFSRTResource]:
        """Get all discovered GTFS-RT resources across all operators."""
        resources = []
        for operator in self._operators.values():
            resources.extend(operator.resources)
        return resources


async def main() -> None:
    """Demo/test function for the Omniscience Engine."""
    logging.basicConfig(level=logging.INFO)
    
    async with OmniscienceEngine() as engine:
        operators = await engine.discover_all_feeds()
        
        print(f"\n{'='*80}")
        print(f"OMNISCIENCE ENGINE - DISCOVERY REPORT")
        print(f"{'='*80}\n")
        
        print(f"Total Operators Discovered: {len(operators)}")
        print(f"Total Resources: {len(engine.get_all_resources())}\n")
        
        for operator in operators.values():
            if operator.resources:
                print(f"\n{operator.name} ({operator.organization_id})")
                print(f"  Resources: {len(operator.resources)}")
                for resource in operator.resources[:3]:  # Show first 3
                    print(f"    - [{resource.resource_type}] {resource.title[:60]}")
                if len(operator.resources) > 3:
                    print(f"    ... and {len(operator.resources) - 3} more")


if __name__ == "__main__":
    asyncio.run(main())
