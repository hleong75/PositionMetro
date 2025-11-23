#!/usr/bin/env python3
"""
Panoptique Ferroviaire - Main Entry Point
==========================================
HNPS v5.0 (Hybrid Neuro-Physical System)

This is the main orchestrator that brings together:
- Omniscience Engine (Auto-discovery)
- Harvester (Data ingestion)
- Fusion Engine (Physics simulation & state estimation)

Author: Supreme Architecture Team
Version: 5.0.0
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

import structlog
import yaml

# Enable uvloop if available for better performance
try:
    import uvloop
    if os.getenv('UVLOOP_ENABLED', 'true').lower() == 'true':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("✓ uvloop enabled for high-performance async operations")
except ImportError:
    print("⚠ uvloop not available, using standard asyncio")

from src.core.omniscience import OmniscienceEngine
from src.ingestion.harvester import GTFSRTHarvester
from src.engine.fusion import HybridFusionEngine


class PanoptiqueFerroviaire:
    """
    Main orchestrator for the Panoptique Ferroviaire system.
    
    This class coordinates all subsystems and manages the lifecycle of:
    - Auto-discovery of GTFS-RT feeds
    - Continuous data harvesting
    - Real-time fusion and simulation
    """
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the Panoptique Ferroviaire system.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        self.logger = structlog.get_logger(__name__)
        self.logger.info(
            "panoptique_initializing",
            version=self.config['application']['version']
        )
        
        # Initialize subsystems
        self.omniscience: Optional[OmniscienceEngine] = None
        self.harvester: Optional[GTFSRTHarvester] = None
        self.fusion_engine: Optional[HybridFusionEngine] = None
        
        self._running = False
        self._tasks = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file with environment variable substitution."""
        if not os.path.exists(config_path):
            print(f"⚠ Config file not found: {config_path}, using defaults")
            return self._default_config()
            
        with open(config_path, 'r') as f:
            config_str = f.read()
            
        # Simple environment variable substitution
        import re
        def replace_env(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default)
            
        config_str = re.sub(r'\$\{(\w+)(?::([^}]*))?\}', replace_env, config_str)
        
        return yaml.safe_load(config_str)
        
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'application': {
                'name': 'Panoptique Ferroviaire',
                'version': '5.0.0'
            },
            'kafka': {
                'bootstrap_servers': 'localhost:9092',
                'topic_raw_telemetry': 'raw_telemetry',
                'consumer_group_id': 'neural_engine'
            },
            'harvester': {
                'harvest_interval': 30.0,
                'max_concurrent_harvests': 20
            },
            'fusion': {
                'simulation_rate': 1.0
            },
            'logging': {
                'level': 'INFO'
            },
            'features': {
                'auto_discovery': True,
                'continuous_harvesting': True,
                'physics_simulation': True
            }
        }
        
    def _setup_logging(self) -> None:
        """Setup structured logging."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level)
        )
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
    async def start(self) -> None:
        """Start the Panoptique Ferroviaire system."""
        self.logger.info("panoptique_starting")
        
        print("\n" + "="*80)
        print("🚂 PANOPTIQUE FERROVIAIRE - HNPS v5.0")
        print("   Hybrid Neuro-Physical System for Railway Surveillance")
        print("="*80 + "\n")
        
        self._running = True
        
        # Phase 1: Auto-Discovery
        if self.config['features']['auto_discovery']:
            await self._run_discovery()
        
        # Phase 2: Start Fusion Engine
        if self.config['features']['physics_simulation']:
            await self._start_fusion_engine()
        
        # Phase 3: Start Continuous Harvesting
        if self.config['features']['continuous_harvesting']:
            await self._start_harvesting()
        
        self.logger.info("panoptique_started")
        print("\n✓ All systems operational\n")
        
    async def _run_discovery(self) -> None:
        """Run auto-discovery phase."""
        print("🔍 Phase 1: Auto-Discovery")
        print("   Scanning French transport data platform...")
        
        self.omniscience = OmniscienceEngine(
            max_concurrent_requests=self.config.get('omniscience', {}).get(
                'max_concurrent_requests', 10
            )
        )
        
        async with self.omniscience:
            operators = await self.omniscience.discover_all_feeds()
            
            total_resources = len(self.omniscience.get_all_resources())
            
            print(f"   ✓ Discovered {len(operators)} operators")
            print(f"   ✓ Found {total_resources} GTFS-RT resources")
            
            if total_resources > 0:
                print("\n   Top operators:")
                for i, operator in enumerate(list(operators.values())[:5], 1):
                    if operator.resources:
                        print(f"   {i}. {operator.name}: {len(operator.resources)} feeds")
                        
    async def _start_fusion_engine(self) -> None:
        """Start the fusion engine."""
        print("\n🧠 Phase 2: Fusion Engine")
        print("   Initializing hybrid neuro-physics system...")
        
        kafka_config = self.config.get('kafka', {})
        
        self.fusion_engine = HybridFusionEngine(
            kafka_bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            kafka_topic=kafka_config.get('topic_raw_telemetry', 'raw_telemetry'),
            kafka_group_id=kafka_config.get('consumer_group_id', 'neural_engine')
        )
        
        await self.fusion_engine.start()
        
        print("   ✓ Fusion engine started")
        print("   ✓ Physics simulation active")
        print("   ✓ Kalman filters initialized")
        
    async def _start_harvesting(self) -> None:
        """Start continuous harvesting."""
        print("\n📡 Phase 3: Data Harvesting")
        print("   Starting continuous feed monitoring...")
        
        if not self.omniscience:
            print("   ⚠ No discovery data available, skipping harvesting")
            return
            
        resources = self.omniscience.get_all_resources()
        
        if not resources:
            print("   ⚠ No resources discovered, skipping harvesting")
            return
            
        kafka_config = self.config.get('kafka', {})
        harvester_config = self.config.get('harvester', {})
        
        self.harvester = GTFSRTHarvester(
            kafka_bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            kafka_topic=kafka_config.get('topic_raw_telemetry', 'raw_telemetry')
        )
        
        await self.harvester.start()
        
        print(f"   ✓ Harvester initialized for {len(resources)} feeds")
        print(f"   ✓ Harvest interval: {harvester_config.get('harvest_interval', 30)}s")
        
        # Start harvesting task
        harvest_interval = harvester_config.get('harvest_interval', 30.0)
        task = asyncio.create_task(
            self.harvester.harvest_continuously(resources, harvest_interval)
        )
        self._tasks.append(task)
        
    async def stop(self) -> None:
        """Stop the Panoptique Ferroviaire system."""
        self.logger.info("panoptique_stopping")
        print("\n🛑 Stopping Panoptique Ferroviaire...")
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Stop subsystems
        if self.harvester:
            await self.harvester.stop()
            
        if self.fusion_engine:
            await self.fusion_engine.stop()
            
        self.logger.info("panoptique_stopped")
        print("✓ System stopped gracefully\n")
        
    async def run(self) -> None:
        """Run the main event loop."""
        try:
            await self.start()
            
            # Run until interrupted
            print("📊 System Status:")
            print("   Press Ctrl+C to stop\n")
            
            while self._running:
                await asyncio.sleep(10)
                
                # Print periodic status
                if self.fusion_engine:
                    trains = self.fusion_engine.get_all_trains()
                    print(f"   Active trains: {len(trains)}", end="\r")
                    
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\n\n⚠ Keyboard interrupt received")
        except Exception as e:
            self.logger.error("panoptique_error", error=str(e))
            print(f"\n❌ Error: {e}")
        finally:
            await self.stop()


def setup_signal_handlers(panoptique: PanoptiqueFerroviaire) -> None:
    """Setup signal handlers for graceful shutdown."""
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        print(f"\n⚠ Received signal {sig}")
        loop.create_task(panoptique.stop())
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))


async def main() -> None:
    """Main entry point."""
    # Print banner
    print("\n" + "="*80)
    print("🚂 PANOPTIQUE FERROVIAIRE - HNPS v5.0")
    print("   Hybrid Neuro-Physical System for Railway Surveillance")
    print("   Supreme Architecture - État de l'Art")
    print("="*80 + "\n")
    
    # Initialize system
    panoptique = PanoptiqueFerroviaire()
    
    # Setup signal handlers
    if sys.platform != "win32":
        setup_signal_handlers(panoptique)
    
    # Run
    await panoptique.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✓ Shutdown complete")
        sys.exit(0)
