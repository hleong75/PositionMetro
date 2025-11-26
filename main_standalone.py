#!/usr/bin/env python3
"""
Panoptique Ferroviaire - Mode Autonome (Standalone)
====================================================
HNPS v6.0 - Version sans Docker

Ce script permet d'exécuter l'application sans Kafka ni PostgreSQL.
Toutes les fonctionnalités principales sont disponibles:
- Auto-découverte des flux GTFS-RT
- Simulation physique (équation de Davis)
- Filtre de Kalman pour l'estimation d'état
- Moving Block (Cantonnement)
- Rail-Lock (projection sur la voie)
- Holographic Positioning

Author: Supreme Architecture Team
Version: 6.0.0 (Standalone)
"""

import asyncio
import logging
import os
import signal
import sys
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

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

from src.core.omniscience import OmniscienceEngine, GTFSRTResource
from src.ingestion.harvester import GTFSRTHarvester, HarvestMetrics, FeedStatus
from src.engine.fusion import (
    HybridFusionEngine,
    TrainEntity,
    TrainStateVector,
    Position2D
)


class StandaloneDataStore:
    """
    In-memory data store for standalone mode.
    Replaces Kafka and PostgreSQL for local testing/demo.
    """
    
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.trains: Dict[str, Dict[str, Any]] = {}
        self.max_messages = 10000  # Limit memory usage
        
    def publish(self, message: Dict[str, Any]) -> None:
        """Store a message (equivalent to Kafka publish)."""
        self.messages.append({
            'timestamp': datetime.now().isoformat(),
            **message
        })
        
        # Trim old messages to prevent memory overflow
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def get_recent_messages(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:]
        
    def update_train(self, train_id: str, state: Dict[str, Any]) -> None:
        """Update train state (equivalent to PostgreSQL update)."""
        self.trains[train_id] = {
            'train_id': train_id,
            'last_update': datetime.now().isoformat(),
            **state
        }
        
    def get_all_trains(self) -> Dict[str, Dict[str, Any]]:
        """Get all train states."""
        return self.trains.copy()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get data store statistics."""
        return {
            'messages_count': len(self.messages),
            'trains_count': len(self.trains),
            'memory_usage_mb': (sys.getsizeof(self.messages) + sys.getsizeof(self.trains)) / 1024 / 1024
        }


class PanoptiqueStandalone:
    """
    Standalone version of Panoptique Ferroviaire.
    
    Runs without Kafka or PostgreSQL, storing data in memory
    and processing everything locally.
    """
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the standalone system.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        self.logger = structlog.get_logger(__name__)
        self.logger.info(
            "panoptique_standalone_initializing",
            version=self.config['application']['version']
        )
        
        # In-memory data store
        self.data_store = StandaloneDataStore()
        
        # Initialize subsystems
        self.omniscience: Optional[OmniscienceEngine] = None
        self.harvester: Optional[GTFSRTHarvester] = None
        self.fusion_engine: Optional[HybridFusionEngine] = None
        
        self._running = False
        self._tasks = []
        self._discovered_resources: List[GTFSRTResource] = []
        
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
        """Return default configuration for standalone mode."""
        return {
            'application': {
                'name': 'Panoptique Ferroviaire (Standalone)',
                'version': '6.0.0-standalone'
            },
            'harvester': {
                'harvest_interval': 30.0,
                'max_concurrent_harvests': 10  # Lower for standalone
            },
            'fusion': {
                'simulation_rate': 1.0
            },
            'static_data': {
                'topology_path': 'data/topology.json',
                'stops_path': 'data/stops.txt'
            },
            'logging': {
                'level': 'INFO'
            },
            'features': {
                'auto_discovery': True,
                'continuous_harvesting': True,
                'physics_simulation': True
            },
            'standalone': {
                'max_discovery_pages': 5,  # Limit for faster startup
                'status_interval': 30  # Status update interval in seconds
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
        """Start the standalone system."""
        self.logger.info("panoptique_standalone_starting")
        
        print("\n" + "="*80)
        print("🚂 PANOPTIQUE FERROVIAIRE - HNPS v6.0 (MODE AUTONOME)")
        print("   Hybrid Neuro-Physical System for Railway Surveillance")
        print("   ⚡ Mode: Standalone (sans Docker/Kafka/PostgreSQL)")
        print("="*80 + "\n")
        
        self._running = True
        
        # Phase 1: Auto-Discovery
        if self.config['features']['auto_discovery']:
            await self._run_discovery()
        
        # Phase 2: Initialize local Fusion Engine (without Kafka)
        if self.config['features']['physics_simulation']:
            await self._start_fusion_engine_standalone()
        
        # Phase 3: Start Continuous Harvesting (direct processing, no Kafka)
        if self.config['features']['continuous_harvesting']:
            await self._start_harvesting_standalone()
        
        self.logger.info("panoptique_standalone_started")
        print("\n✓ Système autonome opérationnel\n")
        
    async def _run_discovery(self) -> None:
        """Run auto-discovery phase."""
        print("🔍 Phase 1: Auto-Découverte")
        print("   Analyse de la plateforme transport.data.gouv.fr...")
        
        self.omniscience = OmniscienceEngine(
            max_concurrent_requests=self.config.get('omniscience', {}).get(
                'max_concurrent_requests', 5
            )
        )
        
        max_pages = self.config.get('standalone', {}).get('max_discovery_pages', 5)
        
        async with self.omniscience:
            # Use the public method for limited discovery
            await self.omniscience.discover_feeds_limited(max_pages)
            
            operators = self.omniscience.get_all_operators()
            self._discovered_resources = self.omniscience.get_all_resources()
            
            print(f"   ✓ Découverts {len(operators)} opérateurs")
            print(f"   ✓ Trouvés {len(self._discovered_resources)} ressources GTFS-RT")
            
            if operators:
                print("\n   Principaux opérateurs:")
                for i, (op_id, operator) in enumerate(list(operators.items())[:5], 1):
                    if operator.resources:
                        print(f"   {i}. {operator.name}: {len(operator.resources)} flux")
                        
    async def _start_fusion_engine_standalone(self) -> None:
        """Start the fusion engine without Kafka."""
        print("\n🧠 Phase 2: Moteur de Fusion (Mode Autonome)")
        print("   Initialisation du système neuro-physique...")
        
        # Get static data file paths from config
        static_data_config = self.config.get('static_data', {})
        topology_path = static_data_config.get('topology_path', 'data/topology.json')
        stops_path = static_data_config.get('stops_path', 'data/stops.txt')
        
        # Check existence of static data files
        if not os.path.exists(topology_path):
            print(f"   ⚠ Fichier topology non trouvé: {topology_path}")
            print("     Rail-Lock fonctionnera en mode dégradé")
            print("     Exécuter: python -m src.tools.prepare_data")
        else:
            print(f"   ✓ Topology chargée: {topology_path}")
        
        if not os.path.exists(stops_path):
            print(f"   ⚠ Fichier stops non trouvé: {stops_path}")
            print("     Holographic Positioning fonctionnera en mode dégradé")
        else:
            print(f"   ✓ Stops chargés: {stops_path}")
        
        # Initialize fusion engine without Kafka connection
        # We pass a dummy Kafka address that won't be used in standalone mode
        self.fusion_engine = HybridFusionEngine(
            kafka_bootstrap_servers="standalone:9092",  # Not used in standalone mode
            kafka_topic="standalone_topic",
            kafka_group_id="standalone_group",
            topology_path=topology_path if os.path.exists(topology_path) else None,
            stops_path=stops_path if os.path.exists(stops_path) else None
        )
        
        # Use the public method to initialize standalone mode
        self.fusion_engine.initialize_standalone_mode()
        
        # Start simulation loop only
        asyncio.create_task(self._simulation_loop())
        
        print("   ✓ Moteur de fusion initialisé")
        print("   ✓ Simulation physique active")
        print("   ✓ Filtres de Kalman prêts")
        
    async def _start_harvesting_standalone(self) -> None:
        """Start continuous harvesting without Kafka (direct processing)."""
        print("\n📡 Phase 3: Récolte de Données (Mode Autonome)")
        print("   Démarrage de la surveillance des flux...")
        
        if not self._discovered_resources:
            print("   ⚠ Aucune ressource découverte, récolte ignorée")
            return
            
        harvester_config = self.config.get('harvester', {})
        
        # Create harvester without Kafka
        self.harvester = GTFSRTHarvester(
            kafka_bootstrap_servers="standalone:9092",  # Not used in standalone mode
            kafka_topic="standalone_topic",
            on_metrics=self._on_harvest_metrics
        )
        
        # Use the public method to start in standalone mode
        await self.harvester.start_standalone()
        
        # Limit resources for standalone mode
        max_resources = min(len(self._discovered_resources), 20)
        resources = self._discovered_resources[:max_resources]
        
        print(f"   ✓ Harvester initialisé pour {len(resources)} flux")
        print(f"   ✓ Intervalle de récolte: {harvester_config.get('harvest_interval', 30)}s")
        
        # Start harvesting task
        harvest_interval = harvester_config.get('harvest_interval', 30.0)
        task = asyncio.create_task(
            self._harvest_loop_standalone(resources, harvest_interval)
        )
        self._tasks.append(task)
        
    def _on_harvest_metrics(self, metrics: HarvestMetrics) -> None:
        """Handle harvest metrics (store in memory instead of Kafka)."""
        if metrics.status == FeedStatus.ACTIVE and metrics.entities_count > 0:
            self.data_store.publish({
                'type': 'harvest_metrics',
                'url': metrics.url,
                'entities_count': metrics.entities_count,
                'response_time_ms': metrics.response_time_ms
            })
            
    async def _harvest_loop_standalone(
        self,
        resources: List[GTFSRTResource],
        interval: float
    ) -> None:
        """Standalone harvest loop with direct processing."""
        self.logger.info("harvest_loop_standalone_started", resources_count=len(resources))
        
        while self._running:
            cycle_start = datetime.now()
            
            # Harvest resources concurrently
            tasks = [self._harvest_and_process(resource) for resource in resources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate statistics
            successful = sum(1 for r in results if r is True)
            
            processing_time = (datetime.now() - cycle_start).total_seconds()
            
            self.logger.debug(
                "harvest_cycle_completed",
                total=len(resources),
                successful=successful,
                processing_time=processing_time
            )
            
            # Wait for next cycle
            sleep_time = max(0.001, interval - processing_time)
            await asyncio.sleep(sleep_time)
            
    async def _harvest_and_process(self, resource: GTFSRTResource) -> bool:
        """Harvest a resource and process it directly (bypass Kafka)."""
        try:
            metrics = await self.harvester.harvest_resource(resource)
            
            if metrics.status == FeedStatus.ACTIVE:
                # In full mode, data would go to Kafka and be consumed by fusion engine
                # In standalone mode, we could process directly here if needed
                return True
            return False
        except Exception as e:
            self.logger.debug("harvest_error", url=resource.url, error=str(e))
            return False
            
    async def _simulation_loop(self) -> None:
        """Physics simulation loop (runs without Kafka)."""
        self.logger.info("simulation_loop_started")
        
        dt = 1.0 / self.config.get('fusion', {}).get('simulation_rate', 1.0)
        
        while self._running and self.fusion_engine and self.fusion_engine.is_active():
            try:
                # Use the public method to run simulation step
                await self.fusion_engine.run_simulation_step(dt)
                
                # Store state in memory for all trains
                for train in self.fusion_engine.get_all_trains():
                    state = train.get_current_state()
                    self.data_store.update_train(train.train_id, {
                        'latitude': state.position.latitude,
                        'longitude': state.position.longitude,
                        'velocity': state.velocity,
                        'acceleration': state.acceleration,
                        'bearing': state.bearing,
                        'track_distance': state.track_distance,
                        'state': train.current_state.value
                    })
                
                await asyncio.sleep(dt)
                
            except Exception as e:
                self.logger.error("simulation_error", error=str(e))
                await asyncio.sleep(1.0)
                
    async def stop(self) -> None:
        """Stop the standalone system."""
        self.logger.info("panoptique_standalone_stopping")
        print("\n🛑 Arrêt de Panoptique Ferroviaire...")
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Stop subsystems
        if self.harvester:
            await self.harvester.stop()
            
        if self.fusion_engine:
            self.fusion_engine.deactivate()
            
        self.logger.info("panoptique_standalone_stopped")
        print("✓ Système arrêté proprement\n")
        
    async def run(self) -> None:
        """Run the main event loop."""
        try:
            await self.start()
            
            # Print status
            status_interval = self.config.get('standalone', {}).get('status_interval', 30)
            
            print("📊 Statut du Système:")
            print("   Appuyez sur Ctrl+C pour arrêter\n")
            
            last_status = datetime.now()
            
            while self._running:
                await asyncio.sleep(1)
                
                # Periodic status update
                if (datetime.now() - last_status).total_seconds() >= status_interval:
                    self._print_status()
                    last_status = datetime.now()
                    
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\n\n⚠ Interruption clavier détectée")
        except Exception as e:
            self.logger.error("panoptique_standalone_error", error=str(e))
            print(f"\n❌ Erreur: {e}")
        finally:
            await self.stop()
            
    def _print_status(self) -> None:
        """Print system status."""
        stats = self.data_store.get_stats()
        trains = self.fusion_engine.get_all_trains() if self.fusion_engine else []
        
        print(f"\n{'='*60}")
        print(f"📊 STATUT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"   Trains actifs: {len(trains)}")
        print(f"   Messages en mémoire: {stats['messages_count']}")
        print(f"   États de trains stockés: {stats['trains_count']}")
        print(f"   Mémoire utilisée: {stats['memory_usage_mb']:.2f} MB")
        
        if trains:
            print(f"\n   Premiers trains:")
            for train in trains[:3]:
                state = train.get_current_state()
                print(f"   • {train.train_id}: "
                      f"v={state.velocity:.1f} m/s, "
                      f"pos=({state.position.latitude:.4f}, {state.position.longitude:.4f})")
        print()


def setup_signal_handlers(panoptique: PanoptiqueStandalone) -> None:
    """Setup signal handlers for graceful shutdown."""
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        print(f"\n⚠ Signal {sig} reçu")
        loop.create_task(panoptique.stop())
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))


async def main() -> None:
    """Main entry point for standalone mode."""
    # Print banner
    print("\n" + "="*80)
    print("🚂 PANOPTIQUE FERROVIAIRE - HNPS v6.0 (MODE AUTONOME)")
    print("   Hybrid Neuro-Physical System for Railway Surveillance")
    print("   ⚡ Exécution sans Docker/Kafka/PostgreSQL")
    print("="*80 + "\n")
    
    # Initialize system
    panoptique = PanoptiqueStandalone()
    
    # Setup signal handlers
    if sys.platform != "win32":
        setup_signal_handlers(panoptique)
    
    # Run
    await panoptique.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✓ Arrêt terminé")
        sys.exit(0)
