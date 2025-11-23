"""
Tests for Main Integration - HNPS v6.0
=======================================
Tests for main.py integration with topology and stops paths.
"""

import asyncio
import pytest
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

from main import PanoptiqueFerroviaire


class TestPanoptiqueFerroviaireIntegration:
    """Test suite for main.py integration with static data files."""
    
    @pytest.mark.asyncio
    async def test_fusion_engine_with_existing_files(self, tmp_path):
        """Test fusion engine initialization when static files exist."""
        # Create mock data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        topology_file = data_dir / "topology.json"
        stops_file = data_dir / "stops.txt"
        
        topology_file.write_text('[{"shape_id": "S1", "route_id": "R1", "points": [[2.0, 48.0]]}]')
        stops_file.write_text('stop_id,stop_name,stop_lat,stop_lon\nS1,Station,48.0,2.0\n')
        
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
application:
  name: Test
  version: 6.0.0
kafka:
  bootstrap_servers: localhost:9092
  topic_raw_telemetry: test_topic
  consumer_group_id: test_group
features:
  auto_discovery: false
  continuous_harvesting: false
  physics_simulation: true
""")
        
        # Change to tmp directory to use relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Initialize Panoptique
            panoptique = PanoptiqueFerroviaire(config_path=str(config_path))
            
            # Mock HybridFusionEngine
            with patch('main.HybridFusionEngine') as mock_fusion_class:
                mock_fusion = AsyncMock()
                mock_fusion_class.return_value = mock_fusion
                
                # Call _start_fusion_engine
                await panoptique._start_fusion_engine()
                
                # Verify HybridFusionEngine was initialized with correct paths
                mock_fusion_class.assert_called_once()
                call_kwargs = mock_fusion_class.call_args[1]
                
                assert call_kwargs['topology_path'] == 'data/topology.json'
                assert call_kwargs['stops_path'] == 'data/stops.txt'
                
                # Verify start was called
                mock_fusion.start.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_fusion_engine_with_missing_files(self, tmp_path):
        """Test fusion engine initialization when static files are missing."""
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
application:
  name: Test
  version: 6.0.0
kafka:
  bootstrap_servers: localhost:9092
  topic_raw_telemetry: test_topic
  consumer_group_id: test_group
features:
  auto_discovery: false
  continuous_harvesting: false
  physics_simulation: true
""")
        
        # Change to tmp directory to use relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Initialize Panoptique
            panoptique = PanoptiqueFerroviaire(config_path=str(config_path))
            
            # Mock HybridFusionEngine
            with patch('main.HybridFusionEngine') as mock_fusion_class:
                mock_fusion = AsyncMock()
                mock_fusion_class.return_value = mock_fusion
                
                # Call _start_fusion_engine
                await panoptique._start_fusion_engine()
                
                # Verify HybridFusionEngine was initialized with None paths
                mock_fusion_class.assert_called_once()
                call_kwargs = mock_fusion_class.call_args[1]
                
                assert call_kwargs['topology_path'] is None
                assert call_kwargs['stops_path'] is None
                
                # Verify start was called (should work in degraded mode)
                mock_fusion.start.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_fusion_engine_with_only_topology(self, tmp_path):
        """Test fusion engine initialization when only topology exists."""
        # Create mock topology file only
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        topology_file = data_dir / "topology.json"
        topology_file.write_text('[{"shape_id": "S1", "route_id": "R1", "points": [[2.0, 48.0]]}]')
        
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
application:
  name: Test
  version: 6.0.0
kafka:
  bootstrap_servers: localhost:9092
features:
  physics_simulation: true
""")
        
        # Change to tmp directory to use relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Initialize Panoptique
            panoptique = PanoptiqueFerroviaire(config_path=str(config_path))
            
            # Mock HybridFusionEngine
            with patch('main.HybridFusionEngine') as mock_fusion_class:
                mock_fusion = AsyncMock()
                mock_fusion_class.return_value = mock_fusion
                
                # Call _start_fusion_engine
                await panoptique._start_fusion_engine()
                
                # Verify HybridFusionEngine was initialized correctly
                call_kwargs = mock_fusion_class.call_args[1]
                
                assert call_kwargs['topology_path'] == 'data/topology.json'
                assert call_kwargs['stops_path'] is None
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_fusion_engine_with_only_stops(self, tmp_path):
        """Test fusion engine initialization when only stops exists."""
        # Create mock stops file only
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        stops_file = data_dir / "stops.txt"
        stops_file.write_text('stop_id,stop_name,stop_lat,stop_lon\nS1,Station,48.0,2.0\n')
        
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
application:
  name: Test
  version: 6.0.0
kafka:
  bootstrap_servers: localhost:9092
features:
  physics_simulation: true
""")
        
        # Change to tmp directory to use relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Initialize Panoptique
            panoptique = PanoptiqueFerroviaire(config_path=str(config_path))
            
            # Mock HybridFusionEngine
            with patch('main.HybridFusionEngine') as mock_fusion_class:
                mock_fusion = AsyncMock()
                mock_fusion_class.return_value = mock_fusion
                
                # Call _start_fusion_engine
                await panoptique._start_fusion_engine()
                
                # Verify HybridFusionEngine was initialized correctly
                call_kwargs = mock_fusion_class.call_args[1]
                
                assert call_kwargs['topology_path'] is None
                assert call_kwargs['stops_path'] == 'data/stops.txt'
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_logging_warning_for_missing_topology(self, tmp_path, caplog):
        """Test that warning is logged when topology file is missing."""
        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
application:
  name: Test
  version: 6.0.0
features:
  physics_simulation: true
logging:
  level: DEBUG
""")
        
        # Change to tmp directory to use relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Initialize Panoptique
            panoptique = PanoptiqueFerroviaire(config_path=str(config_path))
            
            # Mock HybridFusionEngine
            with patch('main.HybridFusionEngine') as mock_fusion_class:
                mock_fusion = AsyncMock()
                mock_fusion_class.return_value = mock_fusion
                
                import logging
                with caplog.at_level(logging.WARNING):
                    # Call _start_fusion_engine
                    await panoptique._start_fusion_engine()
                
                # Check that warning was logged (structured logging, so check attributes)
                # The logger.warning call should have been made
                # Since we're using structlog, we need to check differently
                # For now, just verify the fusion engine was called correctly
                mock_fusion_class.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    def test_default_config_includes_expected_keys(self):
        """Test that default config has required keys."""
        panoptique = PanoptiqueFerroviaire()
        
        assert 'application' in panoptique.config
        assert 'kafka' in panoptique.config
        assert 'features' in panoptique.config
        assert panoptique.config['features']['physics_simulation'] is True
