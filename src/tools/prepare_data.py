#!/usr/bin/env python3
"""
Data Preparation Tool - HNPS v6.0
==================================
Robust script to automate the acquisition and preparation of static GTFS data
for the HNPS system. This tool downloads GTFS ZIP files, extracts stops.txt,
and generates topology.json for Rail-Lock engine.

Features:
- Downloads GTFS ZIP from URL or reads from local path
- Extracts stops.txt for Holographic Positioning (GPS-less mode)
- Generates topology.json for Rail-Lock spatial awareness
- Production-ready with logging and error handling
- Reuses existing GTFSTopologyConverter for topology generation

Author: HNPS DevOps Engineering Team
Version: 6.0.0
Python: 3.12+
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional
from io import BytesIO

import requests

# Import existing topology converter
from src.tools.gtfs_to_topology import GTFSTopologyConverter


# Default GTFS IDFM URL (Île-de-France Mobilités)
DEFAULT_GTFS_URL = "https://data.iledefrance-mobilites.fr/explore/dataset/offre-horaires-tc-gtfs-idfm/files/b80a5a8f7e1c4e1e8e82e7e3d32c0a0c/download/"

# Output paths
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_STOPS_OUTPUT = DEFAULT_OUTPUT_DIR / "stops.txt"
DEFAULT_TOPOLOGY_OUTPUT = DEFAULT_OUTPUT_DIR / "topology.json"


class DataPreparationTool:
    """
    Automates the preparation of static GTFS data for HNPS.
    
    This class handles:
    1. Downloading GTFS ZIP from URL or loading from local path
    2. Extracting stops.txt to data directory
    3. Generating topology.json using existing GTFSTopologyConverter
    """
    
    def __init__(
        self,
        source: str,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        stops_output: Optional[Path] = None,
        topology_output: Optional[Path] = None
    ):
        """
        Initialize the data preparation tool.
        
        Args:
            source: URL or local path to GTFS ZIP file.
            output_dir: Output directory for generated files.
            stops_output: Path for stops.txt output (overrides output_dir).
            topology_output: Path for topology.json output (overrides output_dir).
        """
        self.source = source
        self.output_dir = output_dir
        
        # Set output paths
        self.stops_output = stops_output or (output_dir / "stops.txt")
        self.topology_output = topology_output or (output_dir / "topology.json")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Temporary storage for downloaded/loaded GTFS
        self.gtfs_zip_path: Optional[Path] = None
        self.is_temporary_file = False
        
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        return source.startswith(('http://', 'https://'))
        
    def _download_gtfs(self) -> Path:
        """
        Download GTFS ZIP from URL.
        
        Returns:
            Path to downloaded ZIP file.
            
        Raises:
            requests.RequestException: If download fails.
        """
        self.logger.info(f"📥 Downloading GTFS from {self.source}...")
        
        try:
            response = requests.get(self.source, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save to temporary file in output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temp_zip = self.output_dir / "gtfs_temp.zip"
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                self.logger.info(f"  Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
                            else:
                                self.logger.info(f"  Downloaded: {downloaded / (1024*1024):.1f} MB")
            
            file_size_mb = temp_zip.stat().st_size / (1024 * 1024)
            self.logger.info(f"  ✓ Downloaded {file_size_mb:.2f} MB")
            
            self.is_temporary_file = True
            return temp_zip
            
        except requests.RequestException as e:
            self.logger.error(f"  ✗ Download failed: {e}")
            raise
            
    def _load_gtfs(self) -> Path:
        """
        Load GTFS ZIP from URL or local path.
        
        Returns:
            Path to GTFS ZIP file.
            
        Raises:
            FileNotFoundError: If local file doesn't exist.
            requests.RequestException: If download fails.
        """
        if self._is_url(self.source):
            return self._download_gtfs()
        else:
            # Local path
            local_path = Path(self.source)
            if not local_path.exists():
                raise FileNotFoundError(f"GTFS file not found: {local_path}")
            
            self.logger.info(f"📂 Using local GTFS file: {local_path}")
            self.is_temporary_file = False
            return local_path
            
    def extract_stops(self) -> None:
        """
        Extract stops.txt from GTFS ZIP to output directory.
        
        Essential for Holographic Positioning (GPS-less mode).
        
        Raises:
            KeyError: If stops.txt is not in the ZIP file.
            zipfile.BadZipFile: If the file is not a valid ZIP.
        """
        self.logger.info(f"📍 Extracting stops.txt to {self.stops_output}...")
        
        if not self.gtfs_zip_path or not self.gtfs_zip_path.exists():
            raise FileNotFoundError("GTFS ZIP file not loaded. Call prepare() first.")
        
        try:
            with zipfile.ZipFile(self.gtfs_zip_path, 'r') as zip_ref:
                if 'stops.txt' not in zip_ref.namelist():
                    raise KeyError("stops.txt not found in GTFS ZIP file")
                
                # Extract stops.txt
                self.stops_output.parent.mkdir(parents=True, exist_ok=True)
                zip_ref.extract('stops.txt', path=self.stops_output.parent)
                
                # Rename if extracted to wrong location (some zips have directory structure)
                extracted_stops = self.stops_output.parent / 'stops.txt'
                if extracted_stops != self.stops_output and extracted_stops.exists():
                    extracted_stops.rename(self.stops_output)
                
                # Get file size
                file_size_kb = self.stops_output.stat().st_size / 1024
                self.logger.info(f"  ✓ Extracted stops.txt ({file_size_kb:.2f} KB)")
                
        except zipfile.BadZipFile:
            self.logger.error(f"  ✗ Invalid ZIP file: {self.gtfs_zip_path}")
            raise
        except Exception as e:
            self.logger.error(f"  ✗ Extraction failed: {e}")
            raise
            
    def generate_topology(self) -> None:
        """
        Generate topology.json by crossing shapes.txt and trips.txt.
        
        Essential for Rail-Lock spatial awareness.
        Reuses existing GTFSTopologyConverter.
        
        Raises:
            Various exceptions from GTFSTopologyConverter.
        """
        self.logger.info(f"🌐 Generating topology.json to {self.topology_output}...")
        
        if not self.gtfs_zip_path or not self.gtfs_zip_path.exists():
            raise FileNotFoundError("GTFS ZIP file not loaded. Call prepare() first.")
        
        try:
            # Use existing GTFSTopologyConverter
            converter = GTFSTopologyConverter(
                input_path=self.gtfs_zip_path,
                output_path=self.topology_output
            )
            
            # Run conversion
            converter.convert()
            
            self.logger.info(f"  ✓ Topology generated successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Topology generation failed: {e}")
            raise
            
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.is_temporary_file and self.gtfs_zip_path and self.gtfs_zip_path.exists():
            self.logger.info("🧹 Cleaning up temporary files...")
            try:
                self.gtfs_zip_path.unlink()
                self.logger.info("  ✓ Temporary files removed")
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to remove temporary file: {e}")
                
    def prepare(self) -> None:
        """
        Execute complete data preparation process.
        
        This is the main entry point that orchestrates:
        1. Loading GTFS ZIP (download or local)
        2. Extracting stops.txt
        3. Generating topology.json
        4. Cleanup of temporary files
        """
        try:
            # Step 1: Load GTFS ZIP
            self.gtfs_zip_path = self._load_gtfs()
            
            # Step 2: Extract stops.txt
            self.extract_stops()
            
            # Step 3: Generate topology.json
            self.generate_topology()
            
            # Success message
            self.logger.info("\n✅ Data preparation completed successfully!")
            self.logger.info(f"   📍 stops.txt: {self.stops_output}")
            self.logger.info(f"   🌐 topology.json: {self.topology_output}")
            
        except Exception as e:
            self.logger.error(f"\n❌ Data preparation failed: {e}")
            raise
        finally:
            # Always cleanup temporary files
            self.cleanup()


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(message)s',
        stream=sys.stdout
    )


def main() -> None:
    """
    CLI entry point for data preparation tool.
    
    Parses command-line arguments and executes data preparation.
    """
    parser = argparse.ArgumentParser(
        description="Prepare static GTFS data for HNPS v6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from default IDFM URL
  %(prog)s

  # Download from custom URL
  %(prog)s --url https://example.com/gtfs.zip

  # Use local GTFS file
  %(prog)s --local /path/to/gtfs.zip

  # Custom output directory
  %(prog)s --output-dir /custom/data

Output Files:
  - data/stops.txt: Station locations for Holographic Positioning
  - data/topology.json: Route geometries for Rail-Lock engine

Requirements:
  - Python 3.12+
  - Internet connection (for URL downloads)
  - pandas, tqdm, requests libraries
        """
    )
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        '--url',
        type=str,
        help=f'URL to download GTFS ZIP (default: IDFM)'
    )
    source_group.add_argument(
        '--local',
        type=str,
        help='Local path to GTFS ZIP file'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for generated files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--stops-output',
        type=Path,
        help='Custom path for stops.txt output (overrides --output-dir)'
    )
    parser.add_argument(
        '--topology-output',
        type=Path,
        help='Custom path for topology.json output (overrides --output-dir)'
    )
    
    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 6.0.0 (HNPS Data Preparation Tool)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Determine source
    if args.local:
        source = args.local
    elif args.url:
        source = args.url
    else:
        # Default to IDFM URL
        source = DEFAULT_GTFS_URL
        logging.info(f"Using default GTFS URL: {source}")
    
    # Print banner
    print("\n" + "="*80)
    print("🚂 HNPS v6.0 - Data Preparation Tool")
    print("   Automated Static Data Acquisition")
    print("="*80 + "\n")
    
    # Create tool and execute
    try:
        tool = DataPreparationTool(
            source=source,
            output_dir=args.output_dir,
            stops_output=args.stops_output,
            topology_output=args.topology_output
        )
        
        tool.prepare()
        
        print("\n" + "="*80)
        print("✅ SUCCESS - Data ready for HNPS v6.0")
        print("="*80 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ FAILED - {e}")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
