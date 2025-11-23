#!/usr/bin/env python3
"""
HNPS v6.0 Rail-Lock Demonstration
==================================
This script demonstrates the Rail-Lock functionality by creating a sample
topology and projecting GPS positions onto it.

Usage:
    python demo_rail_lock.py
"""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from src.engine.topology import TopologyEngine


def create_demo_topology(output_path: Path) -> None:
    """
    Create a demo topology file representing a fictional metro line.
    
    This creates a simple L-shaped route with elevation changes.
    """
    # Create a fictional metro line with two segments:
    # Segment 1: North-South (flat)
    # Segment 2: East-West (with gradient)
    
    topology = [
        {
            "shape_id": "LINE_1_NORTH_SOUTH",
            "route_id": "METRO_LINE_1",
            "points": [
                # Start at Paris center, go north
                [2.3522, 48.8566, 35.0],   # Paris center (35m elevation)
                [2.3524, 48.8580, 35.0],   # Station 1
                [2.3526, 48.8595, 40.0],   # Station 2 (climbing)
                [2.3528, 48.8610, 50.0],   # Station 3 (summit)
            ]
        },
        {
            "shape_id": "LINE_1_EAST_WEST",
            "route_id": "METRO_LINE_1",
            "points": [
                # Continue from Station 3, go east
                [2.3528, 48.8610, 50.0],   # Station 3
                [2.3550, 48.8612, 45.0],   # Station 4 (descending)
                [2.3572, 48.8614, 40.0],   # Station 5
                [2.3594, 48.8616, 40.0],   # Station 6 (flat)
            ]
        },
        {
            "shape_id": "LINE_2_CIRCULAR",
            "route_id": "METRO_LINE_2",
            "points": [
                # Circular route around Paris center
                [2.3400, 48.8600, 30.0],
                [2.3450, 48.8650, 32.0],
                [2.3550, 48.8650, 35.0],
                [2.3600, 48.8600, 33.0],
                [2.3550, 48.8550, 30.0],
                [2.3450, 48.8550, 28.0],
                [2.3400, 48.8600, 30.0],  # Back to start
            ]
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"✓ Created demo topology: {output_path}")


def demo_rail_lock() -> None:
    """Run the Rail-Lock demonstration."""
    print("\n" + "="*80)
    print("HNPS v6.0 RAIL-LOCK DEMONSTRATION")
    print("="*80 + "\n")
    
    # Create temporary topology file
    with TemporaryDirectory() as tmpdir:
        topology_path = Path(tmpdir) / "demo_topology.json"
        create_demo_topology(topology_path)
        
        # Initialize Topology Engine
        print("\n📍 Initializing Topology Engine...")
        engine = TopologyEngine(str(topology_path))
        
        print(f"   Shapes loaded: {len(engine.shapes)}")
        print(f"   Routes loaded: {len(engine.route_to_shapes)}")
        print(f"   Rail-Lock available: {engine.is_available()}")
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Train on LINE_1 at Station 1",
                "lat": 48.8580,
                "lon": 2.3524,
                "route_id": "METRO_LINE_1"
            },
            {
                "name": "Train on LINE_1 climbing to Station 3",
                "lat": 48.8600,
                "lon": 2.3527,
                "route_id": "METRO_LINE_1"
            },
            {
                "name": "Train on LINE_2 (circular route)",
                "lat": 48.8650,
                "lon": 2.3500,
                "route_id": "METRO_LINE_2"
            },
            {
                "name": "Train slightly off-track (5m lateral error)",
                "lat": 48.8580 + 0.00005,  # ~5m north
                "lon": 2.3524,
                "route_id": "METRO_LINE_1"
            },
            {
                "name": "Train far from track (derailment scenario)",
                "lat": 48.9000,  # Far away
                "lon": 2.4000,
                "route_id": "METRO_LINE_1"
            }
        ]
        
        print("\n" + "="*80)
        print("TESTING RAIL-LOCK PROJECTIONS")
        print("="*80)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📊 Scenario {i}: {scenario['name']}")
            print(f"   GPS Position: ({scenario['lat']:.6f}°, {scenario['lon']:.6f}°)")
            print(f"   Route ID: {scenario['route_id']}")
            
            # Get Rail-Lock projection
            projection = engine.get_rail_lock(
                scenario['lat'],
                scenario['lon'],
                scenario['route_id']
            )
            
            if projection:
                # Display results
                print(f"\n   ✓ Rail-Lock SUCCESS")
                print(f"   ├─ Shape ID: {projection.shape_id}")
                print(f"   ├─ Track Distance (PK): {projection.track_distance:.2f} m")
                print(f"   ├─ Cross-Track Error: {projection.cross_track_error:.2f} m")
                print(f"   ├─ Gradient: {math.degrees(projection.gradient):.4f}° ({projection.gradient:.6f} rad)")
                print(f"   └─ Confidence: {projection.confidence:.2%}")
                
                # Analysis
                if projection.cross_track_error < 10.0:
                    status = "🟢 EXCELLENT - Train is on track"
                elif projection.cross_track_error < 50.0:
                    status = "🟡 ACCEPTABLE - Train near track centerline"
                else:
                    status = "🔴 WARNING - High cross-track error (possible derailment)"
                
                print(f"\n   Status: {status}")
                
                # Physics implications
                if abs(projection.gradient) > 0.01:  # > ~0.57 degrees
                    gradient_type = "uphill" if projection.gradient > 0 else "downhill"
                    print(f"   Physics: Significant {gradient_type} gradient detected")
                    print(f"            Davis equation will apply gravity component")
                
            else:
                print(f"\n   ✗ Rail-Lock FAILED - No projection available")
        
        # Summary
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("  • GPS position projection onto track geometry")
        print("  • Precise track distance (PK) calculation for cantonnement")
        print("  • Cross-track error measurement for derailment detection")
        print("  • Gradient extraction for 3D physics simulation")
        print("  • Confidence scoring based on projection quality")
        print("  • Graceful handling of off-track scenarios")
        
        print("\nIntegration with HNPS v5.0:")
        print("  • Track distance enables Moving Block (Cantonnement)")
        print("  • Gradient enables gravity in Davis equation")
        print("  • Cross-track error triggers derailment alerts")
        print("  • Confidence score filters unreliable data")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    demo_rail_lock()
