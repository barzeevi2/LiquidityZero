#!/usr/bin/env python3
"""
Test script to diagnose data loading issues
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.sim.market_simulator import MarketSimulator

async def test_data_loading():
    """Test if data loading works"""
    print("="*60)
    print("Testing Data Loading from TimescaleDB")
    print("="*60)
    
    # Use same time range as training script
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    print(f"\nQuerying data from: {start_time}")
    print(f"To: {end_time}")
    print(f"Symbol: BTC/USDT")
    print("\nAttempting to load data...")
    
    try:
        simulator = MarketSimulator(symbol="BTC/USDT", initial_cash=10000.0)
        
        # This is where it might hang
        await simulator.load_historical_data(start_time=start_time, end_time=end_time)
        
        print(f"\n✓ Successfully loaded {len(simulator.snapshots)} snapshots")
        
        if len(simulator.snapshots) > 0:
            print(f"  First snapshot: {simulator.snapshots[0]['datetime']}")
            print(f"  Last snapshot: {simulator.snapshots[-1]['datetime']}")
            return True
        else:
            print("⚠ No snapshots loaded!")
            return False
            
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_data_loading())
    sys.exit(0 if success else 1)



