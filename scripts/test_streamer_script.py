"""
script to run OrderBookStreamer and see live data
"""

import asyncio
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.streamer import OrderBookStreamer

# config logging using AI help
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def run_streamer_demo(symbol: str = "BTC/USDT", max_updates: int = 10):
    """
    run the streamer and print orderbook updates
    """

    streamer = OrderBookStreamer(
        symbol = symbol,
        max_reconnect_attempts=5,
        initial_backoff = 1.0
    )

    print(f"Starting OrderBookStreamer for {symbol}...")
    print(f"Will display up to {max_updates} updates")
    print("-" * 70)

    update_count = 0

    try:
        async for orderbook in streamer.stream():
            update_count += 1

            print(f"\n[{update_count}] Orderbook Update:")
            print(f"  Symbol: {orderbook['symbol']}")
            print(f"  Best Bid: {orderbook['best_bid']:.2f}")
            print(f"  Best Ask: {orderbook['best_ask']:.2f}")
            print(f"  Spread: {orderbook['spread']:.2f} ({orderbook['spread_pct']:.4f}%)")
            print(f"  Timestamp: {orderbook['datetime']}")

            #show top 3 bids and asks
            print(f"\n Top 3 Bids:")
            for i, bid in enumerate(orderbook['bids'][:3], 1):
                print(f"    {i}. Price: {bid[0]:.2f}, Amount: {bid[1]:.8f}")
            
            print(f"\n Top 3 Asks:")
            for i, ask in enumerate(orderbook['asks'][:3], 1):
                print(f"    {i}. Price: {ask[0]:.2f}, Amount: {ask[1]:.8f}")
            

            print("-" * 70)

            if max_updates > 0 and update_count >= max_updates:
                print(f"\nReached {max_updates} updates. Stopping...")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n Interrupted by user, stopping streamer...")
    
    except Exception as e:
        print(f"\n\n Error occurred: {e}")
    finally:
        await streamer.stop()
        print("Streamer stopped")


if __name__ == "__main__":
    symbol = "BTC/USDT"
    asyncio.run(run_streamer_demo(symbol=symbol, max_updates = 10))
