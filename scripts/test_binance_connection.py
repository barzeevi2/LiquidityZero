"""
Simple test to verify the connection to the Binance API and order book retrieval.
uses ccxt library
utility script for testing / debugging
"""

import asyncio
import sys
import ccxt.pro as ccxt
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ccxt import binance
from app.core.config import settings

async def test_binance_connection():
    """
    test function to connect to binance and fetch order book data
    """
    exchange = ccxt.binance({
        'apiKey': '',
        'secret': '',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }

    })

    try: 
        print("Testing connection to Binance...")
        print(f"fetching order book for {settings.SYMBOL}")
        print("-" * 60)

        #get the top 5 bids and asks
        order_book = await exchange.fetch_order_book(settings.SYMBOL, limit=5)

        print("ORDER BOOK DATA:")
        print(f"Symbol: {settings.SYMBOL}")
        print(f"Timestamp: {order_book['timestamp']}")
        print(f"Nonce: {order_book['nonce']}")

        #display bids
        print("BIDS")
        print("-" * 20)
        for bid in order_book['bids'][:5]:
            price, amount = bid
            print(f"Price: {price:.8f}, Amount: {amount:.8f}")
        print("\n" + "-" * 20)

        #display asks
        print("ASKS")
        print("-" * 20)
        for ask in order_book['asks'][:5]:
            price, amount = ask
            print(f"Price: {price:.8f}, Amount: {amount:.8f}")
        print("\n" + "-" * 20)
        

        #calculate the spread
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = best_ask - best_bid
        spread_percentage = (spread / best_bid) * 100


        print("MARKET SUMMARY:")
        print(f"Best Bid: {best_bid:.8f}")
        print(f"Best Ask: {best_ask:.8f}")
        print(f"Spread: {spread:.8f} ({spread_percentage:.2f}%)")
        print("SUCCESS: Connection to Binance established and order book data retrieved.")
    except Exception as e:
        print(f"ERROR: Failed to connect to Binance: {str(e)}")

    finally:
        print("Closing connection to Binance...")
        await exchange.close()

if __name__ == "__main__":
    print("Starting Binance connection test...")
    print("-" * 60)
    asyncio.run(test_binance_connection())