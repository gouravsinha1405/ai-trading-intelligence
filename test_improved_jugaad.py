#!/usr/bin/env python3
"""
Test script for improved JugaadClient
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.jugaad_client import JugaadDataClient
from datetime import datetime, timedelta
import pandas as pd

def test_jugaad_improvements():
    """Test the improved jugaad client functionality"""
    
    print("=== Testing Improved JugaadClient ===")
    
    # Initialize client
    client = JugaadDataClient()
    
    print("\n1. Testing Numeric String Parsing (_to_float method)")
    test_cases = [
        "12,345.67",   # Comma-separated number
        "₹1,234.56",   # Currency symbol
        "12345.67",    # Regular number
        "12,34,567.89", # Indian number format
        "",            # Empty string
        None,          # None value
        "abc123",      # Invalid string
    ]
    
    for test_case in test_cases:
        result = client._to_float(test_case)
        print(f"  '{test_case}' -> {result}")
    
    print("\n2. Testing Market Status with Timezone")
    status = client.get_market_status()
    print(f"  Market Status: {status}")
    
    print("\n3. Testing Historical Data with Cleaning")
    try:
        # Test with a reliable stock
        symbol = "RELIANCE"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"  Fetching {symbol} data from {start_date} to {end_date}")
        df = client.get_stock_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"  ✓ Successfully fetched {len(df)} records")
            print(f"  ✓ Columns: {list(df.columns)}")
            print(f"  ✓ Date range: {df.index.min()} to {df.index.max()}")
            
            # Check data types for Close column
            if 'Close' in df.columns:
                print(f"  ✓ Close type: {df['Close'].dtype}")
                print(f"  ✓ Close sample: {df['Close'].iloc[0] if len(df) > 0 else 'N/A'}")
                    
        else:
            print("  ✗ No data retrieved")
            
    except Exception as e:
        print(f"  ✗ Error fetching historical data: {e}")
    
    print("\n4. Testing Multiple Stocks Functionality")
    try:
        symbols = ["RELIANCE", "TCS", "INFY"]
        stocks_data = client.get_stocks_data(symbols, start_date, end_date)
        
        print(f"  ✓ Fetched data for {len(stocks_data)} out of {len(symbols)} symbols")
        for symbol, data in stocks_data.items():
            print(f"    {symbol}: {len(data)} records")
            
    except Exception as e:
        print(f"  ✗ Error in multiple stocks test: {e}")
    
    print("\n5. Testing Live Price Functionality")
    try:
        live_price = client.get_live_price("RELIANCE")
        
        if live_price:
            print(f"  ✓ Live price data retrieved")
            print(f"    Symbol: {live_price.get('symbol')}")
            print(f"    Price: {live_price.get('price')}")
            print(f"    Change: {live_price.get('change')}")
            print(f"    Timestamp: {live_price.get('timestamp')}")
        else:
            print("  ⚠ No live price data (market might be closed)")
            
    except Exception as e:
        print(f"  ✗ Error fetching live price: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_jugaad_improvements()
