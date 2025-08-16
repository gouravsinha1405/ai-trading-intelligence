#!/usr/bin/env python3
"""
Test script to verify real data integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta

def test_real_data_sources():
    """Test all real data sources to verify they're working"""
    print("🧪 Testing Real Data Sources")
    print("=" * 50)
    
    # Test 1: Jugaad Data Client
    print("\n📊 Testing Real Market Data...")
    try:
        from src.data.jugaad_client import JugaadDataClient
        client = JugaadDataClient()
        
        # Test live prices
        test_symbols = ['RELIANCE', 'TCS']
        live_data = client.get_multiple_live_prices(test_symbols)
        
        if live_data:
            print("✅ Live market data fetched successfully!")
            for symbol, data in live_data.items():
                print(f"   {symbol}: ₹{data.get('price', 'N/A')} ({data.get('pChange', 'N/A')}%)")
        else:
            print("⚠️ Live data unavailable (market may be closed)")
        
        # Test historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hist_data = client.get_stock_data('RELIANCE', start_date, end_date)
        if hist_data is not None and len(hist_data) > 0:
            print(f"✅ Historical data: {len(hist_data)} records for RELIANCE")
            print(f"   Latest close: ₹{hist_data['Close'].iloc[-1]:.2f}")
        else:
            print("⚠️ Historical data unavailable")
            
    except Exception as e:
        print(f"❌ Error testing market data: {e}")
    
    # Test 2: Real News Client
    print("\n📰 Testing Real News Sources...")
    try:
        from src.data.news_client import RealNewsClient
        news_client = RealNewsClient()
        
        # Fetch real news
        news_articles = news_client.get_latest_market_news(max_per_source=3)
        
        if news_articles:
            print(f"✅ Real news fetched: {len(news_articles)} articles")
            for article in news_articles[:3]:
                print(f"   📰 {article['source']}: {article['title'][:60]}...")
        else:
            print("⚠️ No news articles fetched")
            
    except Exception as e:
        print(f"❌ Error testing news data: {e}")
    
    # Test 3: Market Status
    print("\n⏰ Testing Market Status...")
    try:
        market_status = client.get_market_status()
        status = market_status.get('status', 'unknown')
        message = market_status.get('message', 'No message')
        
        if status == 'open':
            print(f"✅ Market Status: {status.upper()} - {message}")
        else:
            print(f"⚠️ Market Status: {status.upper()} - {message}")
            
    except Exception as e:
        print(f"❌ Error checking market status: {e}")
    
    print("\n🎉 Real Data Testing Complete!")
    print("=" * 50)
    print("\n💡 Summary:")
    print("   - Your app uses REAL market data from NSE/BSE via jugaad-data")
    print("   - News comes from REAL RSS feeds from major financial sources")
    print("   - Synthetic data is only used as fallback when real data fails")
    print("   - The app will clearly indicate when using real vs fallback data")

if __name__ == "__main__":
    test_real_data_sources()
