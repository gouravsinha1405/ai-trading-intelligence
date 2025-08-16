#!/usr/bin/env python3
"""
Test script for enhanced RealNewsClient with production features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.news_client import RealNewsClient
from datetime import datetime, timedelta
import json

def test_enhanced_news_client():
    """Test the enhanced news client functionality"""
    
    print("=== Testing Enhanced RealNewsClient ===")
    
    # Initialize client
    client = RealNewsClient()
    
    print("\n1. Testing RSS Feed Configuration")
    print(f"  ✓ Configured {len(client.rss_feeds)} RSS sources:")
    for source in client.rss_feeds.keys():
        print(f"    - {source}")
    
    print("\n2. Testing Enhanced News Fetching")
    try:
        # Test fetching from all sources
        news = client.get_latest_market_news(max_per_source=3)
        
        if news:
            print(f"  ✓ Successfully fetched {len(news)} articles")
            
            # Check for timezone-aware dates
            dated_articles = [a for a in news if a.get('published')]
            print(f"  ✓ {len(dated_articles)} articles have parsed timestamps")
            
            # Check deduplication
            links = [a.get('link', '') for a in news if a.get('link')]
            unique_links = set(links)
            print(f"  ✓ Deduplication: {len(links)} total -> {len(unique_links)} unique")
            
            # Show sample article
            if news:
                sample = news[0]
                print(f"  ✓ Sample article:")
                print(f"    Title: {sample.get('title', 'N/A')[:80]}...")
                print(f"    Source: {sample.get('source', 'N/A')}")
                print(f"    Published: {sample.get('published', 'N/A')}")
                print(f"    Link: {sample.get('link', 'N/A')[:60]}...")
                
        else:
            print("  ⚠ No news articles fetched")
            
    except Exception as e:
        print(f"  ✗ Error fetching news: {e}")
    
    print("\n3. Testing Enhanced Symbol Search")
    try:
        # Test symbol search with aliases
        reliance_aliases = ['Reliance Industries', 'RIL', 'Reliance Ind']
        tcs_aliases = ['Tata Consultancy', 'TCS Ltd']
        
        if 'news' in locals() and news:
            reliance_news = client.search_news_by_symbol('RELIANCE', news, reliance_aliases)
            tcs_news = client.search_news_by_symbol('TCS', news, tcs_aliases)
            
            print(f"  ✓ RELIANCE news found: {len(reliance_news)} articles")
            print(f"  ✓ TCS news found: {len(tcs_news)} articles")
            
            # Show sample match if available
            if reliance_news:
                sample = reliance_news[0]
                print(f"  ✓ Sample RELIANCE match: {sample.get('title', 'N/A')[:60]}...")
                
        else:
            print("  ⚠ No news available for symbol search test")
            
    except Exception as e:
        print(f"  ✗ Error in symbol search: {e}")
    
    print("\n4. Testing Recent News Filtering")
    try:
        recent_news = client.get_recent_news(hours=48, max_per_source=5)
        
        print(f"  ✓ Recent news (48h): {len(recent_news)} articles")
        
        if recent_news:
            # Check date filtering
            now = datetime.now()
            cutoff = now - timedelta(hours=48)
            valid_recent = [a for a in recent_news if a.get('published') and a['published'].replace(tzinfo=None) >= cutoff.replace(tzinfo=None)]
            print(f"  ✓ Properly filtered: {len(valid_recent)}/{len(recent_news)} within time range")
            
    except Exception as e:
        print(f"  ✗ Error in recent news test: {e}")
    
    print("\n5. Testing Multi-Symbol News")
    try:
        symbols_dict = {
            'RELIANCE': ['Reliance Industries', 'RIL'],
            'TCS': ['Tata Consultancy', 'TCS Ltd'],
            'INFY': ['Infosys', 'Infosys Ltd']
        }
        
        multi_news = client.get_news_by_symbols(symbols_dict, max_per_source=3)
        
        print(f"  ✓ Multi-symbol news results:")
        for symbol, articles in multi_news.items():
            print(f"    {symbol}: {len(articles)} articles")
            
    except Exception as e:
        print(f"  ✗ Error in multi-symbol test: {e}")
    
    print("\n6. Testing Sentiment Analysis")
    try:
        keywords = client.get_news_sentiment_keywords()
        print(f"  ✓ Sentiment keywords loaded:")
        print(f"    Positive: {len(keywords['positive'])} words")
        print(f"    Negative: {len(keywords['negative'])} words")
        
        # Test sentiment analysis on sample article
        if 'news' in locals() and news:
            sample_article = news[0]
            sentiment = client.analyze_article_sentiment(sample_article)
            
            print(f"  ✓ Sample sentiment analysis:")
            print(f"    Article: {sample_article.get('title', 'N/A')[:50]}...")
            print(f"    Sentiment: {sentiment['sentiment']}")
            print(f"    Positive score: {sentiment['positive_score']}")
            print(f"    Negative score: {sentiment['negative_score']}")
            print(f"    Confidence: {sentiment['confidence']:.2f}")
            
    except Exception as e:
        print(f"  ✗ Error in sentiment analysis test: {e}")
    
    print("\n7. Testing Error Handling")
    try:
        # Test with invalid feed URL
        invalid_articles = client.fetch_news_from_feed("https://invalid-url.com/feed", 5)
        print(f"  ✓ Invalid URL handling: {len(invalid_articles)} articles (should be 0)")
        
        # Test empty search
        empty_search = client.search_news_by_symbol("NONEXISTENT", news if 'news' in locals() else [])
        print(f"  ✓ Empty search handling: {len(empty_search)} articles (should be 0)")
        
    except Exception as e:
        print(f"  ✗ Error in error handling test: {e}")
    
    print("\n=== Enhanced News Client Test Complete ===")

if __name__ == "__main__":
    test_enhanced_news_client()
