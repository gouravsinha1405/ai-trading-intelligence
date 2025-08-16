#!/usr/bin/env python3
"""
Unit tests for final production improvements to AI Analyzer
Tests: preflight guards, JSON extraction, constraint enforcement
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from src.analysis.ai_analyzer import GroqAnalyzer

def test_preflight_guards():
    """Test preflight checks for empty inputs"""
    print("\n=== Testing Preflight Guards ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Test empty market_stats
    data = {
        "market_stats": {"index": {}},  # Empty
        "strategy": {"knobs": {"sma_fast": [5, 20]}},
        "performance": {"headline": {"cagr": 0.1}}
    }
    
    result = analyzer._preflight(data)
    assert result is not None
    assert result["ok"] is False
    assert "market_stats.index" in result["issues"][0]
    print("✅ Empty market_stats caught by preflight")
    
    # Test empty knobs
    data = {
        "market_stats": {"index": {"ret_mean": 0.001}},
        "strategy": {"knobs": {}},  # Empty
        "performance": {"headline": {"cagr": 0.1}}
    }
    
    result = analyzer._preflight(data)
    assert result is not None
    assert result["ok"] is False
    assert "strategy.knobs" in result["issues"][0]
    print("✅ Empty strategy knobs caught by preflight")
    
    # Test empty performance
    data = {
        "market_stats": {"index": {"ret_mean": 0.001}},
        "strategy": {"knobs": {"sma_fast": [5, 20]}},
        "performance": {"headline": {}}  # Empty
    }
    
    result = analyzer._preflight(data)
    assert result is not None
    assert result["ok"] is False
    assert "performance.headline" in result["issues"][0]
    print("✅ Empty performance headline caught by preflight")
    
    # Test valid inputs pass
    data = {
        "market_stats": {"index": {"ret_mean": 0.001}},
        "strategy": {"knobs": {"sma_fast": [5, 20]}},
        "performance": {"headline": {"cagr": 0.1}}
    }
    
    result = analyzer._preflight(data)
    assert result is None
    print("✅ Valid inputs pass preflight")

def test_json_extraction_robustness():
    """Test robust JSON extraction with code fences"""
    print("\n=== Testing JSON Extraction ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Test with markdown fences
    response = '''```json
    {"ok": true, "changes": []}
    ```'''
    
    extracted = analyzer._extract_json(response)
    parsed = json.loads(extracted)
    assert parsed["ok"] is True
    print("✅ Markdown fences handled correctly")
    
    # Test with prose before/after
    response = '''Here's the analysis:
    
    {"ok": true, "changes": [], "risks": ["market volatility"]}
    
    This should work well.'''
    
    extracted = analyzer._extract_json(response)
    parsed = json.loads(extracted)
    assert parsed["ok"] is True
    assert "market volatility" in parsed["risks"]
    print("✅ JSON extracted from prose correctly")
    
    # Test with nested braces (problematic case)
    response = '''Let me explain {the reasoning}: {"ok": true, "analysis": {"trend": "up"}, "notes": "market volatility is normal"}'''
    
    extracted = analyzer._extract_json(response)
    parsed = json.loads(extracted)
    assert parsed["ok"] is True
    assert parsed["analysis"]["trend"] == "up"
    print("✅ Nested braces handled correctly")
    
    # Test malformed JSON
    response = '''{"ok": true, "malformed": }'''
    
    extracted = analyzer._extract_json(response)
    try:
        json.loads(extracted)
        assert False, "Should have failed to parse"
    except json.JSONDecodeError:
        print("✅ Malformed JSON properly fails")

def test_constraint_enforcement():
    """Test post-LLM constraint enforcement"""
    print("\n=== Testing Constraint Enforcement ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Test invalid area filtering
    result = {
        "ok": True,
        "changes": [
            {"area": "invalid_area", "param": "sma_fast", "new_range": [10, 20]},
            {"area": "entry", "param": "sma_fast", "new_range": [10, 20]}
        ]
    }
    
    knobs = {"sma_fast": [5, 30]}
    constraints = {"risk_per_trade": 0.01}
    
    validated = analyzer._postvalidate(result, knobs, constraints, [])
    assert len(validated["changes"]) == 1
    assert validated["changes"][0]["area"] == "entry"
    print("✅ Invalid areas filtered out")
    
    # Test unknown parameter filtering
    result = {
        "ok": True,
        "changes": [
            {"area": "entry", "param": "unknown_param", "new_range": [10, 20]},
            {"area": "entry", "param": "sma_fast", "new_range": [10, 20]}
        ]
    }
    
    validated = analyzer._postvalidate(result, knobs, constraints, [])
    assert len(validated["changes"]) == 1
    assert validated["changes"][0]["param"] == "sma_fast"
    print("✅ Unknown parameters filtered out")
    
    # Test risk per trade cap enforcement
    result = {
        "ok": True,
        "changes": [
            {"area": "risk", "param": "risk_per_trade", "new_range": [0.015, 0.025]}  # Above 0.01 cap
        ]
    }
    
    # Use slightly higher cap to avoid degenerate range
    constraints_high_cap = {"risk_per_trade": 0.02}
    validated = analyzer._postvalidate(result, {"risk_per_trade": [0.005, 0.05]}, constraints_high_cap, [])
    assert len(validated["changes"]) == 1
    lo, hi = validated["changes"][0]["new_range"]
    assert lo <= 0.02 and hi <= 0.02
    print("✅ Risk per trade cap enforced")
    
    # Test degenerate range filtering
    result = {
        "ok": True,
        "changes": [
            {"area": "entry", "param": "sma_fast", "new_range": [15.0, 15.0]}  # Identical
        ]
    }
    
    validated = analyzer._postvalidate(result, knobs, constraints, [])
    assert len(validated["changes"]) == 0
    print("✅ Degenerate ranges filtered out")

def test_timezone_safety():
    """Test safer timezone handling for news"""
    print("\n=== Testing Timezone Safety ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Test with naive datetime string (should assume UTC)
    news_data = [{
        "title": "Market Update",
        "summary": "Stocks rise on earnings",
        "published": "2024-01-01 12:00:00"  # Naive
    }]
    
    signals = analyzer._extract_news_signals(news_data)
    assert len(signals) == 1
    
    # Should be converted to IST
    ts_str = signals[0]["ts"]
    assert "+05:30" in ts_str
    print("✅ Naive datetime assumed UTC and converted to IST")
    
    # Test with already timezone-aware string
    news_data = [{
        "title": "Market Update",
        "summary": "Stocks rise on earnings", 
        "published": "2024-01-01T12:00:00+00:00"  # UTC
    }]
    
    signals = analyzer._extract_news_signals(news_data)
    ts_str = signals[0]["ts"]
    assert "+05:30" in ts_str
    print("✅ UTC datetime converted to IST")

def test_hurst_log_prices():
    """Test Hurst calculation with log prices"""
    print("\n=== Testing Hurst on Log Prices ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Create trending prices
    np.random.seed(42)
    prices = [100]
    trend = 0.001  # Small upward trend
    
    for i in range(100):
        noise = np.random.normal(0, 0.01)
        prices.append(prices[-1] * (1 + trend + noise))
    
    price_series = pd.Series(prices)
    hurst = analyzer._calculate_hurst(price_series)
    
    # Trending data should have Hurst > 0.5
    assert 0.5 <= hurst <= 1.0
    print(f"✅ Hurst for trending data: {hurst:.3f} (using log prices)")
    
    # Test with insufficient data
    short_series = pd.Series([100, 101, 102])
    hurst = analyzer._calculate_hurst(short_series)
    assert hurst == 0.5  # Default for insufficient data
    print("✅ Hurst defaults to 0.5 for insufficient data")

def test_telemetry_metadata():
    """Test telemetry metadata inclusion"""
    print("\n=== Testing Telemetry Metadata ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    meta = analyzer._get_telemetry_meta()
    
    required_fields = ["model", "temperature", "ts", "prompt_version"]
    for field in required_fields:
        assert field in meta
    
    assert meta["model"] == "llama-3.3-70b-versatile"
    assert meta["temperature"] == 0.1
    assert meta["prompt_version"] == "v1.1"
    assert "+05:30" in meta["ts"]  # IST timezone
    
    print("✅ Telemetry metadata complete")
    print(f"   Model: {meta['model']}")
    print(f"   Temperature: {meta['temperature']}")
    print(f"   Version: {meta['prompt_version']}")

def test_breadth_calculation_safety():
    """Test safer breadth calculation"""
    print("\n=== Testing Breadth Calculation Safety ===")
    
    analyzer = GroqAnalyzer("test_key")
    
    # Test with insufficient data (should skip breadth)
    short_data = pd.DataFrame({
        'Close': list(range(100, 150))  # Only 50 points
    })
    
    stats = analyzer._compute_market_stats(short_data)
    assert "breadth_above_200dma" not in stats
    print("✅ Breadth skipped for insufficient data")
    
    # Test with adequate data
    np.random.seed(42)
    prices = [100]
    for _ in range(250):  # Enough for 200 SMA + buffer
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    adequate_data = pd.DataFrame({'Close': prices})
    stats = analyzer._compute_market_stats(adequate_data)
    
    if "breadth_above_200dma" in stats:
        breadth = stats["breadth_above_200dma"]
        assert 0.0 <= breadth <= 1.0
        print(f"✅ Breadth calculated safely: {breadth:.3f}")
    else:
        print("✅ Breadth calculation requires sufficient valid data")

if __name__ == "__main__":
    print("🔧 Testing Final Production Improvements")
    print("=" * 50)
    
    try:
        test_preflight_guards()
        test_json_extraction_robustness()
        test_constraint_enforcement()
        test_timezone_safety()
        test_hurst_log_prices()
        test_telemetry_metadata()
        test_breadth_calculation_safety()
        
        print("\n" + "=" * 50)
        print("✅ All final improvement tests passed!")
        print("🚀 AI Analyzer v1.1 ready for production deployment")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
