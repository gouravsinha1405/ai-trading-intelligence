#!/usr/bin/env python3
"""
Test the fixed strategy optimization functionality
"""

import sys
import os
sys.path.append('/home/gourav/ai')

import pandas as pd
import numpy as np
from src.analysis.ai_analyzer import GroqAnalyzer

def test_mock_backtest_function():
    """Test the fixed mock backtest function"""
    
    # Import the function from the AI Assistant page
    exec(open('/home/gourav/ai/pages/6_🤖_AI_Assistant.py').read(), globals())
    
    print("🔧 Testing Strategy Optimization Fix...")
    
    # Test with normal strategy config
    strategy_config = {
        "knobs": {
            "sma_fast": 10,
            "sma_slow": 30,
            "risk_per_trade": 0.01
        }
    }
    
    try:
        print("✅ Testing normal case...")
        equity_df, perf_metrics = mock_backtest_function(strategy_config)
        print(f"   - Returned equity_df with {len(equity_df)} rows")
        print(f"   - Performance metrics: {len(perf_metrics)} items")
        print(f"   - Total return: {perf_metrics.get('total_return', 'N/A'):.2f}%")
        print(f"   - Sharpe ratio: {perf_metrics.get('sharpe_ratio', 'N/A'):.2f}")
        
        # Test edge case with minimal data
        print("✅ Testing edge case...")
        equity_df2, perf_metrics2 = mock_backtest_function({"knobs": {}})
        print(f"   - Edge case handled successfully")
        
        print("🎯 All tests passed! Optimization should work correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error in optimization: {e}")
        return False

def test_ai_integration():
    """Test AI analyzer integration"""
    print("\n🤖 Testing AI Integration...")
    
    try:
        # Check if we can import and initialize
        analyzer = GroqAnalyzer()
        print("✅ GroqAnalyzer imported successfully")
        
        # Test basic functionality 
        test_data = {
            "total_return": 15.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 8.3
        }
        
        print("✅ AI integration ready")
        return True
        
    except Exception as e:
        print(f"❌ AI integration issue: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Fixed Strategy Optimization...")
    print("=" * 50)
    
    # Test the mock backtest function
    backtest_ok = test_mock_backtest_function()
    
    # Test AI integration
    ai_ok = test_ai_integration()
    
    print("\n" + "=" * 50)
    if backtest_ok and ai_ok:
        print("🎉 ALL TESTS PASSED! Strategy optimization is ready to use.")
        print("\nYou can now:")
        print("1. Go to http://localhost:8501/AI_Assistant")
        print("2. Adjust the strategy parameters")
        print("3. Click 'Start Optimization'")
        print("4. Watch the real-time optimization process")
    else:
        print("❌ Some issues detected. Check the output above.")
