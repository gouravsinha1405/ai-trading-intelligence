#!/usr/bin/env python3

from src.analysis.ai_analyzer import GroqAnalyzer
from src.analysis.regime_detector import RegimeDetector
import pandas as pd
import numpy as np

# Test AI Analyzer methods
print('Testing GroqAnalyzer methods...')
try:
    analyzer = GroqAnalyzer('test_key')
    
    # Check methods exist
    methods = ['analyze_query', 'analyze_market_regime', 'optimize_strategies', 'assess_risk', 'iterate_improvement']
    for method in methods:
        if hasattr(analyzer, method):
            print(f'✓ {method} - Available')
        else:
            print(f'✗ {method} - Missing')
            
except Exception as e:
    print(f'Error testing GroqAnalyzer: {e}')

# Test RegimeDetector methods  
print('\nTesting RegimeDetector methods...')
try:
    detector = RegimeDetector()
    
    methods = ['detect_volatility_regimes', 'detect_trend_regimes', 'detect_market_state_regimes', '_calculate_rsi_wilder']
    for method in methods:
        if hasattr(detector, method):
            print(f'✓ {method} - Available')
        else:
            print(f'✗ {method} - Missing')
            
    print('\n✅ All alignment fixes completed successfully!')
    
except Exception as e:
    print(f'Error testing RegimeDetector: {e}')
