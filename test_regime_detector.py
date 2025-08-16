#!/usr/bin/env python3
"""
Test script for the improved regime detector
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.regime_detector import RegimeDetector

def create_test_data(n_periods=1000):
    """Create synthetic test data with different market regimes"""
    np.random.seed(42)
    
    # Create date index
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Simulate different market regimes
    regime_changes = [0, 250, 500, 750, n_periods]
    regime_types = ['bull', 'bear', 'sideways', 'volatile']
    
    prices = []
    volumes = []
    current_price = 100
    
    for i in range(len(regime_changes) - 1):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i + 1]
        regime_length = end_idx - start_idx
        regime_type = regime_types[i]
        
        if regime_type == 'bull':
            # Upward trend, low volatility
            returns = np.random.normal(0.001, 0.015, regime_length)
            trend = np.linspace(0, 0.3, regime_length)
            returns += trend / regime_length
            volume_base = np.random.normal(1000000, 200000, regime_length)
        elif regime_type == 'bear':
            # Downward trend, medium volatility
            returns = np.random.normal(-0.001, 0.025, regime_length)
            trend = np.linspace(0, -0.2, regime_length)
            returns += trend / regime_length
            volume_base = np.random.normal(1200000, 300000, regime_length)
        elif regime_type == 'sideways':
            # No trend, low volatility
            returns = np.random.normal(0, 0.01, regime_length)
            volume_base = np.random.normal(800000, 150000, regime_length)
        else:  # volatile
            # High volatility, no clear trend
            returns = np.random.normal(0, 0.04, regime_length)
            volume_base = np.random.normal(1500000, 500000, regime_length)
        
        # Generate prices
        for j in range(regime_length):
            current_price *= (1 + returns[j])
            prices.append(current_price)
            volumes.append(max(int(volume_base[j]), 100000))
    
    # Create OHLC data
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.array(prices) * (1 + np.random.normal(0, 0.002, n_periods)),
        'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'Close': prices,
        'Volume': volumes
    })
    
    # Ensure OHLC consistency
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data.set_index('Date')

def test_regime_detector():
    """Test the improved regime detector"""
    print("🧪 Testing Improved Regime Detector")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating test data...")
    test_data = create_test_data(1000)
    print(f"✅ Created {len(test_data)} days of test data")
    
    # Initialize detector
    detector = RegimeDetector()
    
    # Test volatility regimes with walk-forward
    print("\n🔄 Testing walk-forward volatility regimes...")
    try:
        vol_results = detector.detect_volatility_regimes(
            test_data, 
            window=30, 
            n_regimes=3, 
            use_walkforward=True,
            train_window=300,
            test_window=30
        )
        print("✅ Walk-forward volatility regimes detected successfully")
        vol_regimes = vol_results['Volatility_Regime'].value_counts()
        print(f"   Regime distribution: {dict(vol_regimes)}")
    except Exception as e:
        print(f"❌ Error in volatility regimes: {e}")
    
    # Test improved trend regimes
    print("\n📈 Testing improved trend regimes...")
    try:
        trend_results = detector.detect_trend_regimes(
            test_data,
            ema_period=200,
            lookback=20,
            r2_window=80,
            r2_threshold=0.30
        )
        print("✅ Improved trend regimes detected successfully")
        trend_regimes = trend_results['Trend_Regime'].value_counts()
        print(f"   Regime distribution: {dict(trend_regimes)}")
    except Exception as e:
        print(f"❌ Error in trend regimes: {e}")
    
    # Test market state regimes with GMM
    print("\n🎯 Testing market state regimes with GMM...")
    try:
        market_results = detector.detect_market_state_regimes(
            test_data,
            window=30,
            use_gmm=True,
            train_window=300,
            test_window=30
        )
        print("✅ GMM market state regimes detected successfully")
        market_regimes = market_results['Market_State_Regime'].value_counts()
        print(f"   Regime distribution: {dict(market_regimes)}")
        
        # Check if probabilities were added
        prob_cols = [col for col in market_results.columns if '_Prob' in col]
        if prob_cols:
            print(f"   Probability columns added: {prob_cols}")
    except Exception as e:
        print(f"❌ Error in market state regimes: {e}")
    
    # Test RSI calculation
    print("\n📊 Testing Wilder's RSI calculation...")
    try:
        rsi_wilder = detector._calculate_rsi_wilder(test_data, 14)
        rsi_legacy = detector._calculate_rsi(test_data, 14)
        
        print("✅ RSI calculations completed")
        print(f"   Wilder's RSI range: {rsi_wilder.min():.2f} - {rsi_wilder.max():.2f}")
        print(f"   Legacy RSI range: {rsi_legacy.min():.2f} - {rsi_legacy.max():.2f}")
        
        # Check for differences
        diff = np.abs(rsi_wilder - rsi_legacy).mean()
        print(f"   Average difference: {diff:.4f}")
    except Exception as e:
        print(f"❌ Error in RSI calculation: {e}")
    
    # Test regime probability prediction
    print("\n🎲 Testing regime probability prediction...")
    try:
        if 'Market_State_Regime' in market_results.columns:
            current_data = test_data.tail(50)
            probabilities = detector.predict_regime_probability(
                current_data,
                market_results,
                'Market_State_Regime',
                use_full_features=True
            )
            print("✅ Regime probabilities calculated successfully")
            print("   Current regime probabilities:")
            for regime, prob in probabilities.items():
                print(f"     {regime}: {prob:.3f}")
    except Exception as e:
        print(f"❌ Error in probability prediction: {e}")
    
    print("\n🎉 Regime detector testing completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_regime_detector()
