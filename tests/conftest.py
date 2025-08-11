"""
Test configuration and utilities for the quantitative trading platform
"""
import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock configuration for testing
class MockConfig:
    OPENAI_API_KEY = "sk-test-mock-key"
    INITIAL_CAPITAL = 1000000
    MAX_POSITION_SIZE = 0.25
    RISK_FREE_RATE = 0.06

@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()

@pytest.fixture
def sample_price_data():
    """Provide sample price data for testing strategies"""
    import pandas as pd
    import numpy as np
    
    # Generate synthetic price data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate multiple assets with realistic price movements
    assets = ['FUND_A', 'FUND_B', 'FUND_C']
    price_data = {}
    
    for asset in assets:
        # Random walk with drift (realistic price behavior)
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Convert to prices
        price_data[asset] = prices
    
    return pd.DataFrame(price_data, index=dates)

@pytest.fixture
def sample_returns_data(sample_price_data):
    """Provide sample returns data for testing"""
    return sample_price_data.pct_change().dropna()
