"""
Test configuration and fixtures for AI Trading Platform
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing"""
    with patch('streamlit.session_state', {}):
        with patch('streamlit.secrets', {'GROQ_API_KEY': 'test_key'}):
            yield

@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager"""
    from src.auth.auth_manager import AuthManager
    
    with patch.object(AuthManager, '_load_users') as mock_load:
        mock_load.return_value = {
            'admin': {
                'password_hash': 'hashed_password',
                'role': 'admin',
                'created_at': '2024-01-01'
            },
            'user': {
                'password_hash': 'hashed_password',
                'role': 'user',
                'created_at': '2024-01-01'
            }
        }
        
        auth_manager = AuthManager()
        yield auth_manager

@pytest.fixture
def mock_data_client():
    """Mock data client for testing"""
    from src.data.jugaad_client import JugaadClient
    
    with patch.object(JugaadClient, 'get_historical_data') as mock_data:
        mock_data.return_value = Mock()
        
        client = JugaadClient()
        yield client

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 102 + np.random.randn(100).cumsum(),
        'Low': 98 + np.random.randn(100).cumsum(),
        'Close': 101 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    return data

@pytest.fixture
def test_environment():
    """Set up test environment variables"""
    test_env = {
        'GROQ_API_KEY': 'test_groq_key',
        'ALPHA_VANTAGE_API_KEY': 'test_alpha_key',
        'NEWS_API_KEY': 'test_news_key',
        'STREAMLIT_ENV': 'test'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env
