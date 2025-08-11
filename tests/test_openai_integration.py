"""
Test the OpenAI API integration and configuration management
"""
import pytest
import os
from unittest.mock import patch, MagicMock

def test_config_import():
    """Test that configuration can be imported without errors"""
    try:
        from config.config_template import INITIAL_CAPITAL, MAX_POSITION_SIZE
        assert INITIAL_CAPITAL > 0
        assert 0 < MAX_POSITION_SIZE <= 1
    except ImportError:
        pytest.skip("Config template not available")

def test_openai_client_initialization():
    """Test OpenAI client can be initialized with mock API key"""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-mock-key'}):
        try:
            from openai import OpenAI
            client = OpenAI(api_key='sk-test-mock-key')
            assert client is not None
        except ImportError:
            pytest.skip("OpenAI package not installed")

@patch('openai.OpenAI')
def test_openai_access_function(mock_openai_client, mock_config):
    """Test the main OpenAI access testing function"""
    # Mock the OpenAI client and responses
    mock_client = MagicMock()
    mock_openai_client.return_value = mock_client
    
    # Mock models list response
    mock_models = MagicMock()
    mock_models.data = [
        MagicMock(id='gpt-3.5-turbo'),
        MagicMock(id='gpt-4'),
        MagicMock(id='text-davinci-003')
    ]
    mock_client.models.list.return_value = mock_models
    
    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test joke response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 15
    mock_response.usage.total_tokens = 25
    mock_client.chat.completions.create.return_value = mock_response
    
    # Import and test the function
    try:
        import sys
        sys.path.append('..')
        from test_openai_access import test_openai_access
        
        result = test_openai_access('sk-test-mock-key')
        assert result == True
        
        # Verify the client was called correctly
        mock_client.models.list.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        
    except ImportError:
        pytest.skip("OpenAI access test module not available")

def test_api_key_security():
    """Test that API keys are properly protected"""
    # Check that config.py is in .gitignore
    gitignore_path = os.path.join('..', '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
            assert 'config/config.py' in gitignore_content
            assert '*.key' in gitignore_content
    
    # Check that template exists but actual config might not
    template_path = os.path.join('..', 'config', 'config_template.py')
    config_path = os.path.join('..', 'config', 'config.py')
    
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template_content = f.read()
            assert 'your-openai-api-key-here' in template_content
            
        # If config.py exists, it should not contain placeholder text
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
                assert 'your-openai-api-key-here' not in config_content
