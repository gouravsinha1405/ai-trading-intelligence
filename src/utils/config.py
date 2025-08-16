import os
from pathlib import Path

from dotenv import load_dotenv, set_key


def get_env_path():
    """Get the path to the .env file"""
    return Path(__file__).parent.parent.parent / ".env"


def load_config():
    """Load configuration from environment variables"""

    # Load environment variables from .env file
    env_path = get_env_path()
    load_dotenv(env_path)

    config = {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "debug": os.getenv("DEBUG", "False").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "virtual_money": float(os.getenv("VIRTUAL_MONEY_AMOUNT", 100000)),
        "commission": float(os.getenv("DEFAULT_COMMISSION", 0.1)),
    }

    return config


def save_api_key(api_key):
    """Save API key to .env file and reload environment"""
    try:
        env_path = get_env_path()

        # Use python-dotenv's set_key for safe writing
        set_key(env_path, "GROQ_API_KEY", api_key)

        # Reload the environment
        load_dotenv(env_path, override=True)

        return True, "API key saved successfully!"
    except Exception as e:
        return False, f"Error saving API key: {str(e)}"


def validate_config(config):
    """Validate that required configuration is present"""
    required_keys = ["groq_api_key"]

    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required configuration: {key}")

    return True
