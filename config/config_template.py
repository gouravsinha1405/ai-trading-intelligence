# Configuration template - copy to config.py and fill in your values
# DO NOT commit config.py to version control!

# OpenAI API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"

# Data Source API Keys
ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-key"
QUANDL_API_KEY = "your-quandl-key"

# Database Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "trading_db"
DB_USER = "your-db-user"
DB_PASSWORD = "your-db-password"

# Trading Configuration
INITIAL_CAPITAL = 1000000  # ₹10L starting capital
MAX_POSITION_SIZE = 0.25   # 25% max per position
RISK_FREE_RATE = 0.06      # 6% risk-free rate
TRANSACTION_COST = 0.001   # 0.1% transaction cost

# Backtesting Configuration
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
BENCHMARK = "^NSEI"  # Nifty 50 as benchmark
