# Algorithmic Trading Platform

## Project Overview
This is a comprehensive algorithmic trading platform built with Streamlit, featuring:
- Real-time market data from jugaad-data (Indian markets)
- AI-powered strategy analysis using Groq API
- Backtesting capabilities
- Virtual trading with paper money
- Multi-timeframe analysis
- News sentiment integration

## Key Features
- **Live Data**: Real-time Indian stock market data via jugaad-data
- **AI Assistant**: Groq-powered market analysis and strategy brainstorming
- **Strategy Builder**: Interactive strategy creation and testing
- **Backtesting**: Historical performance analysis
- **Paper Trading**: Virtual money trading simulation
- **News Analysis**: RSS feeds and sentiment analysis

## Technology Stack
- **Frontend**: Streamlit
- **Data Sources**: jugaad-data, yfinance, Alpha Vantage
- **AI**: Groq API (Llama models)
- **Analysis**: pandas, numpy, scipy, scikit-learn
- **Visualization**: plotly
- **Technical Analysis**: TA-Lib

## Project Structure
- `main.py`: Main Streamlit application
- `pages/`: Streamlit pages for different features
- `src/data/`: Data fetching and processing
- `src/analysis/`: Trading analysis and AI integration
- `src/backtesting/`: Backtesting engine
- `src/trading/`: Virtual trading implementation
- `src/utils/`: Utility functions and configuration

## Development Guidelines
- Use real market data only (no synthetic data)
- Implement proper data cleaning and validation
- Focus on risk-adjusted returns
- Include comprehensive error handling
- Maintain clean, documented code

## API Keys Required
- Groq API key for AI analysis
- Alpha Vantage API key (optional)
- News API key (optional)

## Installation
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables in `.env`
5. Run: `streamlit run main.py`
