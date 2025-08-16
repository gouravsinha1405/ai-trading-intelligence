# 🚀 Algorithmic Trading Platform

A comprehensive algorithmic trading platform built with Streamlit, featuring real-time Indian market data, AI-powered analysis, and virtual trading capabilities.

## ✨ Features

- **📊 Real-time Market Data**: Live Indian stock market data via jugaad-data
- **🤖 AI Assistant**: Groq-powered market analysis and strategy brainstorming  
- **🔧 Strategy Builder**: Interactive strategy creation and testing
- **📈 Live Trading**: Virtual money trading simulation
- **🔄 Backtesting**: Historical performance analysis
- **📰 News Analysis**: RSS feeds and sentiment analysis
- **📋 Dashboard**: Portfolio tracking and performance metrics

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Sources**: jugaad-data, yfinance, Alpha Vantage
- **AI**: Groq API (Llama models)
- **Analysis**: pandas, numpy, scipy, scikit-learn
- **Visualization**: plotly
- **Technical Analysis**: TA-Lib

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd algo-trading-app
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
VIRTUAL_MONEY_AMOUNT=100000
DEFAULT_COMMISSION=0.1
```

### 5. Run the Application
```bash
streamlit run main.py
```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
algo-trading-app/
├── main.py                    # Main Streamlit application
├── pages/                     # Streamlit pages
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔧_Strategy_Builder.py
│   ├── 3_📈_Live_Trading.py
│   ├── 4_📰_News_Analysis.py
│   ├── 5_🔄_Backtesting.py
│   └── 6_🤖_AI_Assistant.py
├── src/                       # Source code
│   ├── data/                  # Data fetching and processing
│   │   ├── jugaad_client.py
│   │   ├── data_cleaner.py
│   │   └── news_feeds.py
│   ├── analysis/              # Trading analysis and AI
│   │   ├── ai_analyzer.py
│   │   ├── regime_detector.py
│   │   ├── indicators.py
│   │   └── signals.py
│   ├── backtesting/           # Backtesting engine
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── portfolio.py
│   ├── trading/               # Virtual trading
│   │   ├── virtual_broker.py
│   │   └── order_manager.py
│   └── utils/                 # Utilities
│       ├── config.py
│       └── helpers.py
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                  # This file
```

## 🔧 Configuration

### API Keys Required

1. **Groq API Key** (Required for AI features)
   - Sign up at [Groq Console](https://console.groq.com/)
   - Create an API key
   - Add to `.env` file

2. **Alpha Vantage API Key** (Optional - for additional data)
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/)
   - Get free API key (500 calls/day)

3. **News API Key** (Optional - for news analysis)
   - Sign up at [NewsAPI](https://newsapi.org/)
   - Get free API key (1000 requests/day)

### Groq Models

The platform supports multiple Groq models:
- **llama-3.3-70b-versatile**: Best for complex analysis (default)
- **llama-3.1-8b-instant**: Fast responses for real-time analysis
- **deepseek-r1-distill-llama-70b**: Advanced reasoning capabilities
- **meta-llama/llama-4-scout-17b-16e-instruct**: Latest Llama 4 model
- **qwen/qwen3-32b**: High-performance alternative model

## 🎯 Usage

### 1. Dashboard
- View portfolio performance and metrics
- Monitor active strategies
- Check market overview
- Review recent trades

### 2. AI Assistant  
- Interactive strategy brainstorming
- Market regime analysis
- Risk assessment
- Strategy optimization suggestions

### 3. Strategy Builder
- Create custom trading strategies
- Define entry/exit rules
- Set risk parameters
- Test strategies

### 4. Live Trading
- Execute trades with virtual money
- Monitor real-time positions
- Track P&L
- Manage orders

### 5. Backtesting
- Test strategies on historical data
- Analyze performance metrics
- Compare multiple strategies
- Optimize parameters

### 6. News Analysis
- Monitor market news and sentiment
- RSS feed integration
- Google Trends analysis
- Impact assessment

## ⚠️ Important Notes

- **Virtual Trading Only**: This platform uses virtual money for trading simulation
- **Indian Markets**: Primarily focused on NSE/BSE listed securities
- **Real Data Only**: No synthetic or fabricated data is used
- **Rate Limits**: Respect API rate limits for data sources
- **Educational Purpose**: For learning and strategy development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Review existing issues
- Create a new issue with detailed description

## 🔮 Future Enhancements

- [ ] Multi-asset support (crypto, forex)
- [ ] Advanced ML models for prediction
- [ ] Social sentiment analysis
- [ ] Real broker integration
- [ ] Mobile app
- [ ] Advanced risk analytics

---

**⚡ Happy Trading! ⚡**
