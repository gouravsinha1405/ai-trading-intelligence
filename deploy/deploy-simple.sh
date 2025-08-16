#!/bin/bash

# AI Trading Platform - Simple Deployment (No Docker Required)
echo "🚀 AI Trading Platform - Simple Deployment"
echo "========================================="

# Check if Python 3.11+ is available
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.10 python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$VERSION >= 3.10" | bc -l 2>/dev/null || echo "1") == "1" ]]; then
            PYTHON_CMD=$cmd
            echo "✅ Found Python $VERSION at $cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.10+ not found. Please install Python 3.10 or newer."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️ No .env file found. Creating template..."
    cat > .env << EOF
# AI Trading Platform Configuration
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
STREAMLIT_SHARING=false
LOG_LEVEL=WARNING

# Trading Configuration
VIRTUAL_MONEY_AMOUNT=100000
DEFAULT_COMMISSION=0.1
EOF
    echo "📝 Created .env template. Please edit it with your API keys."
    echo "   nano .env"
    echo ""
fi

# Test the application
echo "🧪 Testing application..."
python -c "
import sys
sys.path.append('src')
from src.utils.config import load_config
print('✅ Configuration test passed')
"

if [ $? -eq 0 ]; then
    echo "✅ Application test successful!"
    echo ""
    echo "🚀 Starting the application..."
    echo "   Access at: http://localhost:8501"
    echo ""
    echo "🛑 To stop: Press Ctrl+C"
    echo ""
    
    # Start Streamlit
    streamlit run main.py --server.port=8501 --server.address=0.0.0.0
else
    echo "❌ Application test failed"
    exit 1
fi
