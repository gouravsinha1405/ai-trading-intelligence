#!/bin/bash

# AI Trading Platform Deployment Script
echo "🚀 AI Trading Platform - Docker Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << EOF
# AI Trading Platform Configuration
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
STREAMLIT_SHARING=false
LOG_LEVEL=WARNING
EOF
    echo "📝 Created .env template. Please edit it with your API keys."
    echo "   nano .env"
    echo ""
fi

# Build and run the application
echo "🔨 Building Docker image..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    echo ""
    echo "🚀 Starting the application..."
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo "✅ Application started successfully!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   http://localhost:8501"
        echo ""
        echo "📊 To view logs:"
        echo "   docker-compose logs -f"
        echo ""
        echo "🛑 To stop the application:"
        echo "   docker-compose down"
        echo ""
        echo "🔄 To restart:"
        echo "   docker-compose restart"
    else
        echo "❌ Failed to start the application"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi
