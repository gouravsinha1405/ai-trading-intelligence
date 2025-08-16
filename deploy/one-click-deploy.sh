#!/bin/bash

# One-Command Railway Deployment
# Usage: ./one-click-deploy.sh [groq_api_key] [admin_password]

echo "🚀 AI Trading Platform - One-Click Deploy"
echo "========================================="

# Get API key and password from command line or prompt
GROQ_KEY=${1:-""}
ADMIN_PASS=${2:-""}

if [ -z "$GROQ_KEY" ]; then
    read -p "Enter your Groq API key: " GROQ_KEY
fi

if [ -z "$ADMIN_PASS" ]; then
    read -s -p "Enter admin password: " ADMIN_PASS
    echo ""
fi

# Generate secure secret
AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")

echo "🔧 Installing Railway CLI..."
npm install -g @railway/cli &>/dev/null

echo "🔐 Logging into Railway..."
railway login

echo "🚀 Creating and deploying project..."
railway init --name ai-trading-platform

echo "⚙️ Setting environment variables..."
railway variables set AUTH_SECRET_KEY="$AUTH_SECRET"
railway variables set ADMIN_USERNAME="admin"
railway variables set ADMIN_PASSWORD="$ADMIN_PASS"
railway variables set GROQ_API_KEY="$GROQ_KEY"
railway variables set VIRTUAL_MONEY_AMOUNT="100000"
railway variables set DEFAULT_COMMISSION="0.1"
railway variables set LOG_LEVEL="INFO"

echo "🚀 Deploying application..."
railway up

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "✅ Your AI Trading Platform is now live!"
echo "📱 URL: $(railway status --json | python3 -c "import json,sys; print(json.load(sys.stdin).get('deployments',[{}])[0].get('url','Check Railway dashboard'))" 2>/dev/null || echo "Check Railway dashboard")"
echo "👤 Admin Login: admin / $ADMIN_PASS"
echo ""
echo "🔧 To manage: railway dashboard"
echo "📊 To check logs: railway logs"
