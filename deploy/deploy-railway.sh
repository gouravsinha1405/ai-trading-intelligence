#!/bin/bash

# Railway Deployment Script for AI Trading Platform
echo "🚀 AI Trading Platform - Railway Deployment Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Railway CLI is installed
check_railway_cli() {
    if command -v railway &> /dev/null; then
        print_status "Railway CLI is already installed"
        return 0
    else
        print_warning "Railway CLI not found. Installing..."
        
        # Check if npm is available
        if command -v npm &> /dev/null; then
            npm install -g @railway/cli
            return $?
        else
            print_error "npm not found. Please install Node.js first or use manual installation:"
            echo "curl -fsSL https://railway.app/install.sh | sh"
            return 1
        fi
    fi
}

# Generate secure secrets
generate_secrets() {
    print_info "Generating secure authentication secrets..."
    
    # Generate a secure AUTH_SECRET_KEY (32 characters)
    AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
    
    echo ""
    echo "🔐 SECURITY CONFIGURATION"
    echo "========================="
    echo ""
    echo "Generated AUTH_SECRET_KEY: $AUTH_SECRET"
    echo ""
    print_warning "IMPORTANT: Save these credentials securely!"
    echo ""
    echo "Recommended Railway Environment Variables:"
    echo "----------------------------------------"
    echo "AUTH_SECRET_KEY=$AUTH_SECRET"
    echo "ADMIN_USERNAME=youradmin"
    echo "ADMIN_PASSWORD=YourSecurePassword123!"
    echo "GROQ_API_KEY=your_groq_api_key_here"
    echo ""
}

# Test local deployment
test_local() {
    print_info "Testing local deployment with Docker..."
    
    if command -v docker &> /dev/null; then
        # Build Docker image
        print_info "Building Docker image..."
        docker build -t ai-trading-platform-railway .
        
        if [ $? -eq 0 ]; then
            print_status "Docker image built successfully"
            
            # Test container
            print_info "Testing container (will run for 10 seconds)..."
            docker run -d --name test-ai-trading -p 8501:8501 \
                -e AUTH_SECRET_KEY="test-secret-key-32-characters-long" \
                -e ADMIN_USERNAME="admin" \
                -e ADMIN_PASSWORD="admin123" \
                -e GROQ_API_KEY="test-key" \
                ai-trading-platform-railway
            
            sleep 10
            
            # Check if container is running
            if docker ps | grep -q test-ai-trading; then
                print_status "Container is running successfully"
                print_info "Test URL: http://localhost:8501"
            else
                print_error "Container failed to start"
                docker logs test-ai-trading
            fi
            
            # Cleanup
            docker stop test-ai-trading 2>/dev/null
            docker rm test-ai-trading 2>/dev/null
            
        else
            print_error "Docker build failed"
            return 1
        fi
    else
        print_warning "Docker not found. Skipping local test."
        print_info "Install Docker to test locally before deploying"
    fi
}

# Initialize Railway project
init_railway() {
    print_info "Initializing Railway project..."
    
    if [ ! -f "railway.toml" ]; then
        print_error "railway.toml not found. Creating default configuration..."
        
        cat > railway.toml << EOF
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run main.py --server.port \$PORT --server.address 0.0.0.0"
healthcheckPath = "/"
healthcheckTimeout = 60
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
EOF
    fi
    
    # Login to Railway
    print_info "Logging into Railway..."
    railway login
    
    if [ $? -eq 0 ]; then
        print_status "Railway login successful"
        
        # Initialize project
        railway init
        
        if [ $? -eq 0 ]; then
            print_status "Railway project initialized"
            return 0
        else
            print_error "Failed to initialize Railway project"
            return 1
        fi
    else
        print_error "Railway login failed"
        return 1
    fi
}

# Set environment variables
set_variables() {
    print_info "Setting up environment variables..."
    
    echo ""
    echo "🔧 ENVIRONMENT VARIABLE SETUP"
    echo "============================="
    echo ""
    
    read -p "Enter your Groq API key: " GROQ_KEY
    read -p "Enter admin username (default: admin): " ADMIN_USER
    ADMIN_USER=${ADMIN_USER:-admin}
    read -s -p "Enter admin password: " ADMIN_PASS
    echo ""
    
    # Generate secret key
    AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
    
    print_info "Setting Railway environment variables..."
    
    railway variables set AUTH_SECRET_KEY="$AUTH_SECRET"
    railway variables set ADMIN_USERNAME="$ADMIN_USER"
    railway variables set ADMIN_PASSWORD="$ADMIN_PASS"
    railway variables set GROQ_API_KEY="$GROQ_KEY"
    railway variables set VIRTUAL_MONEY_AMOUNT="100000"
    railway variables set DEFAULT_COMMISSION="0.1"
    railway variables set LOG_LEVEL="INFO"
    railway variables set DEBUG="false"
    
    print_status "Environment variables configured"
}

# Deploy to Railway
deploy() {
    print_info "Deploying to Railway..."
    
    railway up
    
    if [ $? -eq 0 ]; then
        print_status "Deployment successful!"
        
        # Get the deployment URL
        URL=$(railway status --json | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('deployments', [{}])[0].get('url', 'Check Railway dashboard'))
except:
    print('Check Railway dashboard')
")
        
        echo ""
        echo "🎉 DEPLOYMENT COMPLETE!"
        echo "======================"
        echo ""
        print_status "Your AI Trading Platform is now live!"
        echo "URL: $URL"
        echo ""
        echo "📋 Next Steps:"
        echo "1. Visit your app URL"
        echo "2. Login with your admin credentials"
        echo "3. Change default passwords immediately"
        echo "4. Create additional user accounts"
        echo "5. Test all features"
        echo ""
        
    else
        print_error "Deployment failed"
        echo ""
        echo "🔧 Troubleshooting:"
        echo "1. Check Railway logs: railway logs"
        echo "2. Verify environment variables: railway variables"
        echo "3. Check build status in Railway dashboard"
        return 1
    fi
}

# Main deployment flow
main() {
    echo ""
    print_info "Starting Railway deployment process..."
    echo ""
    
    # Step 1: Check Railway CLI
    if ! check_railway_cli; then
        print_error "Please install Railway CLI and try again"
        exit 1
    fi
    
    # Step 2: Generate secrets for reference
    generate_secrets
    
    # Step 3: Test local deployment (optional)
    read -p "Do you want to test locally with Docker first? (y/N): " test_choice
    if [[ $test_choice =~ ^[Yy]$ ]]; then
        test_local
    fi
    
    # Step 4: Initialize Railway
    if ! init_railway; then
        print_error "Railway initialization failed"
        exit 1
    fi
    
    # Step 5: Set environment variables
    set_variables
    
    # Step 6: Deploy
    deploy
    
    echo ""
    print_status "Railway deployment process complete!"
    echo ""
    print_info "For more information, see RAILWAY_DEPLOYMENT.md"
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
