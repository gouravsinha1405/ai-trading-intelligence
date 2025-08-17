#!/bin/bash

# ========================================
# Railway Terraform Deployment Script
# AI Trading Intelligence Platform
# ========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  AI Trading Platform - Railway Deploy   "
    echo "=========================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if required tools are installed
check_dependencies() {
    print_step "Checking dependencies..."
    
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install it first:"
        echo "  https://learn.hashicorp.com/tutorials/terraform/install-cli"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install it first."
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Check if Railway CLI is installed
check_railway_cli() {
    if ! command -v railway &> /dev/null; then
        print_info "Railway CLI not found. Installing..."
        
        # Install Railway CLI
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -fsSL https://railway.app/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # Mac OS
            curl -fsSL https://railway.app/install.sh | sh
        else
            print_error "Unsupported OS. Please install Railway CLI manually:"
            echo "  https://docs.railway.app/quick-start"
            exit 1
        fi
        
        # Add to PATH
        export PATH="$HOME/.railway/bin:$PATH"
        
        print_success "Railway CLI installed"
    else
        print_success "Railway CLI found"
    fi
}

# Get Railway token
setup_railway_auth() {
    print_step "Setting up Railway authentication..."
    
    if [ -z "$RAILWAY_TOKEN" ]; then
        print_info "Railway token not found in environment."
        echo "Please follow these steps:"
        echo "1. Go to https://railway.app/account/tokens"
        echo "2. Create a new token"
        echo "3. Run: export RAILWAY_TOKEN=your_token_here"
        echo "4. Or add it to your ~/.bashrc or ~/.zshrc"
        echo ""
        read -p "Enter your Railway token: " RAILWAY_TOKEN
        export RAILWAY_TOKEN
    fi
    
    # Test authentication
    if railway auth &> /dev/null; then
        print_success "Railway authentication successful"
    else
        print_error "Railway authentication failed. Please check your token."
        exit 1
    fi
}

# Create terraform.tfvars if it doesn't exist
setup_terraform_vars() {
    print_step "Setting up Terraform variables..."
    
    cd terraform
    
    if [ ! -f "terraform.tfvars" ]; then
        print_info "terraform.tfvars not found. Creating from example..."
        cp terraform.tfvars.example terraform.tfvars
        
        echo ""
        print_info "Please edit terraform/terraform.tfvars with your actual values:"
        echo "  - auth_secret_key: Generate a 32+ character secret"
        echo "  - admin_username: Your admin username"
        echo "  - admin_password: Your secure admin password"
        echo "  - groq_api_key: Your Groq API key"
        echo ""
        
        read -p "Press Enter after editing terraform.tfvars..." 
    fi
    
    print_success "Terraform variables ready"
}

# Validate Terraform configuration
validate_terraform() {
    print_step "Validating Terraform configuration..."
    
    cd terraform
    
    terraform init
    terraform validate
    
    print_success "Terraform configuration is valid"
}

# Plan deployment
plan_deployment() {
    print_step "Planning deployment..."
    
    cd terraform
    
    terraform plan -out=tfplan
    
    echo ""
    print_info "Terraform plan created. Review the changes above."
    read -p "Do you want to proceed with deployment? (y/N): " confirm
    
    if [[ $confirm != [yY] ]]; then
        print_info "Deployment cancelled."
        exit 0
    fi
}

# Deploy to Railway
deploy() {
    print_step "Deploying to Railway..."
    
    cd terraform
    
    terraform apply tfplan
    
    print_success "Deployment completed!"
}

# Show deployment info
show_deployment_info() {
    print_step "Getting deployment information..."
    
    cd terraform
    
    APP_URL=$(terraform output -raw app_url 2>/dev/null || echo "Not available")
    PROJECT_ID=$(terraform output -raw project_id 2>/dev/null || echo "Not available")
    CUSTOM_DOMAIN=$(terraform output -raw custom_domain_url 2>/dev/null || echo "Not configured")
    
    echo ""
    print_success "🚀 Deployment Information:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📱 App URL:        $APP_URL"
    echo "🆔 Project ID:     $PROJECT_ID"
    echo "🌐 Custom Domain:  $CUSTOM_DOMAIN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🎉 Your AI Trading Platform is now live!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Visit your app URL to test the deployment"
    echo "2. Login with your admin credentials"
    echo "3. Test the AI features with your Groq API key"
    echo "4. Configure any additional settings needed"
    echo ""
    echo "🔧 To manage your deployment:"
    echo "• Railway Dashboard: https://railway.app/dashboard"
    echo "• View logs: railway logs"
    echo "• Update variables: railway variables"
    echo ""
}

# Cleanup function
cleanup() {
    cd terraform
    rm -f tfplan
}

# Main execution
main() {
    # Set trap for cleanup
    trap cleanup EXIT
    
    print_header
    
    # Change to project directory
    cd "$(dirname "$0")"
    
    # Run deployment steps
    check_dependencies
    check_railway_cli
    setup_railway_auth
    setup_terraform_vars
    validate_terraform
    plan_deployment
    deploy
    show_deployment_info
    
    print_success "🎉 Deployment completed successfully!"
}

# Run main function
main "$@"