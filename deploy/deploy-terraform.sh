#!/bin/bash

# Automated Terraform Deployment for Railway
echo "🚀 AI Trading Platform - Terraform Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform not found. Installing..."
        
        # Install Terraform
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt update && sudo apt install terraform
        
        if [ $? -ne 0 ]; then
            print_error "Failed to install Terraform"
            return 1
        fi
    fi
    
    print_status "Terraform found: $(terraform version | head -n1)"
    
    # Check Railway CLI
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not found. Installing..."
        npm install -g @railway/cli
        
        if [ $? -ne 0 ]; then
            print_error "Failed to install Railway CLI"
            return 1
        fi
    fi
    
    print_status "Railway CLI found"
    return 0
}

# Setup Railway authentication
setup_railway_auth() {
    print_info "Setting up Railway authentication..."
    
    # Check if already logged in
    if railway whoami &> /dev/null; then
        print_status "Already logged into Railway"
        return 0
    fi
    
    # Login to Railway
    print_info "Please login to Railway..."
    railway login
    
    if [ $? -eq 0 ]; then
        print_status "Railway login successful"
        
        # Get Railway token for Terraform
        RAILWAY_TOKEN=$(railway auth | grep -o 'token: .*' | cut -d' ' -f2)
        export RAILWAY_TOKEN
        
        print_status "Railway token configured for Terraform"
        return 0
    else
        print_error "Railway login failed"
        return 1
    fi
}

# Generate secure secrets
generate_secrets() {
    print_info "Generating secure secrets..."
    
    # Generate AUTH_SECRET_KEY
    AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")
    print_status "Generated AUTH_SECRET_KEY"
    
    echo "🔐 Generated Secrets:"
    echo "AUTH_SECRET_KEY: $AUTH_SECRET"
    echo ""
}

# Create terraform.tfvars
create_tfvars() {
    print_info "Creating terraform configuration..."
    
    cd terraform
    
    if [ ! -f terraform.tfvars ]; then
        echo ""
        print_warning "Creating terraform.tfvars file..."
        echo "Please provide the following information:"
        echo ""
        
        # Get user inputs
        read -p "Admin Username (default: admin): " ADMIN_USER
        ADMIN_USER=${ADMIN_USER:-admin}
        
        read -s -p "Admin Password (min 8 chars): " ADMIN_PASS
        echo ""
        
        read -p "Groq API Key: " GROQ_KEY
        
        read -p "Custom domain (optional, press Enter to skip): " CUSTOM_DOMAIN
        
        # Create tfvars file
        cat > terraform.tfvars << EOF
# AI Trading Platform Terraform Configuration
auth_secret_key = "$AUTH_SECRET"
admin_username = "$ADMIN_USER"
admin_password = "$ADMIN_PASS"
groq_api_key = "$GROQ_KEY"
custom_domain = "$CUSTOM_DOMAIN"
project_name = "ai-trading-platform"
github_repo = "gouravsinha1405/ai-trading-intelligence"
branch = "master"
EOF
        
        print_status "terraform.tfvars created"
    else
        print_status "terraform.tfvars already exists"
    fi
}

# Deploy with Terraform
deploy_terraform() {
    print_info "Deploying with Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    print_info "Initializing Terraform..."
    terraform init
    
    if [ $? -ne 0 ]; then
        print_error "Terraform init failed"
        return 1
    fi
    
    # Plan deployment
    print_info "Planning deployment..."
    terraform plan
    
    if [ $? -ne 0 ]; then
        print_error "Terraform plan failed"
        return 1
    fi
    
    # Apply deployment
    print_info "Applying deployment..."
    read -p "Do you want to proceed with deployment? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        terraform apply -auto-approve
        
        if [ $? -eq 0 ]; then
            print_status "Deployment successful!"
            
            # Get deployment URL
            APP_URL=$(terraform output -raw app_url 2>/dev/null || echo "Check Railway dashboard")
            CUSTOM_URL=$(terraform output -raw custom_domain_url 2>/dev/null || echo "None")
            
            echo ""
            echo "🎉 DEPLOYMENT COMPLETE!"
            echo "======================"
            echo ""
            print_status "Your AI Trading Platform is now live!"
            echo "App URL: $APP_URL"
            echo "Custom Domain: $CUSTOM_URL"
            echo ""
            echo "📋 Next Steps:"
            echo "1. Visit your app URL"
            echo "2. Login with your admin credentials"
            echo "3. Test all features"
            echo "4. Create additional users if needed"
            echo ""
            
        else
            print_error "Terraform apply failed"
            return 1
        fi
    else
        print_info "Deployment cancelled"
        return 0
    fi
}

# Cleanup function
cleanup() {
    print_info "To destroy the deployment later, run:"
    echo "cd terraform && terraform destroy"
}

# Main deployment process
main() {
    echo ""
    print_info "Starting automated Terraform deployment..."
    echo ""
    
    # Step 1: Check prerequisites
    if ! check_prerequisites; then
        print_error "Prerequisites check failed"
        exit 1
    fi
    
    # Step 2: Setup Railway auth
    if ! setup_railway_auth; then
        print_error "Railway authentication failed"
        exit 1
    fi
    
    # Step 3: Generate secrets
    generate_secrets
    
    # Step 4: Create terraform.tfvars
    create_tfvars
    
    # Step 5: Deploy with Terraform
    if deploy_terraform; then
        print_status "Deployment completed successfully!"
        cleanup
    else
        print_error "Deployment failed"
        exit 1
    fi
}

# Run main function
main "$@"
