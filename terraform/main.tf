# Terraform configuration for Railway deployment
# This will create and deploy your AI Trading Platform automatically

terraform {
  required_providers {
    railway = {
      source  = "railway-app/railway"
      version = "~> 0.3.0"
    }
  }
}

# Configure Railway provider
provider "railway" {
  # Railway token will be set via RAILWAY_TOKEN environment variable
}

# Create Railway project
resource "railway_project" "ai_trading_platform" {
  name        = "ai-trading-platform"
  description = "AI Trading Intelligence Platform with Authentication"
  
  # Make project public (optional)
  is_public = false
}

# Create service for the main application
resource "railway_service" "app" {
  project_id = railway_project.ai_trading_platform.id
  name       = "ai-trading-app"
  
  # Connect to GitHub repository
  source {
    repo   = "gouravsinha1405/ai-trading-intelligence"
    branch = "master"
  }
}

# Set environment variables
resource "railway_variable" "auth_secret_key" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "AUTH_SECRET_KEY"
  value        = var.auth_secret_key
}

resource "railway_variable" "admin_username" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "ADMIN_USERNAME"
  value        = var.admin_username
}

resource "railway_variable" "admin_password" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "ADMIN_PASSWORD"
  value        = var.admin_password
  sensitive    = true
}

resource "railway_variable" "groq_api_key" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "GROQ_API_KEY"
  value        = var.groq_api_key
  sensitive    = true
}

resource "railway_variable" "virtual_money" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "VIRTUAL_MONEY_AMOUNT"
  value        = "100000"
}

resource "railway_variable" "commission" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "DEFAULT_COMMISSION"
  value        = "0.1"
}

resource "railway_variable" "log_level" {
  project_id   = railway_project.ai_trading_platform.id
  environment_id = railway_project.ai_trading_platform.default_environment.id
  name         = "LOG_LEVEL"
  value        = "INFO"
}

# Create custom domain (optional)
resource "railway_custom_domain" "app_domain" {
  count      = var.custom_domain != "" ? 1 : 0
  project_id = railway_project.ai_trading_platform.id
  service_id = railway_service.app.id
  domain     = var.custom_domain
}

# Output the deployment URL
output "app_url" {
  description = "The URL where your AI Trading Platform is deployed"
  value       = railway_service.app.url
}

output "project_id" {
  description = "Railway project ID"
  value       = railway_project.ai_trading_platform.id
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.custom_domain != "" ? "https://${var.custom_domain}" : "Not configured"
}
