# Terraform variables for Railway deployment
# AI Trading Intelligence Platform

# Required: Generate a 32+ character secret key
auth_secret_key = "ai-trading-platform-secret-key-32chars-change-this-in-production"

# Required: Set your admin credentials  
admin_username = "admin"
admin_password = "AdminTrading123!"

# Required: Your Groq API key
groq_api_key = "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"

# Optional: Additional API keys
alpha_vantage_api_key = ""
news_api_key = ""

# Optional: Custom domain (leave empty for Railway subdomain)
custom_domain = ""

# Project configuration
project_name = "ai-trading-platform"
github_repo = "gouravsinha1405/ai-trading-intelligence"
branch = "master"
