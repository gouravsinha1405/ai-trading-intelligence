# Input variables for Terraform deployment

variable "auth_secret_key" {
  description = "Secret key for authentication (32+ characters)"
  type        = string
  sensitive   = true
  default     = ""
  
  validation {
    condition     = length(var.auth_secret_key) >= 32
    error_message = "AUTH_SECRET_KEY must be at least 32 characters long."
  }
}

variable "admin_username" {
  description = "Admin username for the platform"
  type        = string
  default     = "admin"
  
  validation {
    condition     = length(var.admin_username) >= 3
    error_message = "Admin username must be at least 3 characters long."
  }
}

variable "admin_password" {
  description = "Admin password (change from default!)"
  type        = string
  sensitive   = true
  default     = ""
  
  validation {
    condition     = length(var.admin_password) >= 8
    error_message = "Admin password must be at least 8 characters long."
  }
}

variable "groq_api_key" {
  description = "Groq API key for AI features"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key (optional)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "news_api_key" {
  description = "News API key (optional)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "custom_domain" {
  description = "Custom domain for the application (optional)"
  type        = string
  default     = ""
}

variable "project_name" {
  description = "Name for the Railway project"
  type        = string
  default     = "ai-trading-platform"
}

variable "github_repo" {
  description = "GitHub repository in format 'owner/repo'"
  type        = string
  default     = "gouravsinha1405/ai-trading-intelligence"
}

variable "branch" {
  description = "Git branch to deploy"
  type        = string
  default     = "master"
}
