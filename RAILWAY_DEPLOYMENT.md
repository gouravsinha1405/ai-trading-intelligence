# 🚀 Railway Deployment Guide
## AI Trading Platform with Authentication

This guide will help you deploy your AI Trading Platform to Railway with secure authentication and admin controls.

## 📋 Prerequisites

1. **GitHub Account** - Your code repository
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Groq API Key** - Get free key from [console.groq.com](https://console.groq.com)

## 🔧 Pre-Deployment Setup

### 1. Update Environment Variables

Before deploying, update your production credentials in Railway:

```bash
# Required Environment Variables for Railway
AUTH_SECRET_KEY=your-super-secret-key-minimum-32-characters
ADMIN_USERNAME=your-admin-username
ADMIN_PASSWORD=your-secure-admin-password
GROQ_API_KEY=your-groq-api-key
```

### 2. Security Configuration

**🔐 Important Security Settings:**

- **AUTH_SECRET_KEY**: Generate a strong 32+ character secret
- **ADMIN_PASSWORD**: Use a strong password (min 12 characters)
- **ADMIN_USERNAME**: Change from default 'admin'

## 🚀 Railway Deployment Steps

### Method 1: One-Click Deploy (Recommended)

1. **Connect GitHub to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Connect your `ai-trading-intelligence` repository

2. **Configure Environment Variables**
   ```
   AUTH_SECRET_KEY=generate-32-character-random-string
   ADMIN_USERNAME=youradmin
   ADMIN_PASSWORD=YourSecurePassword123!
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ALPHA_VANTAGE_API_KEY=optional_alpha_vantage_key
   NEWS_API_KEY=optional_news_api_key
   VIRTUAL_MONEY_AMOUNT=100000
   DEFAULT_COMMISSION=0.1
   LOG_LEVEL=INFO
   ```

3. **Deploy**
   - Railway will automatically detect your `railway.toml` config
   - Build and deployment will start automatically
   - You'll get a public URL like `your-app.railway.app`

### Method 2: Railway CLI Deploy

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   # or
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. **Login and Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set Environment Variables**
   ```bash
   railway variables set AUTH_SECRET_KEY=your-secret-key
   railway variables set ADMIN_USERNAME=youradmin
   railway variables set ADMIN_PASSWORD=YourSecurePassword123!
   railway variables set GROQ_API_KEY=your-groq-key
   ```

## 🔐 Security Features Included

### Authentication System
- **Secure Login/Logout** with session management
- **Password Hashing** using PBKDF2 with salt
- **Session Timeout** (24 hours default)
- **Role-Based Access** (Admin/User roles)

### Admin Panel Features
- **User Management** - Add, delete, enable/disable users
- **Password Management** - Change passwords securely  
- **System Monitoring** - View user activity and stats
- **Security Dashboard** - Recent logins and activity

### Data Protection
- **User Database** stored securely in JSON format
- **Environment Variables** for sensitive configuration
- **HTTPS Enforcement** on Railway platform
- **Session Security** with secure tokens

## 📊 Post-Deployment Configuration

### 1. First Login
1. Access your Railway URL
2. Login with your admin credentials
3. **Immediately change default passwords**
4. Create additional user accounts if needed

### 2. Admin Panel Access
- Navigate to **👨‍💼 Admin Panel** page
- Manage users, passwords, and system settings
- Monitor user activity and login stats

### 3. User Management
- **Add Users**: Create accounts for team members
- **Role Assignment**: Assign Admin or User roles
- **Access Control**: Enable/disable user accounts
- **Password Policies**: Enforce password changes

## 🛠️ Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AUTH_SECRET_KEY` | ✅ | - | Secret key for password hashing |
| `ADMIN_USERNAME` | ✅ | admin | Default admin username |
| `ADMIN_PASSWORD` | ✅ | admin123 | Default admin password |
| `GROQ_API_KEY` | ✅ | - | Groq API key for AI features |
| `ALPHA_VANTAGE_API_KEY` | ❌ | - | Optional market data API |
| `NEWS_API_KEY` | ❌ | - | Optional news API |
| `VIRTUAL_MONEY_AMOUNT` | ❌ | 100000 | Starting virtual money |
| `DEFAULT_COMMISSION` | ❌ | 0.1 | Trading commission rate |
| `LOG_LEVEL` | ❌ | INFO | Logging level |

### Railway-Specific Settings
```toml
# railway.toml (already configured)
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run main.py --server.port $PORT --server.address 0.0.0.0"
healthcheckPath = "/"
```

## 🔧 Troubleshooting

### Common Issues

1. **Authentication Not Working**
   - Check `AUTH_SECRET_KEY` is set
   - Verify admin credentials are correct
   - Check Railway logs for errors

2. **AI Features Disabled**
   - Ensure `GROQ_API_KEY` is set correctly
   - Verify API key is valid
   - Check API quota limits

3. **App Not Loading**
   - Check Railway deployment logs
   - Verify all required environment variables
   - Check health check endpoint

### Debug Commands
```bash
# View Railway logs
railway logs

# Check environment variables
railway variables

# Restart service
railway redeploy
```

## 📈 Monitoring & Maintenance

### Health Monitoring
- Railway provides automatic health checks
- Monitor via Railway dashboard
- Set up alerts for downtime

### User Activity
- Admin panel shows login activity
- Monitor user access patterns
- Review security logs regularly

### Updates & Maintenance
- Push to GitHub to auto-deploy updates
- Monitor Railway resource usage
- Regular security audits

## 🌟 Additional Features

### Custom Domain (Optional)
1. Purchase domain name
2. Add custom domain in Railway dashboard
3. Configure DNS settings
4. Enable SSL certificate

### Backup Strategy
- User data stored in `users.json`
- Regular backups via Railway volumes
- Export user data from admin panel

### Scaling Options
- Railway auto-scales based on traffic
- Monitor resource usage in dashboard
- Upgrade plan if needed

## 🎯 Production Checklist

- [ ] Strong AUTH_SECRET_KEY set
- [ ] Admin password changed from default
- [ ] GROQ_API_KEY configured
- [ ] HTTPS enabled (automatic on Railway)
- [ ] User accounts created
- [ ] Admin panel tested
- [ ] Authentication flow verified
- [ ] Monitoring configured
- [ ] Backup strategy in place

## 📞 Support

- **Railway Support**: [railway.app/help](https://railway.app/help)
- **Platform Issues**: GitHub Issues
- **Security Concerns**: Change credentials immediately

---

🚀 **Your secure AI Trading Platform is now ready for production use!**
