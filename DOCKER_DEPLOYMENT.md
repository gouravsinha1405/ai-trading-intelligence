# 🚀 AI Trading Platform - Docker Deployment Guide

## Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose installed
- Internet connection for data feeds

### Deployment Steps

1. **Run the deployment script:**
   ```bash
   ./deploy.sh
   ```

2. **Access your application:**
   - Open browser to: http://localhost:8501
   - The app should load with all features working

### Manual Deployment

If you prefer manual control:

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application  
docker-compose down
```

## Configuration

### Environment Variables

Edit the `.env` file to configure your API keys:

```bash
# Required for AI features
GROQ_API_KEY=your_groq_api_key_here

# Optional API keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here

# App settings
LOG_LEVEL=WARNING
VIRTUAL_MONEY_AMOUNT=100000
DEFAULT_COMMISSION=0.1
```

### Getting API Keys

1. **Groq API Key (Required for AI features):**
   - Visit: https://console.groq.com
   - Sign up for free account
   - Generate API key
   - Add to `.env` file

2. **Alpha Vantage (Optional):**
   - Visit: https://www.alphavantage.co/support/#api-key
   - Free tier available

3. **News API (Optional):**
   - Visit: https://newsapi.org
   - Free tier available

## Deployment Options

### 1. Local Development
```bash
./deploy.sh
```

### 2. Cloud Deployment (VPS/Server)

#### DigitalOcean Droplet
```bash
# 1. Create a Ubuntu 22.04 droplet (minimum 2GB RAM)
# 2. SSH into your droplet
ssh root@your-server-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 4. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 5. Clone your repository
git clone https://github.com/gouravsinha1405/ai-trading-intelligence.git
cd ai-trading-intelligence

# 6. Configure environment
nano .env

# 7. Deploy
./deploy.sh

# 8. Access at http://your-server-ip:8501
```

#### AWS EC2
```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Update system
sudo apt update && sudo apt upgrade -y

# 4. Install Docker
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker ubuntu
newgrp docker

# 5. Clone and deploy
git clone https://github.com/gouravsinha1405/ai-trading-intelligence.git
cd ai-trading-intelligence
./deploy.sh

# 6. Access at http://your-instance-ip:8501
```

#### Google Cloud Platform
```bash
# 1. Create Compute Engine instance
# 2. Enable HTTP/HTTPS traffic
# 3. SSH into instance
# 4. Follow AWS EC2 steps above
```

### 3. Container Registries

#### Push to Docker Hub
```bash
# Build and tag
docker build -t yourusername/ai-trading-platform:latest .

# Push to Docker Hub
docker push yourusername/ai-trading-platform:latest

# Deploy anywhere
docker run -d -p 8501:8501 --env-file .env yourusername/ai-trading-platform:latest
```

#### GitHub Container Registry
```bash
# Tag for GitHub
docker tag ai-trading-platform:latest ghcr.io/yourusername/ai-trading-platform:latest

# Login to GitHub Container Registry
echo $CR_PAT | docker login ghcr.io -u yourusername --password-stdin

# Push
docker push ghcr.io/yourusername/ai-trading-platform:latest
```

## Production Considerations

### 1. Reverse Proxy (Nginx)

Create `nginx.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Environment Security
```bash
# Set proper permissions on .env
chmod 600 .env

# Use Docker secrets for production
docker swarm init
echo "your_groq_key" | docker secret create groq_api_key -
```

### 4. Monitoring
```bash
# View application logs
docker-compose logs -f

# Monitor resource usage
docker stats

# Health check
curl http://localhost:8501/_stcore/health
```

## Troubleshooting

### Common Issues

1. **Port 8501 already in use:**
   ```bash
   # Kill existing Streamlit processes
   pkill -f streamlit
   
   # Or change port in docker-compose.yml
   ports:
     - "8502:8501"
   ```

2. **Permission denied:**
   ```bash
   # Fix Docker permissions
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **Out of memory:**
   ```bash
   # Check memory usage
   free -h
   
   # Restart with more memory
   docker-compose down
   docker-compose up -d
   ```

4. **Network issues:**
   ```bash
   # Check network connectivity
   docker network ls
   docker-compose logs
   ```

### Logs and Debugging

```bash
# Application logs
docker-compose logs app

# System logs
journalctl -u docker

# Container inspection
docker inspect algo-trading-app
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501-8503:8501"
    deploy:
      replicas: 3
```

### Load Balancer
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  depends_on:
    - app
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review configuration in `.env`
- Ensure all required ports are open
- Verify API keys are valid

## Security Checklist

- [ ] API keys stored securely in `.env`
- [ ] `.env` file has proper permissions (600)
- [ ] Firewall configured properly
- [ ] SSL certificate installed (production)
- [ ] Regular updates and monitoring enabled
- [ ] Backup strategy in place

---

🎉 **Congratulations!** Your AI Trading Platform is now deployed and ready to use!
