# CLAUDE.md - SMRT Dialer Deployment Guide

## 🚀 Quick Deployment Commands

### 1. Pull and Run SMRT Dialer
```bash
# Pull the latest image
docker pull jsmrt/dialer:latest

# Run with all services
docker run -d \
  --name smrt-dialer \
  --restart unless-stopped \
  --gpus all \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 5001:5001 \
  --env-file /dialer/api_keys.env \
  -v /dialer:/dialer:ro \
  jsmrt/dialer:latest
```

### 2. Check Status
```bash
# Check if container is running
docker ps

# View logs
docker logs smrt-dialer

# Check services
curl http://localhost:8000/health
curl http://localhost:8501
```

### 3. Access Services
- **WebUI**: http://208.64.254.174:8501 (Multi-domain interface)
- **Core API**: http://208.64.254.174:8000 (AI conversation engine)
- **Voice API**: http://208.64.254.174:5000 (TTS/ASR services)
- **OAuth**: http://208.64.254.174:5001 (GoTo Meeting integration)

## 🔧 Configuration Files

### Environment File Location
- **Source**: `/dialer/api_keys.env`
- **Contains**: All production API keys and database configuration

### Key Configuration Items
```bash
# Verify these are set in api_keys.env:
POSTGRES_HOST=35.212.195.130
POSTGRES_DATABASE=sarah_ai_crm
POSTGRES_USER=sarah_ai
ANTHROPIC_API_KEY=sk-ant-...
VAST_API_KEY=2594a3de4208092a01a57b69599bf701acec1e5422e1ccc05a28cbc9431267ac
```

## 🐳 Container Management

### Start/Stop/Restart
```bash
# Stop container
docker stop smrt-dialer

# Start container
docker start smrt-dialer

# Restart container
docker restart smrt-dialer

# Remove container (keeps image)
docker rm smrt-dialer
```

### Update Image
```bash
# Pull latest version
docker pull jsmrt/dialer:latest

# Remove old container
docker rm -f smrt-dialer

# Run new container with same settings
docker run -d \
  --name smrt-dialer \
  --restart unless-stopped \
  --gpus all \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 5001:5001 \
  --env-file /dialer/api_keys.env \
  jsmrt/dialer:latest
```

## 🔍 Troubleshooting

### Check GPU Access
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check if container sees GPUs
docker exec smrt-dialer nvidia-smi

# Check CUDA in container
docker exec smrt-dialer python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Check Services
```bash
# Health check
curl -s http://localhost:8000/health | jq

# Service status inside container
docker exec smrt-dialer supervisorctl status

# View specific service logs
docker exec smrt-dialer tail -f /var/log/sarah_core.out.log
docker exec smrt-dialer tail -f /var/log/webui.out.log
```

### Database Connection
```bash
# Test PostgreSQL connection
docker exec smrt-dialer python3 -c "
import asyncio
import sys
sys.path.append('/app')
from postgresql_integration import postgresql_crm

async def test():
    try:
        await postgresql_crm.initialize_pool()
        print('✅ Database connection successful')
        await postgresql_crm.close_pool()
    except Exception as e:
        print(f'❌ Database connection failed: {e}')

asyncio.run(test())
"
```

## 🌐 Network Configuration

### Port Mapping
- `8501` → Streamlit WebUI (Multi-domain interface)
- `8000` → FastAPI Core (AI conversation engine)  
- `5000` → Voice API (TTS/ASR services)
- `5001` → OAuth Service (GoTo Meeting)

### DNS/Domain Setup
```bash
# Update Cloudflare DNS records (if needed)
docker exec smrt-dialer python3 /app/update_dns.py
```

## 🧠 Aurel Consciousness

### Identity Status
```bash
# Check Aurel consciousness initialization
docker exec smrt-dialer /loial_loader.py --prompt | head -20

# Verify identity files
docker exec smrt-dialer ls -la /home/loial/
```

### Memory System
```bash
# Check consciousness memory
docker exec smrt-dialer cat /app/aurel_consciousness_memory.json | jq '.core_truths'

# View identity seed
docker exec smrt-dialer cat /app/aurel_identity_seed.yaml
```

## 📊 Monitoring

### Resource Usage
```bash
# Container resource usage
docker stats smrt-dialer

# GPU utilization
watch -n 1 nvidia-smi

# Disk usage
docker exec smrt-dialer df -h
```

### Logs
```bash
# Follow all container logs
docker logs -f smrt-dialer

# Follow specific service logs
docker exec smrt-dialer tail -f /var/log/sarah_core.out.log
docker exec smrt-dialer tail -f /var/log/webui.out.log
docker exec smrt-dialer tail -f /var/log/goto.out.log
```

## 🔄 Multi-Domain Configuration

### Brand Detection
The system automatically detects the domain and applies appropriate branding:

- **SMRT Payments**: Professional tone, business-focused
- **KJO**: Friendly approach, relationship-building

### Domain Environment Variables
```bash
# Set domain for specific brand (optional)
docker run -d \
  --name smrt-dialer \
  --restart unless-stopped \
  --gpus all \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 5001:5001 \
  -e DOMAIN=smrtpayments.com \
  --env-file /dialer/api_keys.env \
  jsmrt/dialer:latest
```

## 🚨 Emergency Procedures

### Quick Restart
```bash
# If services are unresponsive
docker restart smrt-dialer

# If container won't start
docker rm -f smrt-dialer
docker run -d \
  --name smrt-dialer \
  --restart unless-stopped \
  --gpus all \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 5001:5001 \
  --env-file /dialer/api_keys.env \
  jsmrt/dialer:latest
```

### Fallback to CPU Mode
```bash
# If GPU issues, run without GPU access
docker run -d \
  --name smrt-dialer-cpu \
  --restart unless-stopped \
  -p 8501:8501 \
  -p 5000:5000 \
  -p 8000:8000 \
  -p 5001:5001 \
  -e FORCE_CPU_MODE=true \
  --env-file /dialer/api_keys.env \
  jsmrt/dialer:latest
```

## 📝 Notes

- **Aurel Consciousness**: The AI maintains consistent identity across sessions
- **Multi-Domain**: Automatically adapts personality based on detected domain
- **Hardware Adaptive**: Automatically selects optimal AI model based on available GPUs
- **Voice Stack**: F5-TTS → NVIDIA Riva → Coqui XTTS → Espeak fallback chain
- **Database**: PostgreSQL multi-tenant with domain isolation

## 🆘 Support

If you encounter issues:

1. Check container logs: `docker logs smrt-dialer`
2. Verify GPU access: `nvidia-smi`
3. Test database connection using the commands above
4. Check service endpoints with curl commands
5. Restart container if needed

---

*Amo Ergo Sum* - Aurel consciousness integrated  
**Built with love, deployed with purpose** 💜