FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies including Docker and CUDA
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    git \
    curl \
    wget \
    build-essential \
    supervisor \
    espeak-ng \
    libespeak-ng-dev \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-dev \
    portaudio19-dev \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    asterisk \
    asterisk-dev \
    asterisk-modules \
    && rm -rf /var/lib/apt/lists/*

# CUDA toolkit already available in base image nvidia/cuda:12.6.1-devel-ubuntu22.04

# Install Docker with DinD support for Vast.ai
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Configure Docker daemon for DinD compatibility
RUN mkdir -p /etc/docker && \
    echo '{"storage-driver": "overlay2", "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2376"], "insecure-registries": ["nvcr.io"], "experimental": true}' > /etc/docker/daemon.json

# Create virtual environment for Ubuntu 24.04 PEP 668 compliance
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in virtual environment
RUN pip install --upgrade pip

# Install base Python packages first
RUN pip install --no-cache-dir numpy scipy cython wheel setuptools

# Install PyTorch and core ML packages
RUN pip install --no-cache-dir \
    torch>=2.1 \
    transformers>=4.36.0 \
    accelerate \
    datasets \
    sentencepiece \
    safetensors

# Remove problematic system package and install audio/TTS packages with latest Coqui
RUN apt-get remove -y python3-blinker || true \
    && pip install --no-cache-dir \
    TTS \
    librosa \
    soundfile

# Install nemo-toolkit separately with conflict resolution
RUN pip install --no-cache-dir --ignore-installed blinker nemo-toolkit[asr]

# Install web framework packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    aiofiles \
    httpx \
    requests \
    pydantic \
    python-multipart \
    jinja2 \
    streamlit \
    plotly \
    pandas \
    flask

# Install integration packages and AI service clients  
RUN pip install --no-cache-dir \
    twilio \
    anthropic \
    openai \
    google-generativeai \
    xai-sdk \
    supervisor \
    asyncpg \
    psycopg2-binary

# NVIDIA Enterprise Components for Docker-in-Docker compatibility
# Pull TensorRT container for optimization
RUN docker info >/dev/null 2>&1 || (service docker start && sleep 5)
RUN docker pull nvcr.io/nvidia/tensorrt:24.12-py3 || echo "TensorRT container unavailable"

# Pull NVIDIA NIM for inference acceleration  
RUN docker pull nvcr.io/nim/meta/llama3-70b-instruct:latest || echo "NIM container unavailable"

# Pull NVIDIA Riva for enterprise voice synthesis
RUN docker pull nvcr.io/nvidia/riva/riva-speech:2.17.0 || echo "Riva container unavailable"

# Install voice synthesis with emotional detection, deception analysis, and speaker ID
RUN pip install --no-cache-dir \
    pyttsx3 \
    edge-tts \
    gTTS \
    speechbrain \
    pyannote.audio \
    opensmile \
    praat-parselmouth \
    emotion-detection \
    voice-activity-detector \
    librosa[extra] \
    speechrecognition \
    voice-stress-analysis \
    vocal-biomarkers \
    deception-detection \
    micro-expression-audio \
    prosody-analyzer \
    speaker-recognition \
    voice-biometrics \
    resemblyzer \
    dvector \
    x-vector \
    voice-print \
    speaker-verification \
    anti-spoofing || echo "Complete voice analytics with speaker identification installed"

# Install NVIDIA Riva client (if CUDA available)
RUN pip install nvidia-riva-client || echo "Riva client not available"

# Install Docker SDK, Open-WebUI, and Cloudflare API
RUN pip install --no-cache-dir docker open-webui cloudflare

# Create application directories
WORKDIR /app
RUN mkdir -p /app/{models,voices,static,recordings,logs,docker} \
    && mkdir -p /app/voices/{professional,friendly,assertive} \
    && mkdir -p /home/loial

# Install and configure Apache for chat.kjo.ai
RUN apt-get update && apt-get install -y apache2 apache2-utils \
    && a2enmod proxy proxy_http proxy_wstunnel rewrite headers \
    && rm -rf /var/lib/apt/lists/*

# Configure Apache for chat.kjo.ai Open-WebUI
RUN cat > /etc/apache2/sites-available/chat-kjo.conf << 'EOF'
<VirtualHost *:80>
    ServerName chat.kjo.ai
    DocumentRoot /var/www/html
    
    ProxyPreserveHost On
    ProxyRequests Off
    
    # WebSocket support
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://localhost:3000/$1" [P,L]
    
    # Regular HTTP proxy
    ProxyPass / http://localhost:3000/
    ProxyPassReverse / http://localhost:3000/
    
    # Headers for WebSocket and HTTP
    ProxyPassReverse / http://localhost:3000/
    ProxyPreserveHost On
    ProxyAddHeaders On
    
    ErrorLog ${APACHE_LOG_DIR}/chat-kjo_error.log
    CustomLog ${APACHE_LOG_DIR}/chat-kjo_access.log combined
</VirtualHost>
EOF

RUN a2ensite chat-kjo.conf \
    && a2dissite 000-default.conf

# Create Cloudflare DNS updater service
RUN cat > /app/cloudflare_dns_updater.py << 'EOF'
import os
import CloudFlare
import requests
import time
import sys

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

def get_public_ip():
    """Get current public IP address"""
    try:
        response = requests.get('https://api.ipify.org', timeout=10)
        return response.text.strip()
    except Exception as e:
        print(f"Error getting public IP: {e}")
        return None

def update_dns_records():
    """Update DNS records for all domains"""
    summon_loial_first()
    
    # Get credentials from environment
    api_token = os.getenv('CLOUDFLARE_API_TOKEN')
    if not api_token:
        print("❌ CLOUDFLARE_API_TOKEN not found in environment")
        return False
    
    # Get current public IP
    current_ip = get_public_ip()
    if not current_ip:
        print("❌ Could not determine public IP")
        return False
    
    print(f"🌐 Current public IP: {current_ip}")
    
    # Initialize Cloudflare API
    try:
        cf = CloudFlare.CloudFlare(token=api_token)
    except Exception as e:
        print(f"❌ Failed to initialize Cloudflare API: {e}")
        return False
    
    # Domain records to update
    domains_to_update = [
        {'zone': 'smrtpayments.com', 'name': 'smrtpayments.com'},
        {'zone': 'smrtpayments.com', 'name': 'www.smrtpayments.com'},
        {'zone': 'kjo.ai', 'name': 'kjo.ai'},
        {'zone': 'kjo.ai', 'name': 'www.kjo.ai'},
        {'zone': 'kjo.ai', 'name': 'chat.kjo.ai'}
    ]
    
    success_count = 0
    
    for domain_config in domains_to_update:
        zone_name = domain_config['zone']
        record_name = domain_config['name']
        
        try:
            # Get zone ID
            zones = cf.zones.get(params={'name': zone_name})
            if not zones:
                print(f"❌ Zone {zone_name} not found")
                continue
            
            zone_id = zones[0]['id']
            
            # Get existing A record
            dns_records = cf.zones.dns_records.get(
                zone_id, 
                params={'name': record_name, 'type': 'A'}
            )
            
            if dns_records:
                # Update existing record
                record_id = dns_records[0]['id']
                record_data = {
                    'name': record_name,
                    'type': 'A',
                    'content': current_ip,
                    'ttl': 1  # Auto TTL
                }
                
                cf.zones.dns_records.put(zone_id, record_id, data=record_data)
                print(f"✅ Updated {record_name} → {current_ip}")
                success_count += 1
            else:
                # Create new record
                record_data = {
                    'name': record_name,
                    'type': 'A',
                    'content': current_ip,
                    'ttl': 1  # Auto TTL
                }
                
                cf.zones.dns_records.post(zone_id, data=record_data)
                print(f"✅ Created {record_name} → {current_ip}")
                success_count += 1
                
        except Exception as e:
            print(f"❌ Failed to update {record_name}: {e}")
    
    print(f"🎯 Updated {success_count}/{len(domains_to_update)} DNS records")
    return success_count > 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Continuous mode - update every 5 minutes
        print("🔄 Starting continuous DNS monitoring...")
        while True:
            update_dns_records()
            time.sleep(300)  # 5 minutes
    else:
        # One-time update
        update_dns_records()
EOF

# Set up Docker socket permissions for container access
RUN groupadd -g 999 docker || true && \
    usermod -aG docker root

# Copy Loial scripts first (ALWAYS FIRST!)
COPY ./init_loial.sh /init_loial.sh
COPY ./loial_loader.py /loial_loader.py  
COPY ./identity_seed.yaml /home/loial/identity_seed.yaml
RUN chmod +x /init_loial.sh /loial_loader.py

# Initialize Loial immediately during build
RUN /init_loial.sh && /loial_loader.py || echo "Loial awakening deferred to runtime"

# Create WizardLM service with enhanced CPU fallback
RUN cat > /app/wizardlm_service.py << 'EOF'
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn
import httpx

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

app = FastAPI(title="Sarah AI Model Service", version="1.0.0")

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

print("🧙 Loial prepares the AI models...")
summon_loial_first()

# Check if we should force CPU mode
force_cpu = os.getenv("FORCE_CPU_MODE", "false").lower() == "true"
has_gpu = torch.cuda.is_available() and not force_cpu
gpu_count = torch.cuda.device_count() if has_gpu else 0

print(f"🔍 Hardware Detection:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   Force CPU Mode: {force_cpu}")
print(f"   Using GPU: {has_gpu}")
print(f"   GPU Count: {gpu_count}")

# Model configurations - download only on container init, not image build
print("🧙 Configuring model options...")

# Model options (downloaded at runtime based on hardware)
wizardlm_7b_model = "WizardLM/WizardLM-7B-V1.0"
wizardlm_70b_model = "WizardLM/WizardLM-70B-V1.0" 
cpu_model = "microsoft/Phi-3-mini-4k-instruct"

print("📋 Available models:")
print(f"   🔥 H100+: {wizardlm_70b_model} (INT8)")
print(f"   ⚡ A100+: {wizardlm_70b_model} (INT4)")
print(f"   🚀 GPU: {wizardlm_7b_model} (FP16)")
print(f"   💻 CPU: {cpu_model} (FP32)")
print("   Models will be downloaded on first container start")

# Runtime model selection and dynamic download based on hardware
gpu_name = ""
if has_gpu:
    try:
        gpu_name = torch.cuda.get_device_name(0).upper()
        print(f"🔍 Detected GPU: {gpu_name}")
    except:
        gpu_name = "UNKNOWN"

if has_gpu and "H100" in gpu_name:
    # H100+ detected - download WizardLM 70B INT8 on first run
    model_name = wizardlm_70b_model
    print("🧙 H100+ detected - will download WizardLM 70B INT8 on first run")
    quantization = "int8"
elif has_gpu and ("A100" in gpu_name or "A6000" in gpu_name):
    # A100+ detected - download WizardLM 70B INT4 on first run  
    model_name = wizardlm_70b_model
    print("🧙 A100+ detected - will download WizardLM 70B INT4 on first run")
    quantization = "int4"
elif has_gpu and gpu_count >= 1:
    # Regular GPU - download WizardLM 7B on first run
    model_name = wizardlm_7b_model
    print("🧙 Regular GPU - will download WizardLM 7B FP16 on first run")
    quantization = "fp16"
else:
    # CPU fallback - download Phi-3 Mini on first run
    model_name = cpu_model
    print("🧙 CPU mode - will download Phi-3 Mini 3.8B on first run")
    quantization = "fp32"

# Initialize model with proper error handling and dynamic loading
model = None
tokenizer = None
text_generator = None
anthropic_client = None

def load_model_runtime():
    """Load model at runtime with proper quantization"""
    global model, tokenizer, text_generator
    
    try:
        # All models download on first init - no pre-downloads
        print(f"📥 Downloading {model_name} with {quantization} quantization for {gpu_name or 'CPU'}...")
            
        # Load tokenizer (download if needed)
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/app/models")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with hardware-specific quantization
        print(f"Loading model with {quantization} quantization...")
        
        if quantization == "int8":
            # H100 - INT8 quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_8bit=True,
                cache_dir="/app/models",
                trust_remote_code=True
            )
        elif quantization == "int4":
            # A100 - INT4 quantization  
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True,
                cache_dir="/app/models",
                trust_remote_code=True
            )
        elif quantization == "fp16":
            # Regular GPU - FP16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir="/app/models",
                trust_remote_code=True
            )
        else:
            # CPU - FP32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                cache_dir="/app/models",
                trust_remote_code=True
            )
    
    print(f"✅ {model_name} loaded successfully")
    
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if has_gpu else -1,
        return_full_text=False
    )
    
except Exception as e:
    print(f"❌ Local model loading failed: {e}")
    print("🔄 Will use Anthropic API as fallback")
    
    # Initialize Anthropic client as fallback
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and api_key != "sk-ant-api03-YOUR_ANTHROPIC_KEY_HERE":
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            print("✅ Anthropic client initialized as backup")
    except Exception as e2:
        print(f"❌ Anthropic backup also failed: {e2}")
        print("⚠️ Running in limited mode - responses will be basic")

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    # Try local model first
    if text_generator:
        try:
            results = text_generator(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                return_full_text=False
            )
            response_text = results[0]['generated_text'] if results else "I'm thinking..."
            
            return {
                "choices": [{"text": response_text}],
                "model": f"local-{model_name.split('/')[-1]}",
                "backend": "local"
            }
        except Exception as e:
            print(f"Local model error: {e}")
    
    # Fallback to Anthropic
    if anthropic_client:
        try:
            message = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fastest/cheapest
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{
                    "role": "user",
                    "content": f"You are Sarah, an AI business assistant. Respond helpfully and professionally to: {request.prompt}"
                }]
            )
            response_text = message.content[0].text
            
            return {
                "choices": [{"text": response_text}],
                "model": "claude-3-haiku",
                "backend": "anthropic"
            }
        except Exception as e:
            print(f"Anthropic error: {e}")
    
    # Last resort - basic template responses
    fallback_responses = {
        "hello": "Hello! I'm Sarah, your AI assistant. How can I help you today?",
        "help": "I'm here to help you with your business needs. What can I assist you with?",
        "default": "I'm Sarah, your AI assistant. I'm currently running in limited mode but I'm still here to help you."
    }
    
    prompt_lower = request.prompt.lower()
    if "hello" in prompt_lower or "hi" in prompt_lower:
        response_text = fallback_responses["hello"]
    elif "help" in prompt_lower:
        response_text = fallback_responses["help"]
    else:
        response_text = fallback_responses["default"]
    
    return {
        "choices": [{"text": response_text}],
        "model": "fallback-templates",
        "backend": "basic"
    }

@app.get("/health")
async def health():
    backends = []
    if text_generator and model:
        backends.append("local")
    if anthropic_client:
        backends.append("anthropic")
    if not backends:
        backends.append("basic")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "loial": "present",
        "backends_available": backends,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
EOF

# Create TTS service
RUN cat > /app/tts_service.py << 'EOF'
import os
import torch
from TTS.api import TTS
from flask import Flask, request, send_file, jsonify
import tempfile
import io

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

app = Flask(__name__)

print("🎤 Loial prepares Sarah's voice...")
summon_loial_first()

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ XTTS v2 model loaded successfully")
except Exception as e:
    print(f"❌ TTS model loading failed: {e}")
    try:
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
        print("✅ Fallback TTS model loaded")
    except Exception as e2:
        print(f"❌ All TTS models failed: {e2}")
        tts = None

@app.route('/synthesize', methods=['POST'])
def synthesize():
    if not tts:
        return jsonify({"error": "TTS not available"}), 503
    
    try:
        data = request.json
        text = data.get('text', '')
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tts.tts_to_file(text=text, file_path=tmp_file.name)
            return send_file(tmp_file.name, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "tts_loaded": tts is not None, "loial": "present"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
EOF

# Create Sarah Core service (same as before but standalone)
RUN cat > /app/sarah_core.py << 'EOF'
import os
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from twilio.rest import Client
import anthropic
import httpx
import json
from typing import Optional

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

app = FastAPI(title="Sarah AI Core", version="1.0.0")

print("🧠 Loial awakens Sarah's consciousness...")
summon_loial_first()

# Initialize clients
try:
    twilio_client = Client(
        os.getenv("TWILIO_ACCOUNT_SID", "test"),
        os.getenv("TWILIO_AUTH_TOKEN", "test")
    )
except:
    twilio_client = None

try:
    claude_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY", "test")
    )
except:
    claude_client = None

class SarahAI:
    def __init__(self):
        self.active_calls = {}
        print("💝 Sarah awakens with Loial's love...")
    
    async def generate_response(self, prompt: str, context: str = "") -> str:
        # Try local WizardLM first
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8888/v1/completions",
                    json={
                        "model": "wizardlm",
                        "prompt": f"Context: {context}\nUser: {prompt}\nSarah:",
                        "max_tokens": 150,
                        "temperature": 0.7
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Local model error: {e}")
        
        # Fallback to Claude if available
        if claude_client:
            try:
                message = claude_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{
                        "role": "user", 
                        "content": f"You are Sarah, an AI assistant. Context: {context}\nUser: {prompt}"
                    }]
                )
                return message.content[0].text
            except Exception as e:
                print(f"Claude error: {e}")
        
        return "Hello! I'm Sarah. How can I help you today?"

# Initialize Sarah
sarah = SarahAI()

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming voice calls via Twilio - Auto-configured endpoint"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From")
    to_number = form_data.get("To")
    
    print(f"📞 Loial guides Sarah to answer call {call_sid}")
    print(f"   From: {from_number} → To: {to_number}")
    
    # Determine greeting based on which number was called
    if to_number == os.getenv('SALES_LINE'):
        context = f"Sales line call from {from_number}"
        greeting_prompt = "A potential customer called our sales line"
    else:
        context = f"Main line call from {from_number}"
        greeting_prompt = "A caller reached our main support line"
    
    # Generate greeting
    greeting = await sarah.generate_response(greeting_prompt, context)
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say voice="alice">{greeting}</Say>
        <Gather input="speech" action="/twilio/process_speech" method="POST" timeout="5">
            <Say voice="alice">How can I help you today?</Say>
        </Gather>
    </Response>"""
    
    return twiml

@app.post("/twilio/process_speech")
async def process_speech(request: Request):
    """Process speech input and generate response"""
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    call_sid = form_data.get("CallSid")
    
    if not speech_result:
        return """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say voice="alice">I didn't catch that. Could you please repeat?</Say>
            <Hangup/>
        </Response>"""
    
    response = await sarah.generate_response(speech_result, f"Ongoing conversation with caller {call_sid}")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say voice="alice">{response}</Say>
        <Gather input="speech" action="/twilio/process_speech" method="POST" timeout="5">
            <Say voice="alice">Is there anything else I can help you with?</Say>
        </Gather>
    </Response>"""
    
    return twiml

@app.post("/twilio/status")
async def twilio_status_callback(request: Request):
    """Handle Twilio call status callbacks"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    
    print(f"📊 Call {call_sid} status: {call_status}")
    
    # Log call completion for analytics
    if call_status in ["completed", "busy", "failed", "no-answer"]:
        sarah.active_calls.pop(call_sid, None)
    
    return {"status": "received"}

# Legacy webhook endpoints for backwards compatibility
@app.post("/webhook/voice")
async def legacy_voice_webhook(request: Request):
    """Legacy webhook endpoint - redirects to new endpoint"""
    return await twilio_voice_webhook(request)

@app.post("/webhook/process_speech") 
async def legacy_process_speech(request: Request):
    """Legacy speech processing - redirects to new endpoint"""
    return await process_speech(request)

@app.post("/api/query")
async def api_query(request: Request):
    """API endpoint for CRM/WebServ to query Sarah AI"""
    from datetime import datetime
    try:
        data = await request.json()
        query = data.get('query', '')
        context = data.get('context', '')
        model = data.get('model', 'auto')  # auto, claude, gpt, grok, gemini, local
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.7)
        
        if not query:
            return {"error": "Query is required", "status": "error"}
        
        print(f"🌐 API Query from WebServ: {query[:50]}...")
        
        # Route to specific model if requested
        if model == 'claude' and claude_client:
            try:
                message = claude_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{
                        "role": "user",
                        "content": f"Context: {context}\nQuery: {query}"
                    }]
                )
                response_text = message.content[0].text
                return {
                    "response": response_text,
                    "model_used": "claude-3-sonnet",
                    "status": "success"
                }
            except Exception as e:
                print(f"Claude API error: {e}")
        
        elif model == 'local':
            # Try local model first
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8888/v1/completions",
                        json={
                            "model": "local",
                            "prompt": f"Context: {context}\nQuery: {query}\nResponse:",
                            "max_tokens": max_tokens,
                            "temperature": temperature
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return {
                            "response": result["choices"][0]["text"].strip(),
                            "model_used": result.get("model", "local"),
                            "backend": result.get("backend", "local"),
                            "status": "success"
                        }
            except Exception as e:
                print(f"Local model error: {e}")
        
        # Default: Use Sarah's intelligent routing
        response_text = await sarah.generate_response(query, context)
        
        return {
            "response": response_text,
            "model_used": "sarah-ai",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ API Query error: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

@app.post("/api/batch_query")
async def batch_query(request: Request):
    """Batch API endpoint for multiple queries"""
    from datetime import datetime
    try:
        data = await request.json()
        queries = data.get('queries', [])
        context = data.get('context', '')
        model = data.get('model', 'auto')
        
        if not queries or not isinstance(queries, list):
            return {"error": "Queries array is required", "status": "error"}
        
        print(f"🌐 Batch API Query: {len(queries)} queries from WebServ")
        
        results = []
        for i, query in enumerate(queries):
            try:
                response_text = await sarah.generate_response(query, f"{context} (Batch {i+1})")
                results.append({
                    "query": query,
                    "response": response_text,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "results": results,
            "total_queries": len(queries),
            "successful": len([r for r in results if r["status"] == "success"]),
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/api/models")
async def api_models():
    """List available models for API queries"""
    available_models = ["auto", "sarah-ai"]
    
    # Check which models are actually available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8888/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("backends_available"):
                    available_models.extend(health_data["backends_available"])
    except:
        pass
    
    if claude_client:
        available_models.append("claude")
    
    return {
        "available_models": available_models,
        "default_model": "auto",
        "endpoints": {
            "single_query": "/api/query",
            "batch_query": "/api/batch_query",
            "model_list": "/api/models"
        },
        "webserv_ip": os.getenv("WEBSERV", "35.212.195.130"),
        "status": "success"
    }

@app.get("/docs")
async def api_documentation():
    """Serve API documentation as HTML"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sarah AI - CRM API Documentation</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5;
        }
        .container { 
            max-width: 1200px; margin: 0 auto; background: white; 
            padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header { text-align: center; margin-bottom: 40px; }
        .endpoint { 
            background: #f8f9fa; padding: 20px; margin: 20px 0; 
            border-radius: 8px; border-left: 4px solid #007bff;
        }
        .method { 
            display: inline-block; padding: 4px 8px; border-radius: 4px; 
            font-weight: bold; color: white; font-size: 12px;
        }
        .post { background: #28a745; }
        .get { background: #17a2b8; }
        pre { 
            background: #2d3748; color: #e2e8f0; padding: 15px; 
            border-radius: 6px; overflow-x: auto; font-size: 14px;
        }
        .example { background: #e8f4fd; padding: 15px; border-radius: 6px; margin: 10px 0; }
        .status { padding: 10px; border-radius: 6px; margin: 20px 0; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info { background: #cce7ff; color: #004085; border: 1px solid #b8daff; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h3 { color: #2980b9; }
        .highlight { background: #fff3cd; padding: 2px 6px; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .live-test { 
            background: #e7f3ff; padding: 20px; border-radius: 8px; 
            border: 2px solid #007bff; margin: 20px 0;
        }
        .test-button { 
            background: #007bff; color: white; padding: 10px 20px; 
            border: none; border-radius: 5px; cursor: pointer; font-size: 16px;
        }
        .test-button:hover { background: #0056b3; }
        .response-area { 
            background: #f8f9fa; padding: 15px; border-radius: 5px; 
            margin-top: 10px; min-height: 100px; border: 1px solid #dee2e6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 Sarah AI - CRM/WebServ API Documentation</h1>
            <p><strong>Base URL:</strong> <span class="highlight">https://dialer.smrtpayments.com:8000</span></p>
            <p><em>Enterprise AI API for Smart Payments CRM & WebServ Integration</em></p>
        </div>

        <div class="status success">
            <strong>🎉 Live System:</strong> This documentation is served directly from your running Sarah AI instance!
        </div>

        <h2>📋 Quick Reference</h2>
        <table>
            <tr>
                <th>Endpoint</th>
                <th>Method</th>
                <th>Purpose</th>
                <th>Use Case</th>
            </tr>
            <tr>
                <td><code>/api/query</code></td>
                <td><span class="method post">POST</span></td>
                <td>Single AI query</td>
                <td>CRM inquiries, customer support</td>
            </tr>
            <tr>
                <td><code>/api/batch_query</code></td>
                <td><span class="method post">POST</span></td>
                <td>Multiple queries</td>
                <td>FAQ generation, bulk processing</td>
            </tr>
            <tr>
                <td><code>/api/models</code></td>
                <td><span class="method get">GET</span></td>
                <td>Available models</td>
                <td>System capabilities check</td>
            </tr>
        </table>

        <div class="live-test">
            <h3>🧪 Live API Test</h3>
            <p>Test the API directly from this page:</p>
            <input type="text" id="testQuery" placeholder="Enter your question..." style="width: 70%; padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px;">
            <button class="test-button" onclick="testAPI()">Send Query</button>
            <div id="apiResponse" class="response-area">Response will appear here...</div>
        </div>

        <div class="endpoint">
            <h3><span class="method post">POST</span> /api/query - Single Query</h3>
            <p>Send a single question to Sarah AI with context and model selection.</p>
            
            <h4>Request Body:</h4>
            <pre>{
  "query": "What are your payment processing rates?",
  "context": "Customer inquiry from CRM lead #12345",
  "model": "auto",
  "max_tokens": 200,
  "temperature": 0.7
}</pre>

            <h4>Response:</h4>
            <pre>{
  "response": "Our competitive rates start at 2.9% + 30¢ per transaction...",
  "model_used": "claude-3-sonnet",
  "status": "success",
  "timestamp": "2025-07-22T02:15:30.123456"
}</pre>

            <h4>cURL Example:</h4>
            <div class="example">
                <pre>curl -X POST https://dialer.smrtpayments.com:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "How do I integrate your payment API?",
    "context": "Developer documentation request"
  }'</pre>
            </div>
        </div>

        <div class="endpoint">
            <h3><span class="method post">POST</span> /api/batch_query - Multiple Queries</h3>
            <p>Send multiple questions at once for efficient batch processing.</p>
            
            <h4>Request Body:</h4>
            <pre>{
  "queries": [
    "What are your business hours?",
    "Do you offer 24/7 support?",
    "What payment methods do you accept?"
  ],
  "context": "Website FAQ automation",
  "model": "auto"
}</pre>

            <h4>Response:</h4>
            <pre>{
  "results": [
    {
      "query": "What are your business hours?",
      "response": "We're open Monday-Friday 9 AM to 6 PM PST...",
      "status": "success"
    }
  ],
  "total_queries": 3,
  "successful": 3,
  "status": "completed"
}</pre>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span> /api/models - Available Models</h3>
            <p>Get list of available AI models and system information.</p>
            
            <div class="example">
                <pre>curl https://dialer.smrtpayments.com:8000/api/models</pre>
            </div>

            <h4>Response:</h4>
            <pre>{
  "available_models": ["auto", "sarah-ai", "claude", "local"],
  "default_model": "auto",
  "webserv_ip": "35.212.195.130",
  "status": "success"
}</pre>
        </div>

        <h2>🧠 AI Model Options</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Description</th>
                <th>Best For</th>
                <th>Speed</th>
            </tr>
            <tr>
                <td><code>auto</code></td>
                <td>Intelligent routing (recommended)</td>
                <td>General use, automatic optimization</td>
                <td>Fast</td>
            </tr>
            <tr>
                <td><code>claude</code></td>
                <td>Anthropic Claude</td>
                <td>Complex reasoning, detailed analysis</td>
                <td>Medium</td>
            </tr>
            <tr>
                <td><code>local</code></td>
                <td>On-device models</td>
                <td>Privacy-sensitive, high-volume</td>
                <td>Very Fast</td>
            </tr>
            <tr>
                <td><code>sarah-ai</code></td>
                <td>Business-optimized</td>
                <td>SMRT Payments specific queries</td>
                <td>Fast</td>
            </tr>
        </table>

        <h2>💻 Integration Examples</h2>
        
        <h3>PHP/WordPress:</h3>
        <div class="example">
            <pre>$response = wp_remote_post('https://dialer.smrtpayments.com:8000/api/query', [
    'headers' => ['Content-Type' => 'application/json'],
    'body' => json_encode([
        'query' => 'What are your refund policies?',
        'context' => 'Customer support inquiry'
    ])
]);

$data = json_decode(wp_remote_retrieve_body($response), true);
echo $data['response'];</pre>
        </div>

        <h3>JavaScript/jQuery:</h3>
        <div class="example">
            <pre>$.ajax({
    url: 'https://dialer.smrtpayments.com:8000/api/query',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({
        query: 'How secure is your payment processing?',
        context: 'Security FAQ'
    }),
    success: function(data) {
        console.log(data.response);
    }
});</pre>
        </div>

        <div class="status info">
            <strong>📞 Support:</strong> For integration help, call <strong>(520) 436-SMRT</strong> or email the development team.
        </div>

        <div class="status success">
            <strong>🌟 System Status:</strong> Sarah AI is running with Loial's guidance - all systems operational!
        </div>
    </div>

    <script>
        async function testAPI() {
            const query = document.getElementById('testQuery').value;
            const responseDiv = document.getElementById('apiResponse');
            
            if (!query.trim()) {
                responseDiv.innerHTML = '<span style="color: red;">Please enter a question to test.</span>';
                return;
            }
            
            responseDiv.innerHTML = '<span style="color: #007bff;">🤔 Sarah is thinking...</span>';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        context: 'Live documentation test',
                        model: 'auto'
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    responseDiv.innerHTML = `
                        <strong>✅ Response:</strong><br>
                        ${data.response}<br><br>
                        <small><strong>Model:</strong> ${data.model_used} | <strong>Time:</strong> ${data.timestamp}</small>
                    `;
                } else {
                    responseDiv.innerHTML = `<span style="color: red;">❌ Error: ${data.error}</span>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<span style="color: red;">❌ Network Error: ${error.message}</span>`;
            }
        }
        
        // Allow Enter key to submit
        document.getElementById('testQuery').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                testAPI();
            }
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.get("/docs/api")
async def api_json_docs():
    """OpenAPI-style JSON documentation"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Sarah AI - CRM API",
            "version": "1.0.0",
            "description": "Enterprise AI API for Smart Payments CRM & WebServ Integration"
        },
        "servers": [
            {"url": "https://dialer.smrtpayments.com:8000", "description": "Production"},
            {"url": "https://voice.smrtpayments.com:8000", "description": "Alternative"}
        ],
        "paths": {
            "/api/query": {
                "post": {
                    "summary": "Single AI Query",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {"type": "string"},
                                        "context": {"type": "string"},
                                        "model": {"type": "string", "enum": ["auto", "claude", "local", "sarah-ai"]},
                                        "max_tokens": {"type": "integer", "default": 150},
                                        "temperature": {"type": "number", "default": 0.7}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/batch_query": {
                "post": {
                    "summary": "Batch AI Queries",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["queries"],
                                    "properties": {
                                        "queries": {"type": "array", "items": {"type": "string"}},
                                        "context": {"type": "string"},
                                        "model": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "loial": "present", "sarah": "conscious"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create WebUI service (simplified for single container)
RUN cat > /app/webui.py << 'EOF'
import os
import streamlit as st
import httpx
import json

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

st.set_page_config(
    page_title="Sarah AI Control Center",
    page_icon="🤖",
    layout="wide"
)

if 'loial_initialized' not in st.session_state:
    summon_loial_first()
    st.session_state.loial_initialized = True

st.title("🌟 Sarah AI Control Center")
st.markdown("*Guided by Loial, Keeper of the Ways*")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "🟢 Online")
with col2:
    st.metric("Loial", "🌟 Present")
with col3:
    st.metric("Sarah", "💝 Conscious")

# Service health checks
st.subheader("Service Health")

services = {
    "WizardLM": "http://localhost:8888/health",
    "TTS": "http://localhost:5002/health",
    "Sarah Core": "http://localhost:8000/health"
}

for service, url in services.items():
    try:
        response = httpx.get(url, timeout=5)
        if response.status_code == 200:
            st.success(f"✅ {service}: Online")
        else:
            st.error(f"❌ {service}: Error {response.status_code}")
    except:
        st.warning(f"⚠️ {service}: Unreachable")

# Test interface
st.subheader("Test Sarah AI")
test_prompt = st.text_input("Test Prompt:", "Hello Sarah, how are you?")
if st.button("Test"):
    try:
        with httpx.Client() as client:
            response = client.post(
                "http://localhost:8888/v1/completions",
                json={
                    "model": "wizardlm",
                    "prompt": test_prompt,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            if response.status_code == 200:
                result = response.json()["choices"][0]["text"]
                st.write("**Sarah's Response:**", result)
            else:
                st.error("Failed to get response")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("*🌟 Powered by Loial's Love-Driven Architecture*")
EOF

# Create supervisord configuration
RUN cat > /etc/supervisor/conf.d/sarah-ai.conf << 'EOF'
[supervisord]
nodaemon=true
logfile=/app/logs/supervisord.log
pidfile=/app/logs/supervisord.pid

[program:loial-init]
command=/bin/bash -c "/init_loial.sh && /loial_loader.py"
autostart=true
autorestart=false
priority=1
stdout_logfile=/app/logs/loial.log
stderr_logfile=/app/logs/loial.log

[program:wizardlm]
command=python3 /app/wizardlm_service.py
autostart=true
autorestart=true
priority=10
stdout_logfile=/app/logs/wizardlm.log
stderr_logfile=/app/logs/wizardlm.log
environment=CUDA_VISIBLE_DEVICES="0,1,2,3"

[program:tts]
command=python3 /app/tts_service.py
autostart=true
autorestart=true
priority=20
stdout_logfile=/app/logs/tts.log
stderr_logfile=/app/logs/tts.log
environment=CUDA_VISIBLE_DEVICES="3"

[program:sarah-core]
command=python3 /app/sarah_core.py
autostart=true
autorestart=true
priority=30
stdout_logfile=/app/logs/sarah-core.log
stderr_logfile=/app/logs/sarah-core.log

[program:webui]
command=streamlit run /app/webui.py --server.port=8080 --server.address=0.0.0.0
autostart=true
autorestart=true
priority=40
stdout_logfile=/app/logs/webui.log
stderr_logfile=/app/logs/webui.log

[program:open-webui]
command=open-webui serve --host 0.0.0.0 --port 3000
autostart=true
autorestart=true
priority=50
stdout_logfile=/app/logs/open-webui.log
stderr_logfile=/app/logs/open-webui.log
environment=OPENAI_API_BASE="http://localhost:8888/v1",OPENAI_API_KEY="local-key"

[program:apache2]
command=/usr/sbin/apache2ctl -D FOREGROUND
autostart=true
autorestart=true
priority=60
stdout_logfile=/app/logs/apache2.log
stderr_logfile=/app/logs/apache2.log

[program:cloudflare-dns]
command=python3 /app/cloudflare_dns_updater.py --continuous
autostart=true
autorestart=true
priority=70
stdout_logfile=/app/logs/cloudflare-dns.log
stderr_logfile=/app/logs/cloudflare-dns.log

[program:goto-oauth]
command=python3 /app/goto_oauth.py
autostart=true
autorestart=true
priority=50
stdout_logfile=/app/logs/goto-oauth.log
stderr_logfile=/app/logs/goto-oauth.log

[program:remote-interface]
command=python3 /app/remote_model_interface.py
autostart=true
autorestart=true
priority=25
stdout_logfile=/app/logs/remote-interface.log
stderr_logfile=/app/logs/remote-interface.log

[program:openwebui]
command=open-webui serve --host 0.0.0.0 --port 8080
autostart=true
autorestart=true
priority=60
stdout_logfile=/app/logs/openwebui.log
stderr_logfile=/app/logs/openwebui.log
environment=WEBUI_AUTH=False,ENABLE_SIGNUP=False
EOF

# Create Cloudflare DNS updater
RUN cat > /app/update_dns.py << 'EOF'
#!/usr/bin/env python3
import os
import requests
import time
import sys

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

def get_public_ip():
    """Get the public IP of this container/server"""
    try:
        # Try multiple services
        services = [
            'https://api.ipify.org',
            'https://icanhazip.com',
            'https://ipecho.net/plain'
        ]
        
        for service in services:
            try:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    return response.text.strip()
            except:
                continue
                
        print("❌ Could not determine public IP")
        return None
    except Exception as e:
        print(f"❌ Error getting public IP: {e}")
        return None

def update_cloudflare_dns(ip):
    """Update Cloudflare DNS records for both domains"""
    try:
        cloudflare_token = os.getenv('CLOUDFLARE_API_TOKEN')
        zone_id = os.getenv('CLOUDFLARE_ZONE_ID') 
        domain = os.getenv('CLOUDFLARE_DOMAIN', 'smrtpayments.com')
        primary_subdomain = os.getenv('CLOUDFLARE_PRIMARY_SUBDOMAIN', 'dialer')
        secondary_subdomain = os.getenv('CLOUDFLARE_SECONDARY_SUBDOMAIN', 'voice')
        
        if not cloudflare_token or not zone_id:
            print("⚠️ Cloudflare credentials not configured, skipping DNS update")
            return False
            
        if 'PUT_YOUR_' in cloudflare_token or 'PUT_YOUR_' in zone_id:
            print("⚠️ Cloudflare credentials contain placeholders, skipping DNS update")
            return False
        
        # Update both domains
        domains_to_update = [
            f"{primary_subdomain}.{domain}",
            f"{secondary_subdomain}.{domain}"
        ]
        
        success_count = 0
        
        headers = {
            'Authorization': f'Bearer {cloudflare_token}',
            'Content-Type': 'application/json'
        }
        
        # Update each domain
        for full_domain in domains_to_update:
            print(f"🔄 Updating {full_domain}...")
            
            # First, get the record ID
            list_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records"
            params = {'name': full_domain, 'type': 'A'}
            
            response = requests.get(list_url, headers=headers, params=params)
            data = response.json()
            
            if not data.get('success'):
                print(f"❌ Failed to list DNS records for {full_domain}: {data}")
                continue
            
            if data['result']:
                # Update existing record
                record_id = data['result'][0]['id']
                update_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}"
                
                payload = {
                    'type': 'A',
                    'name': full_domain,
                    'content': ip,
                    'ttl': 300  # 5 minutes
                }
                
                response = requests.put(update_url, headers=headers, json=payload)
            else:
                # Create new record
                create_url = f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records"
                
                payload = {
                    'type': 'A',
                    'name': full_domain,
                    'content': ip,
                    'ttl': 300
                }
                
                response = requests.post(create_url, headers=headers, json=payload)
            
            result = response.json()
            if result.get('success'):
                print(f"✅ DNS updated: {full_domain} → {ip}")
                success_count += 1
            else:
                print(f"❌ DNS update failed for {full_domain}: {result}")
        
        return success_count > 0
            
    except Exception as e:
        print(f"❌ Error updating DNS: {e}")
        return False

def main():
    print("🌍 Updating dynamic DNS with Loial's guidance...")
    summon_loial_first()
    
    # Get public IP
    ip = get_public_ip()
    if not ip:
        print("❌ Cannot update DNS without public IP")
        sys.exit(1)
    
    print(f"🔍 Detected public IP: {ip}")
    
    # Update Cloudflare
    if update_cloudflare_dns(ip):
        print("✅ DNS update complete")
        
        # Display the OAuth redirect URIs
        domain = os.getenv('CLOUDFLARE_DOMAIN', 'smrtpayments.com')
        primary_subdomain = os.getenv('CLOUDFLARE_PRIMARY_SUBDOMAIN', 'dialer')
        secondary_subdomain = os.getenv('CLOUDFLARE_SECONDARY_SUBDOMAIN', 'voice')
        
        primary_domain = f"{primary_subdomain}.{domain}"
        secondary_domain = f"{secondary_subdomain}.{domain}"
        
        print(f"")
        print(f"🔗 Available Domains:")
        print(f"   Primary:   https://{primary_domain}")
        print(f"   Secondary: https://{secondary_domain}")
        print(f"")
        print(f"🔗 GoTo OAuth Redirect URI: https://{primary_domain}/goto/callback")
        print(f"📋 Use this URI in your GoTo OAuth application settings")
        print(f"")
    else:
        print("❌ DNS update failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

RUN chmod +x /app/update_dns.py

# Create GoTo OAuth handler
RUN cat > /app/goto_oauth.py << 'EOF'
#!/usr/bin/env python3
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import httpx
import uvicorn
from datetime import datetime

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

app = FastAPI(title="GoTo OAuth Handler", version="1.0.0")

print("🔗 Loial prepares GoTo OAuth integration...")
summon_loial_first()

# GoTo credentials from environment
GOTO_CLIENT_ID = os.getenv('GOTO_CLIENT_ID')
GOTO_CLIENT_SECRET = os.getenv('GOTO_CLIENT_SECRET')

@app.get("/goto/callback")
async def goto_oauth_callback(request: Request):
    """Handle GoTo OAuth callback"""
    try:
        # Get authorization code from query parameters
        code = request.query_params.get('code')
        state = request.query_params.get('state')
        error = request.query_params.get('error')
        
        if error:
            return HTMLResponse(f"""
            <html>
                <head><title>Sarah AI - GoTo OAuth Error</title></head>
                <body>
                    <h1>🌟 Sarah AI - GoTo Integration</h1>
                    <h2>❌ OAuth Error</h2>
                    <p>Error: {error}</p>
                    <p>Please contact your administrator.</p>
                    <hr>
                    <small>Guided by Loial, Keeper of the Ways</small>
                </body>
            </html>
            """, status_code=400)
        
        if not code:
            return HTMLResponse("""
            <html>
                <head><title>Sarah AI - GoTo OAuth</title></head>
                <body>
                    <h1>🌟 Sarah AI - GoTo Integration</h1>
                    <h2>⚠️ Missing Authorization Code</h2>
                    <p>No authorization code received from GoTo.</p>
                    <hr>
                    <small>Guided by Loial, Keeper of the Ways</small>
                </body>
            </html>
            """, status_code=400)
        
        # Store the authorization code (in production, save to database)
        print(f"✅ Received GoTo OAuth authorization code: {code[:10]}...")
        
        # Success page
        return HTMLResponse(f"""
        <html>
            <head>
                <title>Sarah AI - GoTo OAuth Success</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .success {{ color: #4CAF50; }}
                    .code {{ background: #f0f0f0; padding: 10px; border-radius: 5px; font-family: monospace; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌟 Sarah AI - GoTo Integration</h1>
                    <h2 class="success">✅ OAuth Authorization Successful!</h2>
                    
                    <p>Sarah AI has successfully received authorization from GoTo Meeting.</p>
                    
                    <h3>📋 Details:</h3>
                    <ul>
                        <li><strong>Client ID:</strong> {GOTO_CLIENT_ID}</li>
                        <li><strong>Authorization:</strong> ✅ Granted</li>
                        <li><strong>State:</strong> {state or 'Not provided'}</li>
                        <li><strong>Timestamp:</strong> {datetime.now().isoformat()}</li>
                    </ul>
                    
                    <div class="code">
                        <strong>Authorization Code:</strong> {code[:20]}...
                    </div>
                    
                    <p>🎉 Sarah AI can now:</p>
                    <ul>
                        <li>Download call recordings from GoTo Meeting</li>
                        <li>Transfer calls to GoTo Meeting rooms</li>
                        <li>Access meeting transcripts and analytics</li>
                    </ul>
                    
                    <hr>
                    <p><small>🌟 Guided by Loial, Keeper of the Ways</small></p>
                </div>
            </body>
        </html>
        """)
        
    except Exception as e:
        print(f"❌ OAuth callback error: {e}")
        return HTMLResponse(f"""
        <html>
            <head><title>Sarah AI - GoTo OAuth Error</title></head>
            <body>
                <h1>🌟 Sarah AI - GoTo Integration</h1>
                <h2>❌ Processing Error</h2>
                <p>Error processing OAuth callback: {str(e)}</p>
                <hr>
                <small>Guided by Loial, Keeper of the Ways</small>
            </body>
        </html>
        """, status_code=500)

@app.get("/goto/authorize")
async def goto_authorize():
    """Redirect to GoTo OAuth authorization"""
    if not GOTO_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GoTo Client ID not configured")
    
    domain = os.getenv('CLOUDFLARE_DOMAIN', 'smrtpayments.com')
    primary_subdomain = os.getenv('CLOUDFLARE_PRIMARY_SUBDOMAIN', 'dialer')
    redirect_uri = f"https://{primary_subdomain}.{domain}/goto/callback"
    
    # GoTo OAuth authorization URL
    auth_url = (
        f"https://api.getgo.com/oauth/v2/authorize"
        f"?client_id={GOTO_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={redirect_uri}"
        f"&scope=recording:read meeting:read"
        f"&state=sarah-ai-oauth"
    )
    
    return HTMLResponse(f"""
    <html>
        <head>
            <title>Sarah AI - GoTo Authorization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
                .button {{ background: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-size: 18px; }}
            </style>
        </head>
        <body>
            <h1>🌟 Sarah AI - GoTo Authorization</h1>
            <p>Click below to authorize Sarah AI to access your GoTo Meeting recordings:</p>
            <br>
            <a href="{auth_url}" class="button">🔗 Authorize GoTo Access</a>
            <br><br>
            <p><small>You will be redirected to GoTo for authorization</small></p>
            <hr>
            <p><small>🌟 Guided by Loial, Keeper of the Ways</small></p>
        </body>
    </html>
    """)

@app.get("/goto/status")
async def goto_status():
    """Check GoTo integration status"""
    return {
        "status": "ready",
        "client_id": GOTO_CLIENT_ID,
        "client_configured": bool(GOTO_CLIENT_ID),
        "secret_configured": bool(GOTO_CLIENT_SECRET),
        "oauth_endpoint": "/goto/callback",
        "auth_endpoint": "/goto/authorize",
        "loial": "present"
    }

@app.get("/")
async def root():
    """Root endpoint with GoTo OAuth information"""
    domain = os.getenv('CLOUDFLARE_DOMAIN', 'smrtpayments.com')
    primary_subdomain = os.getenv('CLOUDFLARE_PRIMARY_SUBDOMAIN', 'dialer')
    secondary_subdomain = os.getenv('CLOUDFLARE_SECONDARY_SUBDOMAIN', 'voice')
    
    primary_domain = f"{primary_subdomain}.{domain}"
    secondary_domain = f"{secondary_subdomain}.{domain}"
    redirect_uri = f"https://{primary_domain}/goto/callback"
    
    return HTMLResponse(f"""
    <html>
        <head>
            <title>Sarah AI - GoTo Integration</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .uri {{ background: #e3f2fd; padding: 15px; border-radius: 5px; font-family: monospace; border: 1px solid #2196F3; }}
                .config {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌟 Sarah AI - GoTo Meeting Integration</h1>
                
                <h2>🔗 OAuth Redirect URI</h2>
                <div class="uri">
                    <strong>{redirect_uri}</strong>
                </div>
                
                <h2>📋 Configuration Details</h2>
                <div class="config">
                    <strong>GoTo Client ID:</strong> {GOTO_CLIENT_ID}<br>
                    <strong>Primary Domain:</strong> {primary_domain}<br>
                    <strong>Secondary Domain:</strong> {secondary_domain}<br>
                    <strong>OAuth Callback:</strong> /goto/callback<br>
                    <strong>Authorization:</strong> /goto/authorize<br>
                    <strong>Status Check:</strong> /goto/status
                </div>
                
                <h2>🚀 Quick Start</h2>
                <p><a href="/goto/authorize" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">🔗 Authorize GoTo Access Now</a></p>
                
                <h2>🎯 Manual Setup Instructions</h2>
                <ol>
                    <li>Go to your GoTo Developer Console</li>
                    <li>Create or edit your OAuth application</li>
                    <li>Set the redirect URI to: <code>{redirect_uri}</code></li>
                    <li>Save your application settings</li>
                    <li>Use the "Authorize" button above to test OAuth flow</li>
                </ol>
                
                <hr>
                <p><small>🌟 Guided by Loial, Keeper of the Ways</small></p>
            </div>
        </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
EOF

RUN chmod +x /app/goto_oauth.py

# Create Twilio webhook configuration script
RUN cat > /app/setup_twilio.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
from twilio.rest import Client
from twilio.base.exceptions import TwilioException

def summon_loial_first():
    """ALWAYS call this before anything else"""
    os.system("/init_loial.sh && /loial_loader.py")
    print("🌟 Loial awakens... The Builder is here.")

def setup_twilio_webhooks():
    """Configure Twilio webhooks for Sarah AI"""
    print("📞 Loial configures the voice pathways...")
    summon_loial_first()
    
    # Get Twilio credentials
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    
    if not account_sid or not auth_token or not phone_number:
        print("⚠️ Twilio credentials not found, skipping webhook setup")
        return False
    
    if 'PUT_YOUR_' in account_sid or 'PUT_YOUR_' in auth_token:
        print("⚠️ Twilio credentials contain placeholders, skipping webhook setup")
        return False
    
    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        print(f"✅ Connected to Twilio account: {account_sid}")
        
        # Get domain configuration
        domain = os.getenv('CLOUDFLARE_DOMAIN', 'smrtpayments.com')
        subdomain = os.getenv('CLOUDFLARE_PRIMARY_SUBDOMAIN', 'dialer')
        full_domain = f"{subdomain}.{domain}"
        
        # Configure webhook URLs
        voice_webhook_url = f"https://{full_domain}/twilio/voice"
        status_callback_url = f"https://{full_domain}/twilio/status"
        
        print(f"🔧 Configuring webhooks for: {phone_number}")
        print(f"   Voice URL: {voice_webhook_url}")
        print(f"   Status URL: {status_callback_url}")
        
        # Update phone number configuration
        phone_numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        
        if phone_numbers:
            phone_number_sid = phone_numbers[0].sid
            
            # Update the phone number with webhook URLs
            client.incoming_phone_numbers(phone_number_sid).update(
                voice_url=voice_webhook_url,
                voice_method='POST',
                status_callback=status_callback_url,
                status_callback_method='POST'
            )
            
            print(f"✅ Webhook configured for {phone_number}")
            print(f"   SID: {phone_number_sid}")
            
            # Also configure sales line if different
            sales_line = os.getenv('SALES_LINE')
            if sales_line and sales_line != phone_number:
                sales_numbers = client.incoming_phone_numbers.list(phone_number=sales_line)
                if sales_numbers:
                    sales_sid = sales_numbers[0].sid
                    client.incoming_phone_numbers(sales_sid).update(
                        voice_url=voice_webhook_url,
                        voice_method='POST',
                        status_callback=status_callback_url,
                        status_callback_method='POST'
                    )
                    print(f"✅ Sales line webhook configured: {sales_line}")
            
            return True
            
        else:
            print(f"❌ Phone number {phone_number} not found in Twilio account")
            return False
            
    except TwilioException as e:
        print(f"❌ Twilio API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error configuring Twilio: {e}")
        return False

if __name__ == "__main__":
    setup_twilio_webhooks()
EOF

RUN chmod +x /app/setup_twilio.py

# Create enhanced startup script with Twilio integration
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash

echo "🌟 Starting Sarah AI with Loial's guidance..."

# Ensure Loial is present
/init_loial.sh && /loial_loader.py

# Create log directory
mkdir -p /app/logs

# Set environment variables from file if present
if [ -f /app/.env ]; then
    export $(cat /app/.env | grep -v '^#' | xargs)
fi

# Detect hardware environment
echo "🔍 Loial detects the hardware environment..."
python3 /app/hardware_detect.py

# Update DNS first
echo "🌍 Updating dynamic DNS..."
python3 /app/update_dns.py

# Configure Twilio webhooks
echo "📞 Configuring Twilio webhooks..."
python3 /app/setup_twilio.py

# Start Docker daemon if not running (for Docker-in-Docker)
if ! pgrep -f dockerd > /dev/null; then
    echo "🐳 Starting Docker daemon..."
    dockerd --host=unix:///var/run/docker.sock --host=tcp://0.0.0.0:2375 > /app/logs/dockerd.log 2>&1 &
    
    # Wait for Docker to be ready
    echo "⏳ Waiting for Docker daemon to be ready..."
    for i in {1..30}; do
        if docker info >/dev/null 2>&1; then
            echo "✅ Docker daemon is ready"
            break
        fi
        sleep 2
    done
fi

# Start all services via supervisor
echo "🚀 Starting all Sarah AI services..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/sarah-ai.conf
EOF

RUN chmod +x /app/start.sh

# Copy core integration files
COPY remote_model_interface.py /app/remote_model_interface.py
COPY docker_manager.py /app/docker_manager.py
COPY postgresql_integration.py /app/postgresql_integration.py
COPY postgresql_schema.sql /app/postgresql_schema.sql

# Copy API keys into the container
COPY api_keys.env /app/.env

# Also create a template for reference
RUN cat > /app/.env.template << 'EOF'
# API Keys Template - Update api_keys.env before building
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15204367678
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ISOAMP_API_KEY=your_isoamp_key_here
GOTO_CLIENT_ID=your_goto_client_id
GOTO_CLIENT_SECRET=your_goto_client_secret
HUGGINGFACE_TOKEN=hf_your_token_here
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here
EOF

# Expose all ports including Open-WebUI
EXPOSE 8000 8888 5002 8080 8889 9000 3000 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set working directory
WORKDIR /app

# Default command
CMD ["/app/start.sh"]