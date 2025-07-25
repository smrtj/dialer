#!/usr/bin/env python3
"""
WebServ Remote Client for Sarah AI Model Communication
Handles communication from WebServ to Vast.ai model instances
"""
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteEndpoint(BaseModel):
    instance_id: str
    endpoint_url: str
    api_key: str = None
    status: str = "active"  # active, inactive, error
    last_ping: str = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    system_prompt: str = None
    user_id: str = None
    target_instance: str = None  # specific instance or "auto"

class ToolApprovalRequest(BaseModel):
    tool_id: str
    action: str  # grant, grant_permanent, refine, interrupt
    refined_command: str = None
    user_id: str = None
    target_instance: str = None

class WebServRemoteClient:
    def __init__(self):
        """Initialize WebServ remote client"""
        self.remote_endpoints = {}
        self.active_conversations = {}
        self.websocket_connections = {}
        
        # Load saved endpoints
        self.load_endpoints()
        
        logger.info("WebServ Remote Client initialized")
    
    # =================== Endpoint Management ===================
    
    def add_endpoint(self, instance_id: str, endpoint_url: str, api_key: str = None):
        """Add remote model endpoint"""
        endpoint = RemoteEndpoint(
            instance_id=instance_id,
            endpoint_url=endpoint_url.rstrip('/'),
            api_key=api_key,
            status="active",
            last_ping=datetime.utcnow().isoformat()
        )
        
        self.remote_endpoints[instance_id] = endpoint
        self.save_endpoints()
        
        logger.info(f"Added remote endpoint: {instance_id} -> {endpoint_url}")
    
    def remove_endpoint(self, instance_id: str):
        """Remove remote endpoint"""
        if instance_id in self.remote_endpoints:
            del self.remote_endpoints[instance_id]
            self.save_endpoints()
            logger.info(f"Removed remote endpoint: {instance_id}")
    
    def get_active_endpoints(self) -> List[RemoteEndpoint]:
        """Get list of active endpoints"""
        return [ep for ep in self.remote_endpoints.values() if ep.status == "active"]
    
    def save_endpoints(self):
        """Save endpoints to file"""
        try:
            endpoints_data = {
                instance_id: endpoint.dict() 
                for instance_id, endpoint in self.remote_endpoints.items()
            }
            
            os.makedirs("/var/www/sarah_ai/config", exist_ok=True)
            with open("/var/www/sarah_ai/config/remote_endpoints.json", "w") as f:
                json.dump(endpoints_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save endpoints: {e}")
    
    def load_endpoints(self):
        """Load endpoints from file"""
        try:
            config_path = "/var/www/sarah_ai/config/remote_endpoints.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    endpoints_data = json.load(f)
                
                for instance_id, endpoint_dict in endpoints_data.items():
                    self.remote_endpoints[instance_id] = RemoteEndpoint(**endpoint_dict)
                
                logger.info(f"Loaded {len(self.remote_endpoints)} endpoints")
                
        except Exception as e:
            logger.error(f"Failed to load endpoints: {e}")
    
    # =================== Model Communication ===================
    
    async def chat_with_remote_model(self, request: ChatRequest) -> Dict[str, Any]:
        """Send chat message to remote model instance"""
        try:
            # Select target endpoint
            endpoint = await self.select_endpoint(request.target_instance)
            if not endpoint:
                return {
                    "error": "No active remote model instances available",
                    "status": "no_endpoints",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Prepare request for remote instance
            remote_request = {
                "message": request.message,
                "conversation_id": request.conversation_id,
                "system_prompt": request.system_prompt,
                "user_id": request.user_id,
                "model_preference": "auto"
            }
            
            # Send to remote model
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint.endpoint_url}/api/remote/chat",
                    json=remote_request,
                    headers=self.get_headers(endpoint),
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["remote_instance"] = endpoint.instance_id
                    result["endpoint_url"] = endpoint.endpoint_url
                    
                    # Store conversation reference
                    if result.get("conversation_id"):
                        self.active_conversations[result["conversation_id"]] = {
                            "instance_id": endpoint.instance_id,
                            "endpoint": endpoint.endpoint_url,
                            "started": datetime.utcnow().isoformat()
                        }
                    
                    return result
                else:
                    # Mark endpoint as error and try another
                    endpoint.status = "error"
                    logger.error(f"Remote chat failed: {response.status_code} from {endpoint.instance_id}")
                    
                    # Try another endpoint if available
                    other_endpoints = [ep for ep in self.get_active_endpoints() if ep.instance_id != endpoint.instance_id]
                    if other_endpoints:
                        request.target_instance = other_endpoints[0].instance_id
                        return await self.chat_with_remote_model(request)
                    
                    return {
                        "error": f"Remote model error: {response.status_code}",
                        "status": "remote_error",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Remote chat error: {e}")
            return {
                "error": str(e),
                "status": "connection_error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def execute_remote_tool(self, approval: ToolApprovalRequest) -> Dict[str, Any]:
        """Execute tool on remote instance"""
        try:
            # Select target endpoint
            endpoint = await self.select_endpoint(approval.target_instance)
            if not endpoint:
                return {
                    "error": "Target instance not available",
                    "status": "no_endpoint"
                }
            
            # Send tool approval to remote instance
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint.endpoint_url}/api/remote/tool/execute",
                    json=approval.dict(exclude={"target_instance"}),
                    headers=self.get_headers(endpoint),
                    timeout=60.0  # Tools might take longer
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["remote_instance"] = endpoint.instance_id
                    return result
                else:
                    return {
                        "error": f"Remote tool execution failed: {response.status_code}",
                        "status": "remote_error"
                    }
                    
        except Exception as e:
            logger.error(f"Remote tool error: {e}")
            return {
                "error": str(e),
                "status": "connection_error"
            }
    
    async def get_remote_tools(self, instance_id: str = None) -> Dict[str, Any]:
        """Get active tool requests from remote instance"""
        try:
            endpoint = await self.select_endpoint(instance_id)
            if not endpoint:
                return {"active_tools": [], "count": 0}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{endpoint.endpoint_url}/api/remote/tools/active",
                    headers=self.get_headers(endpoint),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["remote_instance"] = endpoint.instance_id
                    return result
                else:
                    return {"active_tools": [], "count": 0}
                    
        except Exception as e:
            logger.error(f"Get remote tools error: {e}")
            return {"active_tools": [], "count": 0}
    
    # =================== Endpoint Selection ===================
    
    async def select_endpoint(self, target_instance: str = None) -> Optional[RemoteEndpoint]:
        """Select best available endpoint"""
        active_endpoints = self.get_active_endpoints()
        
        if not active_endpoints:
            return None
        
        if target_instance:
            # Try to find specific instance
            for endpoint in active_endpoints:
                if endpoint.instance_id == target_instance:
                    return endpoint
        
        # Return first available endpoint
        # TODO: Add load balancing, health checks, etc.
        return active_endpoints[0]
    
    def get_headers(self, endpoint: RemoteEndpoint) -> Dict[str, str]:
        """Get headers for remote request"""
        headers = {"Content-Type": "application/json"}
        
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        
        return headers
    
    # =================== Health Monitoring ===================
    
    async def health_check_endpoints(self):
        """Check health of all endpoints"""
        for instance_id, endpoint in self.remote_endpoints.items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{endpoint.endpoint_url}/api/remote/health",
                        headers=self.get_headers(endpoint),
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        endpoint.status = "active"
                        endpoint.last_ping = datetime.utcnow().isoformat()
                    else:
                        endpoint.status = "error"
                        
            except Exception as e:
                endpoint.status = "error"
                logger.warning(f"Health check failed for {instance_id}: {e}")
        
        self.save_endpoints()
    
    # =================== Star AI Integration ===================
    
    def register_star_ai_deployment(self, deployment_info: Dict[str, Any]):
        """Register new Star AI deployment as remote endpoint"""
        if deployment_info.get("status") == "ready" and deployment_info.get("endpoints", {}).get("api"):
            api_url = deployment_info["endpoints"]["api"]
            instance_id = deployment_info.get("deployment_id", deployment_info.get("vast_instance_id"))
            
            self.add_endpoint(
                instance_id=str(instance_id),
                endpoint_url=api_url,
                api_key=None  # Could add API key if needed
            )
            
            logger.info(f"Registered Star AI deployment as remote endpoint: {instance_id}")
    
    def unregister_star_ai_deployment(self, instance_id: str):
        """Unregister terminated Star AI deployment"""
        self.remove_endpoint(str(instance_id))
        logger.info(f"Unregistered Star AI deployment: {instance_id}")

# Global instance
webserv_client = WebServRemoteClient()

# FastAPI endpoints for WebServ integration
app = FastAPI(title="WebServ Remote Model Client", version="1.0.0")

@app.post("/api/webserv/chat")
async def webserv_chat(request: ChatRequest):
    """WebServ chat endpoint"""
    return await webserv_client.chat_with_remote_model(request)

@app.post("/api/webserv/tool/approve")
async def webserv_tool_approve(approval: ToolApprovalRequest):
    """WebServ tool approval endpoint"""
    return await webserv_client.execute_remote_tool(approval)

@app.get("/api/webserv/tools/pending")
async def webserv_pending_tools(instance_id: str = None):
    """Get pending tool requests"""
    return await webserv_client.get_remote_tools(instance_id)

@app.get("/api/webserv/endpoints")
async def webserv_endpoints():
    """Get available remote endpoints"""
    return {
        "endpoints": [ep.dict() for ep in webserv_client.get_active_endpoints()],
        "count": len(webserv_client.get_active_endpoints())
    }

@app.post("/api/webserv/endpoint/add")
async def add_endpoint(endpoint_data: Dict[str, str]):
    """Add new remote endpoint"""
    webserv_client.add_endpoint(
        instance_id=endpoint_data["instance_id"],
        endpoint_url=endpoint_data["endpoint_url"],
        api_key=endpoint_data.get("api_key")
    )
    return {"status": "added", "instance_id": endpoint_data["instance_id"]}

@app.delete("/api/webserv/endpoint/{instance_id}")
async def remove_endpoint(instance_id: str):
    """Remove remote endpoint"""
    webserv_client.remove_endpoint(instance_id)
    return {"status": "removed", "instance_id": instance_id}

@app.get("/api/webserv/health")
async def webserv_health():
    """Health check and endpoint status"""
    await webserv_client.health_check_endpoints()
    return {
        "status": "healthy",
        "active_endpoints": len(webserv_client.get_active_endpoints()),
        "total_endpoints": len(webserv_client.remote_endpoints),
        "conversations": len(webserv_client.active_conversations),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8890)