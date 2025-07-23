#!/usr/bin/env python3
"""
Remote Model Interface for Sarah AI
Handles communication between WebServ and Vast.ai model instances
"""
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import subprocess
import signal
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: str
    timestamp: str = None
    message_id: str = None

class ToolRequest(BaseModel):
    tool_id: str
    command: str
    description: str
    risk_level: str  # low, medium, high, critical
    requires_approval: bool = True

class ToolApproval(BaseModel):
    tool_id: str
    action: str  # grant, grant_permanent, refine, interrupt
    refined_command: str = None
    user_id: str = None

class ModelChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    system_prompt: str = None
    user_id: str = None
    model_preference: str = "auto"  # auto, qwen, llama, deepseek

class RemoteModelInterface:
    def __init__(self):
        """Initialize remote model interface"""
        self.active_conversations = {}
        self.active_tools = {}
        self.user_permissions = {}
        self.tool_whitelist = {}
        self.running_processes = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections = {}
        
        logger.info("Remote Model Interface initialized")
    
    # =================== Model Communication ===================
    
    async def chat_with_model(self, request: ModelChatRequest) -> Dict[str, Any]:
        """Send chat message to local model"""
        try:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Prepare the chat request
            model_request = {
                "model": "local",
                "prompt": self.build_prompt(request),
                "max_tokens": 500,
                "temperature": 0.7,
                "conversation_id": conversation_id
            }
            
            # Send to local model service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8888/v1/completions",
                    json=model_request,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    model_response = response.json()
                    
                    # Parse response for tool requests
                    tool_requests = await self.parse_tool_requests(model_response.get("choices", [{}])[0].get("text", ""))
                    
                    result = {
                        "conversation_id": conversation_id,
                        "response": model_response.get("choices", [{}])[0].get("text", ""),
                        "model_used": model_response.get("model", "unknown"),
                        "tool_requests": tool_requests,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "success"
                    }
                    
                    # Store conversation
                    self.active_conversations[conversation_id] = result
                    
                    return result
                else:
                    raise HTTPException(status_code=500, detail=f"Model service error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def build_prompt(self, request: ModelChatRequest) -> str:
        """Build comprehensive prompt for model"""
        prompt_parts = []
        
        # System prompt (admin/owner only)
        if request.system_prompt and self.can_use_system_prompt(request.user_id):
            prompt_parts.append(f"SYSTEM: {request.system_prompt}")
        
        # Standard system instructions
        prompt_parts.append("""
You are Sarah, an AI assistant with system access tools. You can help users manage their Sarah AI deployment.

Available tools:
- system_command: Execute shell commands (requires approval)
- file_operation: Read/write files (requires approval) 
- service_control: Manage system services (requires approval)
- deployment_manager: Handle Star AI deployments (requires approval)
- configuration_manager: Update system configs (requires approval)

When you need to use a tool, format your request as:
TOOL_REQUEST: {tool_name} | {command} | {description} | {risk_level}

Example: TOOL_REQUEST: system_command | systemctl status sarah-ai-webui | Check Sarah AI service status | low
        """)
        
        # User message
        prompt_parts.append(f"USER: {request.message}")
        prompt_parts.append("SARAH:")
        
        return "\n".join(prompt_parts)
    
    async def parse_tool_requests(self, response: str) -> List[ToolRequest]:
        """Parse tool requests from model response"""
        tool_requests = []
        
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('TOOL_REQUEST:'):
                try:
                    # Parse: TOOL_REQUEST: tool_name | command | description | risk_level
                    parts = line.replace('TOOL_REQUEST:', '').split('|')
                    if len(parts) >= 4:
                        tool_request = ToolRequest(
                            tool_id=str(uuid.uuid4()),
                            command=parts[1].strip(),
                            description=parts[2].strip(),
                            risk_level=parts[3].strip(),
                            requires_approval=self.requires_approval(parts[1].strip())
                        )
                        tool_requests.append(tool_request)
                        self.active_tools[tool_request.tool_id] = tool_request
                        
                except Exception as e:
                    logger.error(f"Failed to parse tool request: {e}")
                    continue
        
        return tool_requests
    
    # =================== Tool Execution ===================
    
    async def execute_tool(self, approval: ToolApproval) -> Dict[str, Any]:
        """Execute approved tool"""
        try:
            if approval.tool_id not in self.active_tools:
                raise HTTPException(status_code=404, detail="Tool request not found")
            
            tool_request = self.active_tools[approval.tool_id]
            
            if approval.action == "interrupt":
                return await self.interrupt_tool(approval.tool_id)
            
            command = approval.refined_command if approval.refined_command else tool_request.command
            
            # Check whitelist for permanent grants
            if approval.action == "grant_permanent":
                self.add_to_whitelist(approval.user_id, command)
            
            # Execute the tool
            result = await self.execute_command(command, approval.tool_id)
            
            # Clean up
            if approval.tool_id in self.active_tools:
                del self.active_tools[approval.tool_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "error": str(e),
                "status": "error",
                "tool_id": approval.tool_id
            }
    
    async def execute_command(self, command: str, tool_id: str) -> Dict[str, Any]:
        """Execute shell command with monitoring"""
        try:
            logger.info(f"Executing command: {command}")
            
            # Start process
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.running_processes[tool_id] = process
            start_time = datetime.utcnow()
            
            # Wait for completion
            stdout, stderr = await process.communicate()
            
            # Clean up
            if tool_id in self.running_processes:
                del self.running_processes[tool_id]
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "tool_id": tool_id,
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else "",
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "status": "completed"
            }
            
            logger.info(f"Command completed in {duration:.2f}s with return code {process.returncode}")
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                "tool_id": tool_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def interrupt_tool(self, tool_id: str) -> Dict[str, Any]:
        """Interrupt running tool"""
        try:
            if tool_id in self.running_processes:
                process = self.running_processes[tool_id]
                
                # Try graceful termination first
                process.terminate()
                
                # Wait briefly for graceful shutdown
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if necessary
                    process.kill()
                    await process.wait()
                
                # Clean up
                del self.running_processes[tool_id]
                
                return {
                    "tool_id": tool_id,
                    "status": "interrupted",
                    "message": "Tool execution interrupted by user"
                }
            else:
                return {
                    "tool_id": tool_id,
                    "status": "not_found",
                    "message": "No running process found for tool"
                }
                
        except Exception as e:
            logger.error(f"Failed to interrupt tool: {e}")
            return {
                "tool_id": tool_id,
                "error": str(e),
                "status": "error"
            }
    
    # =================== Permission Management ===================
    
    def can_use_system_prompt(self, user_id: str) -> bool:
        """Check if user can use system prompts"""
        if not user_id:
            return False
        
        # Only admin and owner can use system prompts
        admin_users = os.getenv("ADMIN_USERS", "admin,owner").split(",")
        return user_id.lower() in [u.strip().lower() for u in admin_users]
    
    def requires_approval(self, command: str) -> bool:
        """Determine if command requires approval"""
        # Always approve for now - whitelist will handle automatic approvals
        dangerous_patterns = [
            "rm -rf", "format", "mkfs", "dd if=", ">/dev/", 
            "shutdown", "reboot", "halt", "init 0", "init 6"
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return True
        
        return True  # Default to requiring approval
    
    def add_to_whitelist(self, user_id: str, command: str):
        """Add command to user's whitelist"""
        if user_id not in self.tool_whitelist:
            self.tool_whitelist[user_id] = set()
        
        self.tool_whitelist[user_id].add(command.strip())
        logger.info(f"Added to whitelist for {user_id}: {command}")
    
    def is_whitelisted(self, user_id: str, command: str) -> bool:
        """Check if command is whitelisted for user"""
        if user_id not in self.tool_whitelist:
            return False
        
        return command.strip() in self.tool_whitelist[user_id]

# Global instance
remote_model = RemoteModelInterface()

# FastAPI app for remote endpoints
app = FastAPI(title="Sarah AI Remote Model Interface", version="1.0.0")

@app.post("/api/remote/chat")
async def remote_chat(request: ModelChatRequest):
    """Remote chat endpoint for WebServ"""
    return await remote_model.chat_with_model(request)

@app.post("/api/remote/tool/execute")
async def execute_remote_tool(approval: ToolApproval):
    """Execute tool with approval"""
    return await remote_model.execute_tool(approval)

@app.get("/api/remote/tools/active")
async def get_active_tools():
    """Get active tool requests"""
    return {
        "active_tools": list(remote_model.active_tools.values()),
        "count": len(remote_model.active_tools)
    }

@app.get("/api/remote/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id in remote_model.active_conversations:
        return remote_model.active_conversations[conversation_id]
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/api/remote/health")
async def remote_health():
    """Health check for remote interface"""
    return {
        "status": "healthy",
        "active_conversations": len(remote_model.active_conversations),
        "active_tools": len(remote_model.active_tools),
        "running_processes": len(remote_model.running_processes),
        "timestamp": datetime.utcnow().isoformat()
    }

# WebSocket for real-time updates
@app.websocket("/api/remote/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time communication"""
    await websocket.accept()
    remote_model.websocket_connections[user_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle messages
            data = await websocket.receive_text()
            # Handle any WebSocket messages if needed
            
    except WebSocketDisconnect:
        if user_id in remote_model.websocket_connections:
            del remote_model.websocket_connections[user_id]
        logger.info(f"WebSocket disconnected: {user_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)