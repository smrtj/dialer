#!/usr/bin/env python3
"""
Docker Manager for Sarah AI
Provides Docker-in-Docker capabilities for Vast.ai instances
"""
import os
import json
import asyncio
import docker
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerContainer(BaseModel):
    container_id: str
    name: str
    image: str
    status: str
    ports: Dict[str, Any] = {}
    created: str
    command: str = ""

class DockerImage(BaseModel):
    image_id: str
    repository: str
    tag: str
    size: str
    created: str

class DockerBuildRequest(BaseModel):
    dockerfile_content: str
    image_name: str
    build_context: Dict[str, str] = {}
    build_args: Dict[str, str] = {}

class DockerRunRequest(BaseModel):
    image: str
    name: str = None
    ports: Dict[str, int] = {}
    environment: Dict[str, str] = {}
    volumes: Dict[str, str] = {}
    command: str = None
    detach: bool = True

class DockerManager:
    def __init__(self):
        """Initialize Docker manager"""
        try:
            # Initialize Docker client
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
            
            # Test Docker connection
            self.client.ping()
            logger.info("Docker daemon connection verified")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Docker is available"""
        return self.client is not None
    
    # =================== Container Management ===================
    
    def list_containers(self, all_containers: bool = False) -> List[DockerContainer]:
        """List Docker containers"""
        try:
            if not self.is_available():
                return []
            
            containers = self.client.containers.list(all=all_containers)
            container_list = []
            
            for container in containers:
                container_info = DockerContainer(
                    container_id=container.id[:12],
                    name=container.name,
                    image=container.image.tags[0] if container.image.tags else container.image.id,
                    status=container.status,
                    ports=container.ports,
                    created=container.attrs.get('Created', ''),
                    command=container.attrs.get('Config', {}).get('Cmd', [''])[0] if container.attrs.get('Config', {}).get('Cmd') else ''
                )
                container_list.append(container_info)
            
            return container_list
            
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []
    
    def start_container(self, container_name_or_id: str) -> Dict[str, Any]:
        """Start a Docker container"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            container = self.client.containers.get(container_name_or_id)
            container.start()
            
            return {
                "container_id": container.id[:12],
                "name": container.name,
                "status": "started",
                "message": f"Container {container.name} started successfully"
            }
            
        except docker.errors.NotFound:
            return {"error": f"Container {container_name_or_id} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return {"error": str(e), "status": "error"}
    
    def stop_container(self, container_name_or_id: str, timeout: int = 10) -> Dict[str, Any]:
        """Stop a Docker container"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            container = self.client.containers.get(container_name_or_id)
            container.stop(timeout=timeout)
            
            return {
                "container_id": container.id[:12],
                "name": container.name,
                "status": "stopped",
                "message": f"Container {container.name} stopped successfully"
            }
            
        except docker.errors.NotFound:
            return {"error": f"Container {container_name_or_id} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return {"error": str(e), "status": "error"}
    
    def remove_container(self, container_name_or_id: str, force: bool = False) -> Dict[str, Any]:
        """Remove a Docker container"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            container = self.client.containers.get(container_name_or_id)
            container_name = container.name
            container.remove(force=force)
            
            return {
                "container_id": container.id[:12] if hasattr(container, 'id') else 'unknown',
                "name": container_name,
                "status": "removed",
                "message": f"Container {container_name} removed successfully"
            }
            
        except docker.errors.NotFound:
            return {"error": f"Container {container_name_or_id} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return {"error": str(e), "status": "error"}
    
    def run_container(self, request: DockerRunRequest) -> Dict[str, Any]:
        """Run a new Docker container"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            # Prepare run arguments
            run_kwargs = {
                "image": request.image,
                "detach": request.detach,
                "remove": False,  # Don't auto-remove so we can manage it
            }
            
            if request.name:
                run_kwargs["name"] = request.name
            
            if request.ports:
                run_kwargs["ports"] = request.ports
            
            if request.environment:
                run_kwargs["environment"] = request.environment
            
            if request.volumes:
                run_kwargs["volumes"] = request.volumes
            
            if request.command:
                run_kwargs["command"] = request.command
            
            # Run the container
            container = self.client.containers.run(**run_kwargs)
            
            return {
                "container_id": container.id[:12],
                "name": container.name,
                "image": request.image,
                "status": "running",
                "message": f"Container {container.name} started successfully"
            }
            
        except docker.errors.ImageNotFound:
            return {"error": f"Image {request.image} not found", "status": "image_not_found"}
        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            return {"error": str(e), "status": "error"}
    
    def get_container_logs(self, container_name_or_id: str, tail: int = 100) -> Dict[str, Any]:
        """Get container logs"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            container = self.client.containers.get(container_name_or_id)
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            
            return {
                "container_id": container.id[:12],
                "name": container.name,
                "logs": logs,
                "status": "success"
            }
            
        except docker.errors.NotFound:
            return {"error": f"Container {container_name_or_id} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return {"error": str(e), "status": "error"}
    
    # =================== Image Management ===================
    
    def list_images(self) -> List[DockerImage]:
        """List Docker images"""
        try:
            if not self.is_available():
                return []
            
            images = self.client.images.list()
            image_list = []
            
            for image in images:
                # Get repository and tag
                repo_tags = image.tags[0].split(':') if image.tags else ['<none>', '<none>']
                repository = repo_tags[0] if len(repo_tags) > 0 else '<none>'
                tag = repo_tags[1] if len(repo_tags) > 1 else '<none>'
                
                image_info = DockerImage(
                    image_id=image.id.split(':')[1][:12],
                    repository=repository,
                    tag=tag,
                    size=f"{image.attrs.get('Size', 0) // 1024 // 1024} MB",
                    created=image.attrs.get('Created', '')
                )
                image_list.append(image_info)
            
            return image_list
            
        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            return []
    
    def pull_image(self, image_name: str) -> Dict[str, Any]:
        """Pull a Docker image"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            logger.info(f"Pulling image: {image_name}")
            image = self.client.images.pull(image_name)
            
            return {
                "image_id": image.id.split(':')[1][:12],
                "image_name": image_name,
                "status": "pulled",
                "message": f"Image {image_name} pulled successfully"
            }
            
        except docker.errors.ImageNotFound:
            return {"error": f"Image {image_name} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to pull image: {e}")
            return {"error": str(e), "status": "error"}
    
    def remove_image(self, image_name_or_id: str, force: bool = False) -> Dict[str, Any]:
        """Remove a Docker image"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            self.client.images.remove(image_name_or_id, force=force)
            
            return {
                "image_name": image_name_or_id,
                "status": "removed",
                "message": f"Image {image_name_or_id} removed successfully"
            }
            
        except docker.errors.ImageNotFound:
            return {"error": f"Image {image_name_or_id} not found", "status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to remove image: {e}")
            return {"error": str(e), "status": "error"}
    
    def build_image(self, request: DockerBuildRequest) -> Dict[str, Any]:
        """Build a Docker image from Dockerfile content"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            # Create temporary build context
            import tempfile
            import tarfile
            import io
            
            # Create build context
            build_context = io.BytesIO()
            tar = tarfile.TarFile(fileobj=build_context, mode='w')
            
            # Add Dockerfile
            dockerfile_data = request.dockerfile_content.encode('utf-8')
            dockerfile_info = tarfile.TarInfo(name='Dockerfile')
            dockerfile_info.size = len(dockerfile_data)
            tar.addfile(dockerfile_info, io.BytesIO(dockerfile_data))
            
            # Add additional files from build context
            for filename, content in request.build_context.items():
                file_data = content.encode('utf-8')
                file_info = tarfile.TarInfo(name=filename)
                file_info.size = len(file_data)
                tar.addfile(file_info, io.BytesIO(file_data))
            
            tar.close()
            build_context.seek(0)
            
            # Build the image
            logger.info(f"Building image: {request.image_name}")
            image, logs = self.client.images.build(
                fileobj=build_context,
                custom_context=True,
                tag=request.image_name,
                buildargs=request.build_args
            )
            
            # Extract build logs
            build_log = []
            for log_line in logs:
                if 'stream' in log_line:
                    build_log.append(log_line['stream'].strip())
            
            return {
                "image_id": image.id.split(':')[1][:12],
                "image_name": request.image_name,
                "status": "built",
                "message": f"Image {request.image_name} built successfully",
                "build_logs": build_log
            }
            
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return {"error": str(e), "status": "error"}
    
    # =================== System Information ===================
    
    def get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information"""
        try:
            if not self.is_available():
                return {"error": "Docker not available", "status": "error"}
            
            info = self.client.info()
            return {
                "containers": info.get('Containers', 0),
                "containers_running": info.get('ContainersRunning', 0),
                "containers_paused": info.get('ContainersPaused', 0),
                "containers_stopped": info.get('ContainersStopped', 0),
                "images": info.get('Images', 0),
                "docker_version": info.get('ServerVersion', 'unknown'),
                "kernel_version": info.get('KernelVersion', 'unknown'),
                "operating_system": info.get('OperatingSystem', 'unknown'),
                "total_memory": f"{info.get('MemTotal', 0) // 1024 // 1024 // 1024} GB",
                "cpu_cores": info.get('NCPU', 0),
                "status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get Docker info: {e}")
            return {"error": str(e), "status": "error"}

# Global Docker manager instance
docker_manager = DockerManager()

# Tool functions for remote model interface
def get_docker_tools() -> List[Dict[str, Any]]:
    """Get available Docker tools"""
    return [
        {
            "name": "docker_list_containers",
            "description": "List Docker containers",
            "risk_level": "low"
        },
        {
            "name": "docker_start_container", 
            "description": "Start a Docker container",
            "risk_level": "medium"
        },
        {
            "name": "docker_stop_container",
            "description": "Stop a Docker container", 
            "risk_level": "medium"
        },
        {
            "name": "docker_run_container",
            "description": "Run a new Docker container",
            "risk_level": "high"
        },
        {
            "name": "docker_build_image",
            "description": "Build a Docker image",
            "risk_level": "high"
        },
        {
            "name": "docker_pull_image",
            "description": "Pull a Docker image",
            "risk_level": "medium"
        },
        {
            "name": "docker_info",
            "description": "Get Docker system information",
            "risk_level": "low"
        }
    ]

async def execute_docker_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Docker tool"""
    try:
        if tool_name == "docker_list_containers":
            containers = docker_manager.list_containers(all_containers=parameters.get('all', False))
            return {"containers": [c.dict() for c in containers], "status": "success"}
        
        elif tool_name == "docker_start_container":
            return docker_manager.start_container(parameters.get('container_name_or_id'))
        
        elif tool_name == "docker_stop_container":
            return docker_manager.stop_container(
                parameters.get('container_name_or_id'),
                timeout=parameters.get('timeout', 10)
            )
        
        elif tool_name == "docker_run_container":
            request = DockerRunRequest(**parameters)
            return docker_manager.run_container(request)
        
        elif tool_name == "docker_build_image":
            request = DockerBuildRequest(**parameters)
            return docker_manager.build_image(request)
        
        elif tool_name == "docker_pull_image":
            return docker_manager.pull_image(parameters.get('image_name'))
        
        elif tool_name == "docker_info":
            return docker_manager.get_docker_info()
        
        else:
            return {"error": f"Unknown Docker tool: {tool_name}", "status": "error"}
            
    except Exception as e:
        logger.error(f"Docker tool execution failed: {e}")
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    # Test Docker availability
    if docker_manager.is_available():
        print("✅ Docker is available")
        info = docker_manager.get_docker_info()
        print(f"📊 Docker Info: {info}")
    else:
        print("❌ Docker is not available")