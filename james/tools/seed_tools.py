"""Seed tools for James consciousness system."""

import os
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import tempfile

from james.core.message import Message, MessageSource, MessagePriority
from james.core.a2a import SubAgentProtocol, A2AMessage, A2AResponse
from james.core.sandbox import SecureSandbox


class SeedTools(SubAgentProtocol):
    """Core seed tools that enable James to bootstrap capabilities."""
    
    def __init__(self, james_home: str = "~/.james") -> None:
        self.agent_id = "seed_tools"
        self.name = "Seed Tools"
        self.description = "Core tools for file operations, terminal commands, messaging, and reflection"
        self.capabilities = [
            "write_file", "read_file", "execute_command", "send_message", 
            "retry_operation", "emit_external", "self_reflection"
        ]
        
        self.james_home = Path(james_home).expanduser()
        self.james_home.mkdir(parents=True, exist_ok=True)
        
        # Initialize secure sandbox
        self.sandbox = SecureSandbox(james_home)
        
    async def handle_message(self, message: A2AMessage) -> A2AResponse:
        """Handle incoming A2A messages."""
        try:
            action = message.action
            payload = message.payload
            
            if action == "write_file":
                result = await self.write_file(
                    payload.get("path", ""),
                    payload.get("content", ""),
                    payload.get("mode", "w")
                )
            elif action == "read_file":
                result = await self.read_file(payload.get("path", ""))
            elif action == "execute_command":
                result = await self.execute_command(
                    payload.get("command", ""),
                    payload.get("working_dir"),
                    payload.get("timeout", 30)
                )
            elif action == "send_message":
                result = await self.send_message(
                    payload.get("content", ""),
                    payload.get("source", "internal"),
                    payload.get("priority", "medium")
                )
            elif action == "retry_operation":
                result = await self.retry_operation(
                    payload.get("operation"),
                    payload.get("max_retries", 3),
                    payload.get("delay", 1)
                )
            elif action == "emit_external":
                result = await self.emit_external(
                    payload.get("message", ""),
                    payload.get("channel", "default")
                )
            elif action == "self_reflection":
                result = await self.self_reflection(
                    payload.get("topic", ""),
                    payload.get("context", {})
                )
            else:
                return A2AResponse(
                    message_id=message.message_id,
                    success=False,
                    error=f"Unknown action: {action}"
                )
            
            return A2AResponse(
                message_id=message.message_id,
                success=True,
                result=result
            )
            
        except Exception as e:
            return A2AResponse(
                message_id=message.message_id,
                success=False,
                error=str(e)
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the seed tools."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "james_home": str(self.james_home),
            "sandbox_info": self.sandbox.get_sandbox_info(),
            "operational": True
        }
    
    async def write_file(self, path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """Write content to a file in the James home directory."""
        try:
            # Ensure path is within james_home for security
            file_path = self._secure_path(path)
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode('utf-8')),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read content from a file."""
        try:
            file_path = self._secure_path(path)
            
            if not file_path.exists():
                return {"success": False, "error": "File not found"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_command(self, command: str, working_dir: Optional[str] = None, 
                            timeout: int = 30, use_docker: bool = True) -> Dict[str, Any]:
        """Execute a terminal command with enhanced sandboxing."""
        try:
            result = await self.sandbox.execute_command(
                command=command,
                use_docker=use_docker,
                working_dir=working_dir,
                timeout=timeout
            )
            
            # Add timestamp to result
            result["timestamp"] = datetime.now().isoformat()
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
    
    async def send_message(self, content: str, source: str = "internal", 
                          priority: str = "medium") -> Dict[str, Any]:
        """Send a message to the consciousness queue."""
        try:
            # This would integrate with the main consciousness system
            # For now, just log the message
            message_data = {
                "content": content,
                "source": source,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "sender": self.agent_id
            }
            
            # Write to a messages log file
            messages_file = self.james_home / "internal_messages.jsonl"
            with open(messages_file, 'a') as f:
                f.write(json.dumps(message_data) + '\n')
            
            return {
                "success": True,
                "message_logged": True,
                "message_id": message_data["timestamp"]  # Simple ID for now
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def retry_operation(self, operation: Dict[str, Any], max_retries: int = 3, 
                            delay: float = 1.0) -> Dict[str, Any]:
        """Retry a failed operation with exponential backoff."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # This would execute the operation based on the operation dict
                # For now, just simulate based on operation type
                op_type = operation.get("type", "unknown")
                
                if op_type == "write_file":
                    result = await self.write_file(
                        operation.get("path", ""),
                        operation.get("content", "")
                    )
                elif op_type == "execute_command":
                    result = await self.execute_command(
                        operation.get("command", "")
                    )
                else:
                    return {"success": False, "error": f"Unknown operation type: {op_type}"}
                
                if result.get("success"):
                    return {
                        "success": True,
                        "attempts": attempt + 1,
                        "result": result
                    }
                else:
                    last_error = result.get("error", "Unknown error")
                    
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
        
        return {
            "success": False,
            "attempts": max_retries + 1,
            "final_error": last_error
        }
    
    async def emit_external(self, message: str, channel: str = "default") -> Dict[str, Any]:
        """Emit a message to external systems (like the web UI)."""
        try:
            # Write to external messages file
            external_file = self.james_home / f"external_{channel}.jsonl"
            
            message_data = {
                "message": message,
                "channel": channel,
                "timestamp": datetime.now().isoformat(),
                "sender": self.agent_id
            }
            
            with open(external_file, 'a') as f:
                f.write(json.dumps(message_data) + '\n')
            
            return {
                "success": True,
                "message": message,
                "channel": channel,
                "emitted_at": message_data["timestamp"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def self_reflection(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-reflection and log insights."""
        try:
            reflection_data = {
                "topic": topic,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "reflector": self.agent_id,
                "insights": [
                    "This is a placeholder for actual reflection logic",
                    "Would integrate with LLM for deeper introspection",
                    "Could analyze patterns in behavior and decision-making"
                ]
            }
            
            # Log reflection
            reflection_file = self.james_home / "reflections.jsonl"
            with open(reflection_file, 'a') as f:
                f.write(json.dumps(reflection_data) + '\n')
            
            return {
                "success": True,
                "reflection_logged": True,
                "topic": topic,
                "insights_count": len(reflection_data["insights"])
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _secure_path(self, path: str, base_dir: Optional[Path] = None) -> Path:
        """Ensure path is within the allowed directory."""
        if base_dir is None:
            base_dir = self.james_home
        
        # Convert to absolute path and resolve
        if os.path.isabs(path):
            # If absolute, make relative to base_dir
            path = path.lstrip('/')
        
        target_path = (base_dir / path).resolve()
        
        # Ensure the path is within base_dir
        if not str(target_path).startswith(str(base_dir.resolve())):
            raise ValueError(f"Path {path} is outside allowed directory {base_dir}")
        
        return target_path