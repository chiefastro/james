"""LangSmith integration for observability of James consciousness system."""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager


class JamesObservability:
    """Observability system for James using LangSmith."""
    
    def __init__(self) -> None:
        self.langsmith_client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None
        
        # Initialize LangSmith if configured
        if self._is_langsmith_configured():
            self._initialize_langsmith()
    
    def _is_langsmith_configured(self) -> bool:
        """Check if LangSmith is properly configured."""
        return (
            os.getenv("LANGCHAIN_API_KEY") is not None and
            os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        )
    
    def _initialize_langsmith(self) -> None:
        """Initialize LangSmith client and tracer."""
        try:
            self.langsmith_client = Client(
                api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
                api_key=os.getenv("LANGCHAIN_API_KEY")
            )
            
            self.tracer = LangChainTracer(
                project_name=os.getenv("LANGCHAIN_PROJECT", "james"),
                client=self.langsmith_client
            )
            
            print("LangSmith observability initialized")
            
        except Exception as e:
            print(f"Failed to initialize LangSmith: {e}")
            self.langsmith_client = None
            self.tracer = None
    
    def get_callback_manager(self) -> CallbackManager:
        """Get a callback manager with LangSmith tracing."""
        callbacks = []
        if self.tracer:
            callbacks.append(self.tracer)
        
        return CallbackManager(callbacks)
    
    def create_run(self, name: str, run_type: str, inputs: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new run in LangSmith."""
        if not self.langsmith_client:
            return "no-langsmith"
        
        try:
            run = self.langsmith_client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                project_name=os.getenv("LANGCHAIN_PROJECT", "james"),
                start_time=datetime.now(),
                extra=metadata or {}
            )
            return str(run.id)
        except Exception as e:
            print(f"Failed to create LangSmith run: {e}")
            return "error"
    
    def update_run(self, run_id: str, outputs: Dict[str, Any], 
                   error: Optional[str] = None) -> None:
        """Update a run with outputs and end time."""
        if not self.langsmith_client or run_id in ["no-langsmith", "error"]:
            return
        
        try:
            self.langsmith_client.update_run(
                run_id=run_id,
                outputs=outputs,
                end_time=datetime.now(),
                error=error
            )
        except Exception as e:
            print(f"Failed to update LangSmith run: {e}")
    
    def log_consciousness_cycle(self, message_id: str, message_content: str, 
                              observer_decision: Dict[str, Any],
                              delegation_results: Optional[Dict[str, Any]] = None) -> str:
        """Log a complete consciousness processing cycle."""
        if not self.langsmith_client:
            return "no-langsmith"
        
        inputs = {
            "message_id": message_id,
            "message_content": message_content,
            "timestamp": datetime.now().isoformat()
        }
        
        outputs = {
            "observer_decision": observer_decision,
            "delegation_results": delegation_results
        }
        
        metadata = {
            "component": "consciousness_cycle",
            "james_version": "0.1.0"
        }
        
        run_id = self.create_run(
            name="James Consciousness Cycle",
            run_type="chain",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, outputs)
        return run_id
    
    def log_observer_decision(self, message: str, decision: Dict[str, Any]) -> str:
        """Log an Observer agent decision."""
        inputs = {"message": message}
        outputs = {"decision": decision}
        metadata = {"component": "observer", "agent": "observer"}
        
        run_id = self.create_run(
            name="Observer Decision",
            run_type="llm",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, outputs)
        return run_id
    
    def log_delegation(self, task: str, selected_agents: list, 
                      delegation_strategy: str, results: list) -> str:
        """Log a delegation operation."""
        inputs = {
            "task": task,
            "selected_agents": selected_agents,
            "strategy": delegation_strategy
        }
        
        outputs = {"results": results}
        metadata = {"component": "delegator", "agent": "delegator"}
        
        run_id = self.create_run(
            name="Task Delegation",
            run_type="chain",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, outputs)
        return run_id
    
    def log_memory_operation(self, operation: str, query: str, 
                           results: Dict[str, Any]) -> str:
        """Log a memory system operation."""
        inputs = {
            "operation": operation,
            "query": query
        }
        
        outputs = {"results": results}
        metadata = {"component": "memory", "system": "memory"}
        
        run_id = self.create_run(
            name=f"Memory {operation.title()}",
            run_type="retriever",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, outputs)
        return run_id
    
    def log_tool_execution(self, tool_name: str, parameters: Dict[str, Any], 
                          result: Dict[str, Any]) -> str:
        """Log a tool execution."""
        inputs = {
            "tool": tool_name,
            "parameters": parameters
        }
        
        outputs = {"result": result}
        metadata = {"component": "tools", "tool": tool_name}
        
        run_id = self.create_run(
            name=f"Tool: {tool_name}",
            run_type="tool",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, outputs)
        return run_id
    
    def log_error(self, component: str, error: str, context: Dict[str, Any]) -> str:
        """Log an error with context."""
        inputs = {"component": component, "context": context}
        metadata = {"component": component, "error_type": "system_error"}
        
        run_id = self.create_run(
            name=f"Error in {component}",
            run_type="chain",
            inputs=inputs,
            metadata=metadata
        )
        
        self.update_run(run_id, {}, error=error)
        return run_id
    
    def get_project_stats(self) -> Optional[Dict[str, Any]]:
        """Get project statistics from LangSmith."""
        if not self.langsmith_client:
            return None
        
        try:
            project_name = os.getenv("LANGCHAIN_PROJECT", "james")
            # Note: This would require additional LangSmith API calls
            # that might not be available in the current client
            return {
                "project": project_name,
                "observability_enabled": True,
                "client_initialized": True
            }
        except Exception as e:
            print(f"Failed to get project stats: {e}")
            return None


# Global observability instance
observability = JamesObservability()