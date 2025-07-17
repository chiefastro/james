"""
Builder Subagent for creating new capabilities.

This subagent provides James with the ability to create new subagents,
tools, and capabilities based on identified needs and requirements.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


class BuilderSubagent(BaseSubagent):
    """
    Subagent that handles capability building and creation tasks.
    
    Capabilities:
    - Create new subagent templates
    - Generate tool implementations
    - Build capability specifications
    - Create integration code
    - Generate documentation
    """
    
    def __init__(self, subagent_id: Optional[str] = None):
        """Initialize the builder subagent."""
        super().__init__(
            subagent_id=subagent_id or f"builder_agent_{uuid4().hex[:8]}",
            name="Builder Agent",
            description="Creates new capabilities, subagents, and tools based on identified needs",
            capabilities=[
                "subagent_creation",
                "tool_generation",
                "capability_building",
                "code_generation",
                "template_creation",
                "integration_development",
                "documentation_generation"
            ]
        )
        self._build_history: List[Dict[str, Any]] = []
        self._templates_cache: Dict[str, str] = {}
    
    async def process_task(self, task_id: str, task_description: str, input_data: Dict[str, Any]) -> SubagentResult:
        """
        Process a building task.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of what to build
            input_data: Specifications and requirements
            
        Returns:
            SubagentResult with build artifacts
        """
        try:
            self.logger.info(f"Processing build task: {task_id}")
            
            # Determine the type of build requested
            build_type = input_data.get("build_type", "subagent")
            
            if build_type == "subagent":
                result = await self._build_subagent(input_data)
            elif build_type == "tool":
                result = await self._build_tool(input_data)
            elif build_type == "capability":
                result = await self._build_capability(input_data)
            elif build_type == "integration":
                result = await self._build_integration(input_data)
            elif build_type == "documentation":
                result = await self._build_documentation(input_data)
            else:
                result = self._create_error_result(f"Unknown build type: {build_type}")
            
            # Store build in history
            if result.success:
                build_record = {
                    "task_id": task_id,
                    "build_type": build_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "specifications": self._summarize_specs(input_data),
                    "artifacts_created": self._count_artifacts(result.data)
                }
                self._build_history.append(build_record)
                
                # Keep only last 50 builds
                if len(self._build_history) > 50:
                    self._build_history = self._build_history[-50:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in build task {task_id}: {e}")
            return self._create_error_result(f"Build failed: {str(e)}")
    
    async def _build_subagent(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Build a new subagent based on specifications."""
        specs = input_data.get("specifications", {})
        name = specs.get("name", "New Subagent")
        description = specs.get("description", "A new subagent")
        capabilities = specs.get("capabilities", [])
        input_schema = specs.get("input_schema", {})
        output_schema = specs.get("output_schema", {})
        
        if not name or not description:
            return self._create_error_result("Name and description are required for subagent creation")
        
        # Generate subagent code
        subagent_code = self._generate_subagent_code(name, description, capabilities, input_schema, output_schema)
        
        # Generate test code
        test_code = self._generate_subagent_test_code(name, capabilities)
        
        # Generate registration code
        registration_code = self._generate_registration_code(name, description, capabilities)
        
        artifacts = {
            "subagent_code": subagent_code,
            "test_code": test_code,
            "registration_code": registration_code,
            "metadata": {
                "name": name,
                "description": description,
                "capabilities": capabilities,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return self._create_success_result(
            data=artifacts,
            metadata={
                "build_type": "subagent",
                "name": name,
                "capabilities_count": len(capabilities)
            }
        )
    
    async def _build_tool(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Build a new tool based on specifications."""
        specs = input_data.get("specifications", {})
        name = specs.get("name", "New Tool")
        description = specs.get("description", "A new tool")
        functionality = specs.get("functionality", "")
        parameters = specs.get("parameters", {})
        
        if not name or not functionality:
            return self._create_error_result("Name and functionality are required for tool creation")
        
        # Generate tool code
        tool_code = self._generate_tool_code(name, description, functionality, parameters)
        
        # Generate test code
        test_code = self._generate_tool_test_code(name, parameters)
        
        artifacts = {
            "tool_code": tool_code,
            "test_code": test_code,
            "metadata": {
                "name": name,
                "description": description,
                "functionality": functionality,
                "parameters": parameters,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return self._create_success_result(
            data=artifacts,
            metadata={
                "build_type": "tool",
                "name": name,
                "parameter_count": len(parameters)
            }
        )
    
    async def _build_capability(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Build a new capability specification."""
        specs = input_data.get("specifications", {})
        capability_name = specs.get("name", "new_capability")
        description = specs.get("description", "A new capability")
        requirements = specs.get("requirements", [])
        interfaces = specs.get("interfaces", {})
        
        capability_spec = {
            "name": capability_name,
            "description": description,
            "version": "1.0.0",
            "requirements": requirements,
            "interfaces": interfaces,
            "implementation_notes": [
                f"Implement {capability_name} functionality",
                "Add appropriate error handling",
                "Include comprehensive tests",
                "Document all public methods"
            ],
            "integration_points": [
                "Register with subagent registry",
                "Add to capability discovery system",
                "Update documentation"
            ],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Generate implementation template
        implementation_template = self._generate_capability_template(capability_name, description, interfaces)
        
        artifacts = {
            "specification": capability_spec,
            "implementation_template": implementation_template,
            "metadata": {
                "capability_name": capability_name,
                "requirements_count": len(requirements),
                "interfaces_count": len(interfaces)
            }
        }
        
        return self._create_success_result(
            data=artifacts,
            metadata={
                "build_type": "capability",
                "name": capability_name
            }
        )
    
    async def _build_integration(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Build integration code between components."""
        specs = input_data.get("specifications", {})
        source_component = specs.get("source", "")
        target_component = specs.get("target", "")
        integration_type = specs.get("type", "api")
        data_format = specs.get("data_format", "json")
        
        if not source_component or not target_component:
            return self._create_error_result("Source and target components are required for integration")
        
        # Generate integration code based on type
        if integration_type == "api":
            integration_code = self._generate_api_integration(source_component, target_component, data_format)
        elif integration_type == "event":
            integration_code = self._generate_event_integration(source_component, target_component, data_format)
        elif integration_type == "data":
            integration_code = self._generate_data_integration(source_component, target_component, data_format)
        else:
            integration_code = self._generate_generic_integration(source_component, target_component, data_format)
        
        # Generate configuration
        config = self._generate_integration_config(source_component, target_component, integration_type)
        
        artifacts = {
            "integration_code": integration_code,
            "configuration": config,
            "metadata": {
                "source": source_component,
                "target": target_component,
                "type": integration_type,
                "data_format": data_format,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return self._create_success_result(
            data=artifacts,
            metadata={
                "build_type": "integration",
                "integration_type": integration_type
            }
        )
    
    async def _build_documentation(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Build documentation for components or capabilities."""
        specs = input_data.get("specifications", {})
        doc_type = specs.get("type", "api")
        component_name = specs.get("component", "Component")
        description = specs.get("description", "")
        methods = specs.get("methods", [])
        examples = specs.get("examples", [])
        
        if doc_type == "api":
            documentation = self._generate_api_documentation(component_name, description, methods, examples)
        elif doc_type == "user_guide":
            documentation = self._generate_user_guide(component_name, description, examples)
        elif doc_type == "technical":
            documentation = self._generate_technical_documentation(component_name, description, methods)
        else:
            documentation = self._generate_generic_documentation(component_name, description)
        
        artifacts = {
            "documentation": documentation,
            "metadata": {
                "type": doc_type,
                "component": component_name,
                "methods_count": len(methods),
                "examples_count": len(examples),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return self._create_success_result(
            data=artifacts,
            metadata={
                "build_type": "documentation",
                "doc_type": doc_type
            }
        )
    
    def _generate_subagent_code(self, name: str, description: str, capabilities: List[str], 
                               input_schema: Dict[str, Any], output_schema: Dict[str, Any]) -> str:
        """Generate Python code for a new subagent."""
        class_name = name.replace(" ", "").replace("-", "").replace("_", "") + "Subagent"
        module_name = name.lower().replace(" ", "_").replace("-", "_") + "_agent"
        
        code = f'''"""
{name} Subagent.

{description}
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


class {class_name}(BaseSubagent):
    """
    {description}
    
    Capabilities:
{chr(10).join(f"    - {cap}" for cap in capabilities)}
    """
    
    def __init__(self, subagent_id: Optional[str] = None):
        """Initialize the {name.lower()} subagent."""
        super().__init__(
            subagent_id=subagent_id or f"{module_name}_{{uuid4().hex[:8]}}",
            name="{name}",
            description="{description}",
            capabilities={capabilities}
        )
    
    async def process_task(self, task_id: str, task_description: str, input_data: Dict[str, Any]) -> SubagentResult:
        """
        Process a task for {name.lower()}.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the task
            input_data: Input data for processing
            
        Returns:
            SubagentResult with processing outcome
        """
        try:
            self.logger.info(f"Processing task: {{task_id}}")
            
            # TODO: Implement task processing logic
            result_data = {{
                "task_id": task_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed"
            }}
            
            return self._create_success_result(
                data=result_data,
                metadata={{
                    "task_type": input_data.get("task_type", "unknown"),
                    "processing_time": 0.0
                }}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing task {{task_id}}: {{e}}")
            return self._create_error_result(f"Task processing failed: {{str(e)}}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for input data."""
        return {json.dumps(input_schema, indent=8)}
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for output data."""
        return {json.dumps(output_schema, indent=8)}
'''
        
        return code
    
    def _generate_subagent_test_code(self, name: str, capabilities: List[str]) -> str:
        """Generate test code for a subagent."""
        class_name = name.replace(" ", "").replace("-", "").replace("_", "") + "Subagent"
        module_name = name.lower().replace(" ", "_").replace("-", "_") + "_agent"
        
        code = f'''"""
Tests for {name} Subagent.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from backend.subagents.{module_name} import {class_name}


@pytest.fixture
def {module_name}():
    """Create a {name} subagent instance for testing."""
    return {class_name}()


@pytest.mark.asyncio
async def test_subagent_initialization({module_name}):
    """Test subagent initialization."""
    assert {module_name}.name == "{name}"
    assert "{capabilities[0] if capabilities else 'test_capability'}" in {module_name}.capabilities
    assert {module_name}.subagent_id is not None


@pytest.mark.asyncio
async def test_process_task_success({module_name}):
    """Test successful task processing."""
    task_id = "test_task_123"
    task_description = "Test task"
    input_data = {{"task_type": "test"}}
    
    result = await {module_name}.process_task(task_id, task_description, input_data)
    
    assert result.success is True
    assert result.data is not None
    assert result.data["task_id"] == task_id


@pytest.mark.asyncio
async def test_get_schemas({module_name}):
    """Test schema retrieval."""
    input_schema = {module_name}.get_input_schema()
    output_schema = {module_name}.get_output_schema()
    
    assert isinstance(input_schema, dict)
    assert isinstance(output_schema, dict)


@pytest.mark.asyncio
async def test_health_check({module_name}):
    """Test health check functionality."""
    health = await {module_name}.health_check()
    
    assert health["name"] == "{name}"
    assert health["status"] == "healthy"
    assert "capabilities" in health
'''
        
        return code
    
    def _generate_registration_code(self, name: str, description: str, capabilities: List[str]) -> str:
        """Generate registration code for a subagent."""
        class_name = name.replace(" ", "").replace("-", "").replace("_", "") + "Subagent"
        module_name = name.lower().replace(" ", "_").replace("-", "_") + "_agent"
        
        code = f'''"""
Registration script for {name} Subagent.
"""

import asyncio
from backend.subagents.{module_name} import {class_name}
from backend.registry.subagent_registry import SubagentRegistry


async def register_{module_name}():
    """Register the {name} subagent."""
    # Initialize registry
    registry = SubagentRegistry()
    
    # Create subagent instance
    subagent = {class_name}()
    
    # Get metadata for registration
    metadata = subagent.get_subagent_metadata()
    
    # Register with the registry
    await registry.register_subagent(metadata)
    
    print(f"✅ Registered {{subagent.name}} with ID: {{subagent.subagent_id}}")
    
    return subagent.subagent_id


if __name__ == "__main__":
    asyncio.run(register_{module_name}())
'''
        
        return code
    
    def _generate_tool_code(self, name: str, description: str, functionality: str, parameters: Dict[str, Any]) -> str:
        """Generate Python code for a new tool."""
        class_name = name.replace(" ", "").replace("-", "").replace("_", "") + "Tool"
        
        code = f'''"""
{name} Tool.

{description}
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class {class_name}(BaseTool):
    """
    {description}
    
    Functionality: {functionality}
    """
    
    def __init__(self):
        """Initialize the {name.lower()} tool."""
        super().__init__("{name.lower().replace(' ', '_')}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the {name.lower()} tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution outcome
        """
        try:
            # Validate required parameters
            required_params = {list(parameters.keys())}
            validation_error = self._validate_required_params(kwargs, required_params)
            if validation_error:
                return self._create_error_result(validation_error)
            
            # TODO: Implement tool functionality
            # {functionality}
            
            result_data = {{
                "status": "completed",
                "message": "{name} executed successfully"
            }}
            
            return self._create_success_result(
                data=result_data,
                metadata={{"tool_name": "{name}"}}
            )
            
        except Exception as e:
            return self._create_error_result(f"{name} execution failed: {{str(e)}}")
'''
        
        return code
    
    def _generate_tool_test_code(self, name: str, parameters: Dict[str, Any]) -> str:
        """Generate test code for a tool."""
        class_name = name.replace(" ", "").replace("-", "").replace("_", "") + "Tool"
        
        code = f'''"""
Tests for {name} Tool.
"""

import pytest
import asyncio

from backend.tools.{name.lower().replace(' ', '_')}_tool import {class_name}


@pytest.fixture
def {name.lower().replace(' ', '_')}_tool():
    """Create a {name} tool instance for testing."""
    return {class_name}()


@pytest.mark.asyncio
async def test_tool_execution({name.lower().replace(' ', '_')}_tool):
    """Test tool execution."""
    # TODO: Add test parameters based on tool requirements
    test_params = {{}}
    
    result = await {name.lower().replace(' ', '_')}_tool.execute(**test_params)
    
    # TODO: Add specific assertions based on expected behavior
    assert result is not None
'''
        
        return code
    
    def _generate_capability_template(self, name: str, description: str, interfaces: Dict[str, Any]) -> str:
        """Generate implementation template for a capability."""
        template = f'''"""
{name.title()} Capability Implementation Template.

{description}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class {name.title().replace('_', '')}Interface(ABC):
    """Interface for {name} capability."""
    
    @abstractmethod
    async def execute_{name}(self, **kwargs) -> Dict[str, Any]:
        """Execute {name} functionality."""
        pass


class {name.title().replace('_', '')}Implementation({name.title().replace('_', '')}Interface):
    """Implementation of {name} capability."""
    
    def __init__(self):
        """Initialize {name} implementation."""
        pass
    
    async def execute_{name}(self, **kwargs) -> Dict[str, Any]:
        """
        Execute {name} functionality.
        
        Args:
            **kwargs: Implementation-specific parameters
            
        Returns:
            Result dictionary
        """
        # TODO: Implement {name} logic
        return {{"status": "completed", "result": "placeholder"}}
'''
        
        return template
    
    def _generate_api_integration(self, source: str, target: str, data_format: str) -> str:
        """Generate API integration code."""
        return f'''"""
API Integration between {source} and {target}.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class {source.title()}{target.title()}Integration:
    """Integration between {source} and {target} via API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize integration."""
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def send_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data from {source} to {target}."""
        if not self.session:
            raise RuntimeError("Integration not initialized. Use async context manager.")
        
        headers = {{"Content-Type": "application/{data_format}"}}
        if self.api_key:
            headers["Authorization"] = f"Bearer {{self.api_key}}"
        
        async with self.session.post(
            f"{{self.base_url}}/api/data",
            json=data if data_format == "json" else None,
            data=data if data_format != "json" else None,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"API request failed: {{response.status}}")
'''
    
    def _generate_event_integration(self, source: str, target: str, data_format: str) -> str:
        """Generate event-based integration code."""
        return f'''"""
Event Integration between {source} and {target}.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class {source.title()}{target.title()}EventIntegration:
    """Event-based integration between {source} and {target}."""
    
    def __init__(self):
        """Initialize event integration."""
        self.event_handlers = {{}}
        self.running = False
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {{e}}")
    
    async def start(self):
        """Start the event integration."""
        self.running = True
        logger.info("Event integration started")
    
    async def stop(self):
        """Stop the event integration."""
        self.running = False
        logger.info("Event integration stopped")
'''
    
    def _generate_data_integration(self, source: str, target: str, data_format: str) -> str:
        """Generate data integration code."""
        return f'''"""
Data Integration between {source} and {target}.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class {source.title()}{target.title()}DataIntegration:
    """Data integration between {source} and {target}."""
    
    def __init__(self, data_path: str):
        """Initialize data integration."""
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    async def export_data(self, data: Dict[str, Any], filename: str) -> str:
        """Export data from {source}."""
        file_path = self.data_path / filename
        
        if "{data_format}" == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Handle other formats as needed
            with open(file_path, 'w') as f:
                f.write(str(data))
        
        logger.info(f"Data exported to {{file_path}}")
        return str(file_path)
    
    async def import_data(self, filename: str) -> Dict[str, Any]:
        """Import data to {target}."""
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {{file_path}}")
        
        if "{data_format}" == "json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r') as f:
                data = {{"content": f.read()}}
        
        logger.info(f"Data imported from {{file_path}}")
        return data
'''
    
    def _generate_generic_integration(self, source: str, target: str, data_format: str) -> str:
        """Generate generic integration code."""
        return f'''"""
Generic Integration between {source} and {target}.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class {source.title()}{target.title()}Integration:
    """Generic integration between {source} and {target}."""
    
    def __init__(self):
        """Initialize integration."""
        self.connected = False
    
    async def connect(self):
        """Establish connection between components."""
        # TODO: Implement connection logic
        self.connected = True
        logger.info("Integration connected")
    
    async def disconnect(self):
        """Disconnect components."""
        self.connected = False
        logger.info("Integration disconnected")
    
    async def transfer_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer data between {source} and {target}."""
        if not self.connected:
            raise RuntimeError("Integration not connected")
        
        # TODO: Implement data transfer logic
        result = {{"status": "transferred", "data": data}}
        
        logger.info("Data transferred successfully")
        return result
'''
    
    def _generate_integration_config(self, source: str, target: str, integration_type: str) -> Dict[str, Any]:
        """Generate integration configuration."""
        return {
            "integration": {
                "name": f"{source}_{target}_integration",
                "type": integration_type,
                "source": {
                    "component": source,
                    "endpoint": f"/{source}/api",
                    "format": "json"
                },
                "target": {
                    "component": target,
                    "endpoint": f"/{target}/api",
                    "format": "json"
                },
                "settings": {
                    "timeout": 30,
                    "retry_attempts": 3,
                    "batch_size": 100
                }
            }
        }
    
    def _generate_api_documentation(self, component: str, description: str, methods: List[Dict], examples: List[Dict]) -> str:
        """Generate API documentation."""
        doc = f'''# {component} API Documentation

{description}

## Overview

This document describes the API endpoints and usage for {component}.

## Base URL

```
https://api.example.com/v1/{component.lower()}
```

## Authentication

All API requests require authentication using an API key in the header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

'''
        
        for method in methods:
            method_name = method.get("name", "unknown")
            method_desc = method.get("description", "")
            http_method = method.get("http_method", "GET")
            endpoint = method.get("endpoint", f"/{method_name}")
            
            doc += f'''### {method_name}

{method_desc}

**HTTP Method:** `{http_method}`  
**Endpoint:** `{endpoint}`

'''
        
        if examples:
            doc += "## Examples\n\n"
            for i, example in enumerate(examples, 1):
                doc += f'''### Example {i}

```bash
{example.get("curl", "curl -X GET https://api.example.com/v1/endpoint")}
```

Response:
```json
{json.dumps(example.get("response", {}), indent=2)}
```

'''
        
        return doc
    
    def _generate_user_guide(self, component: str, description: str, examples: List[Dict]) -> str:
        """Generate user guide documentation."""
        return f'''# {component} User Guide

{description}

## Getting Started

This guide will help you get started with {component}.

## Installation

```bash
pip install {component.lower().replace(' ', '-')}
```

## Basic Usage

```python
from {component.lower().replace(' ', '_')} import {component.replace(' ', '')}

# Initialize
{component.lower().replace(' ', '_')} = {component.replace(' ', '')}()

# Use the component
result = await {component.lower().replace(' ', '_')}.process()
```

## Examples

{chr(10).join(f"### {example.get('title', f'Example {i}')}{chr(10)}{example.get('description', '')}{chr(10)}" for i, example in enumerate(examples, 1))}

## Troubleshooting

Common issues and solutions will be documented here.
'''
    
    def _generate_technical_documentation(self, component: str, description: str, methods: List[Dict]) -> str:
        """Generate technical documentation."""
        return f'''# {component} Technical Documentation

{description}

## Architecture

Technical architecture and design decisions for {component}.

## API Reference

{chr(10).join(f"### {method.get('name', 'unknown')}{chr(10)}{method.get('description', '')}{chr(10)}" for method in methods)}

## Implementation Details

Technical implementation details and considerations.

## Performance

Performance characteristics and optimization notes.
'''
    
    def _generate_generic_documentation(self, component: str, description: str) -> str:
        """Generate generic documentation."""
        return f'''# {component} Documentation

{description}

## Overview

Documentation for {component}.

## Usage

How to use {component}.

## Configuration

Configuration options and settings.

## Examples

Usage examples and code samples.
'''
    
    def _summarize_specs(self, input_data: Dict[str, Any]) -> str:
        """Create a brief summary of specifications."""
        specs = input_data.get("specifications", {})
        return f"Specs with {len(specs)} fields"
    
    def _count_artifacts(self, data: Any) -> int:
        """Count the number of artifacts created."""
        if isinstance(data, dict):
            return len([k for k in data.keys() if k != "metadata"])
        return 1
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for input data."""
        return {
            "type": "object",
            "properties": {
                "build_type": {
                    "type": "string",
                    "enum": ["subagent", "tool", "capability", "integration", "documentation"],
                    "description": "Type of artifact to build"
                },
                "specifications": {
                    "type": "object",
                    "description": "Specifications for the build",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the artifact"},
                        "description": {"type": "string", "description": "Description of the artifact"},
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of capabilities"
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of requirements"
                        }
                    }
                }
            },
            "required": ["build_type", "specifications"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for output data."""
        return {
            "type": "object",
            "properties": {
                "subagent_code": {"type": "string", "description": "Generated subagent code"},
                "tool_code": {"type": "string", "description": "Generated tool code"},
                "test_code": {"type": "string", "description": "Generated test code"},
                "documentation": {"type": "string", "description": "Generated documentation"},
                "configuration": {"type": "object", "description": "Generated configuration"},
                "metadata": {
                    "type": "object",
                    "description": "Build metadata and information"
                }
            }
        }
    
    def get_build_history(self) -> List[Dict[str, Any]]:
        """Get the history of builds performed."""
        return self._build_history.copy()