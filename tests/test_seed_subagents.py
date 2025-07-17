"""
Integration tests for seed subagents.

Tests the functionality of the three core seed subagents and their
interactions with the registry and A2A protocol.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from backend.subagents.reflection_agent import ReflectionSubagent
from backend.subagents.builder_agent import BuilderSubagent
from backend.subagents.external_input_agent import ExternalInputSubagent
from backend.subagents.register_seed_subagents import (
    register_seed_subagents, verify_registrations, unregister_seed_subagents
)
from backend.registry.subagent_registry import SubagentRegistry
from backend.protocol.a2a_models import A2AMessage, A2AHeader, A2APayload, A2AMessageType


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def reflection_agent():
    """Create a reflection subagent for testing."""
    return ReflectionSubagent()


@pytest.fixture
def builder_agent():
    """Create a builder subagent for testing."""
    return BuilderSubagent()


@pytest.fixture
def external_input_agent():
    """Create an external input subagent for testing."""
    return ExternalInputSubagent()


class TestReflectionSubagent:
    """Test the Reflection Subagent functionality."""
    
    @pytest.mark.asyncio
    async def test_reflection_agent_initialization(self, reflection_agent):
        """Test reflection agent initialization."""
        assert reflection_agent.name == "Reflection Agent"
        assert "self_reflection" in reflection_agent.capabilities
        assert "pattern_analysis" in reflection_agent.capabilities
        assert reflection_agent.subagent_id is not None
    
    @pytest.mark.asyncio
    async def test_decision_analysis(self, reflection_agent):
        """Test decision analysis functionality."""
        input_data = {
            "reflection_type": "decision_analysis",
            "decisions": [
                {
                    "type": "task_delegation",
                    "outcome": "success",
                    "factors": ["time_pressure", "complexity", "available_resources"]
                },
                {
                    "type": "priority_setting",
                    "outcome": "success",
                    "factors": ["urgency", "importance"]
                },
                {
                    "type": "resource_allocation",
                    "outcome": "failure",
                    "factors": ["time_pressure", "insufficient_data"]
                }
            ]
        }
        
        result = await reflection_agent.process_task("test_task", "Analyze decisions", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "total_decisions" in result.data
        assert result.data["total_decisions"] == 3
        assert "success_rate" in result.data
        assert "recommendations" in result.data
        assert len(result.data["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self, reflection_agent):
        """Test pattern analysis functionality."""
        input_data = {
            "reflection_type": "pattern_analysis",
            "events": [
                {"type": "message_processing", "timestamp": "2024-01-01T10:00:00Z"},
                {"type": "task_delegation", "timestamp": "2024-01-01T10:15:00Z"},
                {"type": "message_processing", "timestamp": "2024-01-01T10:30:00Z"},
                {"type": "error_handling", "timestamp": "2024-01-01T10:45:00Z"},
                {"type": "message_processing", "timestamp": "2024-01-01T11:00:00Z"}
            ]
        }
        
        result = await reflection_agent.process_task("test_task", "Analyze patterns", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "temporal_patterns" in result.data
        assert "frequency_patterns" in result.data
        assert "message_processing" in result.data["frequency_patterns"]
        assert result.data["frequency_patterns"]["message_processing"] == 3
    
    @pytest.mark.asyncio
    async def test_performance_assessment(self, reflection_agent):
        """Test performance assessment functionality."""
        input_data = {
            "reflection_type": "performance_assessment",
            "metrics": {
                "task_completion_rate": {
                    "current": 85,
                    "target": 90,
                    "previous": 80
                },
                "response_time": {
                    "current": 2.5,
                    "target": 2.0,
                    "previous": 3.0
                },
                "error_rate": {
                    "current": 5,
                    "target": 2,
                    "previous": 8
                }
            },
            "time_period": "last_week"
        }
        
        result = await reflection_agent.process_task("test_task", "Assess performance", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "overall_score" in result.data
        assert "metric_scores" in result.data
        assert "strengths" in result.data
        assert "weaknesses" in result.data
        assert result.data["overall_score"] > 0
    
    @pytest.mark.asyncio
    async def test_a2a_message_handling(self, reflection_agent):
        """Test A2A message handling."""
        # Create a task request message
        header = A2AHeader(
            sender_id="test_sender",
            recipient_id=reflection_agent.subagent_id,
            message_type=A2AMessageType.TASK_REQUEST
        )
        
        payload = A2APayload(
            task_id="test_task_123",
            task_description="Perform reflection analysis",
            input_data={
                "reflection_type": "general",
                "data": {"test": "data"}
            }
        )
        
        message = A2AMessage(header=header, payload=payload)
        
        response = await reflection_agent.handle_a2a_message(message)
        
        assert response is not None
        assert response.header.message_type == A2AMessageType.TASK_RESPONSE
        assert response.payload.task_id == "test_task_123"
        assert response.payload.status.value in ["completed", "failed"]


class TestBuilderSubagent:
    """Test the Builder Subagent functionality."""
    
    @pytest.mark.asyncio
    async def test_builder_agent_initialization(self, builder_agent):
        """Test builder agent initialization."""
        assert builder_agent.name == "Builder Agent"
        assert "subagent_creation" in builder_agent.capabilities
        assert "tool_generation" in builder_agent.capabilities
        assert builder_agent.subagent_id is not None
    
    @pytest.mark.asyncio
    async def test_subagent_building(self, builder_agent):
        """Test subagent building functionality."""
        input_data = {
            "build_type": "subagent",
            "specifications": {
                "name": "Test Analyzer",
                "description": "Analyzes test data",
                "capabilities": ["data_analysis", "testing", "validation"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"}
                    }
                }
            }
        }
        
        result = await builder_agent.process_task("test_task", "Build subagent", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "subagent_code" in result.data
        assert "test_code" in result.data
        assert "registration_code" in result.data
        assert "metadata" in result.data
        assert result.data["metadata"]["name"] == "Test Analyzer"
        
        # Check that generated code contains expected elements
        subagent_code = result.data["subagent_code"]
        assert "class TestAnalyzerSubagent" in subagent_code
        assert "data_analysis" in subagent_code
        assert "async def process_task" in subagent_code
    
    @pytest.mark.asyncio
    async def test_tool_building(self, builder_agent):
        """Test tool building functionality."""
        input_data = {
            "build_type": "tool",
            "specifications": {
                "name": "Data Validator",
                "description": "Validates input data",
                "functionality": "Check data format and constraints",
                "parameters": {
                    "data": {"type": "string", "required": True},
                    "format": {"type": "string", "required": False}
                }
            }
        }
        
        result = await builder_agent.process_task("test_task", "Build tool", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "tool_code" in result.data
        assert "test_code" in result.data
        assert "metadata" in result.data
        
        # Check that generated code contains expected elements
        tool_code = result.data["tool_code"]
        assert "class DataValidatorTool" in tool_code
        assert "async def execute" in tool_code
        assert "Check data format and constraints" in tool_code
    
    @pytest.mark.asyncio
    async def test_documentation_building(self, builder_agent):
        """Test documentation building functionality."""
        input_data = {
            "build_type": "documentation",
            "specifications": {
                "type": "api",
                "component": "TestAPI",
                "description": "API for testing purposes",
                "methods": [
                    {
                        "name": "get_data",
                        "description": "Retrieve test data",
                        "http_method": "GET",
                        "endpoint": "/data"
                    }
                ],
                "examples": [
                    {
                        "title": "Get Data Example",
                        "curl": "curl -X GET https://api.test.com/data",
                        "response": {"status": "success", "data": []}
                    }
                ]
            }
        }
        
        result = await builder_agent.process_task("test_task", "Build documentation", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "documentation" in result.data
        assert "metadata" in result.data
        
        # Check that generated documentation contains expected elements
        documentation = result.data["documentation"]
        assert "# TestAPI API Documentation" in documentation
        assert "get_data" in documentation
        assert "GET" in documentation
        assert "/data" in documentation


class TestExternalInputSubagent:
    """Test the External Input Subagent functionality."""
    
    @pytest.mark.asyncio
    async def test_external_input_agent_initialization(self, external_input_agent):
        """Test external input agent initialization."""
        assert external_input_agent.name == "External Input Agent"
        assert "input_validation" in external_input_agent.capabilities
        assert "content_sanitization" in external_input_agent.capabilities
        assert external_input_agent.subagent_id is not None
    
    @pytest.mark.asyncio
    async def test_input_validation(self, external_input_agent):
        """Test input validation functionality."""
        input_data = {
            "processing_type": "validation",
            "content": "This is a test message with some content",
            "source": "test_user",
            "type": "text"
        }
        
        result = await external_input_agent.process_task("test_task", "Validate input", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "is_valid" in result.data
        assert "validation_errors" in result.data
        assert "risk_level" in result.data
        assert result.data["is_valid"] is True
        assert result.data["risk_level"] == "low"
    
    @pytest.mark.asyncio
    async def test_threat_detection(self, external_input_agent):
        """Test threat detection functionality."""
        input_data = {
            "processing_type": "threat_detection",
            "content": "<script>alert('xss')</script>This is malicious content",
            "threat_types": ["xss", "injection"]
        }
        
        result = await external_input_agent.process_task("test_task", "Detect threats", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "threats_detected" in result.data
        assert "risk_level" in result.data
        assert len(result.data["threats_detected"]) > 0
        assert result.data["risk_level"] in ["medium", "high", "critical"]
        
        # Check that XSS threat was detected
        threat_types = [threat["type"] for threat in result.data["threats_detected"]]
        assert "xss" in threat_types
    
    @pytest.mark.asyncio
    async def test_content_sanitization(self, external_input_agent):
        """Test content sanitization functionality."""
        input_data = {
            "processing_type": "sanitization",
            "content": "<script>alert('test')</script><p>Safe content</p>",
            "sanitization_level": "standard"
        }
        
        result = await external_input_agent.process_task("test_task", "Sanitize content", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "sanitized_content" in result.data
        assert "sanitization_actions" in result.data
        assert "content_changed" in result.data
        
        # Check that script tags were removed
        sanitized = result.data["sanitized_content"]
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
        assert result.data["content_changed"] is True
        assert len(result.data["sanitization_actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_data_extraction(self, external_input_agent):
        """Test data extraction functionality."""
        input_data = {
            "processing_type": "extraction",
            "content": "Contact us at test@example.com or visit https://example.com. Call 555-123-4567.",
            "targets": ["emails", "urls", "phones"]
        }
        
        result = await external_input_agent.process_task("test_task", "Extract data", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "emails" in result.data
        assert "urls" in result.data
        assert "phones" in result.data
        
        # Check extracted data
        assert "test@example.com" in result.data["emails"]
        assert "https://example.com" in result.data["urls"]
        assert "555-123-4567" in result.data["phones"]
    
    @pytest.mark.asyncio
    async def test_full_analysis(self, external_input_agent):
        """Test full analysis functionality."""
        input_data = {
            "processing_type": "full_analysis",
            "content": "Hello, this is a test message from a user",
            "source": "trusted_user",
            "type": "text"
        }
        
        result = await external_input_agent.process_task("test_task", "Full analysis", input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "validation" in result.data
        assert "classification" in result.data
        assert "extraction" in result.data
        assert "threats" in result.data
        assert "source_verification" in result.data
        assert "overall_assessment" in result.data
        
        # Check overall assessment
        assessment = result.data["overall_assessment"]
        assert "safe_to_process" in assessment
        assert "confidence" in assessment
        assert "recommendations" in assessment


class TestSubagentRegistration:
    """Test subagent registration functionality."""
    
    @pytest.mark.asyncio
    async def test_register_seed_subagents(self, temp_registry):
        """Test registering all seed subagents."""
        results = await register_seed_subagents(temp_registry)
        
        assert results["total"] == 3
        assert len(results["registered"]) >= 0  # May be 0 if already registered
        assert len(results["failed"]) == 0
        
        # Check that all expected subagents are in registered or already exist
        registered_names = [sa["name"] for sa in results["registered"]]
        expected_names = ["Reflection Agent", "Builder Agent", "External Input Agent"]
        
        for name in expected_names:
            # Should be either newly registered or already existing
            assert any(sa["name"] == name for sa in results["registered"])
    
    @pytest.mark.asyncio
    async def test_verify_registrations(self, temp_registry):
        """Test verifying seed subagent registrations."""
        # First register the subagents
        await register_seed_subagents(temp_registry)
        
        # Then verify them
        results = await verify_registrations(temp_registry)
        
        assert "error" not in results
        assert len(results["verified"]) == 3
        assert len(results["missing"]) == 0
        
        # Check that all expected subagents are verified
        verified_names = [sa["name"] for sa in results["verified"]]
        expected_names = ["Reflection Agent", "Builder Agent", "External Input Agent"]
        
        for name in expected_names:
            assert name in verified_names
    
    @pytest.mark.asyncio
    async def test_unregister_seed_subagents(self, temp_registry):
        """Test unregistering seed subagents."""
        # First register the subagents
        await register_seed_subagents(temp_registry)
        
        # Then unregister them
        results = await unregister_seed_subagents(temp_registry)
        
        assert "error" not in results
        assert len(results["unregistered"]) == 3
        assert len(results["failed"]) == 0
        
        # Verify they are no longer registered
        verify_results = await verify_registrations(temp_registry)
        assert len(verify_results["verified"]) == 0
        assert len(verify_results["missing"]) == 3


class TestSubagentInteractions:
    """Test interactions between subagents."""
    
    @pytest.mark.asyncio
    async def test_reflection_builder_interaction(self, reflection_agent, builder_agent):
        """Test interaction between reflection and builder agents."""
        # Reflection agent analyzes the need for a new capability
        reflection_input = {
            "reflection_type": "improvement_identification",
            "context": {"current_system": "conscious_agent"},
            "current_capabilities": ["reflection", "building", "input_processing"],
            "recent_challenges": [
                {"type": "data_processing", "frequency": 3},
                {"type": "api_integration", "frequency": 2}
            ],
            "goals": [
                {
                    "description": "Improve data processing efficiency",
                    "required_capabilities": ["data_analysis", "optimization"]
                }
            ]
        }
        
        reflection_result = await reflection_agent.process_task(
            "reflection_task", "Identify improvements", reflection_input
        )
        
        assert reflection_result.success is True
        assert "capability_gaps" in reflection_result.data
        
        # Builder agent creates the needed capability
        builder_input = {
            "build_type": "capability",
            "specifications": {
                "name": "data_analysis",
                "description": "Advanced data analysis capability",
                "requirements": ["pandas", "numpy"],
                "interfaces": {
                    "analyze_data": {
                        "input": {"data": "array"},
                        "output": {"analysis": "object"}
                    }
                }
            }
        }
        
        builder_result = await builder_agent.process_task(
            "builder_task", "Build capability", builder_input
        )
        
        assert builder_result.success is True
        assert "specification" in builder_result.data
        assert builder_result.data["specification"]["name"] == "data_analysis"
    
    @pytest.mark.asyncio
    async def test_external_input_reflection_interaction(self, external_input_agent, reflection_agent):
        """Test interaction between external input and reflection agents."""
        # External input agent processes some input
        input_data = {
            "processing_type": "full_analysis",
            "content": "Please analyze the performance of the system over the last week",
            "source": "system_admin",
            "type": "text"
        }
        
        input_result = await external_input_agent.process_task(
            "input_task", "Process external input", input_data
        )
        
        assert input_result.success is True
        
        # Use the classification result to inform reflection
        classification = input_result.data.get("classification", {})
        
        reflection_input = {
            "reflection_type": "general",
            "data": {
                "user_request": input_data["content"],
                "classification": classification,
                "processing_result": input_result.data
            },
            "focus_areas": ["performance", "system_analysis"]
        }
        
        reflection_result = await reflection_agent.process_task(
            "reflection_task", "Reflect on user request", reflection_input
        )
        
        assert reflection_result.success is True
        assert "key_insights" in reflection_result.data
        assert len(reflection_result.data["key_insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_registry_integration(self, temp_registry):
        """Test integration with the subagent registry."""
        # Register subagents
        registration_results = await register_seed_subagents(temp_registry)
        assert registration_results["total"] == 3
        
        # Create registry instance and verify subagents are accessible
        registry = SubagentRegistry(registry_path=temp_registry)
        
        # Test retrieving subagents by capability
        reflection_agents = await registry.search_by_capabilities(["self_reflection"])
        assert len(reflection_agents) >= 1
        assert any(agent.name == "Reflection Agent" for agent in reflection_agents)
        
        builder_agents = await registry.search_by_capabilities(["subagent_creation"])
        assert len(builder_agents) >= 1
        assert any(agent.name == "Builder Agent" for agent in builder_agents)
        
        input_agents = await registry.search_by_capabilities(["input_validation"])
        assert len(input_agents) >= 1
        assert any(agent.name == "External Input Agent" for agent in input_agents)
        
        # Test getting subagent by name
        reflection_agent = await registry.get_subagent_by_name("Reflection Agent")
        assert reflection_agent is not None
        assert reflection_agent.name == "Reflection Agent"
        assert "self_reflection" in reflection_agent.capabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])