"""
Seed subagents for the Conscious Agent System.

This package contains the core subagents that provide essential functionality
for James, including self-reflection, capability building, and external input processing.
"""

from .base import BaseSubagent, SubagentResult
from .reflection_agent import ReflectionSubagent
from .builder_agent import BuilderSubagent
from .external_input_agent import ExternalInputSubagent

__all__ = [
    "BaseSubagent",
    "SubagentResult", 
    "ReflectionSubagent",
    "BuilderSubagent",
    "ExternalInputSubagent"
]