#!/usr/bin/env python3
"""
Quick smoke test to debug CSV registry issues.
"""

import asyncio
import tempfile
import csv
from backend.models.core import Subagent
from backend.registry.subagent_registry import SubagentRegistry


async def main():
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    
    print(f"Using temp file: {temp_path}")
    
    # Create registry
    registry = SubagentRegistry(registry_path=temp_path)
    print(f"Registry created, file exists: {registry.registry_path.exists()}")
    
    # Check initial file content
    with open(temp_path, 'r') as f:
        content = f.read()
        print(f"Initial file content: '{content}'")
        print(f"Initial file size: {len(content)}")
    
    # Create a test subagent
    subagent = Subagent(
        id="test-123",
        name="Test Agent",
        description="A test subagent",
        import_path="test.path",
        capabilities=["testing"]
    )
    
    print(f"Created subagent: {subagent.id}")
    
    # Register the subagent
    await registry.register_subagent(subagent)
    print("Subagent registered")
    
    # Check file content after registration
    with open(temp_path, 'r') as f:
        content = f.read()
        print(f"File content after registration: '{content}'")
        print(f"File size after registration: {len(content)}")
    
    # Try to read it back using CSV reader
    with open(temp_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        print(f"CSV rows found: {len(rows)}")
        for i, row in enumerate(rows):
            print(f"Row {i}: {dict(row)}")
    
    # Try to retrieve the subagent
    try:
        retrieved = await registry.get_subagent_by_id(subagent.id)
        print(f"Retrieved subagent: {retrieved.name if retrieved else 'None'}")
    except Exception as e:
        print(f"Error retrieving subagent: {e}")
    
    # Clean up
    import os
    os.unlink(temp_path)


if __name__ == "__main__":
    asyncio.run(main())