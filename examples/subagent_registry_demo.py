#!/usr/bin/env python3
"""
Demo script showing SubagentRegistry functionality.

This script demonstrates the basic CRUD operations of the SubagentRegistry
including registration, retrieval, updates, and deletion of subagents.
"""

import asyncio
import tempfile
from pathlib import Path

from backend.models.core import Subagent
from backend.registry.subagent_registry import SubagentRegistry


async def main():
    """Demonstrate SubagentRegistry functionality."""
    print("🤖 Subagent Registry Demo")
    print("=" * 50)
    
    # Create a temporary registry for demo purposes
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    
    print(f"📁 Using temporary registry: {temp_path}")
    
    # Initialize registry
    registry = SubagentRegistry(registry_path=temp_path)
    print("✅ Registry initialized")
    
    # Create some sample subagents
    agents = [
        Subagent(
            name="Code Analyzer",
            description="Analyzes code for bugs and improvements",
            import_path="agents.code_analyzer",
            capabilities=["code_analysis", "bug_detection", "optimization"]
        ),
        Subagent(
            name="Documentation Writer",
            description="Generates documentation from code",
            import_path="agents.doc_writer",
            capabilities=["documentation", "writing", "code_analysis"]
        ),
        Subagent(
            name="Test Generator",
            description="Creates unit tests for code",
            import_path="agents.test_generator",
            capabilities=["testing", "code_generation", "validation"]
        )
    ]
    
    print(f"\n📝 Registering {len(agents)} subagents...")
    
    # Register all agents
    for agent in agents:
        await registry.register_subagent(agent)
        print(f"   ✅ Registered: {agent.name}")
    
    # List all subagents
    print(f"\n📋 Listing all subagents:")
    all_agents = await registry.list_subagents(active_only=False)
    for agent in all_agents:
        print(f"   🤖 {agent.name} - {agent.description}")
        print(f"      📍 Path: {agent.import_path}")
        print(f"      🔧 Capabilities: {', '.join(agent.capabilities)}")
        print()
    
    # Search by capabilities
    print("🔍 Searching for agents with 'code_analysis' capability:")
    code_agents = await registry.search_by_capabilities(["code_analysis"])
    for agent in code_agents:
        print(f"   🎯 {agent.name}")
    
    print(f"\n🔍 Searching for agents with 'testing' capability:")
    test_agents = await registry.search_by_capabilities(["testing"])
    for agent in test_agents:
        print(f"   🎯 {agent.name}")
    
    # Get agent by name
    print(f"\n🔎 Retrieving 'Code Analyzer' by name:")
    code_analyzer = await registry.get_subagent_by_name("Code Analyzer")
    if code_analyzer:
        print(f"   ✅ Found: {code_analyzer.name} (ID: {code_analyzer.id})")
    
    # Mark agent as used
    print(f"\n⏰ Marking 'Code Analyzer' as used:")
    await registry.mark_subagent_used(code_analyzer.id)
    print("   ✅ Usage timestamp updated")
    
    # Update an agent
    print(f"\n✏️  Updating 'Documentation Writer':")
    doc_writer = await registry.get_subagent_by_name("Documentation Writer")
    doc_writer.description = "Advanced documentation generator with AI assistance"
    doc_writer.capabilities.append("ai_assistance")
    await registry.update_subagent(doc_writer)
    print("   ✅ Agent updated")
    
    # Get registry statistics
    print(f"\n📊 Registry Statistics:")
    stats = await registry.get_registry_stats()
    print(f"   📈 Total subagents: {stats['total_subagents']}")
    print(f"   🟢 Active subagents: {stats['active_subagents']}")
    print(f"   🔴 Inactive subagents: {stats['inactive_subagents']}")
    print(f"   ⏱️  Used subagents: {stats['used_subagents']}")
    print(f"   💤 Unused subagents: {stats['unused_subagents']}")
    print(f"   💾 Registry file size: {stats['registry_file_size']} bytes")
    
    # Create backup
    print(f"\n💾 Creating backup:")
    backup_path = await registry.backup_registry()
    print(f"   ✅ Backup created: {backup_path}")
    
    # Delete an agent
    print(f"\n🗑️  Deleting 'Test Generator':")
    test_gen = await registry.get_subagent_by_name("Test Generator")
    await registry.delete_subagent(test_gen.id)
    print("   ✅ Agent deleted")
    
    # Final count
    final_agents = await registry.list_subagents(active_only=False)
    print(f"\n📋 Final count: {len(final_agents)} subagents remaining")
    
    # Cleanup
    import os
    os.unlink(temp_path)
    os.unlink(backup_path)
    print(f"\n🧹 Cleanup completed")
    
    print(f"\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())