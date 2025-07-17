"""
Registration script for seed subagents.

This script registers all the core seed subagents with the subagent registry,
making them available for delegation and task processing.
"""

import asyncio
import logging
from pathlib import Path

from .reflection_agent import ReflectionSubagent
from .builder_agent import BuilderSubagent
from .external_input_agent import ExternalInputSubagent
from ..registry.subagent_registry import SubagentRegistry, SubagentRegistryError

logger = logging.getLogger(__name__)


async def register_all_seed_subagents(registry: SubagentRegistry) -> dict:
    """
    Register all seed subagents with the provided registry.
    
    This function is used during system initialization to ensure
    all core subagents are available.
    
    Args:
        registry: The SubagentRegistry instance to register with
        
    Returns:
        Dictionary with registration results
    """
    results = {
        "registered": [],
        "failed": [],
        "total": 0
    }
    
    try:
        # Create seed subagent instances
        subagents = [
            ReflectionSubagent(),
            BuilderSubagent(),
            ExternalInputSubagent()
        ]
        
        results["total"] = len(subagents)
        
        # Register each subagent
        for subagent in subagents:
            try:
                # Get metadata for registration
                metadata = subagent.get_subagent_metadata()
                
                # Register with the registry
                await registry.register_subagent(metadata)
                
                results["registered"].append({
                    "id": subagent.subagent_id,
                    "name": subagent.name,
                    "capabilities": subagent.capabilities
                })
                
                logger.info(f"✅ Registered {subagent.name} with ID: {subagent.subagent_id}")
                
            except SubagentRegistryError as e:
                if "already exists" in str(e):
                    logger.warning(f"⚠️  {subagent.name} already registered, skipping")
                    results["registered"].append({
                        "id": subagent.subagent_id,
                        "name": subagent.name,
                        "capabilities": subagent.capabilities,
                        "status": "already_exists"
                    })
                else:
                    logger.error(f"❌ Failed to register {subagent.name}: {e}")
                    results["failed"].append({
                        "name": subagent.name,
                        "error": str(e)
                    })
            except Exception as e:
                logger.error(f"❌ Unexpected error registering {subagent.name}: {e}")
                results["failed"].append({
                    "name": subagent.name,
                    "error": str(e)
                })
        
        # Get registry statistics
        stats = await registry.get_registry_stats()
        logger.info(f"📊 Registry now contains {stats['active_subagents']} active subagents")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to register seed subagents: {e}")
        results["failed"].append({
            "name": "registration_process",
            "error": str(e)
        })
        return results


async def register_seed_subagents(registry_path: str = None) -> dict:
    """
    Register all seed subagents with the registry.
    
    Args:
        registry_path: Optional path to the registry file
        
    Returns:
        Dictionary with registration results
    """
    results = {
        "registered": [],
        "failed": [],
        "total": 0
    }
    
    try:
        # Initialize registry
        registry = SubagentRegistry(registry_path=registry_path)
        logger.info("Initialized subagent registry")
        
        # Create seed subagent instances
        subagents = [
            ReflectionSubagent(),
            BuilderSubagent(),
            ExternalInputSubagent()
        ]
        
        results["total"] = len(subagents)
        
        # Register each subagent
        for subagent in subagents:
            try:
                # Get metadata for registration
                metadata = subagent.get_subagent_metadata()
                
                # Register with the registry
                await registry.register_subagent(metadata)
                
                results["registered"].append({
                    "id": subagent.subagent_id,
                    "name": subagent.name,
                    "capabilities": subagent.capabilities
                })
                
                logger.info(f"✅ Registered {subagent.name} with ID: {subagent.subagent_id}")
                
            except SubagentRegistryError as e:
                if "already exists" in str(e):
                    logger.warning(f"⚠️  {subagent.name} already registered, skipping")
                    results["registered"].append({
                        "id": subagent.subagent_id,
                        "name": subagent.name,
                        "capabilities": subagent.capabilities,
                        "status": "already_exists"
                    })
                else:
                    logger.error(f"❌ Failed to register {subagent.name}: {e}")
                    results["failed"].append({
                        "name": subagent.name,
                        "error": str(e)
                    })
            except Exception as e:
                logger.error(f"❌ Unexpected error registering {subagent.name}: {e}")
                results["failed"].append({
                    "name": subagent.name,
                    "error": str(e)
                })
        
        # Get registry statistics
        stats = await registry.get_registry_stats()
        logger.info(f"📊 Registry now contains {stats['active_subagents']} active subagents")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize registry or register subagents: {e}")
        results["failed"].append({
            "name": "registry_initialization",
            "error": str(e)
        })
        return results


async def verify_registrations(registry_path: str = None) -> dict:
    """
    Verify that all seed subagents are properly registered.
    
    Args:
        registry_path: Optional path to the registry file
        
    Returns:
        Dictionary with verification results
    """
    verification_results = {
        "verified": [],
        "missing": [],
        "total_expected": 3
    }
    
    try:
        registry = SubagentRegistry(registry_path=registry_path)
        
        # Expected seed subagents
        expected_subagents = [
            "Reflection Agent",
            "Builder Agent", 
            "External Input Agent"
        ]
        
        # Get all registered subagents
        all_subagents = await registry.list_subagents(active_only=True)
        registered_names = [sa.name for sa in all_subagents]
        
        # Check each expected subagent
        for expected_name in expected_subagents:
            if expected_name in registered_names:
                # Find the subagent details
                subagent = next(sa for sa in all_subagents if sa.name == expected_name)
                verification_results["verified"].append({
                    "name": subagent.name,
                    "id": subagent.id,
                    "capabilities": subagent.capabilities,
                    "created_at": subagent.created_at.isoformat(),
                    "is_active": subagent.is_active
                })
                logger.info(f"✅ Verified {expected_name} is registered")
            else:
                verification_results["missing"].append(expected_name)
                logger.warning(f"⚠️  {expected_name} is not registered")
        
        return verification_results
        
    except Exception as e:
        logger.error(f"❌ Failed to verify registrations: {e}")
        return {
            "error": str(e),
            "verified": [],
            "missing": expected_subagents,
            "total_expected": 3
        }


async def unregister_seed_subagents(registry_path: str = None) -> dict:
    """
    Unregister all seed subagents (for cleanup/testing).
    
    Args:
        registry_path: Optional path to the registry file
        
    Returns:
        Dictionary with unregistration results
    """
    results = {
        "unregistered": [],
        "failed": [],
        "not_found": []
    }
    
    try:
        registry = SubagentRegistry(registry_path=registry_path)
        
        # Expected seed subagents
        seed_subagent_names = [
            "Reflection Agent",
            "Builder Agent",
            "External Input Agent"
        ]
        
        # Get all registered subagents
        all_subagents = await registry.list_subagents(active_only=False)
        
        # Find and unregister seed subagents
        for name in seed_subagent_names:
            try:
                subagent = next((sa for sa in all_subagents if sa.name == name), None)
                if subagent:
                    await registry.delete_subagent(subagent.id)
                    results["unregistered"].append({
                        "name": name,
                        "id": subagent.id
                    })
                    logger.info(f"✅ Unregistered {name}")
                else:
                    results["not_found"].append(name)
                    logger.warning(f"⚠️  {name} not found in registry")
                    
            except Exception as e:
                logger.error(f"❌ Failed to unregister {name}: {e}")
                results["failed"].append({
                    "name": name,
                    "error": str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to unregister seed subagents: {e}")
        return {
            "error": str(e),
            "unregistered": [],
            "failed": [],
            "not_found": []
        }


async def main():
    """Main function for running registration operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage seed subagent registrations")
    parser.add_argument("--action", choices=["register", "verify", "unregister"], 
                       default="register", help="Action to perform")
    parser.add_argument("--registry-path", help="Path to registry file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 Seed Subagent Registration Manager")
    print("=" * 50)
    
    if args.action == "register":
        print("📝 Registering seed subagents...")
        results = await register_seed_subagents(args.registry_path)
        
        print(f"\n📊 Registration Results:")
        print(f"   ✅ Successfully registered: {len(results['registered'])}")
        print(f"   ❌ Failed to register: {len(results['failed'])}")
        print(f"   📈 Total attempted: {results['total']}")
        
        if results["registered"]:
            print(f"\n✅ Registered subagents:")
            for subagent in results["registered"]:
                status = f" ({subagent.get('status', 'new')})" if subagent.get('status') else ""
                print(f"   - {subagent['name']}{status}")
                print(f"     ID: {subagent['id']}")
                print(f"     Capabilities: {', '.join(subagent['capabilities'][:3])}{'...' if len(subagent['capabilities']) > 3 else ''}")
        
        if results["failed"]:
            print(f"\n❌ Failed registrations:")
            for failure in results["failed"]:
                print(f"   - {failure['name']}: {failure['error']}")
    
    elif args.action == "verify":
        print("🔍 Verifying seed subagent registrations...")
        results = await verify_registrations(args.registry_path)
        
        if "error" in results:
            print(f"❌ Verification failed: {results['error']}")
        else:
            print(f"\n📊 Verification Results:")
            print(f"   ✅ Verified: {len(results['verified'])}/{results['total_expected']}")
            print(f"   ⚠️  Missing: {len(results['missing'])}")
            
            if results["verified"]:
                print(f"\n✅ Verified subagents:")
                for subagent in results["verified"]:
                    print(f"   - {subagent['name']} (ID: {subagent['id']})")
                    print(f"     Active: {subagent['is_active']}")
                    print(f"     Capabilities: {len(subagent['capabilities'])}")
            
            if results["missing"]:
                print(f"\n⚠️  Missing subagents:")
                for missing in results["missing"]:
                    print(f"   - {missing}")
    
    elif args.action == "unregister":
        print("🗑️  Unregistering seed subagents...")
        results = await unregister_seed_subagents(args.registry_path)
        
        if "error" in results:
            print(f"❌ Unregistration failed: {results['error']}")
        else:
            print(f"\n📊 Unregistration Results:")
            print(f"   ✅ Unregistered: {len(results['unregistered'])}")
            print(f"   ❌ Failed: {len(results['failed'])}")
            print(f"   ⚠️  Not found: {len(results['not_found'])}")
            
            if results["unregistered"]:
                print(f"\n✅ Unregistered subagents:")
                for subagent in results["unregistered"]:
                    print(f"   - {subagent['name']} (ID: {subagent['id']})")
    
    print(f"\n🎉 Operation completed!")


if __name__ == "__main__":
    asyncio.run(main())