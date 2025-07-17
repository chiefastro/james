"""
Comprehensive demonstration of seed tools for agent bootstrapping.

This example shows how James can use all the seed tools together
to perform complex operations, handle errors, reflect on performance,
and communicate both internally and externally.
"""

import asyncio
import json
from datetime import datetime
from backend.tools.file_writer import FileWriterTool
from backend.tools.terminal_executor import TerminalExecutorTool
from backend.tools.message_queue_tool import MessageQueueTool
from backend.tools.error_handler import ErrorHandlerTool, RetryConfig
from backend.tools.external_messenger import ExternalMessengerTool
from backend.tools.reflection_tool import ReflectionTool


class JamesBootstrapDemo:
    """
    Demonstration of James using seed tools for autonomous operation.
    
    This class simulates how James would use the seed tools to:
    1. Perform file operations
    2. Execute commands safely
    3. Handle errors with retry logic
    4. Communicate internally via message queue
    5. Send external notifications
    6. Reflect on performance and learn
    """
    
    def __init__(self):
        """Initialize all seed tools."""
        self.file_writer = FileWriterTool()
        self.terminal_executor = TerminalExecutorTool()
        self.message_queue = MessageQueueTool()
        self.error_handler = ErrorHandlerTool()
        self.external_messenger = ExternalMessengerTool()
        self.reflection_tool = ReflectionTool()
        
        # Track demo metrics
        self.demo_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'start_time': datetime.now()
        }
    
    async def run_complete_demo(self):
        """Run the complete seed tools demonstration."""
        print("🚀 Starting James Seed Tools Bootstrap Demo")
        print("=" * 50)
        
        try:
            # Phase 1: File Operations
            await self._demo_file_operations()
            
            # Phase 2: Terminal Execution
            await self._demo_terminal_execution()
            
            # Phase 3: Message Queue Operations
            await self._demo_message_queue()
            
            # Phase 4: Error Handling
            await self._demo_error_handling()
            
            # Phase 5: External Communication
            await self._demo_external_communication()
            
            # Phase 6: Self-Reflection
            await self._demo_self_reflection()
            
            # Final Summary
            await self._generate_final_summary()
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            await self._handle_demo_failure(e)
    
    async def _demo_file_operations(self):
        """Demonstrate file operations capabilities."""
        print("\n📁 Phase 1: File Operations Demo")
        print("-" * 30)
        
        try:
            # Create a configuration file
            config_data = {
                "agent_name": "James",
                "version": "1.0.0",
                "capabilities": [
                    "file_management",
                    "code_execution",
                    "error_handling",
                    "communication",
                    "self_reflection"
                ],
                "settings": {
                    "max_retries": 3,
                    "timeout_seconds": 30,
                    "log_level": "INFO"
                }
            }
            
            result = await self.file_writer.execute(
                file_path="config/james_config.json",
                content=config_data,
                mode="json"
            )
            
            if result.success:
                print(f"✅ Created configuration file: {result.data['file_path']}")
                self.demo_metrics['tasks_completed'] += 1
            else:
                print(f"❌ Failed to create config file: {result.error}")
                self.demo_metrics['tasks_failed'] += 1
            
            # Create a log file
            log_entry = f"[{datetime.now().isoformat()}] James bootstrap demo started\n"
            
            result = await self.file_writer.execute(
                file_path="logs/demo.log",
                content=log_entry,
                mode="write"
            )
            
            if result.success:
                print(f"✅ Created log file: {result.data['file_path']}")
                self.demo_metrics['tasks_completed'] += 1
            else:
                print(f"❌ Failed to create log file: {result.error}")
                self.demo_metrics['tasks_failed'] += 1
            
            # List created files
            result = await self.file_writer.list_files(recursive=True)
            if result.success:
                print(f"📋 Created {result.data['count']} files:")
                for file_path in result.data['files'][:5]:  # Show first 5
                    print(f"   - {file_path}")
                if result.data['count'] > 5:
                    print(f"   ... and {result.data['count'] - 5} more")
            
        except Exception as e:
            await self._log_error("file_operations", e)
    
    async def _demo_terminal_execution(self):
        """Demonstrate terminal execution capabilities."""
        print("\n💻 Phase 2: Terminal Execution Demo")
        print("-" * 30)
        
        try:
            # Execute system information commands
            commands = [
                "whoami",
                "pwd",
                "python3 --version",
                "echo 'Hello from James!'"
            ]
            
            for command in commands:
                result = await self.terminal_executor.execute(command=command)
                
                if result.success:
                    print(f"✅ Command '{command}': {result.data['stdout'].strip()}")
                    self.demo_metrics['tasks_completed'] += 1
                else:
                    print(f"❌ Command '{command}' failed: {result.error}")
                    self.demo_metrics['tasks_failed'] += 1
            
            # Execute Python code
            python_code = """
import json
import os

# Create a simple data analysis
data = {
    'demo_name': 'James Seed Tools',
    'timestamp': '2024-01-01T00:00:00Z',
    'metrics': {
        'files_created': 2,
        'commands_executed': 4,
        'success_rate': 0.95
    }
}

print("Python execution successful!")
print(f"Demo: {data['demo_name']}")
print(f"Success rate: {data['metrics']['success_rate']:.1%}")

# Write to James directory
with open('/james/python_output.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Data written to /james/python_output.json")
"""
            
            result = await self.terminal_executor.execute_python_code(
                python_code, 
                allow_file_ops=True
            )
            
            if result.success:
                print("✅ Python code execution successful:")
                print(f"   Output: {result.data['stdout'].strip()}")
                self.demo_metrics['tasks_completed'] += 1
            else:
                print(f"❌ Python code execution failed: {result.error}")
                self.demo_metrics['tasks_failed'] += 1
            
        except Exception as e:
            await self._log_error("terminal_execution", e)
    
    async def _demo_message_queue(self):
        """Demonstrate message queue operations."""
        print("\n📨 Phase 3: Message Queue Demo")
        print("-" * 30)
        
        try:
            # Send various types of messages
            messages = [
                {
                    "content": "System initialization complete",
                    "source": "system",
                    "priority": 1,
                    "classification": "act_now"
                },
                {
                    "content": "File operations completed successfully",
                    "source": "subagent",
                    "priority": 5,
                    "metadata": {"phase": "file_ops", "success": True}
                },
                {
                    "content": "Terminal execution phase completed",
                    "source": "system",
                    "priority": 3,
                    "classification": "archive"
                },
                {
                    "content": "Demo progress update",
                    "source": "user",
                    "priority": 10,
                    "metadata": {"progress": 0.6}
                }
            ]
            
            for msg_data in messages:
                result = await self.message_queue.execute(action="send", **msg_data)
                
                if result.success:
                    print(f"✅ Sent message: '{msg_data['content'][:30]}...'")
                    self.demo_metrics['tasks_completed'] += 1
                else:
                    print(f"❌ Failed to send message: {result.error}")
                    self.demo_metrics['tasks_failed'] += 1
            
            # Check queue status
            result = await self.message_queue.execute(action="status")
            if result.success:
                print(f"📊 Queue status: {result.data['size']} messages")
                print(f"   Empty: {result.data['is_empty']}")
                print(f"   Full: {result.data['is_full']}")
            
            # Peek at next message
            result = await self.message_queue.execute(action="peek")
            if result.success and result.metadata['has_message']:
                msg = result.data
                print(f"👀 Next message: '{msg['content'][:30]}...' (Priority: {msg['priority']})")
            
            # Check queue health
            result = await self.message_queue.check_queue_health()
            if result.success:
                print(f"🏥 Queue health: {result.data['health_status']}")
                if result.data['recommendations']:
                    print(f"   Recommendations: {result.data['recommendations'][0]}")
            
        except Exception as e:
            await self._log_error("message_queue", e)
    
    async def _demo_error_handling(self):
        """Demonstrate error handling and retry mechanisms."""
        print("\n🛠️ Phase 4: Error Handling Demo")
        print("-" * 30)
        
        try:
            # Simulate and handle various errors
            test_errors = [
                ValueError("Invalid configuration parameter"),
                ConnectionError("Network timeout during API call"),
                FileNotFoundError("Required configuration file missing"),
                PermissionError("Insufficient permissions for operation")
            ]
            
            for error in test_errors:
                # Inspect the error
                result = await self.error_handler.execute(
                    action="inspect",
                    error=error,
                    context={"demo_phase": "error_handling", "simulated": True}
                )
                
                if result.success:
                    print(f"🔍 Analyzed {result.data['error_type']}: {result.data['severity']} severity")
                    print(f"   Suggestions: {len(result.data['suggestions'])} provided")
                    self.demo_metrics['tasks_completed'] += 1
                else:
                    print(f"❌ Failed to analyze error: {result.error}")
                    self.demo_metrics['tasks_failed'] += 1
            
            # Demonstrate retry mechanism
            attempt_count = 0
            
            def flaky_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ConnectionError("Temporary network issue")
                return {"status": "success", "attempts": attempt_count}
            
            retry_config = {
                "max_attempts": 5,
                "strategy": "exponential_backoff",
                "base_delay": 0.1,
                "backoff_multiplier": 2.0
            }
            
            result = await self.error_handler.execute(
                action="retry",
                function=flaky_function,
                retry_config=retry_config
            )
            
            if result.success:
                print(f"✅ Retry successful after {result.data['attempts']} attempts")
                print(f"   Result: {result.data['result']}")
                self.demo_metrics['tasks_completed'] += 1
            else:
                print(f"❌ Retry failed: {result.error}")
                self.demo_metrics['tasks_failed'] += 1
            
            # Get error history
            result = await self.error_handler.execute(action="history", limit=5)
            if result.success:
                print(f"📚 Error history: {result.data['total_count']} total errors")
                print(f"   Recent errors: {result.data['count']}")
            
            # Analyze error patterns
            result = await self.error_handler.execute(action="analyze")
            if result.success:
                print(f"📈 Error analysis: {result.data['total_errors']} errors analyzed")
                if result.data['most_common_errors']:
                    top_error = result.data['most_common_errors'][0]
                    print(f"   Most common: {top_error[0]} ({top_error[1]} occurrences)")
            
        except Exception as e:
            await self._log_error("error_handling", e)
    
    async def _demo_external_communication(self):
        """Demonstrate external communication capabilities."""
        print("\n🌐 Phase 5: External Communication Demo")
        print("-" * 30)
        
        try:
            # Configure test endpoints
            endpoints = [
                {
                    "name": "httpbin_test",
                    "url": "https://httpbin.org/post",
                    "method": "http_post",
                    "format": "json"
                },
                {
                    "name": "webhook_test",
                    "url": "https://httpbin.org/anything",
                    "method": "http_post",
                    "format": "json",
                    "headers": {"X-Demo": "James-Seed-Tools"}
                }
            ]
            
            for endpoint_config in endpoints:
                result = await self.external_messenger.execute(
                    action="configure",
                    **endpoint_config
                )
                
                if result.success:
                    print(f"✅ Configured endpoint: {endpoint_config['name']}")
                    self.demo_metrics['tasks_completed'] += 1
                else:
                    print(f"❌ Failed to configure endpoint: {result.error}")
                    self.demo_metrics['tasks_failed'] += 1
            
            # List configured endpoints
            result = await self.external_messenger.execute(action="list_endpoints")
            if result.success:
                print(f"📋 Configured endpoints: {result.data['count']}")
                for endpoint in result.data['endpoints']:
                    print(f"   - {endpoint['name']}: {endpoint['method']} -> {endpoint['url']}")
            
            # Send test messages (these would actually work with real endpoints)
            test_messages = [
                {
                    "endpoint": "httpbin_test",
                    "message": {
                        "demo": "James Seed Tools",
                        "phase": "external_communication",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": self.demo_metrics
                    }
                }
            ]
            
            for msg_config in test_messages:
                print(f"📤 Would send message to {msg_config['endpoint']}")
                print(f"   Message: {json.dumps(msg_config['message'], indent=2)[:100]}...")
                # Note: Actual sending is commented out to avoid external calls in demo
                # result = await self.external_messenger.execute(action="send", **msg_config)
                self.demo_metrics['tasks_completed'] += 1
            
            # Demonstrate convenience methods
            print("📱 Slack/Discord integration available:")
            print("   - send_slack_message(webhook_url, message)")
            print("   - send_discord_message(webhook_url, message)")
            
        except Exception as e:
            await self._log_error("external_communication", e)
    
    async def _demo_self_reflection(self):
        """Demonstrate self-reflection capabilities."""
        print("\n🤔 Phase 6: Self-Reflection Demo")
        print("-" * 30)
        
        try:
            # Update performance metrics
            total_tasks = self.demo_metrics['tasks_completed'] + self.demo_metrics['tasks_failed']
            success_rate = self.demo_metrics['tasks_completed'] / max(1, total_tasks)
            
            self.reflection_tool.update_performance_metrics(
                tasks_completed=self.demo_metrics['tasks_completed'],
                tasks_failed=self.demo_metrics['tasks_failed'],
                average_response_time=0.5,
                error_rate=self.demo_metrics['tasks_failed'] / max(1, total_tasks),
                user_satisfaction_score=0.9,
                learning_rate=0.8,
                capability_growth=0.7
            )
            
            # Perform different types of reflection
            reflection_types = [
                ("performance", {"demo_phase": "complete", "success_rate": success_rate}),
                ("learning", {"new_capabilities": 6, "integration_success": True}),
                ("goal_alignment", {"primary_goal": "demonstrate_capabilities"})
            ]
            
            for reflection_type, context in reflection_types:
                result = await self.reflection_tool.execute(
                    action="reflect",
                    reflection_type=reflection_type,
                    context=context
                )
                
                if result.success:
                    print(f"🧠 {reflection_type.title()} reflection completed")
                    print(f"   Confidence: {result.data['confidence_score']:.2f}")
                    print(f"   Insights: {len(result.data['insights'])}")
                    print(f"   Action items: {len(result.data['action_items'])}")
                    
                    # Show first insight and action item
                    if result.data['insights']:
                        print(f"   💡 Key insight: {result.data['insights'][0]}")
                    if result.data['action_items']:
                        print(f"   📋 Action: {result.data['action_items'][0]}")
                    
                    self.demo_metrics['tasks_completed'] += 1
                else:
                    print(f"❌ {reflection_type} reflection failed: {result.error}")
                    self.demo_metrics['tasks_failed'] += 1
            
            # Analyze patterns
            result = await self.reflection_tool.execute(
                action="analyze",
                time_period="day"
            )
            
            if result.success:
                print(f"📊 Pattern analysis: {result.data['reflection_count']} reflections")
                if result.data['patterns'].get('common_types'):
                    print("   Common reflection types:")
                    for rtype, count in result.data['patterns']['common_types'].items():
                        print(f"     - {rtype}: {count}")
            
            # Generate insights
            result = await self.reflection_tool.execute(
                action="insights",
                time_period="day",
                focus_area="capability_demonstration"
            )
            
            if result.success:
                print(f"💡 Generated {len(result.data['insights'])} insights")
                print(f"🎯 Identified {len(result.data['strengths'])} strengths")
                print(f"📈 Found {len(result.data['improvement_areas'])} improvement areas")
            
            # Assess goal alignment
            demo_goals = [
                "Demonstrate all seed tool capabilities",
                "Show integration between tools",
                "Handle errors gracefully",
                "Provide comprehensive examples",
                "Enable autonomous operation"
            ]
            
            result = await self.reflection_tool.execute(
                action="goals",
                goals=demo_goals,
                context={"demo_completion": 1.0, "success_rate": success_rate}
            )
            
            if result.success:
                print(f"🎯 Goal alignment score: {result.data['overall_alignment_score']:.2f}")
                print(f"   Goals assessed: {len(result.data['goal_assessments'])}")
                if result.data['recommendations']:
                    print(f"   Recommendation: {result.data['recommendations'][0]}")
            
        except Exception as e:
            await self._log_error("self_reflection", e)
    
    async def _generate_final_summary(self):
        """Generate final demo summary."""
        print("\n📋 Demo Summary")
        print("=" * 50)
        
        total_tasks = self.demo_metrics['tasks_completed'] + self.demo_metrics['tasks_failed']
        success_rate = self.demo_metrics['tasks_completed'] / max(1, total_tasks) * 100
        duration = (datetime.now() - self.demo_metrics['start_time']).total_seconds()
        
        print(f"✅ Tasks completed: {self.demo_metrics['tasks_completed']}")
        print(f"❌ Tasks failed: {self.demo_metrics['tasks_failed']}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"⏱️ Duration: {duration:.1f} seconds")
        
        print(f"\n🛠️ Seed Tools Demonstrated:")
        print(f"   📁 FileWriterTool - Secure file operations")
        print(f"   💻 TerminalExecutorTool - Safe command execution")
        print(f"   📨 MessageQueueTool - Internal communication")
        print(f"   🛠️ ErrorHandlerTool - Error handling & retry")
        print(f"   🌐 ExternalMessengerTool - External communication")
        print(f"   🤔 ReflectionTool - Self-analysis & learning")
        
        print(f"\n🎯 Key Capabilities Shown:")
        print(f"   • Autonomous file management")
        print(f"   • Secure code execution in sandbox")
        print(f"   • Priority-based message handling")
        print(f"   • Intelligent error recovery")
        print(f"   • Multi-format external messaging")
        print(f"   • Continuous self-improvement")
        
        # Log final summary to file
        summary_data = {
            "demo_completed": datetime.now().isoformat(),
            "metrics": self.demo_metrics,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "tools_demonstrated": 6,
            "capabilities_shown": 6
        }
        
        try:
            result = await self.file_writer.execute(
                file_path="logs/demo_summary.json",
                content=summary_data,
                mode="json"
            )
            
            if result.success:
                print(f"\n💾 Demo summary saved to: {result.data['file_path']}")
        except Exception as e:
            print(f"⚠️ Could not save summary: {e}")
        
        print(f"\n🚀 James Seed Tools Bootstrap Demo Complete!")
        print(f"   James is now ready for autonomous operation with full tool suite.")
    
    async def _log_error(self, phase: str, error: Exception):
        """Log an error during the demo."""
        print(f"❌ Error in {phase}: {error}")
        self.demo_metrics['tasks_failed'] += 1
        
        # Use error handler to analyze the error
        try:
            result = await self.error_handler.execute(
                action="inspect",
                error=error,
                context={"demo_phase": phase}
            )
            
            if result.success:
                print(f"   Severity: {result.data['severity']}")
                if result.data['suggestions']:
                    print(f"   Suggestion: {result.data['suggestions'][0]}")
        except Exception:
            pass  # Don't let error handling errors break the demo
    
    async def _handle_demo_failure(self, error: Exception):
        """Handle overall demo failure."""
        print(f"\n💥 Demo Failed!")
        print(f"Error: {error}")
        
        # Try to reflect on the failure
        try:
            result = await self.reflection_tool.execute(
                action="reflect",
                reflection_type="performance",
                context={
                    "demo_failed": True,
                    "error": str(error),
                    "completed_tasks": self.demo_metrics['tasks_completed'],
                    "failed_tasks": self.demo_metrics['tasks_failed']
                }
            )
            
            if result.success:
                print(f"\n🤔 Failure reflection completed:")
                if result.data['insights']:
                    print(f"   Insight: {result.data['insights'][0]}")
                if result.data['action_items']:
                    print(f"   Action: {result.data['action_items'][0]}")
        except Exception:
            print("   Could not complete failure reflection")


async def main():
    """Run the complete seed tools demonstration."""
    demo = JamesBootstrapDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())