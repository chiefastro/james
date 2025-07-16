#!/usr/bin/env python3
"""Main entry point for James consciousness system."""

import asyncio
import sys
import argparse
from pathlib import Path

from james.core.consciousness import JamesConsciousness
from james.tools.seed_tools import SeedTools
from james.core.message import MessageSource, MessagePriority


async def main():
    """Main function to start James consciousness system."""
    parser = argparse.ArgumentParser(description="James Consciousness System")
    parser.add_argument("--james-home", default="~/.james", help="James home directory")
    parser.add_argument("--message", help="Send a single message to James")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--api-only", action="store_true", help="Start API server only")
    
    args = parser.parse_args()
    
    # Initialize consciousness
    print("Initializing James consciousness system...")
    consciousness = JamesConsciousness(args.james_home)
    
    # Register seed tools
    seed_tools = SeedTools(args.james_home)
    consciousness.a2a.register_agent(seed_tools)
    
    print(f"James home: {consciousness.james_home}")
    print(f"Registered agents: {len(consciousness.a2a.agents)}")
    
    if args.message:
        # Send single message and exit
        print(f"Sending message: {args.message}")
        await consciousness.add_message(args.message, MessageSource.USER, MessagePriority.HIGH)
        result = await consciousness.process_single_message()
        if result:
            print(f"Response: {result}")
        else:
            print("No response")
        return
    
    elif args.api_only:
        # Start API server
        import uvicorn
        from james.api.main import app
        print("Starting James API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    
    elif args.interactive:
        # Interactive mode
        print("Starting interactive mode. Type 'quit' to exit.")
        print("James is listening...")
        
        # Start consciousness loop in background
        consciousness_task = asyncio.create_task(consciousness.consciousness_loop())
        
        try:
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if user_input:
                        await consciousness.add_message(
                            user_input, 
                            MessageSource.USER, 
                            MessagePriority.MEDIUM
                        )
                        print("Message sent to James...")
                        
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                    
        finally:
            print("\nShutting down James...")
            consciousness.stop()
            consciousness_task.cancel()
            try:
                await consciousness_task
            except asyncio.CancelledError:
                pass
    
    else:
        # Default: start consciousness loop
        print("Starting James consciousness loop...")
        print("Press Ctrl+C to stop")
        
        try:
            await consciousness.start()
        except KeyboardInterrupt:
            print("\nShutting down James...")
            consciousness.stop()


if __name__ == "__main__":
    asyncio.run(main())