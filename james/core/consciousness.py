"""Master consciousness graph for James."""

import asyncio
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
import os
from pathlib import Path

from james.core.message import Message, MessageQueue, MessageSource, MessagePriority
from james.agents.observer import ObserverAgent, ObserverAction
from james.agents.delegator import DelegatorAgent
from james.core.registry import SubAgentRegistry
from james.core.a2a import A2AProtocol
from james.core.observability import observability


class ConsciousnessState(BaseModel):
    messages: list[BaseMessage] = []
    current_message: Optional[Message] = None
    observer_decision: Optional[Dict[str, Any]] = None
    delegation_responses: Optional[list[Dict[str, Any]]] = None
    should_continue: bool = True


class JamesConsciousness:
    """Master consciousness system for James."""
    
    def __init__(self, james_home: str = "~/.james") -> None:
        self.james_home = Path(james_home).expanduser()
        self.james_home.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.message_queue = MessageQueue()
        self.registry = SubAgentRegistry(james_home)
        self.a2a = A2AProtocol()
        
        # Initialize agents
        self.observer = ObserverAgent()
        self.delegator = DelegatorAgent(self.registry, self.a2a)
        
        # Build the consciousness graph
        self.graph = self._build_graph()
        
        # State for the consciousness loop
        self.running = False
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for consciousness processing."""
        
        # Define the state structure
        graph = StateGraph(ConsciousnessState)
        
        # Add nodes
        graph.add_node("observer", self._observer_node)
        graph.add_node("delegator", self._delegator_node)
        
        # Add edges
        graph.set_entry_point("observer")
        graph.add_conditional_edges(
            "observer",
            self._should_delegate,
            {
                "delegate": "delegator",
                "end": END
            }
        )
        graph.add_edge("delegator", END)
        
        return graph.compile()
    
    async def _observer_node(self, state: ConsciousnessState) -> Dict[str, Any]:
        """Observer node that classifies incoming messages."""
        if not state.current_message:
            return {"should_continue": False}
        
        # Classify the message
        decision = await self.observer.classify_message(state.current_message)
        
        # Log to observability
        observability.log_observer_decision(
            message=state.current_message.content,
            decision={
                "action": decision.action.value,
                "reasoning": decision.reasoning,
                "delay_seconds": decision.delay_seconds,
                "priority_adjustment": decision.priority_adjustment
            }
        )
        
        # Handle different actions
        if decision.action == ObserverAction.IGNORE:
            # Delete the message
            return {
                "observer_decision": {"action": "ignore", "reasoning": decision.reasoning},
                "should_continue": False
            }
        
        elif decision.action == ObserverAction.DELAY:
            # Delay processing
            if decision.delay_seconds:
                await asyncio.sleep(decision.delay_seconds)
            # Re-add to queue with delay
            await self.message_queue.put(state.current_message)
            return {
                "observer_decision": {"action": "delay", "reasoning": decision.reasoning},
                "should_continue": False
            }
        
        elif decision.action == ObserverAction.ARCHIVE:
            # Archive in memory (implement memory storage)
            # For now, just log
            return {
                "observer_decision": {"action": "archive", "reasoning": decision.reasoning},
                "should_continue": False
            }
        
        elif decision.action == ObserverAction.ACT_NOW:
            # Proceed to delegation
            return {
                "observer_decision": {"action": "act_now", "reasoning": decision.reasoning},
                "should_continue": True
            }
        
        return {"should_continue": False}
    
    async def _delegator_node(self, state: ConsciousnessState) -> Dict[str, Any]:
        """Delegator node that handles task delegation."""
        if not state.current_message:
            return {"delegation_responses": []}
        
        # Process message through delegator
        responses = await self.delegator.process_message(state.current_message)
        
        # Convert responses to serializable format
        serialized_responses = []
        if responses:
            for response in responses:
                serialized_responses.append({
                    "message_id": response.message_id,
                    "success": response.success,
                    "result": response.result,
                    "error": response.error,
                    "timestamp": response.timestamp.isoformat()
                })
            
            # Log delegation to observability
            observability.log_delegation(
                task=state.current_message.content,
                selected_agents=[r.get("receiver_id", "unknown") for r in serialized_responses],
                delegation_strategy="auto",
                results=serialized_responses
            )
        
        return {"delegation_responses": serialized_responses}
    
    def _should_delegate(self, state: ConsciousnessState) -> str:
        """Conditional edge function to determine if we should delegate."""
        if (state.observer_decision and 
            state.observer_decision.get("action") == "act_now" and 
            state.should_continue):
            return "delegate"
        return "end"
    
    async def add_message(self, content: str, source: MessageSource = MessageSource.USER, 
                         priority: MessagePriority = MessagePriority.MEDIUM, 
                         metadata: Optional[Dict[str, Any]] = None,
                         sender_id: Optional[str] = None) -> None:
        """Add a message to the consciousness queue."""
        message = Message(
            content=content,
            source=source,
            priority=priority,
            metadata=metadata or {},
            sender_id=sender_id
        )
        await self.message_queue.put_prioritized(message)
    
    async def process_single_message(self) -> Optional[Dict[str, Any]]:
        """Process a single message from the queue."""
        if self.message_queue.empty():
            return None
        
        message = await self.message_queue.get()
        
        # Initialize state
        initial_state = ConsciousnessState(current_message=message)
        
        # Run through the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Log complete consciousness cycle
        consciousness_result = {
            "message_id": message.id,
            "content": message.content,
            "observer_decision": result.get("observer_decision"),
            "delegation_responses": result.get("delegation_responses"),
            "processed_at": message.timestamp.isoformat()
        }
        
        observability.log_consciousness_cycle(
            message_id=message.id,
            message_content=message.content,
            observer_decision=result.get("observer_decision", {}),
            delegation_results=result.get("delegation_responses")
        )
        
        return consciousness_result
    
    async def consciousness_loop(self) -> None:
        """Main consciousness loop - processes messages continuously."""
        self.running = True
        
        while self.running:
            try:
                result = await self.process_single_message()
                if result:
                    # Log or emit the result
                    print(f"Processed message: {result['message_id']}")
                else:
                    # No messages to process, wait briefly
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in consciousness loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    def stop(self) -> None:
        """Stop the consciousness loop."""
        self.running = False
    
    async def start(self) -> None:
        """Start the consciousness system."""
        print("James consciousness starting...")
        await self.consciousness_loop()