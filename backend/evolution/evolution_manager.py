"""
Evolution manager for coordinating agent adaptation and improvement.

This module manages the overall evolution process, coordinating between
learning insights, adaptation strategies, and system changes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from uuid import uuid4

from .learning_engine import LearningEngine, LearningInsight, LearningType
from ..memory.memory_manager import MemoryManager
from ..memory.memory_types import MemoryType, MemoryQuery
from ..registry.subagent_registry import SubagentRegistry
from ..observability.metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class EvolutionType(Enum):
    """Types of evolution that can occur."""
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    CAPABILITY_ENHANCEMENT = "capability_enhancement"
    BEHAVIOR_MODIFICATION = "behavior_modification"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class EvolutionStatus(Enum):
    """Status of an evolution event."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    IMPLEMENTING = "implementing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class EvolutionEvent:
    """Represents an evolution event or change."""
    id: str
    evolution_type: EvolutionType
    description: str
    proposed_changes: Dict[str, Any]
    justification: str
    confidence: float
    expected_impact: float
    status: EvolutionStatus
    timestamp: datetime
    implementing_agent: Optional[str] = None
    completion_timestamp: Optional[datetime] = None
    actual_impact: Optional[float] = None
    rollback_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result of an evolution attempt."""
    success: bool
    changes_applied: Dict[str, Any]
    performance_delta: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvolutionStrategy:
    """Base class for evolution strategies."""
    
    def __init__(self, name: str):
        """Initialize evolution strategy."""
        self.name = name
    
    async def propose_evolution(
        self,
        insights: List[LearningInsight],
        current_state: Dict[str, Any]
    ) -> List[EvolutionEvent]:
        """
        Propose evolution events based on learning insights.
        
        Args:
            insights: Learning insights to base evolution on
            current_state: Current system state
            
        Returns:
            List of proposed evolution events
        """
        raise NotImplementedError
    
    async def implement_evolution(
        self,
        event: EvolutionEvent,
        current_state: Dict[str, Any]
    ) -> EvolutionResult:
        """
        Implement a proposed evolution event.
        
        Args:
            event: Evolution event to implement
            current_state: Current system state
            
        Returns:
            Result of the evolution implementation
        """
        raise NotImplementedError


class ParameterTuningStrategy(EvolutionStrategy):
    """Strategy for tuning system parameters based on performance."""
    
    def __init__(self):
        """Initialize parameter tuning strategy."""
        super().__init__("ParameterTuningStrategy")
        self.tunable_parameters = {
            "confidence_threshold": {"min": 0.1, "max": 0.9, "step": 0.05},
            "timeout_seconds": {"min": 5, "max": 60, "step": 5},
            "retry_attempts": {"min": 1, "max": 5, "step": 1},
            "batch_size": {"min": 1, "max": 100, "step": 5}
        }
    
    async def propose_evolution(
        self,
        insights: List[LearningInsight],
        current_state: Dict[str, Any]
    ) -> List[EvolutionEvent]:
        """Propose parameter tuning based on performance insights."""
        proposals = []
        
        for insight in insights:
            if insight.learning_type == LearningType.PERFORMANCE_OPTIMIZATION:
                # Analyze the insight to determine parameter adjustments
                if "low performance" in insight.description.lower():
                    # Propose reducing confidence threshold for better recall
                    current_threshold = current_state.get("confidence_threshold", 0.7)
                    new_threshold = max(
                        self.tunable_parameters["confidence_threshold"]["min"],
                        current_threshold - self.tunable_parameters["confidence_threshold"]["step"]
                    )
                    
                    if new_threshold != current_threshold:
                        event = EvolutionEvent(
                            id=str(uuid4()),
                            evolution_type=EvolutionType.PARAMETER_TUNING,
                            description=f"Reduce confidence threshold from {current_threshold} to {new_threshold}",
                            proposed_changes={"confidence_threshold": new_threshold},
                            justification=f"Performance insight suggests lowering threshold: {insight.description}",
                            confidence=insight.confidence * 0.8,
                            expected_impact=insight.impact_score,
                            status=EvolutionStatus.PROPOSED,
                            timestamp=datetime.now(timezone.utc),
                            metadata={
                                "source_insight_id": insight.id,
                                "current_value": current_threshold,
                                "proposed_value": new_threshold
                            }
                        )
                        proposals.append(event)
            
            elif insight.learning_type == LearningType.ERROR_REDUCTION:
                # Propose increasing retry attempts for error-prone operations
                if "error" in insight.description.lower():
                    current_retries = current_state.get("retry_attempts", 3)
                    new_retries = min(
                        self.tunable_parameters["retry_attempts"]["max"],
                        current_retries + self.tunable_parameters["retry_attempts"]["step"]
                    )
                    
                    if new_retries != current_retries:
                        event = EvolutionEvent(
                            id=str(uuid4()),
                            evolution_type=EvolutionType.PARAMETER_TUNING,
                            description=f"Increase retry attempts from {current_retries} to {new_retries}",
                            proposed_changes={"retry_attempts": new_retries},
                            justification=f"Error insight suggests more retries needed: {insight.description}",
                            confidence=insight.confidence * 0.7,
                            expected_impact=insight.impact_score * 0.5,
                            status=EvolutionStatus.PROPOSED,
                            timestamp=datetime.now(timezone.utc),
                            metadata={
                                "source_insight_id": insight.id,
                                "current_value": current_retries,
                                "proposed_value": new_retries
                            }
                        )
                        proposals.append(event)
        
        return proposals
    
    async def implement_evolution(
        self,
        event: EvolutionEvent,
        current_state: Dict[str, Any]
    ) -> EvolutionResult:
        """Implement parameter tuning changes."""
        try:
            changes_applied = {}
            
            for param_name, new_value in event.proposed_changes.items():
                if param_name in self.tunable_parameters:
                    # Validate the new value is within acceptable range
                    param_config = self.tunable_parameters[param_name]
                    if param_config["min"] <= new_value <= param_config["max"]:
                        # In a real implementation, this would update the actual system parameters
                        # For now, we'll just record the change
                        changes_applied[param_name] = new_value
                        logger.info(f"Applied parameter change: {param_name} = {new_value}")
                    else:
                        return EvolutionResult(
                            success=False,
                            changes_applied={},
                            error_message=f"Parameter {param_name} value {new_value} out of range"
                        )
                else:
                    return EvolutionResult(
                        success=False,
                        changes_applied={},
                        error_message=f"Unknown parameter: {param_name}"
                    )
            
            return EvolutionResult(
                success=True,
                changes_applied=changes_applied,
                metadata={"strategy": self.name}
            )
            
        except Exception as e:
            logger.error(f"Error implementing parameter tuning: {e}")
            return EvolutionResult(
                success=False,
                changes_applied={},
                error_message=str(e)
            )


class BehaviorAdaptationStrategy(EvolutionStrategy):
    """Strategy for adapting agent behavior patterns."""
    
    def __init__(self):
        """Initialize behavior adaptation strategy."""
        super().__init__("BehaviorAdaptationStrategy")
    
    async def propose_evolution(
        self,
        insights: List[LearningInsight],
        current_state: Dict[str, Any]
    ) -> List[EvolutionEvent]:
        """Propose behavior adaptations based on insights."""
        proposals = []
        
        for insight in insights:
            if insight.learning_type == LearningType.BEHAVIORAL_ADAPTATION:
                # Propose behavior modifications based on patterns
                event = EvolutionEvent(
                    id=str(uuid4()),
                    evolution_type=EvolutionType.BEHAVIOR_MODIFICATION,
                    description=f"Adapt behavior based on pattern: {insight.description}",
                    proposed_changes={
                        "behavior_pattern": insight.metadata.get("pattern", "unknown"),
                        "adaptation_type": "pattern_based"
                    },
                    justification=f"Behavioral insight: {insight.description}",
                    confidence=insight.confidence,
                    expected_impact=insight.impact_score,
                    status=EvolutionStatus.PROPOSED,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "source_insight_id": insight.id,
                        "pattern_data": insight.metadata
                    }
                )
                proposals.append(event)
        
        return proposals
    
    async def implement_evolution(
        self,
        event: EvolutionEvent,
        current_state: Dict[str, Any]
    ) -> EvolutionResult:
        """Implement behavior adaptation changes."""
        try:
            # In a real implementation, this would modify agent behavior patterns
            # For now, we'll simulate the change
            changes_applied = event.proposed_changes.copy()
            
            logger.info(f"Applied behavior adaptation: {event.description}")
            
            return EvolutionResult(
                success=True,
                changes_applied=changes_applied,
                metadata={"strategy": self.name}
            )
            
        except Exception as e:
            logger.error(f"Error implementing behavior adaptation: {e}")
            return EvolutionResult(
                success=False,
                changes_applied={},
                error_message=str(e)
            )


class EvolutionManager:
    """
    Manages the overall evolution process for the conscious agent system.
    
    Coordinates between learning insights, evolution strategies, and system changes
    to continuously improve agent performance and capabilities.
    """
    
    def __init__(
        self,
        learning_engine: Optional[LearningEngine] = None,
        memory_manager: Optional[MemoryManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        subagent_registry: Optional[SubagentRegistry] = None
    ):
        """
        Initialize the evolution manager.
        
        Args:
            learning_engine: Learning engine for insights
            memory_manager: Memory manager for storing evolution history
            metrics_collector: Metrics collector for performance tracking
            subagent_registry: Registry for subagent management
        """
        self.learning_engine = learning_engine
        self.memory_manager = memory_manager
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.subagent_registry = subagent_registry
        
        # Evolution strategies
        self.strategies: List[EvolutionStrategy] = [
            ParameterTuningStrategy(),
            BehaviorAdaptationStrategy()
        ]
        
        # Evolution state
        self.evolution_events: List[EvolutionEvent] = []
        self.evolution_callbacks: List[Callable[[EvolutionEvent], None]] = []
        self.approval_callbacks: List[Callable[[EvolutionEvent], bool]] = []
        
        # Evolution configuration
        self.auto_approve_threshold = 0.8  # Auto-approve high-confidence changes
        self.evolution_interval_hours = 6
        self.max_concurrent_evolutions = 3
        
        # Evolution task
        self._evolution_task: Optional[asyncio.Task] = None
        self._evolution_running = False
        
        logger.info("EvolutionManager initialized")
    
    def add_strategy(self, strategy: EvolutionStrategy) -> None:
        """Add an evolution strategy."""
        self.strategies.append(strategy)
        logger.info(f"Added evolution strategy: {strategy.name}")
    
    def add_evolution_callback(
        self,
        callback: Callable[[EvolutionEvent], None]
    ) -> None:
        """Add a callback for evolution events."""
        self.evolution_callbacks.append(callback)
        logger.info("Added evolution callback")
    
    def add_approval_callback(
        self,
        callback: Callable[[EvolutionEvent], bool]
    ) -> None:
        """Add a callback for evolution approval."""
        self.approval_callbacks.append(callback)
        logger.info("Added evolution approval callback")
    
    async def start_evolution(self) -> None:
        """Start the continuous evolution process."""
        if self._evolution_running:
            logger.warning("Evolution process is already running")
            return
        
        self._evolution_running = True
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        logger.info("Started continuous evolution process")
    
    async def stop_evolution(self) -> None:
        """Stop the continuous evolution process."""
        if not self._evolution_running:
            return
        
        self._evolution_running = False
        
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous evolution process")
    
    async def _evolution_loop(self) -> None:
        """Main evolution loop."""
        while self._evolution_running:
            try:
                await self.perform_evolution_cycle()
                await asyncio.sleep(self.evolution_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def perform_evolution_cycle(self) -> List[EvolutionEvent]:
        """
        Perform a complete evolution cycle.
        
        Returns:
            List of evolution events processed
        """
        logger.info("Starting evolution cycle")
        
        if not self.learning_engine:
            logger.warning("No learning engine available for evolution")
            return []
        
        # Get recent learning insights
        insights = await self.learning_engine.get_insights(
            min_confidence=0.5,
            limit=20
        )
        
        if not insights:
            logger.info("No learning insights available for evolution")
            return []
        
        # Get current system state
        current_state = await self._get_current_state()
        
        # Generate evolution proposals from all strategies
        proposed_events = []
        for strategy in self.strategies:
            try:
                strategy_proposals = await strategy.propose_evolution(insights, current_state)
                proposed_events.extend(strategy_proposals)
                logger.debug(f"Strategy {strategy.name} proposed {len(strategy_proposals)} evolutions")
            except Exception as e:
                logger.error(f"Error in evolution strategy {strategy.name}: {e}")
        
        if not proposed_events:
            logger.info("No evolution proposals generated")
            return []
        
        # Process evolution proposals
        processed_events = []
        for event in proposed_events:
            try:
                # Check if we should approve this evolution
                if await self._should_approve_evolution(event):
                    event.status = EvolutionStatus.APPROVED
                    
                    # Implement the evolution
                    result = await self._implement_evolution(event)
                    
                    if result.success:
                        event.status = EvolutionStatus.COMPLETED
                        event.completion_timestamp = datetime.now(timezone.utc)
                        event.actual_impact = result.performance_delta
                        logger.info(f"Successfully implemented evolution: {event.description}")
                    else:
                        event.status = EvolutionStatus.FAILED
                        event.rollback_reason = result.error_message
                        logger.warning(f"Failed to implement evolution: {event.description} - {result.error_message}")
                else:
                    logger.info(f"Evolution not approved: {event.description}")
                
                # Store the event
                self.evolution_events.append(event)
                processed_events.append(event)
                
                # Store in memory if available
                if self.memory_manager:
                    try:
                        await self.memory_manager.store_memory(
                            content=f"Evolution event: {event.description}",
                            memory_type=MemoryType.PROCEDURAL,
                            metadata={
                                "event_id": event.id,
                                "evolution_type": event.evolution_type.value,
                                "status": event.status.value,
                                "confidence": event.confidence,
                                "expected_impact": event.expected_impact,
                                "actual_impact": event.actual_impact
                            },
                            importance_score=event.expected_impact,
                            tags=["evolution", event.evolution_type.value, event.status.value]
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store evolution event in memory: {e}")
                
                # Notify callbacks
                for callback in self.evolution_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in evolution callback: {e}")
                
            except Exception as e:
                logger.error(f"Error processing evolution event: {e}")
        
        logger.info(f"Evolution cycle completed: {len(processed_events)} events processed")
        return processed_events
    
    async def _get_current_state(self) -> Dict[str, Any]:
        """Get the current system state for evolution decisions."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_threshold": 0.7,  # Default values
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "batch_size": 10
        }
        
        # Add metrics if available
        if self.metrics_collector:
            try:
                metrics_summary = self.metrics_collector.get_agent_performance_summary()
                state["performance_metrics"] = metrics_summary
            except Exception as e:
                logger.warning(f"Failed to get performance metrics: {e}")
        
        return state
    
    async def _should_approve_evolution(self, event: EvolutionEvent) -> bool:
        """Determine if an evolution should be approved."""
        # Auto-approve high-confidence, low-risk changes
        if event.confidence >= self.auto_approve_threshold and event.expected_impact < 0.5:
            return True
        
        # Check approval callbacks
        for callback in self.approval_callbacks:
            try:
                if not callback(event):
                    return False
            except Exception as e:
                logger.error(f"Error in approval callback: {e}")
                return False
        
        # Default approval logic
        return event.confidence > 0.6
    
    async def _implement_evolution(self, event: EvolutionEvent) -> EvolutionResult:
        """Implement an approved evolution event."""
        event.status = EvolutionStatus.IMPLEMENTING
        
        # Find the appropriate strategy to implement this evolution
        for strategy in self.strategies:
            if (event.evolution_type == EvolutionType.PARAMETER_TUNING and 
                isinstance(strategy, ParameterTuningStrategy)) or \
               (event.evolution_type == EvolutionType.BEHAVIOR_MODIFICATION and 
                isinstance(strategy, BehaviorAdaptationStrategy)):
                
                current_state = await self._get_current_state()
                return await strategy.implement_evolution(event, current_state)
        
        # No suitable strategy found
        return EvolutionResult(
            success=False,
            changes_applied={},
            error_message=f"No strategy available for evolution type: {event.evolution_type.value}"
        )
    
    async def get_evolution_history(
        self,
        evolution_type: Optional[EvolutionType] = None,
        status: Optional[EvolutionStatus] = None,
        limit: int = 50
    ) -> List[EvolutionEvent]:
        """
        Get evolution history with optional filtering.
        
        Args:
            evolution_type: Filter by evolution type
            status: Filter by status
            limit: Maximum number of events to return
            
        Returns:
            List of evolution events
        """
        filtered_events = self.evolution_events
        
        if evolution_type:
            filtered_events = [
                event for event in filtered_events
                if event.evolution_type == evolution_type
            ]
        
        if status:
            filtered_events = [
                event for event in filtered_events
                if event.status == status
            ]
        
        # Sort by timestamp (most recent first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about the evolution process."""
        total_events = len(self.evolution_events)
        
        # Count by status
        status_counts = {}
        for event in self.evolution_events:
            status = event.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by type
        type_counts = {}
        for event in self.evolution_events:
            evo_type = event.evolution_type.value
            type_counts[evo_type] = type_counts.get(evo_type, 0) + 1
        
        # Calculate success rate
        completed_events = [e for e in self.evolution_events if e.status == EvolutionStatus.COMPLETED]
        success_rate = len(completed_events) / total_events if total_events > 0 else 0.0
        
        # Calculate average impact
        avg_expected_impact = 0.0
        avg_actual_impact = 0.0
        
        if self.evolution_events:
            avg_expected_impact = sum(e.expected_impact for e in self.evolution_events) / len(self.evolution_events)
            
            actual_impacts = [e.actual_impact for e in completed_events if e.actual_impact is not None]
            if actual_impacts:
                avg_actual_impact = sum(actual_impacts) / len(actual_impacts)
        
        return {
            "evolution_status": "running" if self._evolution_running else "stopped",
            "total_events": total_events,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "success_rate": success_rate,
            "avg_expected_impact": avg_expected_impact,
            "avg_actual_impact": avg_actual_impact,
            "strategies_count": len(self.strategies),
            "callbacks_count": len(self.evolution_callbacks),
            "approval_callbacks_count": len(self.approval_callbacks),
            "evolution_interval_hours": self.evolution_interval_hours,
            "auto_approve_threshold": self.auto_approve_threshold,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global evolution manager instance
_global_manager: Optional[EvolutionManager] = None


def get_evolution_manager() -> EvolutionManager:
    """Get the global evolution manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = EvolutionManager()
    return _global_manager


async def setup_evolution_manager(
    learning_engine: Optional[LearningEngine] = None,
    memory_manager: Optional[MemoryManager] = None,
    subagent_registry: Optional[SubagentRegistry] = None
) -> EvolutionManager:
    """Set up and start the evolution manager."""
    global _global_manager
    _global_manager = EvolutionManager(
        learning_engine=learning_engine,
        memory_manager=memory_manager,
        subagent_registry=subagent_registry
    )
    await _global_manager.start_evolution()
    return _global_manager