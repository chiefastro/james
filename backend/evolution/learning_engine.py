"""
Core learning engine for agent evolution and adaptation.

This module provides the main learning capabilities that allow agents to
improve their performance based on experience, feedback, and outcomes.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
import statistics

from ..memory.memory_manager import MemoryManager
from ..memory.memory_types import MemoryType, MemoryQuery
from ..observability.metrics_collector import MetricsCollector, get_metrics_collector
from ..observability.trace_analyzer import TraceAnalyzer, TraceInsight

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of learning that can be performed."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DECISION_IMPROVEMENT = "decision_improvement"
    ERROR_REDUCTION = "error_reduction"
    EFFICIENCY_ENHANCEMENT = "efficiency_enhancement"
    PATTERN_RECOGNITION = "pattern_recognition"
    BEHAVIORAL_ADAPTATION = "behavioral_adaptation"


@dataclass
class LearningEvent:
    """Represents a learning event or experience."""
    id: str
    event_type: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    success: bool
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5


@dataclass
class LearningInsight:
    """Represents an insight gained from learning."""
    id: str
    learning_type: LearningType
    description: str
    confidence: float
    supporting_evidence: List[str]
    recommended_actions: List[str]
    impact_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningStrategy(ABC):
    """Abstract base class for learning strategies."""
    
    @abstractmethod
    async def learn_from_events(
        self,
        events: List[LearningEvent]
    ) -> List[LearningInsight]:
        """
        Learn from a collection of events.
        
        Args:
            events: List of learning events to analyze
            
        Returns:
            List of insights gained from the events
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this learning strategy."""
        pass


class PerformanceLearningStrategy(LearningStrategy):
    """Learning strategy focused on performance optimization."""
    
    def __init__(self, performance_threshold: float = 0.8):
        """
        Initialize performance learning strategy.
        
        Args:
            performance_threshold: Minimum performance score to consider successful
        """
        self.performance_threshold = performance_threshold
    
    async def learn_from_events(
        self,
        events: List[LearningEvent]
    ) -> List[LearningInsight]:
        """Learn from performance-related events."""
        insights = []
        
        if not events:
            return insights
        
        # Group events by context similarity
        context_groups = self._group_events_by_context(events)
        
        for context_key, group_events in context_groups.items():
            # Analyze performance patterns
            performance_scores = [
                event.outcome.get("performance_score", 0.0)
                for event in group_events
                if "performance_score" in event.outcome
            ]
            
            if not performance_scores:
                continue
            
            avg_performance = statistics.mean(performance_scores)
            
            # Generate insights based on performance patterns
            if avg_performance < self.performance_threshold:
                insight = LearningInsight(
                    id=str(uuid4()),
                    learning_type=LearningType.PERFORMANCE_OPTIMIZATION,
                    description=f"Low performance detected in context '{context_key}' (avg: {avg_performance:.2f})",
                    confidence=0.8,
                    supporting_evidence=[
                        f"Average performance: {avg_performance:.2f}",
                        f"Events analyzed: {len(group_events)}",
                        f"Performance threshold: {self.performance_threshold}"
                    ],
                    recommended_actions=[
                        "Review and optimize decision-making logic for this context",
                        "Increase training data for similar scenarios",
                        "Consider adjusting confidence thresholds"
                    ],
                    impact_score=1.0 - avg_performance,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "context": context_key,
                        "event_count": len(group_events),
                        "performance_scores": performance_scores
                    }
                )
                insights.append(insight)
        
        return insights
    
    def _group_events_by_context(
        self,
        events: List[LearningEvent]
    ) -> Dict[str, List[LearningEvent]]:
        """Group events by similar context."""
        groups = {}
        
        for event in events:
            # Create a context key based on event type and key context attributes
            context_attrs = []
            context_attrs.append(event.event_type)
            
            # Add key context attributes
            if "operation" in event.context:
                context_attrs.append(event.context["operation"])
            if "category" in event.context:
                context_attrs.append(event.context["category"])
            
            context_key = "_".join(context_attrs)
            
            if context_key not in groups:
                groups[context_key] = []
            groups[context_key].append(event)
        
        return groups
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "PerformanceLearningStrategy"


class ErrorLearningStrategy(LearningStrategy):
    """Learning strategy focused on error reduction."""
    
    def __init__(self, error_threshold: int = 3):
        """
        Initialize error learning strategy.
        
        Args:
            error_threshold: Minimum number of similar errors to trigger learning
        """
        self.error_threshold = error_threshold
    
    async def learn_from_events(
        self,
        events: List[LearningEvent]
    ) -> List[LearningInsight]:
        """Learn from error events."""
        insights = []
        
        # Filter to error events only
        error_events = [event for event in events if not event.success]
        
        if not error_events:
            return insights
        
        # Group errors by type
        error_groups = {}
        for event in error_events:
            error_type = event.outcome.get("error_type", "unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(event)
        
        # Generate insights for frequent error patterns
        for error_type, group_events in error_groups.items():
            if len(group_events) >= self.error_threshold:
                # Analyze error patterns
                common_contexts = self._find_common_contexts(group_events)
                
                insight = LearningInsight(
                    id=str(uuid4()),
                    learning_type=LearningType.ERROR_REDUCTION,
                    description=f"Recurring error pattern detected: {error_type} ({len(group_events)} occurrences)",
                    confidence=min(0.9, len(group_events) / 10.0),
                    supporting_evidence=[
                        f"Error type: {error_type}",
                        f"Occurrences: {len(group_events)}",
                        f"Common contexts: {common_contexts}"
                    ],
                    recommended_actions=[
                        f"Implement specific error handling for {error_type}",
                        "Add validation for common failure scenarios",
                        "Review and improve error recovery mechanisms"
                    ],
                    impact_score=len(group_events) / len(events),
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "error_type": error_type,
                        "event_count": len(group_events),
                        "common_contexts": common_contexts
                    }
                )
                insights.append(insight)
        
        return insights
    
    def _find_common_contexts(self, events: List[LearningEvent]) -> List[str]:
        """Find common context attributes across error events."""
        context_counts = {}
        
        for event in events:
            for key, value in event.context.items():
                context_key = f"{key}:{value}"
                context_counts[context_key] = context_counts.get(context_key, 0) + 1
        
        # Return contexts that appear in at least 50% of events
        threshold = len(events) * 0.5
        common_contexts = [
            context for context, count in context_counts.items()
            if count >= threshold
        ]
        
        return common_contexts
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "ErrorLearningStrategy"


class LearningEngine:
    """
    Main learning engine that coordinates different learning strategies
    and manages the overall learning process for agents.
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        trace_analyzer: Optional[TraceAnalyzer] = None
    ):
        """
        Initialize the learning engine.
        
        Args:
            memory_manager: Memory manager for storing learning experiences
            metrics_collector: Metrics collector for performance data
            trace_analyzer: Trace analyzer for operational insights
        """
        self.memory_manager = memory_manager
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.trace_analyzer = trace_analyzer or TraceAnalyzer()
        
        # Learning strategies
        self.strategies: List[LearningStrategy] = [
            PerformanceLearningStrategy(),
            ErrorLearningStrategy()
        ]
        
        # Learning state
        self.learning_events: List[LearningEvent] = []
        self.insights: List[LearningInsight] = []
        self.learning_callbacks: List[Callable[[LearningInsight], None]] = []
        
        # Learning configuration
        self.learning_interval_minutes = 30
        self.max_events_to_analyze = 1000
        self.insight_retention_days = 30
        
        # Learning task
        self._learning_task: Optional[asyncio.Task] = None
        self._learning_running = False
        
        logger.info("LearningEngine initialized")
    
    def add_strategy(self, strategy: LearningStrategy) -> None:
        """Add a learning strategy to the engine."""
        self.strategies.append(strategy)
        logger.info(f"Added learning strategy: {strategy.get_strategy_name()}")
    
    def add_learning_callback(
        self,
        callback: Callable[[LearningInsight], None]
    ) -> None:
        """Add a callback to be called when new insights are generated."""
        self.learning_callbacks.append(callback)
        logger.info("Added learning callback")
    
    async def record_learning_event(
        self,
        event_type: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        confidence: float = 0.5,
        importance_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LearningEvent:
        """
        Record a learning event for future analysis.
        
        Args:
            event_type: Type of event (e.g., "classification", "delegation")
            context: Context information when the event occurred
            outcome: Outcome of the event
            success: Whether the event was successful
            confidence: Confidence in the outcome
            importance_score: Importance score for learning (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            The created learning event
        """
        event = LearningEvent(
            id=str(uuid4()),
            event_type=event_type,
            context=context,
            outcome=outcome,
            success=success,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            importance_score=importance_score
        )
        
        self.learning_events.append(event)
        
        # Store in memory if available
        if self.memory_manager:
            try:
                await self.memory_manager.store_memory(
                    content=f"Learning event: {event_type} - {'Success' if success else 'Failure'}",
                    memory_type=MemoryType.EPISODIC,
                    metadata={
                        "event_id": event.id,
                        "event_type": event_type,
                        "success": success,
                        "confidence": confidence,
                        "context": context,
                        "outcome": outcome
                    },
                    importance_score=importance_score,
                    tags=["learning", "experience", event_type]
                )
            except Exception as e:
                logger.warning(f"Failed to store learning event in memory: {e}")
        
        # Trim old events if we have too many
        if len(self.learning_events) > self.max_events_to_analyze:
            # Keep the most recent and most important events
            self.learning_events.sort(
                key=lambda x: (x.timestamp, x.importance_score),
                reverse=True
            )
            self.learning_events = self.learning_events[:self.max_events_to_analyze]
        
        logger.debug(f"Recorded learning event: {event_type} ({'success' if success else 'failure'})")
        return event
    
    async def start_learning(self) -> None:
        """Start the continuous learning process."""
        if self._learning_running:
            logger.warning("Learning process is already running")
            return
        
        self._learning_running = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Started continuous learning process")
    
    async def stop_learning(self) -> None:
        """Stop the continuous learning process."""
        if not self._learning_running:
            return
        
        self._learning_running = False
        
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous learning process")
    
    async def _learning_loop(self) -> None:
        """Main learning loop that runs continuously."""
        while self._learning_running:
            try:
                await self.perform_learning_cycle()
                await asyncio.sleep(self.learning_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def perform_learning_cycle(self) -> List[LearningInsight]:
        """
        Perform a complete learning cycle.
        
        Returns:
            List of new insights generated
        """
        logger.info("Starting learning cycle")
        
        # Get recent events for analysis
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_events = [
            event for event in self.learning_events
            if event.timestamp >= cutoff_time
        ]
        
        if not recent_events:
            logger.info("No recent events to learn from")
            return []
        
        # Apply all learning strategies
        new_insights = []
        for strategy in self.strategies:
            try:
                strategy_insights = await strategy.learn_from_events(recent_events)
                new_insights.extend(strategy_insights)
                logger.debug(f"Strategy {strategy.get_strategy_name()} generated {len(strategy_insights)} insights")
            except Exception as e:
                logger.error(f"Error in learning strategy {strategy.get_strategy_name()}: {e}")
        
        # Store new insights
        for insight in new_insights:
            self.insights.append(insight)
            
            # Store in memory if available
            if self.memory_manager:
                try:
                    await self.memory_manager.store_memory(
                        content=f"Learning insight: {insight.description}",
                        memory_type=MemoryType.SEMANTIC,
                        metadata={
                            "insight_id": insight.id,
                            "learning_type": insight.learning_type.value,
                            "confidence": insight.confidence,
                            "impact_score": insight.impact_score,
                            "recommended_actions": insight.recommended_actions
                        },
                        importance_score=insight.impact_score,
                        tags=["learning", "insight", insight.learning_type.value]
                    )
                except Exception as e:
                    logger.warning(f"Failed to store learning insight in memory: {e}")
            
            # Notify callbacks
            for callback in self.learning_callbacks:
                try:
                    callback(insight)
                except Exception as e:
                    logger.error(f"Error in learning callback: {e}")
        
        # Clean up old insights
        await self._cleanup_old_insights()
        
        logger.info(f"Learning cycle completed: {len(new_insights)} new insights generated")
        return new_insights
    
    async def _cleanup_old_insights(self) -> None:
        """Clean up old insights to prevent memory bloat."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.insight_retention_days)
        
        original_count = len(self.insights)
        self.insights = [
            insight for insight in self.insights
            if insight.timestamp >= cutoff_time
        ]
        
        cleaned_count = original_count - len(self.insights)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old insights")
    
    async def get_insights(
        self,
        learning_type: Optional[LearningType] = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[LearningInsight]:
        """
        Get learning insights with optional filtering.
        
        Args:
            learning_type: Filter by learning type
            min_confidence: Minimum confidence threshold
            limit: Maximum number of insights to return
            
        Returns:
            List of matching insights
        """
        filtered_insights = self.insights
        
        # Apply filters
        if learning_type:
            filtered_insights = [
                insight for insight in filtered_insights
                if insight.learning_type == learning_type
            ]
        
        if min_confidence > 0.0:
            filtered_insights = [
                insight for insight in filtered_insights
                if insight.confidence >= min_confidence
            ]
        
        # Sort by impact score and timestamp
        filtered_insights.sort(
            key=lambda x: (x.impact_score, x.timestamp),
            reverse=True
        )
        
        return filtered_insights[:limit]
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        total_events = len(self.learning_events)
        total_insights = len(self.insights)
        
        # Count events by type
        event_types = {}
        success_rate_by_type = {}
        
        for event in self.learning_events:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            if event_type not in success_rate_by_type:
                success_rate_by_type[event_type] = {"total": 0, "success": 0}
            
            success_rate_by_type[event_type]["total"] += 1
            if event.success:
                success_rate_by_type[event_type]["success"] += 1
        
        # Calculate success rates
        for event_type in success_rate_by_type:
            stats = success_rate_by_type[event_type]
            stats["rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Count insights by type
        insight_types = {}
        for insight in self.insights:
            insight_type = insight.learning_type.value
            insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
        
        return {
            "learning_status": "running" if self._learning_running else "stopped",
            "total_events": total_events,
            "total_insights": total_insights,
            "event_types": event_types,
            "success_rates": {
                event_type: stats["rate"]
                for event_type, stats in success_rate_by_type.items()
            },
            "insight_types": insight_types,
            "strategies_count": len(self.strategies),
            "callbacks_count": len(self.learning_callbacks),
            "learning_interval_minutes": self.learning_interval_minutes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global learning engine instance
_global_engine: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """Get the global learning engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = LearningEngine()
    return _global_engine


async def setup_learning_engine(
    memory_manager: Optional[MemoryManager] = None,
    metrics_collector: Optional[MetricsCollector] = None
) -> LearningEngine:
    """Set up and start the learning engine."""
    global _global_engine
    _global_engine = LearningEngine(
        memory_manager=memory_manager,
        metrics_collector=metrics_collector
    )
    await _global_engine.start_learning()
    return _global_engine