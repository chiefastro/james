"""
Adaptation strategies for agent evolution and learning.

This module provides different strategies for adapting agent behavior
based on performance feedback, user interactions, and environmental changes.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import statistics

from .learning_engine import LearningEvent, LearningInsight, LearningType
from ..observability.metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of adaptations that can be performed."""
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    STRATEGY_SELECTION = "strategy_selection"
    RESOURCE_ALLOCATION = "resource_allocation"
    RESPONSE_TIMING = "response_timing"
    ERROR_HANDLING = "error_handling"
    LEARNING_RATE = "learning_rate"


@dataclass
class AdaptationRecommendation:
    """Represents a recommended adaptation."""
    adaptation_type: AdaptationType
    description: str
    current_value: Any
    recommended_value: Any
    confidence: float
    expected_improvement: float
    justification: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptationStrategy(ABC):
    """Abstract base class for adaptation strategies."""
    
    @abstractmethod
    async def analyze_performance(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """
        Analyze performance and recommend adaptations.
        
        Args:
            events: Recent learning events to analyze
            current_config: Current system configuration
            
        Returns:
            List of adaptation recommendations
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this adaptation strategy."""
        pass


class PerformanceBasedAdaptation(AdaptationStrategy):
    """
    Adaptation strategy based on performance metrics and outcomes.
    
    Analyzes success rates, response times, and quality metrics to
    recommend adjustments to system parameters.
    """
    
    def __init__(
        self,
        performance_window_hours: int = 24,
        min_events_threshold: int = 10,
        improvement_threshold: float = 0.05
    ):
        """
        Initialize performance-based adaptation strategy.
        
        Args:
            performance_window_hours: Time window for performance analysis
            min_events_threshold: Minimum events needed for analysis
            improvement_threshold: Minimum improvement needed to recommend change
        """
        self.performance_window_hours = performance_window_hours
        self.min_events_threshold = min_events_threshold
        self.improvement_threshold = improvement_threshold
    
    async def analyze_performance(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze performance metrics and recommend adaptations."""
        recommendations = []
        
        if len(events) < self.min_events_threshold:
            logger.debug(f"Insufficient events for analysis: {len(events)} < {self.min_events_threshold}")
            return recommendations
        
        # Filter to recent events
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.performance_window_hours)
        recent_events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return recommendations
        
        # Analyze success rates by event type
        success_rates = self._calculate_success_rates(recent_events)
        
        for event_type, (success_rate, event_count) in success_rates.items():
            if event_count >= self.min_events_threshold:
                # Recommend threshold adjustments based on success rate
                if success_rate < 0.7:  # Low success rate
                    recommendations.extend(
                        self._recommend_threshold_adjustments(
                            event_type, success_rate, current_config
                        )
                    )
                elif success_rate > 0.95:  # Very high success rate
                    recommendations.extend(
                        self._recommend_efficiency_improvements(
                            event_type, success_rate, current_config
                        )
                    )
        
        # Analyze response times
        response_time_recommendations = self._analyze_response_times(recent_events, current_config)
        recommendations.extend(response_time_recommendations)
        
        # Analyze error patterns
        error_recommendations = self._analyze_error_patterns(recent_events, current_config)
        recommendations.extend(error_recommendations)
        
        return recommendations
    
    def _calculate_success_rates(
        self,
        events: List[LearningEvent]
    ) -> Dict[str, Tuple[float, int]]:
        """Calculate success rates by event type."""
        type_stats = {}
        
        for event in events:
            event_type = event.event_type
            if event_type not in type_stats:
                type_stats[event_type] = {"total": 0, "success": 0}
            
            type_stats[event_type]["total"] += 1
            if event.success:
                type_stats[event_type]["success"] += 1
        
        success_rates = {}
        for event_type, stats in type_stats.items():
            success_rate = stats["success"] / stats["total"]
            success_rates[event_type] = (success_rate, stats["total"])
        
        return success_rates
    
    def _recommend_threshold_adjustments(
        self,
        event_type: str,
        success_rate: float,
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Recommend threshold adjustments for low success rates."""
        recommendations = []
        
        # Recommend lowering confidence threshold
        current_threshold = current_config.get("confidence_threshold", 0.7)
        if current_threshold > 0.3:
            new_threshold = max(0.3, current_threshold - 0.1)
            expected_improvement = (0.8 - success_rate) * 0.5  # Estimate improvement
            
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.THRESHOLD_ADJUSTMENT,
                description=f"Lower confidence threshold for {event_type} to improve success rate",
                current_value=current_threshold,
                recommended_value=new_threshold,
                confidence=0.7,
                expected_improvement=expected_improvement,
                justification=f"Success rate of {success_rate:.2f} is below target, lowering threshold may help",
                metadata={
                    "event_type": event_type,
                    "current_success_rate": success_rate,
                    "parameter": "confidence_threshold"
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_efficiency_improvements(
        self,
        event_type: str,
        success_rate: float,
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Recommend efficiency improvements for high success rates."""
        recommendations = []
        
        # Recommend increasing confidence threshold for efficiency
        current_threshold = current_config.get("confidence_threshold", 0.7)
        if current_threshold < 0.9:
            new_threshold = min(0.9, current_threshold + 0.05)
            expected_improvement = 0.1  # Small efficiency gain
            
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.THRESHOLD_ADJUSTMENT,
                description=f"Increase confidence threshold for {event_type} to improve efficiency",
                current_value=current_threshold,
                recommended_value=new_threshold,
                confidence=0.6,
                expected_improvement=expected_improvement,
                justification=f"High success rate of {success_rate:.2f} suggests threshold can be increased",
                metadata={
                    "event_type": event_type,
                    "current_success_rate": success_rate,
                    "parameter": "confidence_threshold"
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_response_times(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze response times and recommend timing adjustments."""
        recommendations = []
        
        # Extract response times from events
        response_times = []
        for event in events:
            if "response_time_ms" in event.outcome:
                response_times.append(event.outcome["response_time_ms"])
        
        if len(response_times) < self.min_events_threshold:
            return recommendations
        
        avg_response_time = statistics.mean(response_times)
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        
        # Recommend timeout adjustments if response times are consistently high
        current_timeout = current_config.get("timeout_seconds", 30) * 1000  # Convert to ms
        
        if p95_response_time > current_timeout * 0.8:  # Close to timeout
            new_timeout = int((p95_response_time * 1.2) / 1000)  # 20% buffer, convert to seconds
            
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.RESPONSE_TIMING,
                description="Increase timeout to reduce timeout-related failures",
                current_value=current_timeout / 1000,
                recommended_value=new_timeout,
                confidence=0.8,
                expected_improvement=0.15,
                justification=f"95th percentile response time ({p95_response_time:.0f}ms) is close to timeout",
                metadata={
                    "avg_response_time_ms": avg_response_time,
                    "p95_response_time_ms": p95_response_time,
                    "parameter": "timeout_seconds"
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_error_patterns(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze error patterns and recommend error handling improvements."""
        recommendations = []
        
        # Count error types
        error_types = {}
        for event in events:
            if not event.success and "error_type" in event.outcome:
                error_type = event.outcome["error_type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        total_errors = sum(error_types.values())
        if total_errors < 5:  # Not enough errors to analyze
            return recommendations
        
        # Recommend retry adjustments for timeout errors
        timeout_errors = error_types.get("timeout", 0)
        if timeout_errors / total_errors > 0.3:  # More than 30% timeout errors
            current_retries = current_config.get("retry_attempts", 3)
            if current_retries < 5:
                new_retries = min(5, current_retries + 1)
                
                recommendation = AdaptationRecommendation(
                    adaptation_type=AdaptationType.ERROR_HANDLING,
                    description="Increase retry attempts to handle timeout errors",
                    current_value=current_retries,
                    recommended_value=new_retries,
                    confidence=0.7,
                    expected_improvement=0.2,
                    justification=f"Timeout errors account for {timeout_errors/total_errors:.1%} of failures",
                    metadata={
                        "error_type": "timeout",
                        "error_percentage": timeout_errors / total_errors,
                        "parameter": "retry_attempts"
                    }
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "PerformanceBasedAdaptation"


class FeedbackBasedAdaptation(AdaptationStrategy):
    """
    Adaptation strategy based on user feedback and interaction patterns.
    
    Analyzes user satisfaction, feedback scores, and interaction patterns
    to recommend behavioral and response adjustments.
    """
    
    def __init__(
        self,
        feedback_window_hours: int = 48,
        min_feedback_threshold: int = 5
    ):
        """
        Initialize feedback-based adaptation strategy.
        
        Args:
            feedback_window_hours: Time window for feedback analysis
            min_feedback_threshold: Minimum feedback events needed
        """
        self.feedback_window_hours = feedback_window_hours
        self.min_feedback_threshold = min_feedback_threshold
    
    async def analyze_performance(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze feedback and recommend adaptations."""
        recommendations = []
        
        # Filter to feedback events
        feedback_events = [
            e for e in events
            if e.event_type == "user_feedback" or "feedback_score" in e.outcome
        ]
        
        if len(feedback_events) < self.min_feedback_threshold:
            logger.debug(f"Insufficient feedback events: {len(feedback_events)} < {self.min_feedback_threshold}")
            return recommendations
        
        # Filter to recent feedback
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.feedback_window_hours)
        recent_feedback = [e for e in feedback_events if e.timestamp >= cutoff_time]
        
        if not recent_feedback:
            return recommendations
        
        # Analyze feedback scores
        feedback_scores = []
        for event in recent_feedback:
            if "feedback_score" in event.outcome:
                feedback_scores.append(event.outcome["feedback_score"])
        
        if feedback_scores:
            avg_feedback = statistics.mean(feedback_scores)
            
            # Recommend adaptations based on feedback
            if avg_feedback < 3.0:  # Low satisfaction (assuming 1-5 scale)
                recommendations.extend(
                    self._recommend_satisfaction_improvements(avg_feedback, current_config)
                )
        
        # Analyze feedback categories
        category_recommendations = self._analyze_feedback_categories(recent_feedback, current_config)
        recommendations.extend(category_recommendations)
        
        return recommendations
    
    def _recommend_satisfaction_improvements(
        self,
        avg_feedback: float,
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Recommend improvements based on low satisfaction scores."""
        recommendations = []
        
        # Recommend reducing response aggressiveness
        current_threshold = current_config.get("confidence_threshold", 0.7)
        if current_threshold > 0.5:
            new_threshold = max(0.5, current_threshold - 0.1)
            
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.THRESHOLD_ADJUSTMENT,
                description="Lower confidence threshold to provide more helpful responses",
                current_value=current_threshold,
                recommended_value=new_threshold,
                confidence=0.6,
                expected_improvement=0.3,
                justification=f"Low user satisfaction ({avg_feedback:.1f}/5.0) suggests responses are too restrictive",
                metadata={
                    "avg_feedback_score": avg_feedback,
                    "parameter": "confidence_threshold"
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_feedback_categories(
        self,
        feedback_events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze feedback by category and recommend specific improvements."""
        recommendations = []
        
        # Count feedback by category
        category_feedback = {}
        for event in feedback_events:
            category = event.context.get("category", "general")
            if category not in category_feedback:
                category_feedback[category] = []
            
            if "feedback_score" in event.outcome:
                category_feedback[category].append(event.outcome["feedback_score"])
        
        # Analyze each category
        for category, scores in category_feedback.items():
            if len(scores) >= 3:  # Minimum for meaningful analysis
                avg_score = statistics.mean(scores)
                
                if avg_score < 2.5:  # Poor performance in this category
                    recommendation = AdaptationRecommendation(
                        adaptation_type=AdaptationType.STRATEGY_SELECTION,
                        description=f"Improve strategy for {category} interactions",
                        current_value="default",
                        recommended_value=f"enhanced_{category}",
                        confidence=0.5,
                        expected_improvement=0.4,
                        justification=f"Poor feedback in {category} category (avg: {avg_score:.1f}/5.0)",
                        metadata={
                            "category": category,
                            "avg_score": avg_score,
                            "sample_size": len(scores)
                        }
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "FeedbackBasedAdaptation"


class ExperienceBasedAdaptation(AdaptationStrategy):
    """
    Adaptation strategy based on accumulated experience and learning patterns.
    
    Analyzes long-term trends, learning curves, and experience patterns
    to recommend strategic adjustments to learning and behavior.
    """
    
    def __init__(
        self,
        experience_window_days: int = 7,
        learning_rate_adjustment: float = 0.1
    ):
        """
        Initialize experience-based adaptation strategy.
        
        Args:
            experience_window_days: Time window for experience analysis
            learning_rate_adjustment: Rate of learning adjustments
        """
        self.experience_window_days = experience_window_days
        self.learning_rate_adjustment = learning_rate_adjustment
    
    async def analyze_performance(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze experience patterns and recommend adaptations."""
        recommendations = []
        
        if len(events) < 20:  # Need substantial experience
            return recommendations
        
        # Filter to experience window
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.experience_window_days)
        recent_events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return recommendations
        
        # Analyze learning trends
        trend_recommendations = self._analyze_learning_trends(recent_events, current_config)
        recommendations.extend(trend_recommendations)
        
        # Analyze experience diversity
        diversity_recommendations = self._analyze_experience_diversity(recent_events, current_config)
        recommendations.extend(diversity_recommendations)
        
        # Analyze confidence patterns
        confidence_recommendations = self._analyze_confidence_patterns(recent_events, current_config)
        recommendations.extend(confidence_recommendations)
        
        return recommendations
    
    def _analyze_learning_trends(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze learning trends over time."""
        recommendations = []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.timestamp)
        
        # Calculate success rate trend
        window_size = max(10, len(sorted_events) // 5)
        success_rates = []
        
        for i in range(window_size, len(sorted_events)):
            window_events = sorted_events[i-window_size:i]
            success_rate = sum(1 for e in window_events if e.success) / len(window_events)
            success_rates.append(success_rate)
        
        if len(success_rates) >= 3:
            # Check if learning is plateauing
            recent_rates = success_rates[-3:]
            if max(recent_rates) - min(recent_rates) < 0.05:  # Very stable
                current_learning_rate = current_config.get("learning_rate", 0.1)
                new_learning_rate = min(0.3, current_learning_rate + self.learning_rate_adjustment)
                
                recommendation = AdaptationRecommendation(
                    adaptation_type=AdaptationType.LEARNING_RATE,
                    description="Increase learning rate to overcome performance plateau",
                    current_value=current_learning_rate,
                    recommended_value=new_learning_rate,
                    confidence=0.6,
                    expected_improvement=0.1,
                    justification="Performance has plateaued, increasing learning rate may help",
                    metadata={
                        "recent_success_rates": recent_rates,
                        "parameter": "learning_rate"
                    }
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_experience_diversity(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze diversity of experiences."""
        recommendations = []
        
        # Count unique contexts
        contexts = set()
        for event in events:
            context_key = tuple(sorted(event.context.items()))
            contexts.add(context_key)
        
        diversity_ratio = len(contexts) / len(events)
        
        # If diversity is low, recommend exploration
        if diversity_ratio < 0.3:  # Low diversity
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.STRATEGY_SELECTION,
                description="Increase exploration to gain more diverse experience",
                current_value="exploitation",
                recommended_value="exploration",
                confidence=0.5,
                expected_improvement=0.2,
                justification=f"Low experience diversity ({diversity_ratio:.2f}) may limit learning",
                metadata={
                    "diversity_ratio": diversity_ratio,
                    "unique_contexts": len(contexts),
                    "total_events": len(events)
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_confidence_patterns(
        self,
        events: List[LearningEvent],
        current_config: Dict[str, Any]
    ) -> List[AdaptationRecommendation]:
        """Analyze confidence patterns in decisions."""
        recommendations = []
        
        # Extract confidence scores
        confidence_scores = [e.confidence for e in events if e.confidence is not None]
        
        if len(confidence_scores) < 10:
            return recommendations
        
        avg_confidence = statistics.mean(confidence_scores)
        confidence_std = statistics.stdev(confidence_scores)
        
        # If confidence is consistently low, recommend threshold adjustment
        if avg_confidence < 0.6:
            current_threshold = current_config.get("confidence_threshold", 0.7)
            new_threshold = max(0.3, current_threshold - 0.1)
            
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.THRESHOLD_ADJUSTMENT,
                description="Lower confidence threshold due to consistently low confidence",
                current_value=current_threshold,
                recommended_value=new_threshold,
                confidence=0.7,
                expected_improvement=0.15,
                justification=f"Average confidence ({avg_confidence:.2f}) is consistently low",
                metadata={
                    "avg_confidence": avg_confidence,
                    "confidence_std": confidence_std,
                    "parameter": "confidence_threshold"
                }
            )
            recommendations.append(recommendation)
        
        # If confidence variance is high, recommend stability improvements
        elif confidence_std > 0.3:
            recommendation = AdaptationRecommendation(
                adaptation_type=AdaptationType.STRATEGY_SELECTION,
                description="Improve confidence stability through better calibration",
                current_value="default",
                recommended_value="calibrated",
                confidence=0.6,
                expected_improvement=0.1,
                justification=f"High confidence variance ({confidence_std:.2f}) indicates poor calibration",
                metadata={
                    "avg_confidence": avg_confidence,
                    "confidence_std": confidence_std
                }
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "ExperienceBasedAdaptation"