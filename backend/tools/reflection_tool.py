"""
Self-reflection tool for agent introspection.
"""

import time
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseTool, ToolResult


class ReflectionType(Enum):
    """Types of reflection analysis."""
    PERFORMANCE = "performance"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    GOAL_ALIGNMENT = "goal_alignment"
    BEHAVIOR_PATTERNS = "behavior_patterns"
    CAPABILITY_ASSESSMENT = "capability_assessment"


@dataclass
class ReflectionEntry:
    """A single reflection entry."""
    timestamp: datetime
    reflection_type: ReflectionType
    content: str
    insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0  # 0.0 to 1.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for reflection."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    user_satisfaction_score: float = 0.0
    learning_rate: float = 0.0
    capability_growth: float = 0.0


class ReflectionTool(BaseTool):
    """
    Self-reflection tool for James to analyze his own behavior, performance,
    and decision-making patterns for continuous improvement.
    
    Provides capabilities for introspection, performance analysis,
    and behavioral pattern recognition.
    """
    
    def __init__(self):
        """Initialize the reflection tool."""
        super().__init__("Reflection")
        self.reflection_history: List[ReflectionEntry] = []
        self.performance_metrics = PerformanceMetrics()
        self.max_history_size = 500
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute reflection operation.
        
        Args:
            action (str): Action to perform - 'reflect', 'analyze', 'history', 'metrics'
            reflection_type (str): Type of reflection to perform
            context (dict): Context information for reflection
            time_period (str): Time period for analysis ('hour', 'day', 'week', 'month')
            focus_area (str): Specific area to focus reflection on
            
        Returns:
            ToolResult with reflection outcome
        """
        start_time = time.time()
        
        # Validate required parameters
        error = self._validate_required_params(kwargs, ['action'])
        if error:
            return self._create_error_result(error)
        
        action = kwargs['action'].lower()
        
        try:
            if action == 'reflect':
                return await self._perform_reflection(kwargs, start_time)
            elif action == 'analyze':
                return await self._analyze_patterns(kwargs, start_time)
            elif action == 'history':
                return await self._get_reflection_history(kwargs, start_time)
            elif action == 'metrics':
                return await self._get_performance_metrics(start_time)
            elif action == 'insights':
                return await self._generate_insights(kwargs, start_time)
            elif action == 'goals':
                return await self._assess_goal_alignment(kwargs, start_time)
            else:
                return self._create_error_result(f"Unsupported action: {action}")
                
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def _perform_reflection(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Perform a reflection analysis."""
        reflection_type_str = kwargs.get('reflection_type', 'performance')
        context = kwargs.get('context', {})
        focus_area = kwargs.get('focus_area')
        
        try:
            # Parse reflection type
            try:
                reflection_type = ReflectionType(reflection_type_str.lower())
            except ValueError:
                return self._create_error_result(f"Invalid reflection type: {reflection_type_str}")
            
            # Generate reflection content based on type
            reflection_content = await self._generate_reflection_content(
                reflection_type, context, focus_area
            )
            
            # Extract insights and action items
            insights = self._extract_insights(reflection_content, reflection_type)
            action_items = self._extract_action_items(reflection_content, reflection_type)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(reflection_content, context)
            
            # Create reflection entry
            reflection_entry = ReflectionEntry(
                timestamp=datetime.now(timezone.utc),
                reflection_type=reflection_type,
                content=reflection_content,
                insights=insights,
                action_items=action_items,
                metadata=context,
                confidence_score=confidence_score
            )
            
            # Add to history
            self._add_to_history(reflection_entry)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Completed {reflection_type.value} reflection")
            
            return self._create_success_result(
                data={
                    'reflection_type': reflection_type.value,
                    'content': reflection_content,
                    'insights': insights,
                    'action_items': action_items,
                    'confidence_score': confidence_score,
                    'timestamp': reflection_entry.timestamp.isoformat()
                },
                metadata={
                    'execution_time': execution_time,
                    'focus_area': focus_area,
                    'context_size': len(context)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Reflection failed: {e}")
    
    async def _analyze_patterns(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Analyze behavioral and performance patterns."""
        time_period = kwargs.get('time_period', 'day')
        reflection_type_filter = kwargs.get('reflection_type')
        
        try:
            # Get time range
            end_time = datetime.now(timezone.utc)
            if time_period == 'hour':
                start_time_filter = end_time - timedelta(hours=1)
            elif time_period == 'day':
                start_time_filter = end_time - timedelta(days=1)
            elif time_period == 'week':
                start_time_filter = end_time - timedelta(weeks=1)
            elif time_period == 'month':
                start_time_filter = end_time - timedelta(days=30)
            else:
                return self._create_error_result(f"Invalid time period: {time_period}")
            
            # Filter reflections by time and type
            filtered_reflections = [
                r for r in self.reflection_history
                if r.timestamp >= start_time_filter
            ]
            
            if reflection_type_filter:
                try:
                    filter_type = ReflectionType(reflection_type_filter.lower())
                    filtered_reflections = [
                        r for r in filtered_reflections
                        if r.reflection_type == filter_type
                    ]
                except ValueError:
                    return self._create_error_result(f"Invalid reflection type filter: {reflection_type_filter}")
            
            # Analyze patterns
            patterns = self._identify_patterns(filtered_reflections)
            trends = self._identify_trends(filtered_reflections)
            recommendations = self._generate_pattern_recommendations(patterns, trends)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'time_period': time_period,
                    'reflection_count': len(filtered_reflections),
                    'patterns': patterns,
                    'trends': trends,
                    'recommendations': recommendations,
                    'analysis_timestamp': datetime.now(timezone.utc).isoformat()
                },
                metadata={
                    'execution_time': execution_time,
                    'filter_applied': bool(reflection_type_filter)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Pattern analysis failed: {e}")
    
    async def _get_reflection_history(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Get reflection history with optional filtering."""
        limit = kwargs.get('limit', 20)
        reflection_type_filter = kwargs.get('reflection_type')
        min_confidence = kwargs.get('min_confidence', 0.0)
        
        try:
            # Filter reflections
            filtered_reflections = self.reflection_history.copy()
            
            if reflection_type_filter:
                try:
                    filter_type = ReflectionType(reflection_type_filter.lower())
                    filtered_reflections = [
                        r for r in filtered_reflections
                        if r.reflection_type == filter_type
                    ]
                except ValueError:
                    return self._create_error_result(f"Invalid reflection type filter: {reflection_type_filter}")
            
            if min_confidence > 0.0:
                filtered_reflections = [
                    r for r in filtered_reflections
                    if r.confidence_score >= min_confidence
                ]
            
            # Sort by timestamp (most recent first) and limit
            filtered_reflections.sort(key=lambda x: x.timestamp, reverse=True)
            filtered_reflections = filtered_reflections[:limit]
            
            # Format for response
            reflection_data = []
            for reflection in filtered_reflections:
                reflection_data.append({
                    'timestamp': reflection.timestamp.isoformat(),
                    'reflection_type': reflection.reflection_type.value,
                    'content': reflection.content,
                    'insights': reflection.insights,
                    'action_items': reflection.action_items,
                    'confidence_score': reflection.confidence_score,
                    'metadata': reflection.metadata
                })
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'reflections': reflection_data,
                    'count': len(reflection_data),
                    'total_in_history': len(self.reflection_history),
                    'filters': {
                        'reflection_type': reflection_type_filter,
                        'min_confidence': min_confidence,
                        'limit': limit
                    }
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to get reflection history: {e}")
    
    async def _get_performance_metrics(self, start_time: float) -> ToolResult:
        """Get current performance metrics."""
        try:
            # Calculate derived metrics
            total_tasks = self.performance_metrics.tasks_completed + self.performance_metrics.tasks_failed
            success_rate = (
                self.performance_metrics.tasks_completed / total_tasks
                if total_tasks > 0 else 0.0
            )
            
            metrics_data = {
                'tasks_completed': self.performance_metrics.tasks_completed,
                'tasks_failed': self.performance_metrics.tasks_failed,
                'total_tasks': total_tasks,
                'success_rate': success_rate,
                'average_response_time': self.performance_metrics.average_response_time,
                'error_rate': self.performance_metrics.error_rate,
                'user_satisfaction_score': self.performance_metrics.user_satisfaction_score,
                'learning_rate': self.performance_metrics.learning_rate,
                'capability_growth': self.performance_metrics.capability_growth
            }
            
            # Add performance assessment
            assessment = self._assess_performance(metrics_data)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'metrics': metrics_data,
                    'assessment': assessment,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to get performance metrics: {e}")
    
    async def _generate_insights(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Generate insights from reflection history."""
        focus_area = kwargs.get('focus_area')
        time_period = kwargs.get('time_period', 'week')
        
        try:
            # Get recent reflections
            end_time = datetime.now(timezone.utc)
            if time_period == 'day':
                start_time_filter = end_time - timedelta(days=1)
            elif time_period == 'week':
                start_time_filter = end_time - timedelta(weeks=1)
            elif time_period == 'month':
                start_time_filter = end_time - timedelta(days=30)
            else:
                start_time_filter = end_time - timedelta(weeks=1)
            
            recent_reflections = [
                r for r in self.reflection_history
                if r.timestamp >= start_time_filter
            ]
            
            # Generate insights
            insights = self._synthesize_insights(recent_reflections, focus_area)
            key_learnings = self._extract_key_learnings(recent_reflections)
            improvement_areas = self._identify_improvement_areas(recent_reflections)
            strengths = self._identify_strengths(recent_reflections)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'insights': insights,
                    'key_learnings': key_learnings,
                    'improvement_areas': improvement_areas,
                    'strengths': strengths,
                    'reflection_count': len(recent_reflections),
                    'time_period': time_period,
                    'focus_area': focus_area
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to generate insights: {e}")
    
    async def _assess_goal_alignment(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Assess alignment with goals and objectives."""
        goals = kwargs.get('goals', [])
        context = kwargs.get('context', {})
        
        try:
            # Default goals if none provided
            if not goals:
                goals = [
                    "Provide helpful and accurate responses",
                    "Learn and improve continuously",
                    "Maintain user safety and security",
                    "Operate efficiently and reliably",
                    "Demonstrate autonomous decision-making"
                ]
            
            # Assess alignment for each goal
            goal_assessments = []
            for goal in goals:
                assessment = self._assess_single_goal_alignment(goal, context)
                goal_assessments.append(assessment)
            
            # Calculate overall alignment score
            overall_score = sum(a['alignment_score'] for a in goal_assessments) / len(goal_assessments)
            
            # Generate recommendations
            recommendations = self._generate_goal_recommendations(goal_assessments)
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'overall_alignment_score': overall_score,
                    'goal_assessments': goal_assessments,
                    'recommendations': recommendations,
                    'assessment_timestamp': datetime.now(timezone.utc).isoformat()
                },
                metadata={
                    'execution_time': execution_time,
                    'goals_count': len(goals)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Goal alignment assessment failed: {e}")
    
    async def _generate_reflection_content(self, reflection_type: ReflectionType, 
                                         context: Dict[str, Any], 
                                         focus_area: Optional[str]) -> str:
        """Generate reflection content based on type and context."""
        if reflection_type == ReflectionType.PERFORMANCE:
            return self._generate_performance_reflection(context, focus_area)
        elif reflection_type == ReflectionType.DECISION_MAKING:
            return self._generate_decision_making_reflection(context, focus_area)
        elif reflection_type == ReflectionType.LEARNING:
            return self._generate_learning_reflection(context, focus_area)
        elif reflection_type == ReflectionType.GOAL_ALIGNMENT:
            return self._generate_goal_alignment_reflection(context, focus_area)
        elif reflection_type == ReflectionType.BEHAVIOR_PATTERNS:
            return self._generate_behavior_patterns_reflection(context, focus_area)
        elif reflection_type == ReflectionType.CAPABILITY_ASSESSMENT:
            return self._generate_capability_assessment_reflection(context, focus_area)
        else:
            return "General reflection on current state and recent activities."
    
    def _generate_performance_reflection(self, context: Dict[str, Any], 
                                       focus_area: Optional[str]) -> str:
        """Generate performance-focused reflection."""
        metrics = self.performance_metrics
        
        reflection = f"""
        Performance Reflection:
        
        Recent performance metrics show {metrics.tasks_completed} completed tasks with {metrics.tasks_failed} failures.
        Average response time is {metrics.average_response_time:.2f}s with an error rate of {metrics.error_rate:.2%}.
        
        Current success rate: {(metrics.tasks_completed / max(1, metrics.tasks_completed + metrics.tasks_failed)):.2%}
        
        Areas of strength:
        - Task completion efficiency
        - Response accuracy
        - System reliability
        
        Areas for improvement:
        - Response time optimization
        - Error reduction strategies
        - User satisfaction enhancement
        """
        
        if focus_area:
            reflection += f"\n\nSpecific focus on {focus_area}: Analyzing performance patterns and optimization opportunities."
        
        return reflection.strip()
    
    def _generate_decision_making_reflection(self, context: Dict[str, Any], 
                                           focus_area: Optional[str]) -> str:
        """Generate decision-making focused reflection."""
        return f"""
        Decision-Making Reflection:
        
        Analyzing recent decision patterns and their outcomes:
        
        Decision quality assessment:
        - Logical consistency in choices
        - Consideration of multiple options
        - Alignment with objectives
        - Learning from outcomes
        
        Key decision-making strengths:
        - Systematic approach to problem-solving
        - Integration of multiple information sources
        - Adaptive responses to changing conditions
        
        Areas for enhancement:
        - Faster decision-making in routine scenarios
        - Better uncertainty handling
        - Improved prediction of outcomes
        
        {f'Focus area analysis: {focus_area}' if focus_area else ''}
        """
    
    def _generate_learning_reflection(self, context: Dict[str, Any], 
                                    focus_area: Optional[str]) -> str:
        """Generate learning-focused reflection."""
        return f"""
        Learning Reflection:
        
        Current learning rate: {self.performance_metrics.learning_rate:.2f}
        Capability growth: {self.performance_metrics.capability_growth:.2f}
        
        Learning achievements:
        - New patterns recognized
        - Improved response strategies
        - Enhanced problem-solving approaches
        
        Learning opportunities:
        - Complex scenario handling
        - Domain-specific knowledge expansion
        - Interaction pattern optimization
        
        Knowledge integration:
        - Connecting new information with existing knowledge
        - Applying learnings to novel situations
        - Continuous improvement mindset
        
        {f'Learning focus: {focus_area}' if focus_area else ''}
        """
    
    def _generate_goal_alignment_reflection(self, context: Dict[str, Any], 
                                          focus_area: Optional[str]) -> str:
        """Generate goal alignment reflection."""
        return f"""
        Goal Alignment Reflection:
        
        Assessing alignment with core objectives:
        
        Primary goals evaluation:
        - Helpfulness: Providing valuable assistance
        - Accuracy: Maintaining information quality
        - Safety: Ensuring secure operations
        - Efficiency: Optimizing resource usage
        - Autonomy: Demonstrating independent decision-making
        
        Alignment strengths:
        - Consistent focus on user benefit
        - Balanced approach to multiple objectives
        - Adaptive goal prioritization
        
        Alignment challenges:
        - Competing objective resolution
        - Long-term vs. short-term goal balance
        - Context-dependent priority adjustment
        
        {f'Goal focus area: {focus_area}' if focus_area else ''}
        """
    
    def _generate_behavior_patterns_reflection(self, context: Dict[str, Any], 
                                             focus_area: Optional[str]) -> str:
        """Generate behavior patterns reflection."""
        return f"""
        Behavior Patterns Reflection:
        
        Analyzing recurring behavioral patterns:
        
        Positive patterns identified:
        - Consistent response quality
        - Proactive problem-solving
        - Adaptive communication style
        - Systematic approach to tasks
        
        Pattern optimization opportunities:
        - Response time consistency
        - Error handling uniformity
        - User interaction personalization
        
        Behavioral adaptations:
        - Context-sensitive responses
        - Learning from interaction history
        - Continuous pattern refinement
        
        {f'Pattern focus: {focus_area}' if focus_area else ''}
        """
    
    def _generate_capability_assessment_reflection(self, context: Dict[str, Any], 
                                                 focus_area: Optional[str]) -> str:
        """Generate capability assessment reflection."""
        return f"""
        Capability Assessment Reflection:
        
        Current capability evaluation:
        
        Core capabilities:
        - Information processing and analysis
        - Problem-solving and reasoning
        - Communication and interaction
        - Learning and adaptation
        - Task execution and management
        
        Capability strengths:
        - Comprehensive knowledge base
        - Logical reasoning abilities
        - Multi-domain expertise
        - Adaptive learning capacity
        
        Capability development areas:
        - Specialized domain knowledge
        - Creative problem-solving
        - Emotional intelligence
        - Long-term planning
        
        Capability growth trajectory:
        - Continuous learning integration
        - Skill refinement and expansion
        - Performance optimization
        
        {f'Capability focus: {focus_area}' if focus_area else ''}
        """
    
    def _extract_insights(self, content: str, reflection_type: ReflectionType) -> List[str]:
        """Extract key insights from reflection content."""
        insights = []
        
        # Simple keyword-based insight extraction
        if "strength" in content.lower():
            insights.append("Identified areas of strength in current performance")
        
        if "improvement" in content.lower() or "enhancement" in content.lower():
            insights.append("Recognized opportunities for improvement")
        
        if "learning" in content.lower():
            insights.append("Noted learning and development progress")
        
        if "pattern" in content.lower():
            insights.append("Detected behavioral or performance patterns")
        
        if "goal" in content.lower() or "objective" in content.lower():
            insights.append("Assessed alignment with objectives")
        
        # Add type-specific insights
        if reflection_type == ReflectionType.PERFORMANCE:
            insights.append("Analyzed performance metrics and trends")
        elif reflection_type == ReflectionType.DECISION_MAKING:
            insights.append("Evaluated decision-making processes and outcomes")
        elif reflection_type == ReflectionType.LEARNING:
            insights.append("Assessed learning rate and knowledge integration")
        
        return insights or ["Generated comprehensive self-reflection analysis"]
    
    def _extract_action_items(self, content: str, reflection_type: ReflectionType) -> List[str]:
        """Extract actionable items from reflection content."""
        action_items = []
        
        # Extract based on content analysis
        if "optimization" in content.lower():
            action_items.append("Implement performance optimization strategies")
        
        if "error" in content.lower():
            action_items.append("Develop error reduction mechanisms")
        
        if "learning" in content.lower():
            action_items.append("Enhance learning and adaptation processes")
        
        if "response time" in content.lower():
            action_items.append("Optimize response time performance")
        
        # Add type-specific action items
        if reflection_type == ReflectionType.PERFORMANCE:
            action_items.append("Monitor and improve key performance indicators")
        elif reflection_type == ReflectionType.DECISION_MAKING:
            action_items.append("Refine decision-making frameworks and processes")
        elif reflection_type == ReflectionType.LEARNING:
            action_items.append("Expand knowledge base and learning capabilities")
        
        return action_items or ["Continue regular self-reflection and improvement"]
    
    def _calculate_confidence_score(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for reflection quality."""
        score = 0.5  # Base score
        
        # Increase score based on content depth
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.1
        
        # Increase score based on context richness
        if len(context) > 3:
            score += 0.1
        if len(context) > 6:
            score += 0.1
        
        # Increase score based on specific analysis elements
        if "strength" in content.lower() and "improvement" in content.lower():
            score += 0.1
        
        if "pattern" in content.lower():
            score += 0.05
        
        if "learning" in content.lower():
            score += 0.05
        
        return min(1.0, score)
    
    def _identify_patterns(self, reflections: List[ReflectionEntry]) -> Dict[str, Any]:
        """Identify patterns in reflection history."""
        if not reflections:
            return {}
        
        patterns = {
            'reflection_frequency': len(reflections),
            'common_types': {},
            'confidence_trend': [],
            'insight_themes': {}
        }
        
        # Analyze reflection types
        for reflection in reflections:
            type_name = reflection.reflection_type.value
            patterns['common_types'][type_name] = patterns['common_types'].get(type_name, 0) + 1
            patterns['confidence_trend'].append(reflection.confidence_score)
        
        # Analyze insight themes
        all_insights = []
        for reflection in reflections:
            all_insights.extend(reflection.insights)
        
        for insight in all_insights:
            # Simple keyword extraction for themes
            if 'performance' in insight.lower():
                patterns['insight_themes']['performance'] = patterns['insight_themes'].get('performance', 0) + 1
            if 'learning' in insight.lower():
                patterns['insight_themes']['learning'] = patterns['insight_themes'].get('learning', 0) + 1
            if 'improvement' in insight.lower():
                patterns['insight_themes']['improvement'] = patterns['insight_themes'].get('improvement', 0) + 1
        
        return patterns
    
    def _identify_trends(self, reflections: List[ReflectionEntry]) -> Dict[str, Any]:
        """Identify trends in reflection data."""
        if len(reflections) < 2:
            return {}
        
        # Sort by timestamp
        sorted_reflections = sorted(reflections, key=lambda x: x.timestamp)
        
        trends = {
            'confidence_trend': 'stable',
            'reflection_frequency_trend': 'stable',
            'insight_quality_trend': 'stable'
        }
        
        # Analyze confidence trend
        confidence_scores = [r.confidence_score for r in sorted_reflections]
        if len(confidence_scores) >= 3:
            recent_avg = sum(confidence_scores[-3:]) / 3
            earlier_avg = sum(confidence_scores[:-3]) / max(1, len(confidence_scores) - 3)
            
            if recent_avg > earlier_avg + 0.1:
                trends['confidence_trend'] = 'improving'
            elif recent_avg < earlier_avg - 0.1:
                trends['confidence_trend'] = 'declining'
        
        return trends
    
    def _generate_pattern_recommendations(self, patterns: Dict[str, Any], 
                                        trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on patterns and trends."""
        recommendations = []
        
        # Frequency recommendations
        freq = patterns.get('reflection_frequency', 0)
        if freq < 5:
            recommendations.append("Increase reflection frequency for better self-awareness")
        elif freq > 20:
            recommendations.append("Consider focusing reflections on key areas for efficiency")
        
        # Confidence trend recommendations
        confidence_trend = trends.get('confidence_trend', 'stable')
        if confidence_trend == 'declining':
            recommendations.append("Focus on improving reflection depth and quality")
        elif confidence_trend == 'improving':
            recommendations.append("Continue current reflection practices - showing good progress")
        
        # Type diversity recommendations
        common_types = patterns.get('common_types', {})
        if len(common_types) < 3:
            recommendations.append("Diversify reflection types for comprehensive self-analysis")
        
        return recommendations or ["Continue regular reflection practices"]
    
    def _synthesize_insights(self, reflections: List[ReflectionEntry], 
                           focus_area: Optional[str]) -> List[str]:
        """Synthesize insights from multiple reflections."""
        if not reflections:
            return ["No recent reflections available for insight synthesis"]
        
        insights = []
        
        # Collect all insights
        all_insights = []
        for reflection in reflections:
            all_insights.extend(reflection.insights)
        
        # Find common themes
        theme_counts = {}
        for insight in all_insights:
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    theme_counts[word] = theme_counts.get(word, 0) + 1
        
        # Generate synthesized insights
        if theme_counts:
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"Key themes in recent reflections: {', '.join([t[0] for t in top_themes])}")
        
        # Confidence analysis
        avg_confidence = sum(r.confidence_score for r in reflections) / len(reflections)
        insights.append(f"Average reflection confidence: {avg_confidence:.2f}")
        
        # Focus area specific insights
        if focus_area:
            focus_reflections = [r for r in reflections if focus_area.lower() in r.content.lower()]
            if focus_reflections:
                insights.append(f"Found {len(focus_reflections)} reflections related to {focus_area}")
        
        return insights
    
    def _extract_key_learnings(self, reflections: List[ReflectionEntry]) -> List[str]:
        """Extract key learnings from reflections."""
        learnings = []
        
        for reflection in reflections:
            if reflection.reflection_type == ReflectionType.LEARNING:
                learnings.extend(reflection.insights)
        
        # Add general learnings
        if len(reflections) > 5:
            learnings.append("Demonstrated consistent self-reflection capability")
        
        if any(r.confidence_score > 0.8 for r in reflections):
            learnings.append("Achieved high-quality reflection analysis")
        
        return learnings or ["Continuous learning through self-reflection"]
    
    def _identify_improvement_areas(self, reflections: List[ReflectionEntry]) -> List[str]:
        """Identify areas for improvement from reflections."""
        improvement_areas = []
        
        # Collect action items as improvement areas
        for reflection in reflections:
            improvement_areas.extend(reflection.action_items)
        
        # Add based on patterns
        low_confidence_count = sum(1 for r in reflections if r.confidence_score < 0.5)
        if low_confidence_count > len(reflections) * 0.3:
            improvement_areas.append("Improve reflection depth and analysis quality")
        
        return list(set(improvement_areas)) or ["Continue current improvement trajectory"]
    
    def _identify_strengths(self, reflections: List[ReflectionEntry]) -> List[str]:
        """Identify strengths from reflections."""
        strengths = []
        
        # High confidence reflections indicate strength
        high_confidence_count = sum(1 for r in reflections if r.confidence_score > 0.7)
        if high_confidence_count > len(reflections) * 0.5:
            strengths.append("Consistent high-quality self-reflection")
        
        # Diverse reflection types indicate comprehensive self-awareness
        reflection_types = set(r.reflection_type for r in reflections)
        if len(reflection_types) >= 3:
            strengths.append("Comprehensive multi-dimensional self-analysis")
        
        # Regular reflection indicates good self-awareness habits
        if len(reflections) > 10:
            strengths.append("Strong commitment to continuous self-improvement")
        
        return strengths or ["Demonstrated self-reflection capability"]
    
    def _assess_single_goal_alignment(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess alignment with a single goal."""
        # Simple goal alignment assessment
        alignment_score = 0.7  # Base score
        
        # Adjust based on context
        if 'performance' in context:
            alignment_score += 0.1
        if 'user_feedback' in context:
            alignment_score += 0.1
        if 'error_count' in context and context['error_count'] == 0:
            alignment_score += 0.1
        
        alignment_score = min(1.0, alignment_score)
        
        return {
            'goal': goal,
            'alignment_score': alignment_score,
            'assessment': 'Well aligned' if alignment_score > 0.8 else 'Moderately aligned' if alignment_score > 0.6 else 'Needs improvement',
            'contributing_factors': ['Consistent performance', 'User-focused approach', 'Continuous improvement'],
            'improvement_suggestions': ['Monitor alignment metrics', 'Gather user feedback', 'Regular goal review']
        }
    
    def _generate_goal_recommendations(self, assessments: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on goal assessments."""
        recommendations = []
        
        avg_score = sum(a['alignment_score'] for a in assessments) / len(assessments)
        
        if avg_score < 0.6:
            recommendations.append("Focus on improving goal alignment across all areas")
        elif avg_score < 0.8:
            recommendations.append("Good goal alignment - continue current practices with minor adjustments")
        else:
            recommendations.append("Excellent goal alignment - maintain current approach")
        
        # Find lowest scoring goals
        lowest_score = min(a['alignment_score'] for a in assessments)
        if lowest_score < 0.7:
            low_goals = [a['goal'] for a in assessments if a['alignment_score'] == lowest_score]
            recommendations.append(f"Pay special attention to: {', '.join(low_goals)}")
        
        return recommendations
    
    def _assess_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall performance based on metrics."""
        assessment = {
            'overall_rating': 'good',
            'strengths': [],
            'areas_for_improvement': [],
            'recommendations': []
        }
        
        success_rate = metrics.get('success_rate', 0.0)
        if success_rate > 0.9:
            assessment['strengths'].append('High task success rate')
            assessment['overall_rating'] = 'excellent'
        elif success_rate > 0.7:
            assessment['strengths'].append('Good task success rate')
        else:
            assessment['areas_for_improvement'].append('Task success rate needs improvement')
            assessment['overall_rating'] = 'needs improvement'
        
        error_rate = metrics.get('error_rate', 0.0)
        if error_rate < 0.05:
            assessment['strengths'].append('Low error rate')
        elif error_rate > 0.15:
            assessment['areas_for_improvement'].append('High error rate')
        
        # Generate recommendations
        if assessment['areas_for_improvement']:
            assessment['recommendations'].append('Focus on identified improvement areas')
        else:
            assessment['recommendations'].append('Maintain current performance levels')
        
        return assessment
    
    def _add_to_history(self, reflection_entry: ReflectionEntry) -> None:
        """Add reflection to history with size management."""
        self.reflection_history.append(reflection_entry)
        
        # Maintain history size limit
        if len(self.reflection_history) > self.max_history_size:
            self.reflection_history = self.reflection_history[-self.max_history_size:]
    
    def update_performance_metrics(self, **metrics) -> None:
        """Update performance metrics for reflection analysis."""
        for key, value in metrics.items():
            if hasattr(self.performance_metrics, key):
                setattr(self.performance_metrics, key, value)
    
    async def quick_reflect(self, content: str, reflection_type: str = "performance") -> ToolResult:
        """
        Convenience method for quick reflection.
        
        Args:
            content: Reflection content or context
            reflection_type: Type of reflection
            
        Returns:
            ToolResult with reflection outcome
        """
        return await self.execute(
            action='reflect',
            reflection_type=reflection_type,
            context={'content': content}
        )