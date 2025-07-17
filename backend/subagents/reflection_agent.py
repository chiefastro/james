"""
Reflection Subagent for self-analysis and improvement.

This subagent provides James with the ability to reflect on his actions,
analyze his decision-making patterns, and identify areas for improvement.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


class ReflectionSubagent(BaseSubagent):
    """
    Subagent that handles self-reflection and analysis tasks.
    
    Capabilities:
    - Analyze decision-making patterns
    - Identify improvement opportunities
    - Generate self-assessment reports
    - Track behavioral changes over time
    """
    
    def __init__(self, subagent_id: Optional[str] = None):
        """Initialize the reflection subagent."""
        super().__init__(
            subagent_id=subagent_id or f"reflection_agent_{uuid4().hex[:8]}",
            name="Reflection Agent",
            description="Analyzes James' actions and decisions to identify patterns and improvement opportunities",
            capabilities=[
                "self_reflection",
                "pattern_analysis", 
                "decision_analysis",
                "improvement_identification",
                "behavioral_tracking",
                "performance_assessment"
            ]
        )
        self._reflection_history: List[Dict[str, Any]] = []
    
    async def process_task(self, task_id: str, task_description: str, input_data: Dict[str, Any]) -> SubagentResult:
        """
        Process a reflection task.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the reflection task
            input_data: Data to analyze (actions, decisions, outcomes, etc.)
            
        Returns:
            SubagentResult with reflection analysis
        """
        try:
            self.logger.info(f"Processing reflection task: {task_id}")
            
            # Determine the type of reflection requested
            reflection_type = input_data.get("reflection_type", "general")
            
            if reflection_type == "decision_analysis":
                result = await self._analyze_decisions(input_data)
            elif reflection_type == "pattern_analysis":
                result = await self._analyze_patterns(input_data)
            elif reflection_type == "performance_assessment":
                result = await self._assess_performance(input_data)
            elif reflection_type == "improvement_identification":
                result = await self._identify_improvements(input_data)
            elif reflection_type == "behavioral_tracking":
                result = await self._track_behavior(input_data)
            else:
                result = await self._general_reflection(input_data)
            
            # Store reflection in history
            reflection_record = {
                "task_id": task_id,
                "reflection_type": reflection_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_summary": self._summarize_input(input_data),
                "result_summary": self._summarize_result(result)
            }
            self._reflection_history.append(reflection_record)
            
            # Keep only last 100 reflections
            if len(self._reflection_history) > 100:
                self._reflection_history = self._reflection_history[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in reflection task {task_id}: {e}")
            return self._create_error_result(f"Reflection failed: {str(e)}")
    
    async def _analyze_decisions(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Analyze decision-making patterns and outcomes."""
        decisions = input_data.get("decisions", [])
        if not decisions:
            return self._create_error_result("No decisions provided for analysis")
        
        analysis = {
            "total_decisions": len(decisions),
            "decision_types": {},
            "outcome_patterns": {},
            "success_rate": 0.0,
            "common_factors": [],
            "recommendations": []
        }
        
        successful_decisions = 0
        decision_factors = []
        
        for decision in decisions:
            # Categorize decision type
            decision_type = decision.get("type", "unknown")
            analysis["decision_types"][decision_type] = analysis["decision_types"].get(decision_type, 0) + 1
            
            # Analyze outcome
            outcome = decision.get("outcome", "unknown")
            analysis["outcome_patterns"][outcome] = analysis["outcome_patterns"].get(outcome, 0) + 1
            
            if outcome in ["success", "positive", "good"]:
                successful_decisions += 1
            
            # Collect factors that influenced the decision
            factors = decision.get("factors", [])
            decision_factors.extend(factors)
        
        # Calculate success rate
        if len(decisions) > 0:
            analysis["success_rate"] = successful_decisions / len(decisions)
        
        # Identify common factors
        factor_counts = {}
        for factor in decision_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        analysis["common_factors"] = sorted(
            factor_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Generate recommendations
        if analysis["success_rate"] < 0.7:
            analysis["recommendations"].append("Consider spending more time analyzing options before deciding")
        
        if "time_pressure" in [f[0] for f in analysis["common_factors"]]:
            analysis["recommendations"].append("Try to reduce time pressure in decision-making when possible")
        
        analysis["recommendations"].append("Continue tracking decision outcomes to identify improvement patterns")
        
        return self._create_success_result(
            data=analysis,
            metadata={
                "analysis_type": "decision_analysis",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _analyze_patterns(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Analyze behavioral and operational patterns."""
        events = input_data.get("events", [])
        if not events:
            return self._create_error_result("No events provided for pattern analysis")
        
        patterns = {
            "temporal_patterns": {},
            "frequency_patterns": {},
            "sequence_patterns": [],
            "anomalies": [],
            "trends": []
        }
        
        # Analyze temporal patterns (time of day, day of week)
        for event in events:
            timestamp = event.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    day_of_week = dt.strftime('%A')
                    
                    patterns["temporal_patterns"][f"hour_{hour}"] = patterns["temporal_patterns"].get(f"hour_{hour}", 0) + 1
                    patterns["temporal_patterns"][f"day_{day_of_week}"] = patterns["temporal_patterns"].get(f"day_{day_of_week}", 0) + 1
                except:
                    continue
        
        # Analyze event frequency
        event_types = [event.get("type", "unknown") for event in events]
        for event_type in event_types:
            patterns["frequency_patterns"][event_type] = patterns["frequency_patterns"].get(event_type, 0) + 1
        
        # Look for sequence patterns (simplified)
        if len(events) >= 3:
            for i in range(len(events) - 2):
                sequence = [
                    events[i].get("type", "unknown"),
                    events[i+1].get("type", "unknown"),
                    events[i+2].get("type", "unknown")
                ]
                sequence_str = " -> ".join(sequence)
                patterns["sequence_patterns"].append(sequence_str)
        
        # Identify potential anomalies (events that occur very infrequently)
        total_events = len(events)
        for event_type, count in patterns["frequency_patterns"].items():
            if count / total_events < 0.05:  # Less than 5% of events
                patterns["anomalies"].append({
                    "type": event_type,
                    "frequency": count,
                    "percentage": (count / total_events) * 100
                })
        
        # Generate trend insights
        if patterns["temporal_patterns"]:
            most_active_hour = max(
                [(k, v) for k, v in patterns["temporal_patterns"].items() if k.startswith("hour_")],
                key=lambda x: x[1],
                default=(None, 0)
            )
            if most_active_hour[0]:
                hour = most_active_hour[0].split("_")[1]
                patterns["trends"].append(f"Most active during hour {hour}")
        
        return self._create_success_result(
            data=patterns,
            metadata={
                "analysis_type": "pattern_analysis",
                "events_analyzed": len(events),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _assess_performance(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Assess overall performance across different metrics."""
        metrics = input_data.get("metrics", {})
        time_period = input_data.get("time_period", "unknown")
        
        assessment = {
            "overall_score": 0.0,
            "metric_scores": {},
            "strengths": [],
            "weaknesses": [],
            "improvement_areas": [],
            "time_period": time_period
        }
        
        total_score = 0
        metric_count = 0
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                current_value = metric_data.get("current", 0)
                target_value = metric_data.get("target", 1)
                previous_value = metric_data.get("previous", current_value)
                
                # Calculate score (0-100)
                if target_value > 0:
                    score = min(100, (current_value / target_value) * 100)
                else:
                    score = 50  # Neutral score if no target
                
                assessment["metric_scores"][metric_name] = {
                    "score": score,
                    "current": current_value,
                    "target": target_value,
                    "previous": previous_value,
                    "trend": "improving" if current_value > previous_value else "declining" if current_value < previous_value else "stable"
                }
                
                total_score += score
                metric_count += 1
                
                # Identify strengths and weaknesses
                if score >= 80:
                    assessment["strengths"].append(metric_name)
                elif score < 60:
                    assessment["weaknesses"].append(metric_name)
                    assessment["improvement_areas"].append({
                        "metric": metric_name,
                        "current_score": score,
                        "gap_to_target": target_value - current_value
                    })
        
        # Calculate overall score
        if metric_count > 0:
            assessment["overall_score"] = total_score / metric_count
        
        return self._create_success_result(
            data=assessment,
            metadata={
                "analysis_type": "performance_assessment",
                "metrics_evaluated": metric_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _identify_improvements(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Identify specific areas and actions for improvement."""
        context = input_data.get("context", {})
        current_capabilities = input_data.get("current_capabilities", [])
        recent_challenges = input_data.get("recent_challenges", [])
        goals = input_data.get("goals", [])
        
        improvements = {
            "priority_areas": [],
            "specific_actions": [],
            "capability_gaps": [],
            "learning_opportunities": [],
            "resource_needs": []
        }
        
        # Analyze capability gaps
        for goal in goals:
            required_capabilities = goal.get("required_capabilities", [])
            missing_capabilities = [cap for cap in required_capabilities if cap not in current_capabilities]
            
            if missing_capabilities:
                improvements["capability_gaps"].append({
                    "goal": goal.get("description", "Unknown goal"),
                    "missing_capabilities": missing_capabilities
                })
        
        # Analyze recent challenges for improvement opportunities
        challenge_patterns = {}
        for challenge in recent_challenges:
            challenge_type = challenge.get("type", "unknown")
            challenge_patterns[challenge_type] = challenge_patterns.get(challenge_type, 0) + 1
        
        # Prioritize improvement areas based on challenge frequency
        for challenge_type, frequency in sorted(challenge_patterns.items(), key=lambda x: x[1], reverse=True):
            improvements["priority_areas"].append({
                "area": challenge_type,
                "frequency": frequency,
                "priority": "high" if frequency > 3 else "medium" if frequency > 1 else "low"
            })
        
        # Generate specific action recommendations
        if improvements["capability_gaps"]:
            improvements["specific_actions"].append("Develop missing capabilities identified in capability gaps")
        
        if improvements["priority_areas"]:
            top_area = improvements["priority_areas"][0]["area"]
            improvements["specific_actions"].append(f"Focus on improving {top_area} handling")
        
        improvements["specific_actions"].append("Continue regular self-reflection and pattern analysis")
        improvements["specific_actions"].append("Set measurable goals for improvement tracking")
        
        # Identify learning opportunities
        improvements["learning_opportunities"] = [
            "Study successful decision patterns from reflection history",
            "Analyze high-performing agents or systems in similar domains",
            "Experiment with new approaches in low-risk situations",
            "Seek feedback from interactions and outcomes"
        ]
        
        return self._create_success_result(
            data=improvements,
            metadata={
                "analysis_type": "improvement_identification",
                "challenges_analyzed": len(recent_challenges),
                "goals_analyzed": len(goals),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _track_behavior(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Track behavioral changes and trends over time."""
        behaviors = input_data.get("behaviors", [])
        time_window = input_data.get("time_window", "1_week")
        
        tracking = {
            "behavior_trends": {},
            "change_indicators": [],
            "stability_metrics": {},
            "notable_changes": []
        }
        
        # Group behaviors by type and time
        behavior_groups = {}
        for behavior in behaviors:
            behavior_type = behavior.get("type", "unknown")
            timestamp = behavior.get("timestamp")
            
            if behavior_type not in behavior_groups:
                behavior_groups[behavior_type] = []
            
            behavior_groups[behavior_type].append({
                "timestamp": timestamp,
                "value": behavior.get("value", 1),
                "context": behavior.get("context", {})
            })
        
        # Analyze trends for each behavior type
        for behavior_type, behavior_list in behavior_groups.items():
            if len(behavior_list) >= 2:
                # Sort by timestamp
                sorted_behaviors = sorted(behavior_list, key=lambda x: x.get("timestamp", ""))
                
                # Calculate trend (simplified linear trend)
                values = [b.get("value", 1) for b in sorted_behaviors]
                if len(values) > 1:
                    trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
                    
                    tracking["behavior_trends"][behavior_type] = {
                        "trend": trend,
                        "start_value": values[0],
                        "end_value": values[-1],
                        "change_magnitude": abs(values[-1] - values[0]),
                        "data_points": len(values)
                    }
                    
                    # Identify significant changes
                    if abs(values[-1] - values[0]) > 0.5:  # Threshold for significant change
                        tracking["notable_changes"].append({
                            "behavior": behavior_type,
                            "change_type": trend,
                            "magnitude": abs(values[-1] - values[0])
                        })
        
        # Calculate stability metrics
        for behavior_type, trend_data in tracking["behavior_trends"].items():
            variance = 0
            if behavior_type in behavior_groups:
                values = [b.get("value", 1) for b in behavior_groups[behavior_type]]
                if len(values) > 1:
                    mean_value = sum(values) / len(values)
                    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            
            tracking["stability_metrics"][behavior_type] = {
                "variance": variance,
                "stability": "high" if variance < 0.1 else "medium" if variance < 0.5 else "low"
            }
        
        return self._create_success_result(
            data=tracking,
            metadata={
                "analysis_type": "behavioral_tracking",
                "behaviors_analyzed": len(behaviors),
                "behavior_types": len(behavior_groups),
                "time_window": time_window,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _general_reflection(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Perform general reflection on provided data."""
        reflection_data = input_data.get("data", {})
        focus_areas = input_data.get("focus_areas", ["overall"])
        
        reflection = {
            "summary": "",
            "key_insights": [],
            "observations": [],
            "questions_for_consideration": [],
            "next_steps": []
        }
        
        # Generate summary based on available data
        data_types = list(reflection_data.keys())
        reflection["summary"] = f"Reflection on {len(data_types)} data categories: {', '.join(data_types)}"
        
        # Generate insights based on focus areas
        for focus_area in focus_areas:
            if focus_area == "overall":
                reflection["key_insights"].append("Regular reflection helps maintain awareness of patterns and progress")
            elif focus_area == "decision_making":
                reflection["key_insights"].append("Decision quality improves with systematic analysis of outcomes")
            elif focus_area == "learning":
                reflection["key_insights"].append("Continuous learning requires both success and failure analysis")
            elif focus_area == "efficiency":
                reflection["key_insights"].append("Efficiency gains come from identifying and eliminating recurring issues")
            elif focus_area == "performance":
                reflection["key_insights"].append("Performance analysis reveals opportunities for system optimization")
            elif focus_area == "system_analysis":
                reflection["key_insights"].append("System analysis helps identify bottlenecks and improvement areas")
        
        # Always add at least one general insight if none were added
        if not reflection["key_insights"]:
            reflection["key_insights"].append("Reflection provides valuable insights for continuous improvement")
        
        # Generate observations
        reflection["observations"] = [
            "Reflection is most valuable when done consistently over time",
            "Patterns become clearer with larger datasets",
            "Both quantitative and qualitative data provide valuable insights"
        ]
        
        # Generate questions for consideration
        reflection["questions_for_consideration"] = [
            "What patterns am I not seeing that might be important?",
            "How can I better measure progress toward my goals?",
            "What assumptions am I making that should be questioned?",
            "Where am I spending time that doesn't align with priorities?"
        ]
        
        # Suggest next steps
        reflection["next_steps"] = [
            "Continue regular reflection sessions",
            "Collect more specific data for deeper analysis",
            "Set measurable goals based on insights",
            "Implement changes based on identified improvements"
        ]
        
        return self._create_success_result(
            data=reflection,
            metadata={
                "analysis_type": "general_reflection",
                "focus_areas": focus_areas,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> str:
        """Create a brief summary of input data."""
        data_keys = list(input_data.keys())
        return f"Input with {len(data_keys)} fields: {', '.join(data_keys[:3])}{'...' if len(data_keys) > 3 else ''}"
    
    def _summarize_result(self, result: SubagentResult) -> str:
        """Create a brief summary of result data."""
        if result.success and result.data:
            if isinstance(result.data, dict):
                return f"Analysis with {len(result.data)} components"
            else:
                return "Successful analysis"
        else:
            return f"Failed: {result.error}"
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for input data."""
        return {
            "type": "object",
            "properties": {
                "reflection_type": {
                    "type": "string",
                    "enum": [
                        "decision_analysis",
                        "pattern_analysis", 
                        "performance_assessment",
                        "improvement_identification",
                        "behavioral_tracking",
                        "general"
                    ],
                    "description": "Type of reflection to perform"
                },
                "data": {
                    "type": "object",
                    "description": "Data to reflect upon"
                },
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "outcome": {"type": "string"},
                            "factors": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "description": "List of decisions for analysis"
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "timestamp": {"type": "string"}
                        }
                    },
                    "description": "List of events for pattern analysis"
                },
                "metrics": {
                    "type": "object",
                    "description": "Performance metrics for assessment"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas to focus reflection on"
                }
            },
            "required": []
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for output data."""
        return {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis performed"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of reflection"
                },
                "key_insights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key insights from reflection"
                },
                "recommendations": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Recommendations for improvement"
                },
                "patterns": {
                    "type": "object",
                    "description": "Identified patterns"
                },
                "trends": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Observed trends"
                }
            }
        }
    
    def get_reflection_history(self) -> List[Dict[str, Any]]:
        """Get the history of reflections performed."""
        return self._reflection_history.copy()