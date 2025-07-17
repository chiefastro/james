"""
Behavioral pattern analysis and learning for agent evolution.

This module analyzes agent behavior patterns to identify successful strategies,
common failure modes, and opportunities for improvement through learning.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics
import json

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of behavioral patterns that can be identified."""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    EFFICIENCY_PATTERN = "efficiency_pattern"
    DECISION_PATTERN = "decision_pattern"
    INTERACTION_PATTERN = "interaction_pattern"
    ADAPTATION_PATTERN = "adaptation_pattern"


class PatternConfidence(Enum):
    """Confidence levels for pattern identification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class BehavioralPattern:
    """Represents an identified behavioral pattern."""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    confidence: PatternConfidence
    frequency: int
    success_rate: float
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    context: Dict[str, Any]
    discovered_at: datetime
    last_observed: datetime
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence.value,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "conditions": self.conditions,
            "outcomes": self.outcomes,
            "context": self.context,
            "discovered_at": self.discovered_at.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "examples": self.examples,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralPattern':
        """Create pattern from dictionary."""
        return cls(
            id=data["id"],
            pattern_type=PatternType(data["pattern_type"]),
            name=data["name"],
            description=data["description"],
            confidence=PatternConfidence(data["confidence"]),
            frequency=data["frequency"],
            success_rate=data["success_rate"],
            conditions=data["conditions"],
            outcomes=data["outcomes"],
            context=data["context"],
            discovered_at=datetime.fromisoformat(data["discovered_at"]),
            last_observed=datetime.fromisoformat(data["last_observed"]),
            examples=data.get("examples", []),
            metrics=data.get("metrics", {})
        )


@dataclass
class PatternAnalysisResult:
    """Result of behavioral pattern analysis."""
    patterns_discovered: List[BehavioralPattern]
    patterns_updated: List[BehavioralPattern]
    patterns_deprecated: List[str]
    analysis_summary: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class BehavioralPatternAnalyzer:
    """
    Analyzes agent behavior to identify patterns and learning opportunities.
    
    This system examines agent interactions, decisions, and outcomes to discover
    recurring behavioral patterns that can inform evolution and adaptation.
    """
    
    def __init__(self, min_pattern_frequency: int = 5, confidence_threshold: float = 0.7):
        """
        Initialize the behavioral pattern analyzer.
        
        Args:
            min_pattern_frequency: Minimum frequency for pattern recognition
            confidence_threshold: Minimum confidence for pattern acceptance
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.discovered_patterns: Dict[str, BehavioralPattern] = {}
        self.pattern_history: List[BehavioralPattern] = []
        
        # Analysis state
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_ttl_minutes = 30
        
        logger.info("BehavioralPatternAnalyzer initialized")
    
    async def analyze_behavior_data(
        self,
        behavior_data: List[Dict[str, Any]],
        analysis_window_hours: int = 24
    ) -> PatternAnalysisResult:
        """
        Analyze behavior data to identify patterns.
        
        Args:
            behavior_data: List of behavior records to analyze
            analysis_window_hours: Time window for analysis
            
        Returns:
            PatternAnalysisResult with discovered patterns and insights
        """
        logger.info(f"Analyzing {len(behavior_data)} behavior records")
        
        # Filter data to analysis window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=analysis_window_hours)
        recent_data = [
            record for record in behavior_data
            if datetime.fromisoformat(record.get("timestamp", datetime.now().isoformat())) >= cutoff_time
        ]
        
        if not recent_data:
            return PatternAnalysisResult(
                patterns_discovered=[],
                patterns_updated=[],
                patterns_deprecated=[],
                analysis_summary={"error": "No recent behavior data"},
                recommendations=[],
                confidence_score=0.0
            )
        
        # Analyze different pattern types
        success_patterns = await self._analyze_success_patterns(recent_data)
        failure_patterns = await self._analyze_failure_patterns(recent_data)
        efficiency_patterns = await self._analyze_efficiency_patterns(recent_data)
        decision_patterns = await self._analyze_decision_patterns(recent_data)
        interaction_patterns = await self._analyze_interaction_patterns(recent_data)
        
        # Combine all discovered patterns
        all_patterns = (
            success_patterns + failure_patterns + efficiency_patterns +
            decision_patterns + interaction_patterns
        )
        
        # Update existing patterns and identify new ones
        patterns_discovered = []
        patterns_updated = []
        patterns_deprecated = []
        
        for pattern in all_patterns:
            if pattern.id in self.discovered_patterns:
                # Update existing pattern
                existing = self.discovered_patterns[pattern.id]
                existing.frequency += pattern.frequency
                existing.last_observed = pattern.last_observed
                existing.examples.extend(pattern.examples)
                
                # Recalculate metrics
                existing.success_rate = self._calculate_success_rate(existing.examples)
                existing.confidence = self._calculate_confidence(existing)
                
                patterns_updated.append(existing)
            else:
                # New pattern discovered
                if pattern.frequency >= self.min_pattern_frequency:
                    self.discovered_patterns[pattern.id] = pattern
                    patterns_discovered.append(pattern)
        
        # Check for deprecated patterns (not seen recently)
        deprecation_threshold = datetime.now(timezone.utc) - timedelta(hours=analysis_window_hours * 2)
        for pattern_id, pattern in list(self.discovered_patterns.items()):
            if pattern.last_observed < deprecation_threshold:
                patterns_deprecated.append(pattern_id)
                del self.discovered_patterns[pattern_id]
        
        # Generate analysis summary and recommendations
        analysis_summary = self._generate_analysis_summary(recent_data, all_patterns)
        recommendations = self._generate_recommendations(all_patterns)
        confidence_score = self._calculate_overall_confidence(all_patterns)
        
        result = PatternAnalysisResult(
            patterns_discovered=patterns_discovered,
            patterns_updated=patterns_updated,
            patterns_deprecated=patterns_deprecated,
            analysis_summary=analysis_summary,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
        
        logger.info(f"Pattern analysis complete: {len(patterns_discovered)} new, {len(patterns_updated)} updated")
        return result
    
    async def _analyze_success_patterns(self, data: List[Dict[str, Any]]) -> List[BehavioralPattern]:
        """Analyze patterns associated with successful outcomes."""
        success_data = [record for record in data if record.get("success", False)]
        
        if len(success_data) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Group by similar conditions
        condition_groups = self._group_by_conditions(success_data)
        
        for conditions, records in condition_groups.items():
            if len(records) >= self.min_pattern_frequency:
                pattern = self._create_pattern(
                    pattern_type=PatternType.SUCCESS_PATTERN,
                    name=f"Success Pattern: {conditions[:50]}...",
                    description=f"Pattern leading to successful outcomes with conditions: {conditions}",
                    records=records,
                    conditions=json.loads(conditions) if conditions else {}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_failure_patterns(self, data: List[Dict[str, Any]]) -> List[BehavioralPattern]:
        """Analyze patterns associated with failures."""
        failure_data = [record for record in data if not record.get("success", True)]
        
        if len(failure_data) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Group by error types and conditions
        error_groups = defaultdict(list)
        for record in failure_data:
            error_type = record.get("error_type", "unknown")
            error_groups[error_type].append(record)
        
        for error_type, records in error_groups.items():
            if len(records) >= self.min_pattern_frequency:
                pattern = self._create_pattern(
                    pattern_type=PatternType.FAILURE_PATTERN,
                    name=f"Failure Pattern: {error_type}",
                    description=f"Pattern leading to {error_type} failures",
                    records=records,
                    conditions={"error_type": error_type}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_efficiency_patterns(self, data: List[Dict[str, Any]]) -> List[BehavioralPattern]:
        """Analyze patterns related to efficiency and performance."""
        # Focus on records with timing information
        timed_data = [record for record in data if "duration_ms" in record]
        
        if len(timed_data) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Identify fast vs slow operations
        durations = [record["duration_ms"] for record in timed_data]
        median_duration = statistics.median(durations)
        
        fast_operations = [record for record in timed_data if record["duration_ms"] < median_duration * 0.5]
        slow_operations = [record for record in timed_data if record["duration_ms"] > median_duration * 2.0]
        
        if len(fast_operations) >= self.min_pattern_frequency:
            pattern = self._create_pattern(
                pattern_type=PatternType.EFFICIENCY_PATTERN,
                name="High Efficiency Pattern",
                description="Pattern associated with fast operation completion",
                records=fast_operations,
                conditions={"efficiency": "high"}
            )
            patterns.append(pattern)
        
        if len(slow_operations) >= self.min_pattern_frequency:
            pattern = self._create_pattern(
                pattern_type=PatternType.EFFICIENCY_PATTERN,
                name="Low Efficiency Pattern",
                description="Pattern associated with slow operation completion",
                records=slow_operations,
                conditions={"efficiency": "low"}
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_decision_patterns(self, data: List[Dict[str, Any]]) -> List[BehavioralPattern]:
        """Analyze decision-making patterns."""
        decision_data = [record for record in data if "decision" in record]
        
        if len(decision_data) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Group by decision types
        decision_groups = defaultdict(list)
        for record in decision_data:
            decision_type = record["decision"].get("type", "unknown")
            decision_groups[decision_type].append(record)
        
        for decision_type, records in decision_groups.items():
            if len(records) >= self.min_pattern_frequency:
                # Calculate decision quality metrics
                confidences = [r["decision"].get("confidence", 0.5) for r in records]
                avg_confidence = statistics.mean(confidences)
                
                pattern = self._create_pattern(
                    pattern_type=PatternType.DECISION_PATTERN,
                    name=f"Decision Pattern: {decision_type}",
                    description=f"Pattern for {decision_type} decisions",
                    records=records,
                    conditions={"decision_type": decision_type},
                    metrics={"avg_confidence": avg_confidence}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _analyze_interaction_patterns(self, data: List[Dict[str, Any]]) -> List[BehavioralPattern]:
        """Analyze interaction patterns between agents."""
        interaction_data = [record for record in data if "interaction" in record]
        
        if len(interaction_data) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Group by interaction types
        interaction_groups = defaultdict(list)
        for record in interaction_data:
            interaction_type = record["interaction"].get("type", "unknown")
            interaction_groups[interaction_type].append(record)
        
        for interaction_type, records in interaction_groups.items():
            if len(records) >= self.min_pattern_frequency:
                pattern = self._create_pattern(
                    pattern_type=PatternType.INTERACTION_PATTERN,
                    name=f"Interaction Pattern: {interaction_type}",
                    description=f"Pattern for {interaction_type} interactions",
                    records=records,
                    conditions={"interaction_type": interaction_type}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _group_by_conditions(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group records by similar conditions."""
        groups = defaultdict(list)
        
        for record in data:
            # Extract key conditions for grouping
            conditions = {}
            for key in ["operation_type", "input_type", "context_type", "agent_type"]:
                if key in record:
                    conditions[key] = record[key]
            
            # Convert to string for grouping
            condition_key = json.dumps(conditions, sort_keys=True)
            groups[condition_key].append(record)
        
        return dict(groups)
    
    def _create_pattern(
        self,
        pattern_type: PatternType,
        name: str,
        description: str,
        records: List[Dict[str, Any]],
        conditions: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> BehavioralPattern:
        """Create a behavioral pattern from records."""
        from uuid import uuid4
        
        # Calculate pattern metrics
        success_rate = self._calculate_success_rate(records)
        confidence = self._calculate_pattern_confidence(records, success_rate)
        
        # Extract outcomes
        outcomes = self._extract_outcomes(records)
        
        # Create context information
        context = {
            "sample_size": len(records),
            "time_span_hours": self._calculate_time_span(records),
            "data_quality": self._assess_data_quality(records)
        }
        
        now = datetime.now(timezone.utc)
        
        return BehavioralPattern(
            id=str(uuid4()),
            pattern_type=pattern_type,
            name=name,
            description=description,
            confidence=confidence,
            frequency=len(records),
            success_rate=success_rate,
            conditions=conditions,
            outcomes=outcomes,
            context=context,
            discovered_at=now,
            last_observed=now,
            examples=records[:5],  # Store first 5 examples
            metrics=metrics or {}
        )
    
    def _calculate_success_rate(self, records: List[Dict[str, Any]]) -> float:
        """Calculate success rate for a set of records."""
        if not records:
            return 0.0
        
        successes = sum(1 for record in records if record.get("success", False))
        return successes / len(records)
    
    def _calculate_pattern_confidence(self, records: List[Dict[str, Any]], success_rate: float) -> PatternConfidence:
        """Calculate confidence level for a pattern."""
        sample_size = len(records)
        data_quality = self._assess_data_quality(records)
        
        # Base confidence on sample size, success rate consistency, and data quality
        confidence_score = 0.0
        
        # Sample size factor
        if sample_size >= 20:
            confidence_score += 0.4
        elif sample_size >= 10:
            confidence_score += 0.3
        elif sample_size >= 5:
            confidence_score += 0.2
        
        # Success rate consistency factor
        if success_rate >= 0.9 or success_rate <= 0.1:
            confidence_score += 0.3  # Very consistent
        elif success_rate >= 0.8 or success_rate <= 0.2:
            confidence_score += 0.2  # Moderately consistent
        else:
            confidence_score += 0.1  # Less consistent
        
        # Data quality factor
        confidence_score += data_quality * 0.3
        
        if confidence_score >= 0.8:
            return PatternConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return PatternConfidence.HIGH
        elif confidence_score >= 0.4:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW
    
    def _calculate_confidence(self, pattern: BehavioralPattern) -> PatternConfidence:
        """Recalculate confidence for an existing pattern."""
        return self._calculate_pattern_confidence(pattern.examples, pattern.success_rate)
    
    def _extract_outcomes(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common outcomes from records."""
        outcomes = {}
        
        # Success/failure distribution
        successes = sum(1 for record in records if record.get("success", False))
        outcomes["success_rate"] = successes / len(records) if records else 0.0
        
        # Common error types
        errors = [record.get("error_type") for record in records if record.get("error_type")]
        if errors:
            error_counts = Counter(errors)
            outcomes["common_errors"] = dict(error_counts.most_common(3))
        
        # Performance metrics
        durations = [record.get("duration_ms") for record in records if record.get("duration_ms")]
        if durations:
            outcomes["avg_duration_ms"] = statistics.mean(durations)
            outcomes["median_duration_ms"] = statistics.median(durations)
        
        return outcomes
    
    def _calculate_time_span(self, records: List[Dict[str, Any]]) -> float:
        """Calculate time span covered by records in hours."""
        timestamps = []
        for record in records:
            if "timestamp" in record:
                try:
                    timestamps.append(datetime.fromisoformat(record["timestamp"]))
                except ValueError:
                    continue
        
        if len(timestamps) < 2:
            return 0.0
        
        time_span = max(timestamps) - min(timestamps)
        return time_span.total_seconds() / 3600
    
    def _assess_data_quality(self, records: List[Dict[str, Any]]) -> float:
        """Assess the quality of data in records (0.0 to 1.0)."""
        if not records:
            return 0.0
        
        quality_score = 0.0
        
        # Check for required fields
        required_fields = ["timestamp", "success"]
        field_completeness = 0.0
        for field in required_fields:
            field_present = sum(1 for record in records if field in record and record[field] is not None)
            field_completeness += field_present / len(records)
        quality_score += (field_completeness / len(required_fields)) * 0.5
        
        # Check for additional useful fields
        useful_fields = ["duration_ms", "operation_type", "error_type", "decision"]
        useful_completeness = 0.0
        for field in useful_fields:
            field_present = sum(1 for record in records if field in record and record[field] is not None)
            useful_completeness += field_present / len(records)
        quality_score += (useful_completeness / len(useful_fields)) * 0.3
        
        # Check for data consistency
        consistency_score = self._check_data_consistency(records)
        quality_score += consistency_score * 0.2
        
        return min(quality_score, 1.0)
    
    def _check_data_consistency(self, records: List[Dict[str, Any]]) -> float:
        """Check consistency of data in records."""
        if not records:
            return 0.0
        
        consistency_score = 1.0
        
        # Check timestamp ordering
        timestamps = []
        for record in records:
            if "timestamp" in record:
                try:
                    timestamps.append(datetime.fromisoformat(record["timestamp"]))
                except ValueError:
                    consistency_score -= 0.1
        
        # Check for reasonable duration values
        durations = [record.get("duration_ms") for record in records if record.get("duration_ms")]
        if durations:
            # Flag unreasonable durations (negative or extremely large)
            unreasonable = sum(1 for d in durations if d < 0 or d > 300000)  # 5 minutes max
            if unreasonable > 0:
                consistency_score -= (unreasonable / len(durations)) * 0.2
        
        return max(consistency_score, 0.0)
    
    def _generate_analysis_summary(
        self,
        data: List[Dict[str, Any]],
        patterns: List[BehavioralPattern]
    ) -> Dict[str, Any]:
        """Generate summary of pattern analysis."""
        return {
            "total_records_analyzed": len(data),
            "patterns_by_type": {
                pattern_type.value: len([p for p in patterns if p.pattern_type == pattern_type])
                for pattern_type in PatternType
            },
            "avg_pattern_confidence": statistics.mean([
                self._confidence_to_score(p.confidence) for p in patterns
            ]) if patterns else 0.0,
            "success_rate_distribution": self._calculate_success_distribution(data),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_recommendations(self, patterns: List[BehavioralPattern]) -> List[str]:
        """Generate recommendations based on discovered patterns."""
        recommendations = []
        
        # Analyze success patterns
        success_patterns = [p for p in patterns if p.pattern_type == PatternType.SUCCESS_PATTERN]
        if success_patterns:
            high_confidence_success = [p for p in success_patterns if p.confidence in [PatternConfidence.HIGH, PatternConfidence.VERY_HIGH]]
            if high_confidence_success:
                recommendations.append(
                    f"Reinforce {len(high_confidence_success)} high-confidence success patterns to improve overall performance"
                )
        
        # Analyze failure patterns
        failure_patterns = [p for p in patterns if p.pattern_type == PatternType.FAILURE_PATTERN]
        if failure_patterns:
            frequent_failures = [p for p in failure_patterns if p.frequency >= self.min_pattern_frequency * 2]
            if frequent_failures:
                recommendations.append(
                    f"Address {len(frequent_failures)} frequent failure patterns to reduce error rates"
                )
        
        # Analyze efficiency patterns
        efficiency_patterns = [p for p in patterns if p.pattern_type == PatternType.EFFICIENCY_PATTERN]
        low_efficiency = [p for p in efficiency_patterns if "low" in p.conditions.get("efficiency", "")]
        if low_efficiency:
            recommendations.append(
                "Investigate and optimize low-efficiency patterns to improve response times"
            )
        
        # General recommendations
        if len(patterns) < 3:
            recommendations.append("Collect more behavioral data to identify additional patterns")
        
        low_confidence_patterns = [p for p in patterns if p.confidence == PatternConfidence.LOW]
        if len(low_confidence_patterns) > len(patterns) * 0.5:
            recommendations.append("Improve data quality and collection to increase pattern confidence")
        
        return recommendations
    
    def _calculate_overall_confidence(self, patterns: List[BehavioralPattern]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not patterns:
            return 0.0
        
        confidence_scores = [self._confidence_to_score(p.confidence) for p in patterns]
        return statistics.mean(confidence_scores)
    
    def _confidence_to_score(self, confidence: PatternConfidence) -> float:
        """Convert confidence enum to numeric score."""
        mapping = {
            PatternConfidence.LOW: 0.25,
            PatternConfidence.MEDIUM: 0.5,
            PatternConfidence.HIGH: 0.75,
            PatternConfidence.VERY_HIGH: 1.0
        }
        return mapping.get(confidence, 0.0)
    
    def _calculate_success_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of success rates."""
        if not data:
            return {}
        
        successes = sum(1 for record in data if record.get("success", False))
        total = len(data)
        
        return {
            "overall_success_rate": successes / total,
            "total_operations": total,
            "successful_operations": successes,
            "failed_operations": total - successes
        }
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[BehavioralPattern]:
        """Get a specific pattern by ID."""
        return self.discovered_patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[BehavioralPattern]:
        """Get all patterns of a specific type."""
        return [p for p in self.discovered_patterns.values() if p.pattern_type == pattern_type]
    
    def get_high_confidence_patterns(self) -> List[BehavioralPattern]:
        """Get patterns with high or very high confidence."""
        return [
            p for p in self.discovered_patterns.values()
            if p.confidence in [PatternConfidence.HIGH, PatternConfidence.VERY_HIGH]
        ]
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export all discovered patterns for storage or analysis."""
        return {
            "patterns": [pattern.to_dict() for pattern in self.discovered_patterns.values()],
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_patterns": len(self.discovered_patterns),
            "pattern_types": {
                pattern_type.value: len(self.get_patterns_by_type(pattern_type))
                for pattern_type in PatternType
            }
        }
    
    def import_patterns(self, pattern_data: Dict[str, Any]) -> int:
        """Import patterns from exported data."""
        imported_count = 0
        
        for pattern_dict in pattern_data.get("patterns", []):
            try:
                pattern = BehavioralPattern.from_dict(pattern_dict)
                self.discovered_patterns[pattern.id] = pattern
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import pattern: {e}")
        
        logger.info(f"Imported {imported_count} behavioral patterns")
        return imported_count