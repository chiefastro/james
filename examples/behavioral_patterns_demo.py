"""
Demonstration of the behavioral pattern analysis system.

This script shows how the behavioral pattern analyzer works to identify patterns
in agent behavior, analyze them, and generate recommendations for improvement.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import json
from uuid import uuid4

from backend.evolution.behavioral_patterns import (
    BehavioralPatternAnalyzer, PatternType, PatternConfidence
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BehavioralPatternsDemo:
    """Demonstration of the behavioral pattern analysis system."""
    
    def __init__(self):
        """Initialize the behavioral patterns demo."""
        self.pattern_analyzer = BehavioralPatternAnalyzer(
            min_pattern_frequency=3,  # Lower threshold for demo purposes
            confidence_threshold=0.6
        )
        
        # Demo configuration
        self.simulation_days = 5
        self.events_per_day = 30
        self.agent_id = "pattern_demo_agent"
    
    def generate_behavior_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic behavior data for demonstration."""
        logger.info(f"Generating {self.simulation_days} days of behavior data...")
        
        behavior_data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=self.simulation_days)
        
        # Define some behavior patterns to simulate
        patterns = [
            {
                "name": "Fast Classification Success",
                "conditions": {"operation_type": "classification", "input_type": "text"},
                "success_rate": 0.9,
                "avg_duration_ms": 500,
                "frequency": 0.3  # 30% of events
            },
            {
                "name": "Slow Processing Failure",
                "conditions": {"operation_type": "processing", "input_type": "image"},
                "success_rate": 0.2,
                "avg_duration_ms": 3000,
                "frequency": 0.2  # 20% of events
            },
            {
                "name": "Moderate Decision Making",
                "conditions": {"operation_type": "decision", "context_type": "complex"},
                "success_rate": 0.6,
                "avg_duration_ms": 1500,
                "frequency": 0.25  # 25% of events
            },
            # Random events will make up the remaining 25%
        ]
        
        # Generate events for each day
        for day in range(self.simulation_days):
            day_start = base_time + timedelta(days=day)
            
            for event_num in range(self.events_per_day):
                event_time = day_start + timedelta(minutes=event_num * 15)
                
                # Determine which pattern to use (if any)
                pattern_choice = random.random()
                cumulative_freq = 0
                selected_pattern = None
                
                for pattern in patterns:
                    cumulative_freq += pattern["frequency"]
                    if pattern_choice <= cumulative_freq:
                        selected_pattern = pattern
                        break
                
                # Generate event based on selected pattern or random
                if selected_pattern:
                    event = self._generate_patterned_event(selected_pattern, event_time)
                else:
                    event = self._generate_random_event(event_time)
                
                behavior_data.append(event)
        
        logger.info(f"Generated {len(behavior_data)} behavior records")
        return behavior_data
    
    def _generate_patterned_event(
        self,
        pattern: Dict[str, Any],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Generate an event following a specific pattern."""
        # Base event with pattern conditions
        event = {
            "id": str(uuid4()),
            "timestamp": timestamp.isoformat(),
            **pattern["conditions"]
        }
        
        # Determine success based on pattern success rate
        success = random.random() < pattern["success_rate"]
        event["success"] = success
        
        # Add duration with some variability
        base_duration = pattern["avg_duration_ms"]
        variability = base_duration * 0.2  # 20% variability
        duration = base_duration + random.uniform(-variability, variability)
        event["duration_ms"] = max(10, duration)  # Ensure positive duration
        
        # Add pattern-specific details
        if "classification" in pattern["name"].lower():
            event["decision"] = {
                "type": "classification",
                "confidence": 0.7 + random.uniform(-0.1, 0.2),
                "options_considered": random.randint(2, 5)
            }
        elif "processing" in pattern["name"].lower():
            event["processing"] = {
                "complexity": "high",
                "resource_usage": random.uniform(0.6, 0.9),
                "batch_size": random.randint(1, 10)
            }
        elif "decision" in pattern["name"].lower():
            event["decision"] = {
                "type": "multi_criteria",
                "confidence": 0.6 + random.uniform(-0.2, 0.2),
                "criteria_count": random.randint(3, 7)
            }
        
        # Add error information for failures
        if not success:
            error_types = ["timeout", "validation", "resource_limit", "processing"]
            event["error_type"] = random.choice(error_types)
            
            if event["error_type"] == "timeout":
                event["timeout_seconds"] = random.randint(5, 30)
        
        return event
    
    def _generate_random_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a random event not following any specific pattern."""
        operation_types = ["query", "transform", "generate", "validate", "analyze"]
        input_types = ["text", "number", "mixed", "structured", "unstructured"]
        context_types = ["simple", "moderate", "complex"]
        
        event = {
            "id": str(uuid4()),
            "timestamp": timestamp.isoformat(),
            "operation_type": random.choice(operation_types),
            "input_type": random.choice(input_types),
            "context_type": random.choice(context_types),
            "success": random.random() > 0.4,  # 60% success rate
            "duration_ms": random.uniform(100, 5000)
        }
        
        # Add random decision data
        if random.random() > 0.5:
            event["decision"] = {
                "type": random.choice(["binary", "ranking", "selection", "threshold"]),
                "confidence": random.uniform(0.3, 0.9),
                "options_considered": random.randint(2, 10)
            }
        
        # Add error information for failures
        if not event["success"]:
            error_types = ["input_error", "logic_error", "external_failure", "unknown"]
            event["error_type"] = random.choice(error_types)
        
        # Add interaction data
        if random.random() > 0.7:
            event["interaction"] = {
                "type": random.choice(["user", "system", "agent"]),
                "direction": random.choice(["inbound", "outbound"]),
                "complexity": random.choice(["simple", "complex"])
            }
        
        return event
    
    async def analyze_patterns(self, behavior_data: List[Dict[str, Any]]):
        """Analyze patterns in the behavior data."""
        logger.info("Analyzing behavioral patterns...")
        
        # Perform pattern analysis
        analysis_result = await self.pattern_analyzer.analyze_behavior_data(
            behavior_data,
            analysis_window_hours=self.simulation_days * 24
        )
        
        # Display discovered patterns
        logger.info(f"Discovered {len(analysis_result.patterns_discovered)} new patterns:")
        for pattern in analysis_result.patterns_discovered:
            logger.info(f"  - {pattern.pattern_type.value}: {pattern.name}")
            logger.info(f"    Confidence: {pattern.confidence.value}, Success Rate: {pattern.success_rate:.2f}")
            logger.info(f"    Description: {pattern.description}")
            logger.info(f"    Frequency: {pattern.frequency} occurrences")
            logger.info(f"    Conditions: {pattern.conditions}")
        
        # Display updated patterns
        if analysis_result.patterns_updated:
            logger.info(f"Updated {len(analysis_result.patterns_updated)} existing patterns")
        
        # Display deprecated patterns
        if analysis_result.patterns_deprecated:
            logger.info(f"Deprecated {len(analysis_result.patterns_deprecated)} patterns")
        
        # Display recommendations
        logger.info("Pattern analysis recommendations:")
        for recommendation in analysis_result.recommendations:
            logger.info(f"  - {recommendation}")
        
        # Display analysis summary
        logger.info("Analysis Summary:")
        for key, value in analysis_result.analysis_summary.items():
            if isinstance(value, dict):
                logger.info(f"  - {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"      {subkey}: {subvalue}")
            else:
                logger.info(f"  - {key}: {value}")
        
        return analysis_result
    
    async def demonstrate_pattern_extraction(self, behavior_data: List[Dict[str, Any]]):
        """Demonstrate pattern extraction capabilities."""
        logger.info("Demonstrating pattern extraction...")
        
        # Extract success patterns
        success_data = [record for record in behavior_data if record.get("success", False)]
        
        # Group by operation type
        operation_groups = {}
        for record in success_data:
            op_type = record.get("operation_type", "unknown")
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(record)
        
        # Analyze each operation type
        for op_type, records in operation_groups.items():
            if len(records) >= 5:  # Minimum sample size
                logger.info(f"Analyzing success patterns for operation type: {op_type}")
                
                # Calculate success metrics
                durations = [r.get("duration_ms", 0) for r in records]
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                confidence_values = [
                    r.get("decision", {}).get("confidence", 0) 
                    for r in records if "decision" in r
                ]
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                
                logger.info(f"  - Records analyzed: {len(records)}")
                logger.info(f"  - Average duration: {avg_duration:.2f}ms")
                logger.info(f"  - Average confidence: {avg_confidence:.2f}")
                
                # Extract common conditions
                conditions = self._extract_common_conditions(records)
                if conditions:
                    logger.info(f"  - Common conditions:")
                    for key, value in conditions.items():
                        logger.info(f"      {key}: {value}")
    
    def _extract_common_conditions(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract conditions that appear in at least 70% of records."""
        if not records:
            return {}
        
        # Count occurrences of each condition
        condition_counts = defaultdict(lambda: defaultdict(int))
        for record in records:
            for key, value in record.items():
                if key not in ["id", "timestamp", "success", "duration_ms", "decision", "error_type"]:
                    condition_counts[key][value] += 1
        
        # Find conditions that appear in at least 70% of records
        common_conditions = {}
        threshold = len(records) * 0.7
        
        for key, value_counts in condition_counts.items():
            most_common_value = max(value_counts.items(), key=lambda x: x[1])
            if most_common_value[1] >= threshold:
                common_conditions[key] = most_common_value[0]
        
        return common_conditions
    
    async def demonstrate_pattern_evolution(self):
        """Demonstrate how patterns evolve over time."""
        logger.info("Demonstrating pattern evolution...")
        
        # Generate two sets of behavior data with different characteristics
        logger.info("Generating early behavior data...")
        early_data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        
        # Early behavior - more errors, slower processing
        for i in range(50):
            event_time = base_time + timedelta(minutes=i * 30)
            event = {
                "id": str(uuid4()),
                "timestamp": event_time.isoformat(),
                "operation_type": "classification",
                "input_type": "text",
                "success": random.random() > 0.5,  # 50% success rate
                "duration_ms": random.uniform(1000, 3000)  # Slower
            }
            
            if not event["success"]:
                event["error_type"] = random.choice(["timeout", "validation"])
            
            early_data.append(event)
        
        # Analyze early patterns
        early_result = await self.pattern_analyzer.analyze_behavior_data(
            early_data,
            analysis_window_hours=240  # 10 days
        )
        
        logger.info("Early behavior patterns:")
        for pattern in early_result.patterns_discovered:
            logger.info(f"  - {pattern.name}: Success rate {pattern.success_rate:.2f}")
        
        # Generate later behavior data - improved performance
        logger.info("Generating later behavior data...")
        later_data = []
        base_time = datetime.now(timezone.utc) - timedelta(days=2)
        
        # Later behavior - fewer errors, faster processing
        for i in range(50):
            event_time = base_time + timedelta(minutes=i * 30)
            event = {
                "id": str(uuid4()),
                "timestamp": event_time.isoformat(),
                "operation_type": "classification",
                "input_type": "text",
                "success": random.random() > 0.2,  # 80% success rate
                "duration_ms": random.uniform(500, 1500)  # Faster
            }
            
            if not event["success"]:
                event["error_type"] = random.choice(["validation", "input_error"])
            
            later_data.append(event)
        
        # Analyze later patterns
        later_result = await self.pattern_analyzer.analyze_behavior_data(
            later_data,
            analysis_window_hours=48  # 2 days
        )
        
        logger.info("Later behavior patterns:")
        for pattern in later_result.patterns_discovered:
            logger.info(f"  - {pattern.name}: Success rate {pattern.success_rate:.2f}")
        
        # Compare patterns
        logger.info("Pattern evolution:")
        logger.info("  - Early success rate: " + 
                   f"{next((p.success_rate for p in early_result.patterns_discovered if p.pattern_type == PatternType.SUCCESS_PATTERN), 0):.2f}")
        logger.info("  - Later success rate: " + 
                   f"{next((p.success_rate for p in later_result.patterns_discovered if p.pattern_type == PatternType.SUCCESS_PATTERN), 0):.2f}")
        
        early_duration = next((p.outcomes.get("avg_duration_ms", 0) for p in early_result.patterns_discovered), 0)
        later_duration = next((p.outcomes.get("avg_duration_ms", 0) for p in later_result.patterns_discovered), 0)
        
        if early_duration and later_duration:
            improvement = (early_duration - later_duration) / early_duration * 100
            logger.info(f"  - Duration improvement: {improvement:.1f}%")
    
    async def demonstrate_pattern_export_import(self, behavior_data: List[Dict[str, Any]]):
        """Demonstrate pattern export and import capabilities."""
        logger.info("Demonstrating pattern export and import...")
        
        # First analyze some patterns
        await self.pattern_analyzer.analyze_behavior_data(behavior_data)
        
        # Export patterns
        exported_data = self.pattern_analyzer.export_patterns()
        logger.info(f"Exported {exported_data['total_patterns']} patterns")
        
        # Create a new analyzer
        new_analyzer = BehavioralPatternAnalyzer()
        
        # Import patterns
        imported_count = new_analyzer.import_patterns(exported_data)
        logger.info(f"Imported {imported_count} patterns into new analyzer")
        
        # Verify patterns were imported correctly
        logger.info("Verifying imported patterns:")
        for pattern_id, pattern in new_analyzer.discovered_patterns.items():
            logger.info(f"  - {pattern.name}: {pattern.pattern_type.value}")
    
    async def run_demo(self):
        """Run the complete behavioral patterns demonstration."""
        logger.info("Starting Behavioral Patterns Demonstration")
        logger.info("=" * 60)
        
        try:
            # Generate behavior data
            behavior_data = self.generate_behavior_data()
            
            # Analyze patterns
            analysis_result = await self.analyze_patterns(behavior_data)
            
            # Demonstrate pattern extraction
            await self.demonstrate_pattern_extraction(behavior_data)
            
            # Demonstrate pattern evolution
            await self.demonstrate_pattern_evolution()
            
            # Demonstrate pattern export/import
            await self.demonstrate_pattern_export_import(behavior_data)
            
            # Get high confidence patterns
            high_confidence_patterns = self.pattern_analyzer.get_high_confidence_patterns()
            logger.info(f"Found {len(high_confidence_patterns)} high confidence patterns")
            
            logger.info("=" * 60)
            logger.info("Behavioral Patterns Demonstration Complete!")
            
        except Exception as e:
            logger.error(f"Error during demonstration: {e}", exc_info=True)
            raise


async def main():
    """Main function to run the behavioral patterns demo."""
    demo = BehavioralPatternsDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())