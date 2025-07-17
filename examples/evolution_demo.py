"""
Demonstration of the agent evolution and learning system.

This script shows how the evolution system works, including learning from experience,
adapting behavior, distilling knowledge, and evolving agent capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import random

from backend.evolution.learning_engine import (
    LearningEngine, LearningEvent, LearningType, setup_learning_engine
)
from backend.evolution.evolution_manager import (
    EvolutionManager, EvolutionType, setup_evolution_manager
)
from backend.evolution.adaptation_strategies import (
    PerformanceBasedAdaptation, FeedbackBasedAdaptation, ExperienceBasedAdaptation
)
from backend.evolution.knowledge_distillation import (
    KnowledgeDistiller, setup_knowledge_distiller
)
from backend.evolution.behavioral_patterns import BehavioralPatternAnalyzer
from backend.memory.memory_manager import MemoryManager
from backend.observability.metrics_collector import get_metrics_collector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionDemo:
    """Demonstration of the evolution system capabilities."""
    
    def __init__(self):
        """Initialize the evolution demo."""
        self.learning_engine = None
        self.evolution_manager = None
        self.knowledge_distiller = None
        self.pattern_analyzer = None
        self.metrics_collector = get_metrics_collector()
        
        # Demo configuration
        self.agent_id = "demo_agent"
        self.simulation_days = 7
        self.events_per_day = 20
    
    async def setup_components(self):
        """Set up all evolution system components."""
        logger.info("Setting up evolution system components...")
        
        # Initialize components
        self.learning_engine = LearningEngine()
        self.evolution_manager = EvolutionManager(learning_engine=self.learning_engine)
        self.knowledge_distiller = KnowledgeDistiller()
        self.pattern_analyzer = BehavioralPatternAnalyzer()
        
        logger.info("Evolution system components initialized")
    
    async def simulate_agent_experience(self) -> List[LearningEvent]:
        """Simulate agent experiences over time."""
        logger.info(f"Simulating {self.simulation_days} days of agent experience...")
        
        events = []
        base_time = datetime.now(timezone.utc) - timedelta(days=self.simulation_days)
        
        # Simulate different types of experiences
        for day in range(self.simulation_days):
            day_start = base_time + timedelta(days=day)
            
            # Simulate events for this day
            for event_num in range(self.events_per_day):
                event_time = day_start + timedelta(minutes=event_num * 30)
                
                # Create different types of events
                event_type = random.choice([
                    "classification", "delegation", "problem_solving", 
                    "user_interaction", "error_handling"
                ])
                
                # Simulate context based on event type
                context = self._generate_event_context(event_type, day)
                
                # Simulate outcome with some variability
                outcome = self._generate_event_outcome(event_type, day, event_num)
                
                # Success rate improves over time (learning effect)
                base_success_rate = 0.6 + (day / self.simulation_days) * 0.3
                success = random.random() < base_success_rate
                
                # Confidence also improves over time
                confidence = 0.5 + (day / self.simulation_days) * 0.3 + random.uniform(-0.1, 0.1)
                confidence = max(0.1, min(0.9, confidence))
                
                # Record the learning event
                event = await self.learning_engine.record_learning_event(
                    event_type=event_type,
                    context=context,
                    outcome=outcome,
                    success=success,
                    confidence=confidence,
                    importance_score=random.uniform(0.3, 0.8)
                )
                
                events.append(event)
        
        logger.info(f"Simulated {len(events)} learning events")
        return events
    
    def _generate_event_context(self, event_type: str, day: int) -> Dict[str, Any]:
        """Generate context for an event."""
        contexts = {
            "classification": {
                "operation": "classify",
                "category": random.choice(["text", "image", "data"]),
                "complexity": random.choice(["low", "medium", "high"]),
                "domain": random.choice(["technical", "business", "general"])
            },
            "delegation": {
                "operation": "delegate",
                "task_type": random.choice(["analysis", "processing", "generation"]),
                "urgency": random.choice(["low", "medium", "high"]),
                "resources_available": random.choice(["limited", "adequate", "abundant"])
            },
            "problem_solving": {
                "operation": "solve",
                "problem_type": random.choice(["logical", "creative", "analytical"]),
                "constraints": random.choice(["few", "moderate", "many"]),
                "time_pressure": random.choice(["low", "medium", "high"])
            },
            "user_interaction": {
                "operation": "interact",
                "interaction_type": random.choice(["direct", "indirect", "collaborative"]),
                "user_expertise": random.choice(["novice", "intermediate", "expert"]),
                "communication_style": random.choice(["formal", "casual", "technical"])
            },
            "error_handling": {
                "operation": "handle_error",
                "error_type": random.choice(["timeout", "connection", "validation", "processing"]),
                "severity": random.choice(["low", "medium", "high"]),
                "recovery_options": random.choice(["few", "several", "many"])
            }
        }
        
        return contexts.get(event_type, {"operation": "general"})
    
    def _generate_event_outcome(self, event_type: str, day: int, event_num: int) -> Dict[str, Any]:
        """Generate outcome for an event."""
        # Base performance improves over time
        base_performance = 0.6 + (day / self.simulation_days) * 0.3
        performance_score = base_performance + random.uniform(-0.2, 0.2)
        performance_score = max(0.1, min(1.0, performance_score))
        
        # Response time varies by event type and improves over time
        base_response_time = {
            "classification": 1000,
            "delegation": 2000,
            "problem_solving": 3000,
            "user_interaction": 1500,
            "error_handling": 2500
        }.get(event_type, 1500)
        
        # Response time improves over time
        improvement_factor = 1.0 - (day / self.simulation_days) * 0.3
        response_time = base_response_time * improvement_factor + random.uniform(-200, 200)
        response_time = max(100, response_time)
        
        outcome = {
            "performance_score": performance_score,
            "response_time_ms": response_time,
            "quality_score": performance_score + random.uniform(-0.1, 0.1)
        }
        
        # Add event-specific outcome data
        if event_type == "error_handling":
            outcome["retry_count"] = random.randint(0, 3)
            outcome["eventual_success"] = random.random() < 0.7
        elif event_type == "user_interaction":
            outcome["feedback_score"] = random.uniform(2.0, 5.0)
            outcome["satisfaction_rating"] = random.uniform(0.5, 1.0)
        elif event_type == "delegation":
            outcome["subagents_used"] = random.randint(1, 4)
            outcome["coordination_efficiency"] = random.uniform(0.6, 0.9)
        
        return outcome
    
    async def demonstrate_learning(self, events: List[LearningEvent]):
        """Demonstrate the learning capabilities."""
        logger.info("Demonstrating learning capabilities...")
        
        # Perform learning cycle
        insights = await self.learning_engine.perform_learning_cycle()
        
        logger.info(f"Generated {len(insights)} learning insights:")
        for insight in insights[:5]:  # Show first 5 insights
            logger.info(f"  - {insight.learning_type.value}: {insight.description}")
            logger.info(f"    Confidence: {insight.confidence:.2f}, Impact: {insight.impact_score:.2f}")
        
        # Show learning statistics
        stats = await self.learning_engine.get_learning_statistics()
        logger.info(f"Learning Statistics:")
        logger.info(f"  - Total events: {stats['total_events']}")
        logger.info(f"  - Total insights: {stats['total_insights']}")
        logger.info(f"  - Success rates by type: {stats['success_rates']}")
        
        return insights
    
    async def demonstrate_evolution(self, insights: List):
        """Demonstrate the evolution capabilities."""
        logger.info("Demonstrating evolution capabilities...")
        
        # Perform evolution cycle
        evolution_events = await self.evolution_manager.perform_evolution_cycle()
        
        logger.info(f"Generated {len(evolution_events)} evolution events:")
        for event in evolution_events[:3]:  # Show first 3 events
            logger.info(f"  - {event.evolution_type.value}: {event.description}")
            logger.info(f"    Status: {event.status.value}, Confidence: {event.confidence:.2f}")
            if event.proposed_changes:
                logger.info(f"    Changes: {event.proposed_changes}")
        
        # Show evolution statistics
        stats = await self.evolution_manager.get_evolution_statistics()
        logger.info(f"Evolution Statistics:")
        logger.info(f"  - Total events: {stats['total_events']}")
        logger.info(f"  - Success rate: {stats['success_rate']:.2f}")
        logger.info(f"  - Average expected impact: {stats['avg_expected_impact']:.2f}")
        
        return evolution_events
    
    async def demonstrate_adaptation(self, events: List[LearningEvent]):
        """Demonstrate adaptation strategies."""
        logger.info("Demonstrating adaptation strategies...")
        
        current_config = {
            "confidence_threshold": 0.7,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "learning_rate": 0.1
        }
        
        # Test different adaptation strategies
        strategies = [
            ("Performance-Based", PerformanceBasedAdaptation()),
            ("Feedback-Based", FeedbackBasedAdaptation()),
            ("Experience-Based", ExperienceBasedAdaptation())
        ]
        
        for strategy_name, strategy in strategies:
            logger.info(f"Testing {strategy_name} Adaptation:")
            
            recommendations = await strategy.analyze_performance(events, current_config)
            
            logger.info(f"  Generated {len(recommendations)} recommendations:")
            for rec in recommendations[:2]:  # Show first 2 recommendations
                logger.info(f"    - {rec.adaptation_type.value}: {rec.description}")
                logger.info(f"      Current: {rec.current_value} -> Recommended: {rec.recommended_value}")
                logger.info(f"      Expected improvement: {rec.expected_improvement:.2f}")
    
    async def demonstrate_knowledge_distillation(self, events: List[LearningEvent]):
        """Demonstrate knowledge distillation."""
        logger.info("Demonstrating knowledge distillation...")
        
        # Extract knowledge from agent experiences
        extracted_knowledge = await self.knowledge_distiller._extract_agent_knowledge(self.agent_id)
        
        logger.info(f"Extracted {len(extracted_knowledge)} knowledge items:")
        for item in extracted_knowledge[:3]:  # Show first 3 items
            logger.info(f"  - {item.knowledge_type.value}: {item.title}")
            logger.info(f"    Confidence: {item.confidence:.2f}, Applicability: {item.applicability_score:.2f}")
            logger.info(f"    Description: {item.description}")
        
        # Show distillation statistics
        stats = await self.knowledge_distiller.get_distillation_statistics()
        logger.info(f"Knowledge Distillation Statistics:")
        logger.info(f"  - Total knowledge items: {stats['total_knowledge_items']}")
        logger.info(f"  - Knowledge by type: {stats['knowledge_by_type']}")
        logger.info(f"  - Average confidence: {stats['avg_confidence']:.2f}")
        
        return extracted_knowledge
    
    async def demonstrate_behavioral_patterns(self, events: List[LearningEvent]):
        """Demonstrate behavioral pattern analysis."""
        logger.info("Demonstrating behavioral pattern analysis...")
        
        # Analyze behavioral patterns
        patterns = await self.pattern_analyzer.analyze_behavioral_patterns(events, self.agent_id)
        
        logger.info(f"Detected {len(patterns)} behavioral patterns:")
        for pattern in patterns[:3]:  # Show first 3 patterns
            logger.info(f"  - {pattern.behavior_type.value}: {pattern.pattern_name}")
            logger.info(f"    Strength: {pattern.strength.value}, Effectiveness: {pattern.effectiveness:.2f}")
            logger.info(f"    Description: {pattern.description}")
        
        # Generate recommendations
        recommendations = await self.pattern_analyzer.generate_behavior_recommendations(patterns, self.agent_id)
        
        logger.info(f"Generated {len(recommendations)} behavior recommendations:")
        for rec in recommendations[:2]:  # Show first 2 recommendations
            logger.info(f"  - {rec.recommendation_type}: {rec.description}")
            logger.info(f"    Expected improvement: {rec.expected_improvement:.2f}")
        
        # Show pattern statistics
        stats = await self.pattern_analyzer.get_pattern_statistics()
        logger.info(f"Behavioral Pattern Statistics:")
        logger.info(f"  - Total patterns: {stats['total_patterns']}")
        logger.info(f"  - Patterns by type: {stats['patterns_by_type']}")
        logger.info(f"  - Average effectiveness: {stats['avg_effectiveness']:.2f}")
        
        return patterns, recommendations
    
    async def demonstrate_integration(self):
        """Demonstrate integration between all components."""
        logger.info("Demonstrating system integration...")
        
        # Show how components work together
        logger.info("Integration flow:")
        logger.info("1. Agent experiences -> Learning Events")
        logger.info("2. Learning Events -> Learning Insights (via Learning Engine)")
        logger.info("3. Learning Insights -> Evolution Events (via Evolution Manager)")
        logger.info("4. Learning Events -> Behavioral Patterns (via Pattern Analyzer)")
        logger.info("5. Learning Events -> Knowledge Items (via Knowledge Distiller)")
        logger.info("6. All components -> Adaptation Recommendations")
        
        # Show overall system health
        learning_stats = await self.learning_engine.get_learning_statistics()
        evolution_stats = await self.evolution_manager.get_evolution_statistics()
        
        logger.info("System Health Summary:")
        logger.info(f"  - Learning: {learning_stats['learning_status']}")
        logger.info(f"  - Evolution: {evolution_stats['evolution_status']}")
        logger.info(f"  - Total learning events: {learning_stats['total_events']}")
        logger.info(f"  - Total insights generated: {learning_stats['total_insights']}")
        logger.info(f"  - Total evolution events: {evolution_stats['total_events']}")
    
    async def run_complete_demo(self):
        """Run the complete evolution system demonstration."""
        logger.info("Starting Evolution System Demonstration")
        logger.info("=" * 50)
        
        try:
            # Setup
            await self.setup_components()
            
            # Simulate agent experience
            events = await self.simulate_agent_experience()
            
            # Demonstrate each component
            insights = await self.demonstrate_learning(events)
            evolution_events = await self.demonstrate_evolution(insights)
            await self.demonstrate_adaptation(events)
            knowledge_items = await self.demonstrate_knowledge_distillation(events)
            patterns, recommendations = await self.demonstrate_behavioral_patterns(events)
            
            # Show integration
            await self.demonstrate_integration()
            
            logger.info("=" * 50)
            logger.info("Evolution System Demonstration Complete!")
            
            # Summary
            logger.info("\nDemonstration Summary:")
            logger.info(f"  - Simulated {len(events)} agent experiences")
            logger.info(f"  - Generated {len(insights)} learning insights")
            logger.info(f"  - Created {len(evolution_events)} evolution events")
            logger.info(f"  - Extracted {len(knowledge_items)} knowledge items")
            logger.info(f"  - Detected {len(patterns)} behavioral patterns")
            logger.info(f"  - Produced {len(recommendations)} behavior recommendations")
            
        except Exception as e:
            logger.error(f"Error during demonstration: {e}")
            raise


async def main():
    """Main function to run the evolution demo."""
    demo = EvolutionDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())