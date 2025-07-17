"""
Demonstration of the knowledge distillation system.

This script shows how the knowledge distillation system works to extract,
distill, and transfer knowledge between agents based on their experiences.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from uuid import uuid4

from backend.evolution.knowledge_distillation import (
    KnowledgeDistiller, KnowledgeType, KnowledgeItem,
    DecisionPatternExtractor, ErrorHandlingExtractor
)
from backend.evolution.learning_engine import LearningEvent, LearningInsight, LearningType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeDistillationDemo:
    """Demonstration of the knowledge distillation system."""
    
    def __init__(self):
        """Initialize the knowledge distillation demo."""
        self.knowledge_distiller = KnowledgeDistiller()
        
        # Demo configuration
        self.agent_ids = ["expert_agent", "novice_agent", "specialist_agent"]
        self.simulation_days = 5
        self.events_per_agent = 50
    
    def generate_learning_events(self) -> Dict[str, List[LearningEvent]]:
        """Generate synthetic learning events for different agents."""
        logger.info(f"Generating learning events for {len(self.agent_ids)} agents...")
        
        events_by_agent = {}
        base_time = datetime.now(timezone.utc) - timedelta(days=self.simulation_days)
        
        # Define agent characteristics
        agent_characteristics = {
            "expert_agent": {
                "success_rate": 0.85,
                "confidence": 0.8,
                "domains": ["technical", "scientific", "analytical"],
                "error_types": ["validation", "edge_case"]
            },
            "novice_agent": {
                "success_rate": 0.6,
                "confidence": 0.5,
                "domains": ["general", "basic"],
                "error_types": ["timeout", "validation", "input_error", "processing"]
            },
            "specialist_agent": {
                "success_rate": 0.75,
                "confidence": 0.7,
                "domains": ["medical", "financial"],
                "error_types": ["domain_specific", "validation"]
            }
        }
        
        # Generate events for each agent
        for agent_id in self.agent_ids:
            agent_events = []
            characteristics = agent_characteristics.get(agent_id, {})
            
            for i in range(self.events_per_agent):
                event_time = base_time + timedelta(hours=random.randint(0, self.simulation_days * 24))
                
                # Determine event type based on agent's domain
                domains = characteristics.get("domains", ["general"])
                domain = random.choice(domains)
                
                if domain in ["technical", "scientific", "analytical", "medical", "financial"]:
                    event_types = ["classification", "analysis", "prediction", "recommendation"]
                else:
                    event_types = ["classification", "general_query", "simple_task"]
                
                event_type = random.choice(event_types)
                
                # Determine success based on agent's success rate
                success_rate = characteristics.get("success_rate", 0.7)
                success = random.random() < success_rate
                
                # Determine confidence based on agent's characteristics
                base_confidence = characteristics.get("confidence", 0.6)
                confidence = base_confidence + random.uniform(-0.1, 0.1)
                confidence = max(0.1, min(0.95, confidence))
                
                # Create context
                context = {
                    "domain": domain,
                    "complexity": random.choice(["low", "medium", "high"]),
                    "input_type": random.choice(["text", "structured", "numeric", "mixed"])
                }
                
                # Create outcome
                outcome = {
                    "performance_score": random.uniform(0.5, 0.9) if success else random.uniform(0.2, 0.5),
                    "response_time_ms": random.uniform(500, 2000)
                }
                
                # Add error information for failures
                if not success:
                    error_types = characteristics.get("error_types", ["general_error"])
                    outcome["error_type"] = random.choice(error_types)
                    
                    if outcome["error_type"] == "timeout":
                        outcome["retry_count"] = random.randint(0, 3)
                        outcome["eventual_success"] = random.random() < 0.3
                
                # Create learning event
                event = LearningEvent(
                    id=f"{agent_id}_event_{i}",
                    event_type=event_type,
                    context=context,
                    outcome=outcome,
                    success=success,
                    confidence=confidence,
                    timestamp=event_time,
                    importance_score=random.uniform(0.3, 0.8),
                    metadata={"agent_id": agent_id}
                )
                
                agent_events.append(event)
            
            events_by_agent[agent_id] = agent_events
            logger.info(f"Generated {len(agent_events)} events for {agent_id}")
        
        return events_by_agent
    
    def generate_learning_insights(self, events_by_agent: Dict[str, List[LearningEvent]]) -> Dict[str, List[LearningInsight]]:
        """Generate synthetic learning insights based on events."""
        logger.info("Generating learning insights...")
        
        insights_by_agent = {}
        
        for agent_id, events in events_by_agent.items():
            agent_insights = []
            
            # Group events by type
            event_types = {}
            for event in events:
                if event.event_type not in event_types:
                    event_types[event.event_type] = []
                event_types[event.event_type].append(event)
            
            # Generate insights for each event type with sufficient data
            for event_type, type_events in event_types.items():
                if len(type_events) >= 5:  # Minimum threshold for insights
                    # Calculate success rate
                    success_count = sum(1 for e in type_events if e.success)
                    success_rate = success_count / len(type_events)
                    
                    # Generate performance insight
                    if success_rate < 0.7:
                        insight = LearningInsight(
                            id=f"{agent_id}_insight_perf_{event_type}",
                            learning_type=LearningType.PERFORMANCE_OPTIMIZATION,
                            description=f"Low performance detected in {event_type} operations (success rate: {success_rate:.2f})",
                            confidence=0.8,
                            supporting_evidence=[f"{success_count}/{len(type_events)} successful operations"],
                            recommended_actions=[
                                "Review and optimize processing logic",
                                "Improve input validation"
                            ],
                            impact_score=0.7,
                            timestamp=datetime.now(timezone.utc),
                            metadata={"event_type": event_type, "agent_id": agent_id}
                        )
                        agent_insights.append(insight)
                    
                    # Generate error reduction insight if there are failures
                    failures = [e for e in type_events if not e.success]
                    if failures:
                        error_types = {}
                        for failure in failures:
                            error_type = failure.outcome.get("error_type", "unknown")
                            if error_type not in error_types:
                                error_types[error_type] = 0
                            error_types[error_type] += 1
                        
                        most_common_error = max(error_types.items(), key=lambda x: x[1])
                        
                        insight = LearningInsight(
                            id=f"{agent_id}_insight_error_{event_type}",
                            learning_type=LearningType.ERROR_REDUCTION,
                            description=f"Frequent {most_common_error[0]} errors in {event_type} operations",
                            confidence=0.75,
                            supporting_evidence=[f"{most_common_error[1]} occurrences of {most_common_error[0]} errors"],
                            recommended_actions=[
                                f"Implement specific handling for {most_common_error[0]} errors",
                                "Add validation checks"
                            ],
                            impact_score=0.6,
                            timestamp=datetime.now(timezone.utc),
                            metadata={"error_type": most_common_error[0], "agent_id": agent_id}
                        )
                        agent_insights.append(insight)
            
            insights_by_agent[agent_id] = agent_insights
            logger.info(f"Generated {len(agent_insights)} insights for {agent_id}")
        
        return insights_by_agent
    
    async def demonstrate_knowledge_extraction(
        self,
        events_by_agent: Dict[str, List[LearningEvent]],
        insights_by_agent: Dict[str, List[LearningInsight]]
    ):
        """Demonstrate knowledge extraction from agent experiences."""
        logger.info("Demonstrating knowledge extraction...")
        
        extracted_knowledge = {}
        
        for agent_id in self.agent_ids:
            logger.info(f"Extracting knowledge from {agent_id}...")
            
            events = events_by_agent.get(agent_id, [])
            insights = insights_by_agent.get(agent_id, [])
            
            # Extract knowledge using different extractors
            decision_extractor = DecisionPatternExtractor()
            error_extractor = ErrorHandlingExtractor()
            
            decision_knowledge = await decision_extractor.extract_knowledge(agent_id, events, insights)
            error_knowledge = await error_extractor.extract_knowledge(agent_id, events, insights)
            
            all_knowledge = decision_knowledge + error_knowledge
            extracted_knowledge[agent_id] = all_knowledge
            
            logger.info(f"Extracted {len(all_knowledge)} knowledge items from {agent_id}:")
            for item in all_knowledge[:3]:  # Show first 3 items
                logger.info(f"  - {item.knowledge_type.value}: {item.title}")
                logger.info(f"    Confidence: {item.confidence:.2f}, Applicability: {item.applicability_score:.2f}")
                logger.info(f"    Description: {item.description}")
        
        return extracted_knowledge
    
    async def demonstrate_knowledge_transfer(self, extracted_knowledge: Dict[str, List[KnowledgeItem]]):
        """Demonstrate knowledge transfer between agents."""
        logger.info("Demonstrating knowledge transfer...")
        
        # Store knowledge items in the distiller
        for agent_id, knowledge_items in extracted_knowledge.items():
            for item in knowledge_items:
                self.knowledge_distiller.knowledge_items[item.id] = item
        
        # Perform knowledge transfers
        transfers = await self.knowledge_distiller._perform_knowledge_transfers()
        
        logger.info(f"Performed {len(transfers)} knowledge transfers:")
        for transfer in transfers[:5]:  # Show first 5 transfers
            logger.info(f"  - From {transfer.source_agent} to {transfer.target_agent}")
            logger.info(f"    Knowledge items: {len(transfer.knowledge_items)}")
            logger.info(f"    Transfer method: {transfer.transfer_method}")
            logger.info(f"    Success: {transfer.success}")
        
        return transfers
    
    async def demonstrate_knowledge_application(self, extracted_knowledge: Dict[str, List[KnowledgeItem]]):
        """Demonstrate how knowledge can be applied to improve agent performance."""
        logger.info("Demonstrating knowledge application...")
        
        # Select a target agent (novice) and source agent (expert)
        target_agent = "novice_agent"
        source_agent = "expert_agent"
        
        # Get knowledge from expert agent
        expert_knowledge = extracted_knowledge.get(source_agent, [])
        
        if not expert_knowledge:
            logger.info(f"No knowledge available from {source_agent}")
            return
        
        # Filter to high-quality knowledge
        high_quality_knowledge = [
            item for item in expert_knowledge
            if item.confidence > 0.7 and item.applicability_score > 0.6
        ]
        
        logger.info(f"Selected {len(high_quality_knowledge)} high-quality knowledge items from {source_agent}")
        
        # Simulate performance improvement from knowledge application
        logger.info(f"Simulating performance improvement for {target_agent}...")
        
        # Before metrics (from original events)
        original_success_rate = 0.6  # Novice agent's original success rate
        
        # After metrics (simulated improvement)
        improved_success_rate = original_success_rate
        
        for item in high_quality_knowledge:
            # Different knowledge types provide different improvements
            if item.knowledge_type == KnowledgeType.DECISION_PATTERNS:
                improved_success_rate += 0.03
            elif item.knowledge_type == KnowledgeType.ERROR_HANDLING:
                improved_success_rate += 0.02
        
        # Cap at reasonable maximum
        improved_success_rate = min(0.85, improved_success_rate)
        
        improvement_percentage = (improved_success_rate - original_success_rate) / original_success_rate * 100
        
        logger.info(f"Performance improvement simulation results:")
        logger.info(f"  - Original success rate: {original_success_rate:.2f}")
        logger.info(f"  - Improved success rate: {improved_success_rate:.2f}")
        logger.info(f"  - Improvement: {improvement_percentage:.1f}%")
        
        # Simulate specific improvements
        logger.info("Specific improvements from knowledge application:")
        
        for item in high_quality_knowledge[:3]:  # Show first 3 items
            if item.knowledge_type == KnowledgeType.DECISION_PATTERNS:
                logger.info(f"  - Applied decision pattern: {item.title}")
                logger.info(f"    Benefit: Improved decision accuracy in {item.content.get('context', 'unknown')} context")
            elif item.knowledge_type == KnowledgeType.ERROR_HANDLING:
                logger.info(f"  - Applied error handling: {item.title}")
                logger.info(f"    Benefit: Reduced {item.content.get('error_type', 'unknown')} errors")
    
    async def demonstrate_knowledge_distillation_cycle(self):
        """Demonstrate a complete knowledge distillation cycle."""
        logger.info("Demonstrating complete knowledge distillation cycle...")
        
        # Perform distillation cycle
        results = await self.knowledge_distiller.perform_distillation_cycle()
        
        logger.info(f"Distillation cycle results:")
        logger.info(f"  - Agents processed: {results['agents_processed']}")
        logger.info(f"  - Knowledge extracted: {results['knowledge_extracted']}")
        logger.info(f"  - Transfers performed: {results['transfers_performed']}")
        
        if results['errors']:
            logger.info(f"  - Errors encountered: {len(results['errors'])}")
        
        # Show distillation statistics
        stats = await self.knowledge_distiller.get_distillation_statistics()
        logger.info(f"Knowledge Distillation Statistics:")
        logger.info(f"  - Total knowledge items: {stats['total_knowledge_items']}")
        logger.info(f"  - Knowledge by type: {stats['knowledge_by_type']}")
        logger.info(f"  - Average confidence: {stats['avg_confidence']:.2f}")
        logger.info(f"  - Average applicability: {stats['avg_applicability']:.2f}")
    
    async def run_demo(self):
        """Run the complete knowledge distillation demonstration."""
        logger.info("Starting Knowledge Distillation Demonstration")
        logger.info("=" * 60)
        
        try:
            # Generate learning events and insights
            events_by_agent = self.generate_learning_events()
            insights_by_agent = self.generate_learning_insights(events_by_agent)
            
            # Demonstrate knowledge extraction
            extracted_knowledge = await self.demonstrate_knowledge_extraction(events_by_agent, insights_by_agent)
            
            # Demonstrate knowledge transfer
            transfers = await self.demonstrate_knowledge_transfer(extracted_knowledge)
            
            # Demonstrate knowledge application
            await self.demonstrate_knowledge_application(extracted_knowledge)
            
            # Demonstrate complete distillation cycle
            await self.demonstrate_knowledge_distillation_cycle()
            
            logger.info("=" * 60)
            logger.info("Knowledge Distillation Demonstration Complete!")
            
            # Summary
            total_knowledge = sum(len(items) for items in extracted_knowledge.values())
            logger.info("\nDemonstration Summary:")
            logger.info(f"  - Agents involved: {len(self.agent_ids)}")
            logger.info(f"  - Total knowledge items extracted: {total_knowledge}")
            logger.info(f"  - Knowledge transfers performed: {len(transfers)}")
            
        except Exception as e:
            logger.error(f"Error during demonstration: {e}", exc_info=True)
            raise


async def main():
    """Main function to run the knowledge distillation demo."""
    demo = KnowledgeDistillationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())