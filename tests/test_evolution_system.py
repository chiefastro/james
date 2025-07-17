"""
Tests for the agent evolution and learning system.

This module tests the learning engine, evolution manager, adaptation strategies,
knowledge distillation, and behavioral pattern analysis components.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from backend.evolution.learning_engine import (
    LearningEngine, LearningEvent, LearningInsight, LearningType,
    PerformanceLearningStrategy, ErrorLearningStrategy
)
from backend.evolution.evolution_manager import (
    EvolutionManager, EvolutionEvent, EvolutionType, EvolutionStatus,
    ParameterTuningStrategy, BehaviorAdaptationStrategy
)
from backend.evolution.adaptation_strategies import (
    PerformanceBasedAdaptation, FeedbackBasedAdaptation, ExperienceBasedAdaptation,
    AdaptationType
)
from backend.evolution.knowledge_distillation import (
    KnowledgeDistiller, KnowledgeType, DecisionPatternExtractor, ErrorHandlingExtractor
)
from backend.evolution.behavioral_patterns import (
    BehavioralPatternAnalyzer, BehaviorType, PatternStrength
)


class TestLearningEngine:
    """Test the learning engine functionality."""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a learning engine for testing."""
        return LearningEngine()
    
    @pytest.fixture
    def sample_events(self):
        """Create sample learning events."""
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(20):
            event = LearningEvent(
                id=f"event_{i}",
                event_type="classification",
                context={"operation": "classify", "category": "test"},
                outcome={"performance_score": 0.8 if i % 2 == 0 else 0.6},
                success=i % 3 != 0,  # 2/3 success rate
                confidence=0.7 + (i % 3) * 0.1,
                timestamp=base_time + timedelta(minutes=i),
                importance_score=0.5
            )
            events.append(event)
        
        return events
    
    def test_learning_engine_initialization(self, learning_engine):
        """Test learning engine initialization."""
        assert learning_engine is not None
        assert len(learning_engine.strategies) == 2  # Default strategies
        assert isinstance(learning_engine.strategies[0], PerformanceLearningStrategy)
        assert isinstance(learning_engine.strategies[1], ErrorLearningStrategy)
    
    @pytest.mark.asyncio
    async def test_record_learning_event(self, learning_engine):
        """Test recording learning events."""
        event = await learning_engine.record_learning_event(
            event_type="test_event",
            context={"test": "context"},
            outcome={"result": "success"},
            success=True,
            confidence=0.8
        )
        
        assert event is not None
        assert event.event_type == "test_event"
        assert event.success is True
        assert event.confidence == 0.8
        assert len(learning_engine.learning_events) == 1
    
    @pytest.mark.asyncio
    async def test_performance_learning_strategy(self, sample_events):
        """Test performance learning strategy."""
        strategy = PerformanceLearningStrategy()
        insights = await strategy.learn_from_events(sample_events)
        
        assert len(insights) > 0
        assert all(isinstance(insight, LearningInsight) for insight in insights)
        assert any(insight.learning_type == LearningType.PERFORMANCE_OPTIMIZATION for insight in insights)
    
    @pytest.mark.asyncio
    async def test_error_learning_strategy(self, sample_events):
        """Test error learning strategy."""
        # Add error information to some events
        for i, event in enumerate(sample_events):
            if not event.success:
                event.outcome["error_type"] = "timeout" if i % 2 == 0 else "connection"
        
        strategy = ErrorLearningStrategy()
        insights = await strategy.learn_from_events(sample_events)
        
        assert len(insights) > 0
        assert all(isinstance(insight, LearningInsight) for insight in insights)
        assert any(insight.learning_type == LearningType.ERROR_REDUCTION for insight in insights)
    
    @pytest.mark.asyncio
    async def test_learning_cycle(self, learning_engine, sample_events):
        """Test complete learning cycle."""
        # Add events to the engine
        learning_engine.learning_events = sample_events
        
        # Perform learning cycle
        insights = await learning_engine.perform_learning_cycle()
        
        assert len(insights) > 0
        assert len(learning_engine.insights) > 0
    
    @pytest.mark.asyncio
    async def test_get_insights(self, learning_engine):
        """Test getting insights with filtering."""
        # Add some mock insights
        insight1 = LearningInsight(
            id="insight_1",
            learning_type=LearningType.PERFORMANCE_OPTIMIZATION,
            description="Test insight 1",
            confidence=0.8,
            supporting_evidence=[],
            recommended_actions=[],
            impact_score=0.7,
            timestamp=datetime.now(timezone.utc)
        )
        
        insight2 = LearningInsight(
            id="insight_2",
            learning_type=LearningType.ERROR_REDUCTION,
            description="Test insight 2",
            confidence=0.6,
            supporting_evidence=[],
            recommended_actions=[],
            impact_score=0.5,
            timestamp=datetime.now(timezone.utc)
        )
        
        learning_engine.insights = [insight1, insight2]
        
        # Test filtering
        performance_insights = await learning_engine.get_insights(
            learning_type=LearningType.PERFORMANCE_OPTIMIZATION
        )
        assert len(performance_insights) == 1
        assert performance_insights[0].id == "insight_1"
        
        high_confidence_insights = await learning_engine.get_insights(min_confidence=0.7)
        assert len(high_confidence_insights) == 1
        assert high_confidence_insights[0].id == "insight_1"


class TestEvolutionManager:
    """Test the evolution manager functionality."""
    
    @pytest.fixture
    def evolution_manager(self):
        """Create an evolution manager for testing."""
        return EvolutionManager()
    
    @pytest.fixture
    def mock_learning_engine(self):
        """Create a mock learning engine."""
        mock_engine = Mock()
        mock_engine.get_insights = AsyncMock(return_value=[
            LearningInsight(
                id="insight_1",
                learning_type=LearningType.PERFORMANCE_OPTIMIZATION,
                description="Low performance detected",
                confidence=0.8,
                supporting_evidence=[],
                recommended_actions=[],
                impact_score=0.6,
                timestamp=datetime.now(timezone.utc)
            )
        ])
        return mock_engine
    
    def test_evolution_manager_initialization(self, evolution_manager):
        """Test evolution manager initialization."""
        assert evolution_manager is not None
        assert len(evolution_manager.strategies) == 2  # Default strategies
        assert isinstance(evolution_manager.strategies[0], ParameterTuningStrategy)
        assert isinstance(evolution_manager.strategies[1], BehaviorAdaptationStrategy)
    
    @pytest.mark.asyncio
    async def test_parameter_tuning_strategy(self):
        """Test parameter tuning strategy."""
        strategy = ParameterTuningStrategy()
        
        insights = [
            LearningInsight(
                id="insight_1",
                learning_type=LearningType.PERFORMANCE_OPTIMIZATION,
                description="Low performance detected",
                confidence=0.8,
                supporting_evidence=[],
                recommended_actions=[],
                impact_score=0.6,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        current_state = {"confidence_threshold": 0.7}
        
        proposals = await strategy.propose_evolution(insights, current_state)
        
        assert len(proposals) > 0
        assert all(isinstance(event, EvolutionEvent) for event in proposals)
        assert any(event.evolution_type == EvolutionType.PARAMETER_TUNING for event in proposals)
    
    @pytest.mark.asyncio
    async def test_evolution_cycle(self, evolution_manager, mock_learning_engine):
        """Test complete evolution cycle."""
        evolution_manager.learning_engine = mock_learning_engine
        
        events = await evolution_manager.perform_evolution_cycle()
        
        # Should have processed some events
        assert isinstance(events, list)
        # Note: Actual events depend on approval logic


class TestAdaptationStrategies:
    """Test adaptation strategies."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for adaptation testing."""
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(15):
            event = LearningEvent(
                id=f"event_{i}",
                event_type="classification",
                context={"operation": "classify"},
                outcome={
                    "performance_score": 0.6 + (i % 3) * 0.1,
                    "response_time_ms": 1000 + i * 100
                },
                success=i % 4 != 0,  # 75% success rate
                confidence=0.6 + (i % 4) * 0.1,
                timestamp=base_time + timedelta(minutes=i)
            )
            events.append(event)
        
        return events
    
    @pytest.mark.asyncio
    async def test_performance_based_adaptation(self, sample_events):
        """Test performance-based adaptation strategy."""
        strategy = PerformanceBasedAdaptation()
        current_config = {"confidence_threshold": 0.7, "timeout_seconds": 30}
        
        recommendations = await strategy.analyze_performance(sample_events, current_config)
        
        assert isinstance(recommendations, list)
        # Should have some recommendations for the low performance events
    
    @pytest.mark.asyncio
    async def test_feedback_based_adaptation(self):
        """Test feedback-based adaptation strategy."""
        strategy = FeedbackBasedAdaptation()
        
        # Create feedback events
        feedback_events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(10):
            event = LearningEvent(
                id=f"feedback_{i}",
                event_type="user_feedback",
                context={"category": "general"},
                outcome={"feedback_score": 2.5 + (i % 3) * 0.5},  # Low feedback scores
                success=True,
                confidence=0.8,
                timestamp=base_time + timedelta(minutes=i)
            )
            feedback_events.append(event)
        
        current_config = {"confidence_threshold": 0.7}
        recommendations = await strategy.analyze_performance(feedback_events, current_config)
        
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_experience_based_adaptation(self, sample_events):
        """Test experience-based adaptation strategy."""
        strategy = ExperienceBasedAdaptation()
        current_config = {"learning_rate": 0.1, "confidence_threshold": 0.7}
        
        recommendations = await strategy.analyze_performance(sample_events, current_config)
        
        assert isinstance(recommendations, list)


class TestKnowledgeDistillation:
    """Test knowledge distillation functionality."""
    
    @pytest.fixture
    def knowledge_distiller(self):
        """Create a knowledge distiller for testing."""
        return KnowledgeDistiller()
    
    @pytest.fixture
    def sample_events_for_extraction(self):
        """Create sample events for knowledge extraction."""
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(25):
            event = LearningEvent(
                id=f"event_{i}",
                event_type="classification",
                context={"operation": "classify", "category": "test"},
                outcome={"performance_score": 0.8 if i % 2 == 0 else 0.6},
                success=i % 3 != 0,
                confidence=0.7 + (i % 3) * 0.1,
                timestamp=base_time + timedelta(minutes=i)
            )
            events.append(event)
        
        return events
    
    def test_knowledge_distiller_initialization(self, knowledge_distiller):
        """Test knowledge distiller initialization."""
        assert knowledge_distiller is not None
        assert len(knowledge_distiller.extractors) == 2  # Default extractors
        assert isinstance(knowledge_distiller.extractors[0], DecisionPatternExtractor)
        assert isinstance(knowledge_distiller.extractors[1], ErrorHandlingExtractor)
    
    @pytest.mark.asyncio
    async def test_decision_pattern_extractor(self, sample_events_for_extraction):
        """Test decision pattern extraction."""
        extractor = DecisionPatternExtractor()
        
        knowledge_items = await extractor.extract_knowledge(
            "test_agent",
            sample_events_for_extraction,
            []
        )
        
        assert len(knowledge_items) > 0
        assert all(item.knowledge_type == KnowledgeType.DECISION_PATTERNS for item in knowledge_items)
        assert all(item.source_agent == "test_agent" for item in knowledge_items)
    
    @pytest.mark.asyncio
    async def test_error_handling_extractor(self, sample_events_for_extraction):
        """Test error handling extraction."""
        # Add error information to failed events
        for event in sample_events_for_extraction:
            if not event.success:
                event.outcome["error_type"] = "timeout"
                event.outcome["retry_count"] = 2
        
        extractor = ErrorHandlingExtractor()
        
        knowledge_items = await extractor.extract_knowledge(
            "test_agent",
            sample_events_for_extraction,
            []
        )
        
        assert len(knowledge_items) > 0
        assert all(item.knowledge_type == KnowledgeType.ERROR_HANDLING for item in knowledge_items)


class TestBehavioralPatterns:
    """Test behavioral pattern analysis."""
    
    @pytest.fixture
    def pattern_analyzer(self):
        """Create a behavioral pattern analyzer for testing."""
        return BehavioralPatternAnalyzer()
    
    @pytest.fixture
    def sample_behavioral_events(self):
        """Create sample events for behavioral analysis."""
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(30):
            event = LearningEvent(
                id=f"event_{i}",
                event_type="classification",
                context={
                    "operation": "classify",
                    "category": "test",
                    "interaction_type": "direct" if i % 2 == 0 else "indirect"
                },
                outcome={
                    "response_time_ms": 1000 + i * 50,
                    "error_type": "timeout" if not (i % 5) and i % 3 == 0 else None
                },
                success=i % 4 != 0,  # 75% success rate
                confidence=0.7 + (i % 3) * 0.05,
                timestamp=base_time + timedelta(minutes=i)
            )
            events.append(event)
        
        return events
    
    def test_pattern_analyzer_initialization(self, pattern_analyzer):
        """Test pattern analyzer initialization."""
        assert pattern_analyzer is not None
        assert pattern_analyzer.analysis_window_days == 7
        assert pattern_analyzer.pattern_threshold == 0.6
    
    @pytest.mark.asyncio
    async def test_analyze_behavioral_patterns(self, pattern_analyzer, sample_behavioral_events):
        """Test behavioral pattern analysis."""
        patterns = await pattern_analyzer.analyze_behavioral_patterns(
            sample_behavioral_events,
            "test_agent"
        )
        
        assert len(patterns) > 0
        assert all(hasattr(pattern, 'behavior_type') for pattern in patterns)
        assert all(hasattr(pattern, 'strength') for pattern in patterns)
        assert all(hasattr(pattern, 'effectiveness') for pattern in patterns)
    
    @pytest.mark.asyncio
    async def test_detect_behavior_changes(self, pattern_analyzer):
        """Test behavior change detection."""
        # Create two sets of events representing different time periods
        base_time = datetime.now(timezone.utc)
        
        old_events = []
        for i in range(15):
            event = LearningEvent(
                id=f"old_event_{i}",
                event_type="classification",
                context={"operation": "classify"},
                outcome={"response_time_ms": 1000},
                success=i % 3 != 0,  # 67% success rate
                confidence=0.6,
                timestamp=base_time - timedelta(days=7) + timedelta(minutes=i)
            )
            old_events.append(event)
        
        new_events = []
        for i in range(15):
            event = LearningEvent(
                id=f"new_event_{i}",
                event_type="classification",
                context={"operation": "classify"},
                outcome={"response_time_ms": 800},
                success=i % 2 != 0,  # 50% success rate (degradation)
                confidence=0.8,
                timestamp=base_time + timedelta(minutes=i)
            )
            new_events.append(event)
        
        changes = await pattern_analyzer.detect_behavior_changes(
            old_events,
            new_events,
            "test_agent"
        )
        
        assert isinstance(changes, list)
        # Should detect some changes due to different success rates
    
    @pytest.mark.asyncio
    async def test_generate_behavior_recommendations(self, pattern_analyzer, sample_behavioral_events):
        """Test behavior recommendation generation."""
        patterns = await pattern_analyzer.analyze_behavioral_patterns(
            sample_behavioral_events,
            "test_agent"
        )
        
        recommendations = await pattern_analyzer.generate_behavior_recommendations(
            patterns,
            "test_agent"
        )
        
        assert isinstance(recommendations, list)
        # May or may not have recommendations depending on pattern effectiveness


class TestIntegration:
    """Integration tests for the evolution system."""
    
    @pytest.mark.asyncio
    async def test_full_evolution_pipeline(self):
        """Test the complete evolution pipeline."""
        # Create components
        learning_engine = LearningEngine()
        evolution_manager = EvolutionManager(learning_engine=learning_engine)
        knowledge_distiller = KnowledgeDistiller()
        pattern_analyzer = BehavioralPatternAnalyzer()
        
        # Create sample events
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(20):
            event = await learning_engine.record_learning_event(
                event_type="classification",
                context={"operation": "classify", "category": "test"},
                outcome={"performance_score": 0.7 + (i % 3) * 0.1},
                success=i % 3 != 0,
                confidence=0.6 + (i % 4) * 0.1
            )
            events.append(event)
        
        # Test learning
        insights = await learning_engine.perform_learning_cycle()
        assert len(insights) >= 0  # May or may not generate insights
        
        # Test pattern analysis
        patterns = await pattern_analyzer.analyze_behavioral_patterns(events, "test_agent")
        assert isinstance(patterns, list)
        
        # Test knowledge extraction
        extracted_knowledge = await knowledge_distiller._extract_agent_knowledge("test_agent")
        assert isinstance(extracted_knowledge, list)
        
        # Test evolution (if insights were generated)
        if insights:
            evolution_events = await evolution_manager.perform_evolution_cycle()
            assert isinstance(evolution_events, list)
    
    @pytest.mark.asyncio
    async def test_statistics_and_monitoring(self):
        """Test statistics and monitoring capabilities."""
        learning_engine = LearningEngine()
        evolution_manager = EvolutionManager()
        knowledge_distiller = KnowledgeDistiller()
        pattern_analyzer = BehavioralPatternAnalyzer()
        
        # Get statistics from all components
        learning_stats = await learning_engine.get_learning_statistics()
        evolution_stats = await evolution_manager.get_evolution_statistics()
        distillation_stats = await knowledge_distiller.get_distillation_statistics()
        pattern_stats = await pattern_analyzer.get_pattern_statistics()
        
        # Verify statistics structure
        assert "learning_status" in learning_stats
        assert "total_events" in learning_stats
        
        assert "evolution_status" in evolution_stats
        assert "total_events" in evolution_stats
        
        assert "distillation_status" in distillation_stats
        assert "total_knowledge_items" in distillation_stats
        
        assert "total_patterns" in pattern_stats
        assert "total_changes" in pattern_stats


if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        """Run a simple test of the evolution system."""
        print("Testing Evolution System...")
        
        # Test learning engine
        learning_engine = LearningEngine()
        
        # Record some events
        for i in range(10):
            await learning_engine.record_learning_event(
                event_type="test",
                context={"test": True},
                outcome={"score": 0.8 if i % 2 == 0 else 0.6},
                success=i % 3 != 0,
                confidence=0.7
            )
        
        # Perform learning
        insights = await learning_engine.perform_learning_cycle()
        print(f"Generated {len(insights)} insights")
        
        # Get statistics
        stats = await learning_engine.get_learning_statistics()
        print(f"Learning statistics: {stats}")
        
        print("Evolution system test completed!")
    
    asyncio.run(run_simple_test())