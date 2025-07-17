"""
Agent evolution and learning system.

This module provides capabilities for agents to learn from experience,
adapt their behavior, and evolve their decision-making processes over time.
"""

from .learning_engine import LearningEngine, LearningStrategy
from .evolution_manager import EvolutionManager, EvolutionEvent
from .adaptation_strategies import (
    PerformanceBasedAdaptation,
    FeedbackBasedAdaptation,
    ExperienceBasedAdaptation
)
from .knowledge_distillation import KnowledgeDistiller
from .behavioral_patterns import BehavioralPatternAnalyzer

__all__ = [
    "LearningEngine",
    "LearningStrategy", 
    "EvolutionManager",
    "EvolutionEvent",
    "PerformanceBasedAdaptation",
    "FeedbackBasedAdaptation",
    "ExperienceBasedAdaptation",
    "KnowledgeDistiller",
    "BehavioralPatternAnalyzer"
]