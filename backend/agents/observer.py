"""
Observer agent for message classification and routing.

The Observer agent is responsible for analyzing incoming messages and
classifying them based on content, urgency, and relevance to determine
the appropriate processing strategy.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from ..models.core import Message, MessageClassification, MessageSource
from ..queue.message_queue import MessageQueue
from ..observability.langsmith_tracer import trace_observer_operation, get_tracer
from ..observability.metrics_collector import get_metrics_collector


logger = logging.getLogger(__name__)


class ClassificationReason(Enum):
    """Reasons for message classification decisions."""
    SPAM_DETECTED = "spam_detected"
    URGENT_REQUEST = "urgent_request"
    INFORMATION_ONLY = "information_only"
    REQUIRES_ACTION = "requires_action"
    TIME_SENSITIVE = "time_sensitive"
    ROUTINE_QUERY = "routine_query"
    SYSTEM_NOTIFICATION = "system_notification"
    SUBAGENT_COMMUNICATION = "subagent_communication"
    MALICIOUS_CONTENT = "malicious_content"
    DUPLICATE_MESSAGE = "duplicate_message"


@dataclass
class ClassificationResult:
    """Result of message classification analysis."""
    classification: MessageClassification
    priority: int
    delay_seconds: Optional[int] = None
    reason: Optional[ClassificationReason] = None
    confidence: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MessageClassifier:
    """
    LLM-based message classifier with structured output.
    
    Analyzes message content, source, and context to determine
    the appropriate classification and processing strategy.
    """
    
    # Keywords that indicate urgent/immediate action
    URGENT_KEYWORDS = {
        "urgent", "emergency", "critical", "asap", "immediately", "now",
        "help", "error", "failed", "broken", "down", "issue", "problem"
    }
    
    # Keywords that indicate informational content
    INFO_KEYWORDS = {
        "fyi", "information", "update", "status", "report", "summary",
        "notification", "alert", "reminder", "note"
    }
    
    # Keywords that indicate spam or low-value content
    SPAM_KEYWORDS = {
        "spam", "advertisement", "promotion", "offer", "deal", "sale",
        "click here", "free", "winner", "congratulations", "prize"
    }
    
    # Keywords that indicate questions requiring action
    ACTION_KEYWORDS = {
        "how", "what", "when", "where", "why", "can you", "please",
        "help me", "need", "want", "request", "ask", "question"
    }
    
    def __init__(self):
        """Initialize the message classifier."""
        self._classification_history: List[ClassificationResult] = []
        logger.info("MessageClassifier initialized")
    
    def classify_message(self, message: Message) -> ClassificationResult:
        """
        Classify a message using rule-based logic and pattern matching.
        
        In a full implementation, this would use an LLM with structured output.
        For now, we implement rule-based classification that can be easily
        extended with LLM integration.
        
        Args:
            message: The message to classify
            
        Returns:
            ClassificationResult with classification and metadata
        """
        content_lower = message.content.lower().strip()
        
        # Handle empty or very short messages
        if len(content_lower) < 3:
            return ClassificationResult(
                classification=MessageClassification.IGNORE_DELETE,
                priority=70,
                reason=ClassificationReason.SPAM_DETECTED,
                confidence=0.9,
                explanation="Message too short to be meaningful"
            )
        
        # Check for malicious patterns
        if self._is_malicious_content(content_lower):
            return ClassificationResult(
                classification=MessageClassification.IGNORE_DELETE,
                priority=80,
                reason=ClassificationReason.MALICIOUS_CONTENT,
                confidence=0.95,
                explanation="Potentially malicious content detected"
            )
        
        # Check for spam patterns
        if self._is_spam_content(content_lower):
            return ClassificationResult(
                classification=MessageClassification.IGNORE_DELETE,
                priority=75,
                reason=ClassificationReason.SPAM_DETECTED,
                confidence=0.8,
                explanation="Spam patterns detected in content"
            )
        
        # System messages get high priority
        if message.source == MessageSource.SYSTEM:
            return self._classify_system_message(message, content_lower)
        
        # Subagent messages get medium-high priority
        if message.source == MessageSource.SUBAGENT:
            return self._classify_subagent_message(message, content_lower)
        
        # User and external messages
        return self._classify_user_message(message, content_lower)
    
    def _is_malicious_content(self, content: str) -> bool:
        """Check if content contains potentially malicious patterns."""
        malicious_patterns = [
            "rm -rf", "del /", "format c:", "sudo rm",
            "<script", "javascript:", "eval(", "exec(",
            "drop table", "delete from", "truncate",
            "../../", "../../../", "passwd", "/etc/shadow"
        ]
        
        return any(pattern in content for pattern in malicious_patterns)
    
    def _is_spam_content(self, content: str) -> bool:
        """Check if content appears to be spam."""
        spam_score = 0
        
        # Check for spam keywords
        spam_words = sum(1 for keyword in self.SPAM_KEYWORDS if keyword in content)
        spam_score += spam_words * 2
        
        # Check for excessive capitalization
        if content.isupper() and len(content) > 10:
            spam_score += 3
        
        # Check for excessive punctuation
        exclamation_count = content.count('!')
        if exclamation_count > 3:
            spam_score += exclamation_count
        
        # Check for suspicious URLs or email patterns
        if "http://" in content or "www." in content:
            spam_score += 2
        
        return spam_score >= 5
    
    def _classify_system_message(self, message: Message, content: str) -> ClassificationResult:
        """Classify system-generated messages."""
        # System messages are generally important
        if any(keyword in content for keyword in self.URGENT_KEYWORDS):
            return ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=5,
                reason=ClassificationReason.SYSTEM_NOTIFICATION,
                confidence=0.9,
                explanation="Urgent system notification requiring immediate attention"
            )
        
        return ClassificationResult(
            classification=MessageClassification.ACT_NOW,
            priority=10,
            reason=ClassificationReason.SYSTEM_NOTIFICATION,
            confidence=0.8,
            explanation="System notification requiring processing"
        )
    
    def _classify_subagent_message(self, message: Message, content: str) -> ClassificationResult:
        """Classify messages from subagents."""
        # Check if it's a response or status update
        if any(word in content for word in ["completed", "finished", "done", "result"]):
            return ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=15,
                reason=ClassificationReason.SUBAGENT_COMMUNICATION,
                confidence=0.85,
                explanation="Subagent task completion or result"
            )
        
        # Check if it's an error or issue
        if any(word in content for word in ["error", "failed", "exception", "problem"]):
            return ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=12,
                reason=ClassificationReason.SUBAGENT_COMMUNICATION,
                confidence=0.9,
                explanation="Subagent error or issue requiring attention"
            )
        
        # Regular subagent communication
        return ClassificationResult(
            classification=MessageClassification.ACT_NOW,
            priority=20,
            reason=ClassificationReason.SUBAGENT_COMMUNICATION,
            confidence=0.75,
            explanation="Regular subagent communication"
        )
    
    def _classify_user_message(self, message: Message, content: str) -> ClassificationResult:
        """Classify messages from users or external sources."""
        # Check for urgent requests
        urgent_score = sum(1 for keyword in self.URGENT_KEYWORDS if keyword in content)
        if urgent_score >= 2:
            return ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=15,
                reason=ClassificationReason.URGENT_REQUEST,
                confidence=0.8,
                explanation=f"Multiple urgency indicators detected ({urgent_score})"
            )
        
        # Check for time-sensitive content first (before action requests)
        time_keywords = ["today", "tomorrow", "deadline", "due", "schedule", "meeting"]
        if any(keyword in content for keyword in time_keywords):
            return ClassificationResult(
                classification=MessageClassification.DELAY,
                priority=30,
                delay_seconds=1800,  # 30 minutes
                reason=ClassificationReason.TIME_SENSITIVE,
                confidence=0.65,
                explanation="Time-sensitive content, delayed for appropriate timing"
            )
        
        # Check for action requests
        action_score = sum(1 for keyword in self.ACTION_KEYWORDS if keyword in content)
        if action_score >= 2:
            return ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=25,
                reason=ClassificationReason.REQUIRES_ACTION,
                confidence=0.7,
                explanation="Message contains request for action or assistance"
            )
        
        # Check for informational content
        info_score = sum(1 for keyword in self.INFO_KEYWORDS if keyword in content)
        if info_score >= 1:
            return ClassificationResult(
                classification=MessageClassification.ARCHIVE,
                priority=40,
                reason=ClassificationReason.INFORMATION_ONLY,
                confidence=0.6,
                explanation="Informational content for archival"
            )
        
        # Default classification for routine queries
        return ClassificationResult(
            classification=MessageClassification.ACT_NOW,
            priority=35,
            reason=ClassificationReason.ROUTINE_QUERY,
            confidence=0.5,
            explanation="Routine user query requiring standard processing"
        )
    
    def get_classification_history(self) -> List[ClassificationResult]:
        """Get the history of classification results."""
        return self._classification_history.copy()
    
    def clear_history(self) -> None:
        """Clear the classification history."""
        self._classification_history.clear()
        logger.info("Classification history cleared")


class ObserverAgent:
    """
    Observer agent that classifies and routes messages.
    
    The Observer is responsible for analyzing incoming messages,
    classifying them based on content and context, and routing
    them appropriately through the message queue system.
    """
    
    def __init__(self, message_queue: MessageQueue):
        """
        Initialize the Observer agent.
        
        Args:
            message_queue: The message queue for routing classified messages
        """
        self.message_queue = message_queue
        self.classifier = MessageClassifier()
        self._processed_count = 0
        self._classification_stats: Dict[MessageClassification, int] = {
            classification: 0 for classification in MessageClassification
        }
        
        logger.info("ObserverAgent initialized")
    
    async def process_message(self, message: Message) -> ClassificationResult:
        """
        Process and classify a message, then route it appropriately.
        
        Args:
            message: The message to process
            
        Returns:
            ClassificationResult with the classification decision
        """
        # Get tracer and metrics collector
        tracer = get_tracer()
        metrics = get_metrics_collector()
        
        # Start timing for metrics
        start_time = datetime.now(timezone.utc)
        timer_id = metrics.start_timer("message_processing_time_ms")
        
        # Start trace
        trace_id = await tracer.start_trace(
            operation_name="process_message",
            agent_type="Observer",
            inputs={
                "message_id": message.id,
                "message_source": message.source.value,
                "message_length": len(message.content)
            },
            metadata={
                "message_priority": message.priority,
                "message_timestamp": message.timestamp.isoformat()
            }
        )
        
        try:
            # Classify the message
            result = self.classifier.classify_message(message)
            
            # Update message with classification
            message.classification = result.classification
            message.priority = result.priority
            if result.delay_seconds:
                message.delay_seconds = result.delay_seconds
            
            # Add classification metadata
            message.metadata.update({
                "classification_reason": result.reason.value if result.reason else None,
                "classification_confidence": result.confidence,
                "classification_explanation": result.explanation,
                "classified_at": datetime.now(timezone.utc).isoformat(),
                "observer_metadata": result.metadata,
                "classification": {
                    "type": result.classification.value,
                    "confidence": result.confidence,
                    "reason": result.reason.value if result.reason else None
                }
            })
            
            # Route the message based on classification
            await self._route_message(message, result)
            
            # Update statistics
            self._processed_count += 1
            self._classification_stats[result.classification] += 1
            
            # Record metrics
            processing_time = metrics.stop_timer("message_processing_time_ms", timer_id)
            metrics.record_classification_result(
                accuracy=1.0,  # Assume accuracy for now - would be calculated based on feedback
                confidence=result.confidence,
                classification_type=result.classification.value
            )
            
            # End trace successfully
            await tracer.end_trace(
                trace_id=trace_id,
                outputs={
                    "classification": result.classification.value,
                    "priority": result.priority,
                    "confidence": result.confidence,
                    "processing_time_ms": processing_time
                },
                success=True,
                additional_metadata={
                    "classification_reason": result.reason.value if result.reason else None,
                    "routed_successfully": True
                }
            )
            
            logger.info(
                f"Processed message {message.id}: {result.classification.value} "
                f"(priority={result.priority}, confidence={result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            metrics.record_error("classification_error", "Observer", "error")
            
            # End trace with error
            await tracer.end_trace(
                trace_id=trace_id,
                error=str(e),
                success=False
            )
            
            logger.error(f"Error processing message {message.id}: {e}")
            
            # On error, default to safe handling
            error_result = ClassificationResult(
                classification=MessageClassification.ACT_NOW,
                priority=50,
                reason=None,
                confidence=0.0,
                explanation=f"Error during classification: {str(e)}"
            )
            
            message.classification = error_result.classification
            message.priority = error_result.priority
            await self.message_queue.enqueue(message, priority=error_result.priority)
            
            return error_result
    
    async def _route_message(self, message: Message, result: ClassificationResult) -> None:
        """
        Route a classified message to the appropriate destination.
        
        Args:
            message: The classified message
            result: The classification result
        """
        if result.classification == MessageClassification.IGNORE_DELETE:
            # Don't add to queue, effectively deleting the message
            logger.debug(f"Message {message.id} ignored/deleted: {result.explanation}")
            return
        
        elif result.classification == MessageClassification.DELAY:
            # Add to queue with delay metadata
            logger.debug(f"Message {message.id} delayed by {result.delay_seconds} seconds")
            await self.message_queue.enqueue(message, priority=result.priority)
        
        elif result.classification == MessageClassification.ARCHIVE:
            # Add to queue with archive priority
            logger.debug(f"Message {message.id} archived for later processing")
            await self.message_queue.enqueue(message, priority=result.priority)
        
        elif result.classification == MessageClassification.ACT_NOW:
            # Add to queue with high priority
            logger.debug(f"Message {message.id} queued for immediate action")
            await self.message_queue.enqueue(message, priority=result.priority)
        
        else:
            # Fallback: add to queue with default priority
            logger.warning(f"Unknown classification {result.classification}, using default routing")
            await self.message_queue.enqueue(message, priority=result.priority)
    
    async def batch_process_messages(self, messages: List[Message]) -> List[ClassificationResult]:
        """
        Process multiple messages in batch.
        
        Args:
            messages: List of messages to process
            
        Returns:
            List of classification results
        """
        results = []
        for message in messages:
            result = await self.process_message(message)
            results.append(result)
        
        logger.info(f"Batch processed {len(messages)} messages")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics for the Observer agent.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_count": self._processed_count,
            "classification_stats": {
                classification.value: count 
                for classification, count in self._classification_stats.items()
            },
            "classification_history_size": len(self.classifier.get_classification_history()),
            "uptime": datetime.now(timezone.utc).isoformat()
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._processed_count = 0
        self._classification_stats = {
            classification: 0 for classification in MessageClassification
        }
        self.classifier.clear_history()
        logger.info("Observer statistics reset")