"""
Message queue interaction tool for internal communication.
"""

import time
from typing import Dict, List, Optional, Any

from .base import BaseTool, ToolResult
from ..models.core import Message, MessageSource, MessageClassification
from ..queue.message_queue import MessageQueue, PriorityCalculator
from ..queue.exceptions import QueueError, QueueFullError, QueueEmptyError


class MessageQueueTool(BaseTool):
    """
    Tool for James to interact with the internal message queue system.
    
    Provides capabilities to send messages, check queue status,
    and manage message processing.
    """
    
    def __init__(self, message_queue: Optional[MessageQueue] = None):
        """Initialize the message queue tool."""
        super().__init__("MessageQueue")
        self.message_queue = message_queue or MessageQueue()
        self.priority_calculator = PriorityCalculator()
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute message queue operation.
        
        Args:
            action (str): Action to perform - 'send', 'peek', 'status', 'clear'
            content (str): Message content (for 'send' action)
            source (str): Message source (for 'send' action)
            priority (int): Message priority (for 'send' action)
            classification (str): Message classification (for 'send' action)
            metadata (dict): Additional message metadata (for 'send' action)
            
        Returns:
            ToolResult with operation outcome
        """
        start_time = time.time()
        
        # Validate required parameters
        error = self._validate_required_params(kwargs, ['action'])
        if error:
            return self._create_error_result(error)
        
        action = kwargs['action'].lower()
        
        try:
            if action == 'send':
                return await self._send_message(kwargs, start_time)
            elif action == 'peek':
                return await self._peek_message(start_time)
            elif action == 'status':
                return await self._get_queue_status(start_time)
            elif action == 'clear':
                return await self._clear_queue(start_time)
            elif action == 'list':
                return await self._list_messages(kwargs, start_time)
            else:
                return self._create_error_result(f"Unsupported action: {action}")
                
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def _send_message(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Send a message to the queue."""
        # Validate required parameters for send action
        error = self._validate_required_params(kwargs, ['content'])
        if error:
            return self._create_error_result(error)
        
        content = kwargs['content']
        source_str = kwargs.get('source', 'system')
        priority = kwargs.get('priority')
        classification_str = kwargs.get('classification')
        metadata = kwargs.get('metadata', {})
        
        try:
            # Parse source
            try:
                source = MessageSource(source_str.lower())
            except ValueError:
                return self._create_error_result(f"Invalid message source: {source_str}")
            
            # Parse classification if provided
            classification = None
            if classification_str:
                try:
                    classification = MessageClassification(classification_str.lower())
                except ValueError:
                    return self._create_error_result(f"Invalid classification: {classification_str}")
            
            # Create message
            message = Message(
                content=content,
                source=source,
                priority=priority or 0,
                classification=classification,
                metadata=metadata
            )
            
            # Calculate priority if not provided
            calculated_priority = priority if priority is not None else self.priority_calculator.calculate_priority(message)
            
            # Send to queue
            await self.message_queue.enqueue(message, calculated_priority)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Sent message to queue: {message.id}")
            
            return self._create_success_result(
                data={
                    'message_id': message.id,
                    'content': content,
                    'source': source.value,
                    'priority': calculated_priority,
                    'classification': classification.value if classification else None,
                    'timestamp': message.timestamp.isoformat()
                },
                metadata={
                    'execution_time': execution_time,
                    'queue_size_after': await self.message_queue.size()
                }
            )
            
        except QueueFullError as e:
            return self._create_error_result(f"Queue is full: {e}")
        except QueueError as e:
            return self._create_error_result(f"Queue error: {e}")
    
    async def _peek_message(self, start_time: float) -> ToolResult:
        """Peek at the next message without removing it."""
        try:
            message = await self.message_queue.peek_next()
            execution_time = time.time() - start_time
            
            if message:
                return self._create_success_result(
                    data={
                        'message_id': message.id,
                        'content': message.content,
                        'source': message.source.value,
                        'priority': message.priority,
                        'classification': message.classification.value if message.classification else None,
                        'timestamp': message.timestamp.isoformat(),
                        'metadata': message.metadata
                    },
                    metadata={
                        'execution_time': execution_time,
                        'has_message': True
                    }
                )
            else:
                return self._create_success_result(
                    data={'message': None},
                    metadata={
                        'execution_time': execution_time,
                        'has_message': False
                    }
                )
                
        except Exception as e:
            return self._create_error_result(f"Error peeking message: {e}")
    
    async def _get_queue_status(self, start_time: float) -> ToolResult:
        """Get queue status and statistics."""
        try:
            stats = await self.message_queue.get_stats()
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data=stats,
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error getting queue status: {e}")
    
    async def _clear_queue(self, start_time: float) -> ToolResult:
        """Clear all messages from the queue."""
        try:
            cleared_count = await self.message_queue.clear()
            execution_time = time.time() - start_time
            
            self.logger.info(f"Cleared {cleared_count} messages from queue")
            
            return self._create_success_result(
                data={
                    'cleared_count': cleared_count,
                    'queue_size': 0
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error clearing queue: {e}")
    
    async def _list_messages(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """List messages in a priority range."""
        min_priority = kwargs.get('min_priority', 0)
        max_priority = kwargs.get('max_priority', 100)
        
        try:
            messages = await self.message_queue.get_messages_by_priority(min_priority, max_priority)
            execution_time = time.time() - start_time
            
            message_data = []
            for message in messages:
                message_data.append({
                    'message_id': message.id,
                    'content': message.content[:100] + '...' if len(message.content) > 100 else message.content,
                    'source': message.source.value,
                    'priority': message.priority,
                    'classification': message.classification.value if message.classification else None,
                    'timestamp': message.timestamp.isoformat()
                })
            
            return self._create_success_result(
                data={
                    'messages': message_data,
                    'count': len(message_data),
                    'priority_range': {
                        'min': min_priority,
                        'max': max_priority
                    }
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Error listing messages: {e}")
    
    async def send_internal_message(self, content: str, priority: Optional[int] = None,
                                   classification: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Convenience method to send an internal system message.
        
        Args:
            content: Message content
            priority: Message priority
            classification: Message classification
            metadata: Additional metadata
            
        Returns:
            ToolResult with send outcome
        """
        return await self.execute(
            action='send',
            content=content,
            source='system',
            priority=priority,
            classification=classification,
            metadata=metadata or {}
        )
    
    async def send_subagent_message(self, content: str, subagent_id: str,
                                   priority: Optional[int] = None) -> ToolResult:
        """
        Send a message from a subagent.
        
        Args:
            content: Message content
            subagent_id: ID of the sending subagent
            priority: Message priority
            
        Returns:
            ToolResult with send outcome
        """
        metadata = {'subagent_id': subagent_id}
        
        return await self.execute(
            action='send',
            content=content,
            source='subagent',
            priority=priority,
            metadata=metadata
        )
    
    async def check_queue_health(self) -> ToolResult:
        """
        Check the health of the message queue.
        
        Returns:
            ToolResult with health status
        """
        try:
            stats = await self.message_queue.get_stats()
            
            # Determine health status
            health_status = "healthy"
            issues = []
            
            # Check if queue is too full
            if stats.get('is_full', False):
                health_status = "critical"
                issues.append("Queue is at maximum capacity")
            elif stats.get('size', 0) > (stats.get('max_size', 1000) * 0.8):
                health_status = "warning"
                issues.append("Queue is over 80% capacity")
            
            # Check for very old messages (if we have timestamp info)
            if 'avg_priority' in stats and stats['avg_priority'] > 50:
                health_status = "warning"
                issues.append("High average priority indicates many low-priority messages")
            
            return self._create_success_result(
                data={
                    'health_status': health_status,
                    'issues': issues,
                    'stats': stats,
                    'recommendations': self._get_health_recommendations(stats, issues)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Error checking queue health: {e}")
    
    def _get_health_recommendations(self, stats: Dict[str, Any], issues: List[str]) -> List[str]:
        """Get recommendations based on queue health."""
        recommendations = []
        
        if "Queue is at maximum capacity" in issues:
            recommendations.append("Consider increasing queue size or processing messages faster")
        
        if "Queue is over 80% capacity" in issues:
            recommendations.append("Monitor queue size and consider processing optimization")
        
        if "High average priority indicates many low-priority messages" in issues:
            recommendations.append("Consider clearing low-priority messages or adjusting priority calculation")
        
        if not issues:
            recommendations.append("Queue is operating normally")
        
        return recommendations