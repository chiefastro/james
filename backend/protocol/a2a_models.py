"""
A2A Protocol Data Models.

Defines the message format and data structures for Agent-to-Agent communication,
including authentication, validation, and protocol compliance.
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class A2AMessageType(Enum):
    """Types of A2A protocol messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"


class A2AMessageStatus(Enum):
    """Status of A2A messages."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class A2AHeader:
    """
    A2A message header containing metadata and authentication information.
    """
    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: A2AMessageType = A2AMessageType.TASK_REQUEST
    timestamp: float = field(default_factory=time.time)
    protocol_version: str = "1.0"
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    def __post_init__(self) -> None:
        """Validate header data after initialization."""
        if not self.sender_id.strip():
            raise ValueError("Sender ID cannot be empty")
        if not self.recipient_id.strip():
            raise ValueError("Recipient ID cannot be empty")
        if self.ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
    
    def is_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        return time.time() > (self.timestamp + self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert header to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "protocol_version": self.protocol_version,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds
        }


@dataclass
class A2APayload:
    """
    A2A message payload containing the actual data and task information.
    """
    task_id: Optional[str] = None
    task_description: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: A2AMessageStatus = A2AMessageStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities_requested: List[str] = field(default_factory=list)
    capabilities_offered: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "capabilities_requested": self.capabilities_requested,
            "capabilities_offered": self.capabilities_offered
        }


@dataclass
class A2ASignature:
    """
    A2A message signature for authentication and integrity verification.
    """
    algorithm: str = "HMAC-SHA256"
    signature: str = ""
    key_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "signature": self.signature,
            "key_id": self.key_id,
            "timestamp": self.timestamp
        }


@dataclass
class A2AMessage:
    """
    Complete A2A protocol message with header, payload, and signature.
    """
    header: A2AHeader
    payload: A2APayload
    signature: Optional[A2ASignature] = None
    
    def __post_init__(self) -> None:
        """Validate message structure after initialization."""
        if not isinstance(self.header, A2AHeader):
            raise ValueError("Header must be an A2AHeader instance")
        if not isinstance(self.payload, A2APayload):
            raise ValueError("Payload must be an A2APayload instance")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete message to dictionary for serialization."""
        message_dict = {
            "header": self.header.to_dict(),
            "payload": self.payload.to_dict()
        }
        if self.signature:
            message_dict["signature"] = self.signature.to_dict()
        return message_dict
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create A2AMessage from dictionary."""
        # Parse header
        header_data = data.get("header", {})
        header = A2AHeader(
            message_id=header_data.get("message_id", str(uuid4())),
            sender_id=header_data.get("sender_id", ""),
            recipient_id=header_data.get("recipient_id", ""),
            message_type=A2AMessageType(header_data.get("message_type", "task_request")),
            timestamp=header_data.get("timestamp", time.time()),
            protocol_version=header_data.get("protocol_version", "1.0"),
            correlation_id=header_data.get("correlation_id"),
            reply_to=header_data.get("reply_to"),
            ttl_seconds=header_data.get("ttl_seconds", 300)
        )
        
        # Parse payload
        payload_data = data.get("payload", {})
        payload = A2APayload(
            task_id=payload_data.get("task_id"),
            task_description=payload_data.get("task_description"),
            input_data=payload_data.get("input_data", {}),
            output_data=payload_data.get("output_data", {}),
            status=A2AMessageStatus(payload_data.get("status", "pending")),
            error_message=payload_data.get("error_message"),
            metadata=payload_data.get("metadata", {}),
            capabilities_requested=payload_data.get("capabilities_requested", []),
            capabilities_offered=payload_data.get("capabilities_offered", [])
        )
        
        # Parse signature if present
        signature = None
        if "signature" in data:
            sig_data = data["signature"]
            signature = A2ASignature(
                algorithm=sig_data.get("algorithm", "HMAC-SHA256"),
                signature=sig_data.get("signature", ""),
                key_id=sig_data.get("key_id", ""),
                timestamp=sig_data.get("timestamp", time.time())
            )
        
        return cls(header=header, payload=payload, signature=signature)
    
    @classmethod
    def from_json(cls, json_str: str) -> "A2AMessage":
        """Create A2AMessage from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_valid(self) -> bool:
        """Check if message is valid and not expired."""
        try:
            # Check if message is expired
            if self.header.is_expired():
                return False
            
            # Check required fields based on message type
            if self.header.message_type in [A2AMessageType.TASK_REQUEST, A2AMessageType.TASK_RESPONSE]:
                if not self.payload.task_id:
                    return False
            
            return True
        except Exception:
            return False
    
    def create_reply(self, sender_id: str, payload: A2APayload) -> "A2AMessage":
        """Create a reply message to this message."""
        reply_header = A2AHeader(
            sender_id=sender_id,
            recipient_id=self.header.sender_id,
            message_type=A2AMessageType.TASK_RESPONSE,
            correlation_id=self.header.message_id,
            reply_to=self.header.message_id
        )
        
        return A2AMessage(header=reply_header, payload=payload)


# Protocol constants
A2A_PROTOCOL_VERSION = "1.0"
A2A_MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
A2A_DEFAULT_TTL = 300  # 5 minutes
A2A_MAX_TTL = 3600  # 1 hour
A2A_HEARTBEAT_INTERVAL = 30  # 30 seconds