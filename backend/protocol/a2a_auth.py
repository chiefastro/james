"""
A2A Protocol Authentication and Validation.

Provides message authentication, signature verification, and security validation
for Agent-to-Agent communications.
"""

import hashlib
import hmac
import json
import secrets
import time
from typing import Dict, Optional, Set, Tuple

from .a2a_models import A2AMessage, A2ASignature


class A2AAuthenticationError(Exception):
    """Raised when A2A message authentication fails."""
    pass


class A2AValidationError(Exception):
    """Raised when A2A message validation fails."""
    pass


class A2AKeyManager:
    """
    Manages authentication keys for A2A protocol communication.
    """
    
    def __init__(self) -> None:
        """Initialize key manager with empty key store."""
        self._keys: Dict[str, str] = {}
        self._key_metadata: Dict[str, Dict[str, any]] = {}
    
    def generate_key(self, agent_id: str, key_length: int = 32) -> str:
        """
        Generate a new authentication key for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            key_length: Length of the key in bytes
            
        Returns:
            The generated key as a hex string
        """
        key = secrets.token_hex(key_length)
        self._keys[agent_id] = key
        self._key_metadata[agent_id] = {
            "created_at": time.time(),
            "last_used": None,
            "usage_count": 0
        }
        return key
    
    def get_key(self, agent_id: str) -> Optional[str]:
        """
        Retrieve authentication key for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            The authentication key or None if not found
        """
        return self._keys.get(agent_id)
    
    def revoke_key(self, agent_id: str) -> bool:
        """
        Revoke authentication key for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if key was revoked, False if not found
        """
        if agent_id in self._keys:
            del self._keys[agent_id]
            del self._key_metadata[agent_id]
            return True
        return False
    
    def update_key_usage(self, agent_id: str) -> None:
        """Update key usage statistics."""
        if agent_id in self._key_metadata:
            metadata = self._key_metadata[agent_id]
            metadata["last_used"] = time.time()
            metadata["usage_count"] += 1
    
    def list_agents(self) -> Set[str]:
        """Get set of all agents with keys."""
        return set(self._keys.keys())


class A2AAuthenticator:
    """
    Handles authentication and signature verification for A2A messages.
    """
    
    def __init__(self, key_manager: A2AKeyManager) -> None:
        """
        Initialize authenticator with key manager.
        
        Args:
            key_manager: Key manager instance for retrieving keys
        """
        self.key_manager = key_manager
        self._nonce_cache: Set[str] = set()
        self._max_nonce_cache_size = 10000
    
    def sign_message(self, message: A2AMessage, sender_key: str) -> A2AMessage:
        """
        Sign an A2A message with HMAC-SHA256.
        
        Args:
            message: The message to sign
            sender_key: The sender's authentication key
            
        Returns:
            The message with signature added
        """
        # Create message content for signing (without signature)
        message_dict = message.to_dict()
        if "signature" in message_dict:
            del message_dict["signature"]
        
        # Create canonical JSON representation
        message_content = json.dumps(message_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate signature
        signature_bytes = hmac.new(
            sender_key.encode('utf-8'),
            message_content.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Create signature object
        signature = A2ASignature(
            algorithm="HMAC-SHA256",
            signature=signature_bytes,
            key_id=message.header.sender_id,
            timestamp=time.time()
        )
        
        # Add signature to message
        message.signature = signature
        return message
    
    def verify_message(self, message: A2AMessage) -> bool:
        """
        Verify the signature of an A2A message.
        
        Args:
            message: The message to verify
            
        Returns:
            True if signature is valid, False otherwise
            
        Raises:
            A2AAuthenticationError: If authentication fails
        """
        if not message.signature:
            raise A2AAuthenticationError("Message has no signature")
        
        # Get sender's key
        sender_key = self.key_manager.get_key(message.header.sender_id)
        if not sender_key:
            raise A2AAuthenticationError(f"No key found for sender: {message.header.sender_id}")
        
        # Create message content for verification (without signature)
        message_dict = message.to_dict()
        del message_dict["signature"]
        
        # Create canonical JSON representation
        message_content = json.dumps(message_dict, sort_keys=True, separators=(',', ':'))
        
        # Calculate expected signature
        expected_signature = hmac.new(
            sender_key.encode('utf-8'),
            message_content.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Verify signature
        is_valid = hmac.compare_digest(expected_signature, message.signature.signature)
        
        if is_valid:
            self.key_manager.update_key_usage(message.header.sender_id)
        
        return is_valid
    
    def _generate_nonce(self) -> str:
        """Generate a unique nonce for replay attack prevention."""
        return secrets.token_hex(16)
    
    def _check_nonce(self, nonce: str) -> bool:
        """
        Check if nonce has been used before.
        
        Args:
            nonce: The nonce to check
            
        Returns:
            True if nonce is new, False if already used
        """
        if nonce in self._nonce_cache:
            return False
        
        # Add to cache and manage size
        self._nonce_cache.add(nonce)
        if len(self._nonce_cache) > self._max_nonce_cache_size:
            # Remove oldest nonces (simple FIFO)
            oldest_nonces = list(self._nonce_cache)[:1000]
            for old_nonce in oldest_nonces:
                self._nonce_cache.discard(old_nonce)
        
        return True


class A2AValidator:
    """
    Validates A2A messages for protocol compliance and security.
    """
    
    def __init__(self, max_message_size: int = 1024 * 1024) -> None:
        """
        Initialize validator with configuration.
        
        Args:
            max_message_size: Maximum allowed message size in bytes
        """
        self.max_message_size = max_message_size
        self._rate_limits: Dict[str, Dict[str, any]] = {}
        self._blocked_agents: Set[str] = set()
    
    def validate_message(self, message: A2AMessage) -> Tuple[bool, Optional[str]]:
        """
        Validate an A2A message for protocol compliance.
        
        Args:
            message: The message to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if sender is blocked
            if message.header.sender_id in self._blocked_agents:
                return False, f"Sender {message.header.sender_id} is blocked"
            
            # Check message expiration
            if message.header.is_expired():
                return False, "Message has expired"
            
            # Check message size
            message_size = len(message.to_json().encode('utf-8'))
            if message_size > self.max_message_size:
                return False, f"Message size {message_size} exceeds limit {self.max_message_size}"
            
            # Check protocol version
            if message.header.protocol_version != "1.0":
                return False, f"Unsupported protocol version: {message.header.protocol_version}"
            
            # Check required fields based on message type
            validation_result = self._validate_message_type_specific(message)
            if not validation_result[0]:
                return validation_result
            
            # Check rate limits
            if not self._check_rate_limit(message.header.sender_id):
                return False, f"Rate limit exceeded for sender: {message.header.sender_id}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_message_type_specific(self, message: A2AMessage) -> Tuple[bool, Optional[str]]:
        """Validate message based on its specific type."""
        from .a2a_models import A2AMessageType
        
        msg_type = message.header.message_type
        payload = message.payload
        
        if msg_type == A2AMessageType.TASK_REQUEST:
            if not payload.task_id:
                return False, "Task request must have task_id"
            if not payload.task_description:
                return False, "Task request must have task_description"
        
        elif msg_type == A2AMessageType.TASK_RESPONSE:
            if not payload.task_id:
                return False, "Task response must have task_id"
            if not message.header.correlation_id:
                return False, "Task response must have correlation_id"
        
        elif msg_type == A2AMessageType.ERROR_REPORT:
            if not payload.error_message:
                return False, "Error report must have error_message"
        
        elif msg_type == A2AMessageType.CAPABILITY_QUERY:
            if not payload.capabilities_requested:
                return False, "Capability query must specify requested capabilities"
        
        elif msg_type == A2AMessageType.CAPABILITY_RESPONSE:
            if not payload.capabilities_offered:
                return False, "Capability response must specify offered capabilities"
        
        return True, None
    
    def _check_rate_limit(self, agent_id: str, max_messages_per_minute: int = 60) -> bool:
        """
        Check if agent is within rate limits.
        
        Args:
            agent_id: The agent to check
            max_messages_per_minute: Maximum messages per minute allowed
            
        Returns:
            True if within limits, False otherwise
        """
        current_time = time.time()
        
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = {
                "messages": [],
                "last_reset": current_time
            }
        
        agent_limits = self._rate_limits[agent_id]
        
        # Clean old messages (older than 1 minute)
        agent_limits["messages"] = [
            msg_time for msg_time in agent_limits["messages"]
            if current_time - msg_time < 60
        ]
        
        # Check if under limit
        if len(agent_limits["messages"]) >= max_messages_per_minute:
            return False
        
        # Add current message
        agent_limits["messages"].append(current_time)
        return True
    
    def block_agent(self, agent_id: str) -> None:
        """Block an agent from sending messages."""
        self._blocked_agents.add(agent_id)
    
    def unblock_agent(self, agent_id: str) -> None:
        """Unblock an agent."""
        self._blocked_agents.discard(agent_id)
    
    def is_agent_blocked(self, agent_id: str) -> bool:
        """Check if an agent is blocked."""
        return agent_id in self._blocked_agents