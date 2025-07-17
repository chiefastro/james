"""
External Input Processing Subagent.

This subagent handles processing of external inputs from various sources,
including validation, normalization, and routing to appropriate handlers.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from urllib.parse import urlparse

from .base import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)


class ExternalInputSubagent(BaseSubagent):
    """
    Subagent that handles external input processing tasks.
    
    Capabilities:
    - Input validation and sanitization
    - Format normalization
    - Content classification
    - Source verification
    - Data extraction
    - Routing recommendations
    """
    
    def __init__(self, subagent_id: Optional[str] = None):
        """Initialize the external input processing subagent."""
        super().__init__(
            subagent_id=subagent_id or f"external_input_agent_{uuid4().hex[:8]}",
            name="External Input Agent",
            description="Processes and validates external inputs from various sources",
            capabilities=[
                "input_validation",
                "content_sanitization",
                "format_normalization",
                "content_classification",
                "source_verification",
                "data_extraction",
                "routing_recommendation",
                "threat_detection"
            ]
        )
        self._processing_history: List[Dict[str, Any]] = []
        self._known_sources: Dict[str, Dict[str, Any]] = {}
        self._threat_patterns: List[str] = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'import\s+os',  # OS imports
            r'__import__',  # Dynamic imports
        ]
    
    async def process_task(self, task_id: str, task_description: str, input_data: Dict[str, Any]) -> SubagentResult:
        """
        Process an external input processing task.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the processing task
            input_data: External input data to process
            
        Returns:
            SubagentResult with processed input analysis
        """
        try:
            self.logger.info(f"Processing external input task: {task_id}")
            
            # Determine the type of processing requested
            processing_type = input_data.get("processing_type", "full_analysis")
            
            if processing_type == "validation":
                result = await self._validate_input(input_data)
            elif processing_type == "sanitization":
                result = await self._sanitize_input(input_data)
            elif processing_type == "classification":
                result = await self._classify_content(input_data)
            elif processing_type == "extraction":
                result = await self._extract_data(input_data)
            elif processing_type == "source_verification":
                result = await self._verify_source(input_data)
            elif processing_type == "threat_detection":
                result = await self._detect_threats(input_data)
            else:
                result = await self._full_analysis(input_data)
            
            # Store processing in history
            if result.success:
                processing_record = {
                    "task_id": task_id,
                    "processing_type": processing_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_summary": self._summarize_input(input_data),
                    "threats_detected": result.data.get("threats_detected", 0) if isinstance(result.data, dict) else 0
                }
                self._processing_history.append(processing_record)
                
                # Keep only last 100 processing records
                if len(self._processing_history) > 100:
                    self._processing_history = self._processing_history[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in external input processing task {task_id}: {e}")
            return self._create_error_result(f"External input processing failed: {str(e)}")
    
    async def _validate_input(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Validate external input for safety and correctness."""
        content = input_data.get("content", "")
        source = input_data.get("source", "unknown")
        input_type = input_data.get("type", "text")
        
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "warnings": [],
            "sanitization_needed": False,
            "risk_level": "low"
        }
        
        # Basic validation checks
        if not content or not isinstance(content, str):
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Content is empty or not a string")
        
        # Length validation
        if len(content) > 100000:  # 100KB limit
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Content exceeds maximum length limit")
        
        # Encoding validation
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            validation_result["warnings"].append("Content contains non-UTF-8 characters")
        
        # Threat pattern detection
        threats_found = []
        for pattern in self._threat_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats_found.append(pattern)
        
        if threats_found:
            validation_result["sanitization_needed"] = True
            validation_result["risk_level"] = "high"
            validation_result["warnings"].append(f"Potential threats detected: {len(threats_found)} patterns")
        
        # URL validation if content contains URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        suspicious_urls = []
        for url in urls:
            parsed = urlparse(url)
            if not parsed.netloc or parsed.scheme not in ['http', 'https']:
                suspicious_urls.append(url)
        
        if suspicious_urls:
            validation_result["warnings"].append(f"Suspicious URLs found: {len(suspicious_urls)}")
            validation_result["risk_level"] = "medium"
        
        return self._create_success_result(
            data=validation_result,
            metadata={
                "processing_type": "validation",
                "content_length": len(content),
                "source": source,
                "input_type": input_type
            }
        )
    
    async def _sanitize_input(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Sanitize external input to remove potential threats."""
        content = input_data.get("content", "")
        sanitization_level = input_data.get("sanitization_level", "standard")
        
        sanitized_content = content
        sanitization_actions = []
        
        # Remove script tags and JavaScript
        if re.search(r'<script[^>]*>.*?</script>', sanitized_content, re.IGNORECASE | re.DOTALL):
            sanitized_content = re.sub(r'<script[^>]*>.*?</script>', '', sanitized_content, flags=re.IGNORECASE | re.DOTALL)
            sanitization_actions.append("Removed script tags")
        
        # Remove JavaScript URLs
        if 'javascript:' in sanitized_content.lower():
            sanitized_content = re.sub(r'javascript:[^"\'>\s]*', '', sanitized_content, flags=re.IGNORECASE)
            sanitization_actions.append("Removed JavaScript URLs")
        
        # Remove event handlers
        event_handlers = re.findall(r'on\w+\s*=\s*["\'][^"\']*["\']', sanitized_content, re.IGNORECASE)
        if event_handlers:
            sanitized_content = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized_content, flags=re.IGNORECASE)
            sanitization_actions.append(f"Removed {len(event_handlers)} event handlers")
        
        # Advanced sanitization for high level
        if sanitization_level == "strict":
            # Remove all HTML tags
            sanitized_content = re.sub(r'<[^>]+>', '', sanitized_content)
            sanitization_actions.append("Removed all HTML tags")
            
            # Remove special characters that could be used for injection
            sanitized_content = re.sub(r'[<>&"\']', '', sanitized_content)
            sanitization_actions.append("Removed special characters")
        
        # Normalize whitespace
        sanitized_content = re.sub(r'\s+', ' ', sanitized_content).strip()
        
        sanitization_result = {
            "original_content": content,
            "sanitized_content": sanitized_content,
            "sanitization_actions": sanitization_actions,
            "content_changed": content != sanitized_content,
            "safety_level": "safe" if sanitization_actions else "already_safe"
        }
        
        return self._create_success_result(
            data=sanitization_result,
            metadata={
                "processing_type": "sanitization",
                "original_length": len(content),
                "sanitized_length": len(sanitized_content),
                "actions_taken": len(sanitization_actions)
            }
        )
    
    async def _classify_content(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Classify the content type and intent of external input."""
        content = input_data.get("content", "")
        source = input_data.get("source", "unknown")
        
        classification = {
            "content_type": "unknown",
            "intent": "unknown",
            "language": "unknown",
            "sentiment": "neutral",
            "topics": [],
            "confidence": 0.0
        }
        
        # Basic content type detection
        if re.search(r'<[^>]+>', content):
            classification["content_type"] = "html"
        elif content.strip().startswith('{') and content.strip().endswith('}'):
            try:
                json.loads(content)
                classification["content_type"] = "json"
            except:
                classification["content_type"] = "text"
        elif content.strip().startswith('[') and content.strip().endswith(']'):
            try:
                json.loads(content)
                classification["content_type"] = "json_array"
            except:
                classification["content_type"] = "text"
        else:
            classification["content_type"] = "text"
        
        # Intent detection (simplified)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in content.lower() for word in question_words):
            classification["intent"] = "question"
        elif any(word in content.lower() for word in ['please', 'can you', 'could you', 'help']):
            classification["intent"] = "request"
        elif any(word in content.lower() for word in ['create', 'build', 'make', 'generate']):
            classification["intent"] = "creation"
        elif any(word in content.lower() for word in ['analyze', 'review', 'check', 'examine']):
            classification["intent"] = "analysis"
        else:
            classification["intent"] = "statement"
        
        # Simple language detection
        if re.search(r'[а-яё]', content.lower()):
            classification["language"] = "russian"
        elif re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', content.lower()):
            classification["language"] = "european"
        elif re.search(r'[一-龯]', content):
            classification["language"] = "chinese"
        elif re.search(r'[ひらがなカタカナ]', content):
            classification["language"] = "japanese"
        else:
            classification["language"] = "english"
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'problem']
        
        positive_count = sum(1 for word in positive_words if word in content.lower())
        negative_count = sum(1 for word in negative_words if word in content.lower())
        
        if positive_count > negative_count:
            classification["sentiment"] = "positive"
        elif negative_count > positive_count:
            classification["sentiment"] = "negative"
        else:
            classification["sentiment"] = "neutral"
        
        # Topic extraction (simplified keyword extraction)
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 most frequent words as topics
        classification["topics"] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Set confidence based on available indicators
        indicators = sum([
            1 if classification["content_type"] != "unknown" else 0,
            1 if classification["intent"] != "unknown" else 0,
            1 if classification["language"] != "unknown" else 0,
            1 if classification["sentiment"] != "neutral" else 0,
            1 if classification["topics"] else 0
        ])
        classification["confidence"] = indicators / 5.0
        
        return self._create_success_result(
            data=classification,
            metadata={
                "processing_type": "classification",
                "content_length": len(content),
                "word_count": len(words),
                "source": source
            }
        )
    
    async def _extract_data(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Extract structured data from external input."""
        content = input_data.get("content", "")
        extraction_targets = input_data.get("targets", ["urls", "emails", "dates", "numbers"])
        
        extracted_data = {}
        
        # Extract URLs
        if "urls" in extraction_targets:
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            # Clean up URLs by removing trailing punctuation
            cleaned_urls = []
            for url in urls:
                # Remove trailing punctuation
                url = re.sub(r'[.,;!?]+$', '', url)
                cleaned_urls.append(url)
            extracted_data["urls"] = list(set(cleaned_urls))  # Remove duplicates
        
        # Extract email addresses
        if "emails" in extraction_targets:
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            extracted_data["emails"] = list(set(emails))
        
        # Extract dates (simple patterns)
        if "dates" in extraction_targets:
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, content))
            extracted_data["dates"] = list(set(dates))
        
        # Extract numbers
        if "numbers" in extraction_targets:
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
            extracted_data["numbers"] = [float(n) if '.' in n else int(n) for n in numbers]
        
        # Extract phone numbers
        if "phones" in extraction_targets:
            phone_patterns = [
                r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
                r'\b\d{10}\b',  # XXXXXXXXXX
            ]
            phones = []
            for pattern in phone_patterns:
                phones.extend(re.findall(pattern, content))
            extracted_data["phones"] = list(set(phones))
        
        # Extract mentions (@ symbols)
        if "mentions" in extraction_targets:
            mentions = re.findall(r'@\w+', content)
            extracted_data["mentions"] = list(set(mentions))
        
        # Extract hashtags
        if "hashtags" in extraction_targets:
            hashtags = re.findall(r'#\w+', content)
            extracted_data["hashtags"] = list(set(hashtags))
        
        # Calculate extraction statistics
        total_extracted = sum(len(v) if isinstance(v, list) else 1 for v in extracted_data.values())
        
        return self._create_success_result(
            data=extracted_data,
            metadata={
                "processing_type": "extraction",
                "targets_requested": extraction_targets,
                "total_extracted": total_extracted,
                "extraction_types": list(extracted_data.keys())
            }
        )
    
    async def _verify_source(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Verify the source of external input."""
        source = input_data.get("source", "unknown")
        source_metadata = input_data.get("source_metadata", {})
        content = input_data.get("content", "")
        
        verification_result = {
            "source": source,
            "is_known": False,
            "trust_level": "unknown",
            "verification_status": "unverified",
            "risk_indicators": [],
            "recommendations": []
        }
        
        # Check if source is in known sources
        if source in self._known_sources:
            verification_result["is_known"] = True
            known_source = self._known_sources[source]
            verification_result["trust_level"] = known_source.get("trust_level", "unknown")
            verification_result["verification_status"] = "known_source"
        
        # Analyze source metadata for risk indicators
        if source_metadata:
            # Check for suspicious patterns in metadata
            if source_metadata.get("user_agent", "").lower() in ["", "unknown", "bot"]:
                verification_result["risk_indicators"].append("Suspicious or missing user agent")
            
            if source_metadata.get("ip_address", "").startswith("127.") or source_metadata.get("ip_address", "") == "localhost":
                verification_result["risk_indicators"].append("Local/loopback IP address")
            
            if source_metadata.get("referrer", "") == "":
                verification_result["risk_indicators"].append("Missing referrer information")
        
        # Analyze content for source verification clues
        if "http" in content.lower():
            urls_in_content = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            if urls_in_content:
                verification_result["risk_indicators"].append(f"Contains {len(urls_in_content)} URLs")
        
        # Generate recommendations based on verification
        if verification_result["risk_indicators"]:
            verification_result["recommendations"].append("Perform additional validation due to risk indicators")
        
        if not verification_result["is_known"]:
            verification_result["recommendations"].append("Consider adding source to known sources list")
        
        if verification_result["trust_level"] == "unknown":
            verification_result["recommendations"].append("Establish trust level for this source")
        
        return self._create_success_result(
            data=verification_result,
            metadata={
                "processing_type": "source_verification",
                "source": source,
                "risk_indicators_count": len(verification_result["risk_indicators"])
            }
        )
    
    async def _detect_threats(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Detect potential security threats in external input."""
        content = input_data.get("content", "")
        threat_types = input_data.get("threat_types", ["injection", "xss", "malware", "phishing"])
        
        threat_detection = {
            "threats_detected": [],
            "risk_level": "low",
            "threat_score": 0,
            "recommendations": []
        }
        
        threat_score = 0
        
        # SQL Injection detection
        if "injection" in threat_types:
            sql_patterns = [
                r"union\s+select", r"drop\s+table", r"insert\s+into", r"delete\s+from",
                r"update\s+set", r"exec\s*\(", r"execute\s*\(", r"sp_executesql"
            ]
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threat_detection["threats_detected"].append({
                        "type": "sql_injection",
                        "pattern": pattern,
                        "severity": "high"
                    })
                    threat_score += 10
        
        # XSS detection
        if "xss" in threat_types:
            xss_patterns = [
                r"<script[^>]*>", r"javascript:", r"on\w+\s*=", r"eval\s*\(",
                r"document\.cookie", r"window\.location", r"innerHTML"
            ]
            for pattern in xss_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threat_detection["threats_detected"].append({
                        "type": "xss",
                        "pattern": pattern,
                        "severity": "high"
                    })
                    threat_score += 8
        
        # Command injection detection
        if "injection" in threat_types:
            cmd_patterns = [
                r";\s*rm\s+", r";\s*cat\s+", r";\s*ls\s+", r";\s*pwd",
                r"\|\s*nc\s+", r"&&\s*curl", r">\s*/dev/null"
            ]
            for pattern in cmd_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threat_detection["threats_detected"].append({
                        "type": "command_injection",
                        "pattern": pattern,
                        "severity": "critical"
                    })
                    threat_score += 15
        
        # Malware indicators
        if "malware" in threat_types:
            malware_patterns = [
                r"base64_decode", r"eval\s*\(", r"exec\s*\(", r"system\s*\(",
                r"shell_exec", r"passthru", r"file_get_contents"
            ]
            for pattern in malware_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threat_detection["threats_detected"].append({
                        "type": "malware_indicator",
                        "pattern": pattern,
                        "severity": "high"
                    })
                    threat_score += 12
        
        # Phishing indicators
        if "phishing" in threat_types:
            phishing_keywords = [
                "verify your account", "click here immediately", "urgent action required",
                "suspended account", "confirm your identity", "update payment"
            ]
            for keyword in phishing_keywords:
                if keyword.lower() in content.lower():
                    threat_detection["threats_detected"].append({
                        "type": "phishing_indicator",
                        "pattern": keyword,
                        "severity": "medium"
                    })
                    threat_score += 5
        
        # Determine risk level based on threat score
        if threat_score >= 20:
            threat_detection["risk_level"] = "critical"
        elif threat_score >= 10:
            threat_detection["risk_level"] = "high"
        elif threat_score >= 5:
            threat_detection["risk_level"] = "medium"
        else:
            threat_detection["risk_level"] = "low"
        
        threat_detection["threat_score"] = threat_score
        
        # Generate recommendations
        if threat_detection["threats_detected"]:
            threat_detection["recommendations"].append("Block or quarantine this input")
            threat_detection["recommendations"].append("Perform deep security analysis")
            threat_detection["recommendations"].append("Log security incident")
        else:
            threat_detection["recommendations"].append("Input appears safe for processing")
        
        return self._create_success_result(
            data=threat_detection,
            metadata={
                "processing_type": "threat_detection",
                "threats_found": len(threat_detection["threats_detected"]),
                "threat_score": threat_score,
                "risk_level": threat_detection["risk_level"]
            }
        )
    
    async def _full_analysis(self, input_data: Dict[str, Any]) -> SubagentResult:
        """Perform full analysis of external input."""
        # Run all analysis types
        validation_result = await self._validate_input(input_data)
        classification_result = await self._classify_content(input_data)
        extraction_result = await self._extract_data(input_data)
        threat_result = await self._detect_threats(input_data)
        source_result = await self._verify_source(input_data)
        
        # Combine all results
        full_analysis = {
            "validation": validation_result.data if validation_result.success else None,
            "classification": classification_result.data if classification_result.success else None,
            "extraction": extraction_result.data if extraction_result.success else None,
            "threats": threat_result.data if threat_result.success else None,
            "source_verification": source_result.data if source_result.success else None,
            "overall_assessment": {
                "safe_to_process": True,
                "confidence": 0.0,
                "recommendations": []
            }
        }
        
        # Calculate overall safety assessment
        safety_factors = []
        
        if validation_result.success and validation_result.data:
            safety_factors.append(validation_result.data.get("is_valid", False))
        
        if threat_result.success and threat_result.data:
            threat_level = threat_result.data.get("risk_level", "low")
            safety_factors.append(threat_level in ["low", "medium"])
        
        if source_result.success and source_result.data:
            risk_indicators = len(source_result.data.get("risk_indicators", []))
            safety_factors.append(risk_indicators < 3)
        
        # Overall safety decision
        full_analysis["overall_assessment"]["safe_to_process"] = all(safety_factors) if safety_factors else False
        full_analysis["overall_assessment"]["confidence"] = sum(safety_factors) / len(safety_factors) if safety_factors else 0.0
        
        # Generate overall recommendations
        if not full_analysis["overall_assessment"]["safe_to_process"]:
            full_analysis["overall_assessment"]["recommendations"].append("Input requires sanitization before processing")
        
        if full_analysis["overall_assessment"]["confidence"] < 0.7:
            full_analysis["overall_assessment"]["recommendations"].append("Consider manual review due to low confidence")
        
        full_analysis["overall_assessment"]["recommendations"].append("Monitor processing results for anomalies")
        
        return self._create_success_result(
            data=full_analysis,
            metadata={
                "processing_type": "full_analysis",
                "analysis_components": 5,
                "overall_safe": full_analysis["overall_assessment"]["safe_to_process"],
                "confidence": full_analysis["overall_assessment"]["confidence"]
            }
        )
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> str:
        """Create a brief summary of input data."""
        content = input_data.get("content", "")
        source = input_data.get("source", "unknown")
        return f"Input from {source}, {len(content)} chars"
    
    def add_known_source(self, source: str, trust_level: str, metadata: Dict[str, Any] = None):
        """Add a source to the known sources list."""
        self._known_sources[source] = {
            "trust_level": trust_level,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for input data."""
        return {
            "type": "object",
            "properties": {
                "processing_type": {
                    "type": "string",
                    "enum": [
                        "validation",
                        "sanitization",
                        "classification",
                        "extraction",
                        "source_verification",
                        "threat_detection",
                        "full_analysis"
                    ],
                    "description": "Type of processing to perform"
                },
                "content": {
                    "type": "string",
                    "description": "The external input content to process"
                },
                "source": {
                    "type": "string",
                    "description": "Source of the input"
                },
                "type": {
                    "type": "string",
                    "description": "Type of input (text, html, json, etc.)"
                },
                "source_metadata": {
                    "type": "object",
                    "description": "Metadata about the input source"
                },
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data extraction targets"
                },
                "threat_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of threats to detect"
                },
                "sanitization_level": {
                    "type": "string",
                    "enum": ["standard", "strict"],
                    "description": "Level of sanitization to apply"
                }
            },
            "required": ["content"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for output data."""
        return {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean", "description": "Whether input is valid"},
                "sanitized_content": {"type": "string", "description": "Sanitized version of input"},
                "classification": {"type": "object", "description": "Content classification results"},
                "extracted_data": {"type": "object", "description": "Extracted structured data"},
                "threats_detected": {"type": "array", "description": "List of detected threats"},
                "source_verification": {"type": "object", "description": "Source verification results"},
                "risk_level": {"type": "string", "description": "Overall risk level"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Processing recommendations"
                }
            }
        }
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get the history of processing operations."""
        return self._processing_history.copy()
    
    def get_known_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get the list of known sources."""
        return self._known_sources.copy()