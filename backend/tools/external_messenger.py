"""
External message emission capabilities tool.
"""

import time
import json
import aiohttp
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .base import BaseTool, ToolResult


class MessageFormat(Enum):
    """External message formats."""
    JSON = "json"
    TEXT = "text"
    XML = "xml"
    FORM = "form"


class DeliveryMethod(Enum):
    """Message delivery methods."""
    HTTP_POST = "http_post"
    HTTP_GET = "http_get"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"


@dataclass
class ExternalEndpoint:
    """Configuration for an external endpoint."""
    name: str
    url: str
    method: DeliveryMethod
    format: MessageFormat
    headers: Dict[str, str]
    auth: Optional[Dict[str, str]] = None
    timeout: int = 30
    retry_attempts: int = 3


class ExternalMessengerTool(BaseTool):
    """
    Tool for James to send messages to external systems and services.
    
    Provides capabilities to emit messages via HTTP, webhooks, email,
    and various messaging platforms with proper formatting and error handling.
    """
    
    def __init__(self):
        """Initialize the external messenger tool."""
        super().__init__("ExternalMessenger")
        self.endpoints: Dict[str, ExternalEndpoint] = {}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute external message operation.
        
        Args:
            action (str): Action to perform - 'send', 'configure', 'list_endpoints', 'test'
            endpoint (str): Endpoint name (for 'send' and 'test' actions)
            message (str|dict): Message content
            format (str): Message format override
            headers (dict): Additional headers
            url (str): Direct URL (alternative to endpoint)
            method (str): HTTP method for direct sends
            
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
            elif action == 'configure':
                return await self._configure_endpoint(kwargs, start_time)
            elif action == 'list_endpoints':
                return await self._list_endpoints(start_time)
            elif action == 'test':
                return await self._test_endpoint(kwargs, start_time)
            elif action == 'remove_endpoint':
                return await self._remove_endpoint(kwargs, start_time)
            else:
                return self._create_error_result(f"Unsupported action: {action}")
                
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def _send_message(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Send a message to an external endpoint."""
        message = kwargs.get('message')
        endpoint_name = kwargs.get('endpoint')
        url = kwargs.get('url')
        method = kwargs.get('method', 'POST')
        format_override = kwargs.get('format')
        headers_override = kwargs.get('headers', {})
        
        if not message:
            return self._create_error_result("No message provided")
        
        if not endpoint_name and not url:
            return self._create_error_result("Either endpoint name or URL must be provided")
        
        try:
            # Get endpoint configuration or create temporary one
            if endpoint_name:
                if endpoint_name not in self.endpoints:
                    return self._create_error_result(f"Endpoint '{endpoint_name}' not configured")
                endpoint = self.endpoints[endpoint_name]
            else:
                # Create temporary endpoint for direct URL
                endpoint = ExternalEndpoint(
                    name="temp",
                    url=url,
                    method=DeliveryMethod(method.lower()) if method.lower() in [m.value for m in DeliveryMethod] else DeliveryMethod.HTTP_POST,
                    format=MessageFormat(format_override.lower()) if format_override else MessageFormat.JSON,
                    headers=headers_override
                )
            
            # Override format if specified
            if format_override:
                try:
                    endpoint.format = MessageFormat(format_override.lower())
                except ValueError:
                    return self._create_error_result(f"Invalid format: {format_override}")
            
            # Prepare message based on format
            formatted_message, content_type = self._format_message(message, endpoint.format)
            
            # Prepare headers
            headers = endpoint.headers.copy()
            headers.update(headers_override)
            if content_type:
                headers['Content-Type'] = content_type
            
            # Add authentication if configured
            if endpoint.auth:
                headers.update(self._prepare_auth_headers(endpoint.auth))
            
            # Send message
            response_data = await self._send_http_request(
                endpoint.url,
                endpoint.method,
                formatted_message,
                headers,
                endpoint.timeout
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Successfully sent message to {endpoint.name or endpoint.url}")
            
            return self._create_success_result(
                data={
                    'endpoint': endpoint.name,
                    'url': endpoint.url,
                    'method': endpoint.method.value,
                    'format': endpoint.format.value,
                    'response': response_data,
                    'message_size': len(str(formatted_message))
                },
                metadata={
                    'execution_time': execution_time,
                    'headers_sent': len(headers)
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to send message: {e}")
    
    async def _configure_endpoint(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Configure an external endpoint."""
        name = kwargs.get('name')
        url = kwargs.get('url')
        method = kwargs.get('method', 'http_post')
        format_type = kwargs.get('format', 'json')
        headers = kwargs.get('headers', {})
        auth = kwargs.get('auth')
        timeout = kwargs.get('timeout', 30)
        retry_attempts = kwargs.get('retry_attempts', 3)
        
        if not name or not url:
            return self._create_error_result("Name and URL are required for endpoint configuration")
        
        try:
            # Validate method and format
            try:
                delivery_method = DeliveryMethod(method.lower())
            except ValueError:
                return self._create_error_result(f"Invalid delivery method: {method}")
            
            try:
                message_format = MessageFormat(format_type.lower())
            except ValueError:
                return self._create_error_result(f"Invalid message format: {format_type}")
            
            # Create endpoint configuration
            endpoint = ExternalEndpoint(
                name=name,
                url=url,
                method=delivery_method,
                format=message_format,
                headers=headers,
                auth=auth,
                timeout=timeout,
                retry_attempts=retry_attempts
            )
            
            # Store endpoint
            self.endpoints[name] = endpoint
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Configured endpoint: {name}")
            
            return self._create_success_result(
                data={
                    'name': name,
                    'url': url,
                    'method': method,
                    'format': format_type,
                    'configured': True
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to configure endpoint: {e}")
    
    async def _list_endpoints(self, start_time: float) -> ToolResult:
        """List all configured endpoints."""
        try:
            endpoints_data = []
            for name, endpoint in self.endpoints.items():
                endpoints_data.append({
                    'name': endpoint.name,
                    'url': endpoint.url,
                    'method': endpoint.method.value,
                    'format': endpoint.format.value,
                    'timeout': endpoint.timeout,
                    'retry_attempts': endpoint.retry_attempts,
                    'has_auth': bool(endpoint.auth)
                })
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'endpoints': endpoints_data,
                    'count': len(endpoints_data)
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to list endpoints: {e}")
    
    async def _test_endpoint(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Test an endpoint with a simple message."""
        endpoint_name = kwargs.get('endpoint')
        test_message = kwargs.get('message', 'Test message from James')
        
        if not endpoint_name:
            return self._create_error_result("Endpoint name required for testing")
        
        if endpoint_name not in self.endpoints:
            return self._create_error_result(f"Endpoint '{endpoint_name}' not configured")
        
        try:
            # Send test message
            result = await self._send_message({
                'endpoint': endpoint_name,
                'message': test_message
            }, start_time)
            
            if result.success:
                result.data['test_result'] = 'success'
                result.data['test_message'] = test_message
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Endpoint test failed: {e}")
    
    async def _remove_endpoint(self, kwargs: Dict[str, Any], start_time: float) -> ToolResult:
        """Remove an endpoint configuration."""
        name = kwargs.get('name')
        
        if not name:
            return self._create_error_result("Endpoint name required for removal")
        
        try:
            if name in self.endpoints:
                del self.endpoints[name]
                removed = True
            else:
                removed = False
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'name': name,
                    'removed': removed,
                    'remaining_endpoints': len(self.endpoints)
                },
                metadata={'execution_time': execution_time}
            )
            
        except Exception as e:
            return self._create_error_result(f"Failed to remove endpoint: {e}")
    
    def _format_message(self, message: Union[str, Dict, List], format_type: MessageFormat) -> tuple:
        """Format message according to the specified format."""
        if format_type == MessageFormat.JSON:
            if isinstance(message, str):
                try:
                    # Try to parse as JSON first
                    json.loads(message)
                    return message, 'application/json'
                except json.JSONDecodeError:
                    # Wrap string in JSON
                    return json.dumps({'message': message}), 'application/json'
            else:
                return json.dumps(message), 'application/json'
        
        elif format_type == MessageFormat.TEXT:
            if isinstance(message, (dict, list)):
                return str(message), 'text/plain'
            else:
                return str(message), 'text/plain'
        
        elif format_type == MessageFormat.XML:
            if isinstance(message, str):
                return message, 'application/xml'
            else:
                # Simple dict to XML conversion
                xml_content = self._dict_to_xml(message)
                return xml_content, 'application/xml'
        
        elif format_type == MessageFormat.FORM:
            if isinstance(message, dict):
                # Convert to form data
                form_data = aiohttp.FormData()
                for key, value in message.items():
                    form_data.add_field(key, str(value))
                return form_data, 'application/x-www-form-urlencoded'
            else:
                form_data = aiohttp.FormData()
                form_data.add_field('message', str(message))
                return form_data, 'application/x-www-form-urlencoded'
        
        return str(message), 'text/plain'
    
    def _dict_to_xml(self, data: Dict[str, Any], root_tag: str = 'message') -> str:
        """Convert dictionary to simple XML."""
        xml_parts = [f'<{root_tag}>']
        
        for key, value in data.items():
            if isinstance(value, dict):
                xml_parts.append(self._dict_to_xml(value, key))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        xml_parts.append(self._dict_to_xml(item, key))
                    else:
                        xml_parts.append(f'<{key}>{item}</{key}>')
            else:
                xml_parts.append(f'<{key}>{value}</{key}>')
        
        xml_parts.append(f'</{root_tag}>')
        return ''.join(xml_parts)
    
    def _prepare_auth_headers(self, auth_config: Dict[str, str]) -> Dict[str, str]:
        """Prepare authentication headers."""
        headers = {}
        
        auth_type = auth_config.get('type', '').lower()
        
        if auth_type == 'bearer':
            token = auth_config.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'
        
        elif auth_type == 'basic':
            username = auth_config.get('username')
            password = auth_config.get('password')
            if username and password:
                import base64
                credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'
        
        elif auth_type == 'api_key':
            api_key = auth_config.get('api_key')
            header_name = auth_config.get('header_name', 'X-API-Key')
            if api_key:
                headers[header_name] = api_key
        
        return headers
    
    async def _send_http_request(self, url: str, method: DeliveryMethod, 
                                data: Any, headers: Dict[str, str], 
                                timeout: int) -> Dict[str, Any]:
        """Send HTTP request to external endpoint."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            if method == DeliveryMethod.HTTP_POST:
                async with self.session.post(url, data=data, headers=headers, timeout=timeout) as response:
                    return await self._process_response(response)
            
            elif method == DeliveryMethod.HTTP_GET:
                # For GET requests, convert data to query parameters
                params = {}
                if isinstance(data, dict):
                    params = data
                elif isinstance(data, str):
                    try:
                        params = json.loads(data)
                    except json.JSONDecodeError:
                        params = {'message': data}
                
                async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                    return await self._process_response(response)
            
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientTimeout:
            raise Exception(f"Request timeout after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP client error: {e}")
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Process HTTP response."""
        response_data = {
            'status_code': response.status,
            'headers': dict(response.headers),
            'success': 200 <= response.status < 300
        }
        
        try:
            # Try to get response content
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                response_data['content'] = await response.json()
            else:
                response_data['content'] = await response.text()
                
        except Exception as e:
            response_data['content'] = f"Failed to read response: {e}"
        
        if not response_data['success']:
            raise Exception(f"HTTP {response.status}: {response_data.get('content', 'Unknown error')}")
        
        return response_data
    
    async def send_slack_message(self, webhook_url: str, message: str, 
                                channel: Optional[str] = None) -> ToolResult:
        """
        Convenience method to send a Slack message.
        
        Args:
            webhook_url: Slack webhook URL
            message: Message content
            channel: Optional channel override
            
        Returns:
            ToolResult with send outcome
        """
        payload = {'text': message}
        if channel:
            payload['channel'] = channel
        
        return await self.execute(
            action='send',
            url=webhook_url,
            message=payload,
            format='json',
            method='POST'
        )
    
    async def send_discord_message(self, webhook_url: str, message: str) -> ToolResult:
        """
        Convenience method to send a Discord message.
        
        Args:
            webhook_url: Discord webhook URL
            message: Message content
            
        Returns:
            ToolResult with send outcome
        """
        payload = {'content': message}
        
        return await self.execute(
            action='send',
            url=webhook_url,
            message=payload,
            format='json',
            method='POST'
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None