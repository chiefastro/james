"""
File writing tool with security validation.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import time

from .base import BaseTool, ToolResult
from ..filesystem.file_manager import FileManager
from ..filesystem.exceptions import FileSystemError, SecurityError


class FileWriterTool(BaseTool):
    """
    Secure file writing tool for James to create and modify files.
    
    Provides safe file operations within the ~/.james directory with
    security validation and backup capabilities.
    """
    
    def __init__(self):
        """Initialize the file writer tool."""
        super().__init__("FileWriter")
        self.file_manager = FileManager()
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute file writing operation.
        
        Args:
            file_path (str): Path to file relative to ~/.james
            content (str): Content to write to file
            mode (str): Write mode - 'write', 'append', or 'json'
            encoding (str): Text encoding (default: utf-8)
            create_backup (bool): Whether to create backup (default: True)
            
        Returns:
            ToolResult with file operation outcome
        """
        start_time = time.time()
        
        # Validate required parameters
        error = self._validate_required_params(kwargs, ['file_path', 'content'])
        if error:
            return self._create_error_result(error)
        
        file_path = kwargs['file_path']
        content = kwargs['content']
        mode = kwargs.get('mode', 'write')
        encoding = kwargs.get('encoding', 'utf-8')
        create_backup = kwargs.get('create_backup', True)
        
        try:
            # Validate file path security
            if not self._is_safe_path(file_path):
                return self._create_error_result(f"Unsafe file path: {file_path}")
            
            # Execute based on mode
            if mode == 'write':
                checksum = await self._write_text_file(
                    file_path, content, encoding, create_backup
                )
                result_data = {
                    'file_path': file_path,
                    'checksum': checksum,
                    'size': len(content.encode(encoding))
                }
                
            elif mode == 'append':
                await self._append_text_file(file_path, content, encoding)
                result_data = {
                    'file_path': file_path,
                    'appended_size': len(content.encode(encoding))
                }
                
            elif mode == 'json':
                if not isinstance(content, (dict, list)):
                    try:
                        content = json.loads(content) if isinstance(content, str) else content
                    except json.JSONDecodeError:
                        return self._create_error_result("Invalid JSON content")
                
                checksum = await self._write_json_file(file_path, content, create_backup)
                result_data = {
                    'file_path': file_path,
                    'checksum': checksum,
                    'format': 'json'
                }
                
            else:
                return self._create_error_result(f"Unsupported write mode: {mode}")
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Successfully wrote file: {file_path} (mode: {mode})")
            
            return self._create_success_result(
                data=result_data,
                metadata={
                    'execution_time': execution_time,
                    'mode': mode,
                    'encoding': encoding,
                    'backup_created': create_backup
                }
            )
            
        except SecurityError as e:
            return self._create_error_result(f"Security violation: {e}")
        except FileSystemError as e:
            return self._create_error_result(f"File system error: {e}")
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def _write_text_file(self, file_path: str, content: str, 
                              encoding: str, create_backup: bool) -> str:
        """Write text content to file."""
        return await self.file_manager.write_text_file(
            file_path, content, encoding, create_backup
        )
    
    async def _append_text_file(self, file_path: str, content: str, encoding: str) -> None:
        """Append text content to file."""
        await self.file_manager.append_text_to_file(file_path, content, encoding)
    
    async def _write_json_file(self, file_path: str, content: Dict[str, Any], 
                              create_backup: bool) -> str:
        """Write JSON content to file."""
        return await self.file_manager.write_json_file(
            file_path, content, indent=2, create_backup=create_backup
        )
    
    def _is_safe_path(self, file_path: str) -> bool:
        """
        Validate that the file path is safe for writing.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            '../',  # Directory traversal
            '/..',  # Directory traversal
            '/etc/', # System directories
            '/root/', # Root directory
            '/home/', # Other user directories (unless ~/.james)
            '/usr/', # System binaries
            '/var/', # System variables
            '/sys/', # System files
            '/proc/', # Process files
            '/dev/', # Device files
        ]
        
        path_lower = file_path.lower()
        for pattern in dangerous_patterns:
            if pattern in path_lower:
                # Allow ~/.james paths
                if '/james' not in path_lower:
                    return False
        
        # Check for executable extensions
        dangerous_extensions = [
            '.sh', '.bash', '.zsh', '.fish',  # Shell scripts
            '.py', '.pl', '.rb', '.js',       # Scripting languages (allow .py for James)
            '.exe', '.bat', '.cmd',           # Windows executables
            '.so', '.dll', '.dylib',          # Libraries
        ]
        
        path_obj = Path(file_path)
        if path_obj.suffix.lower() in dangerous_extensions:
            # Allow .py files for James' code
            if path_obj.suffix.lower() != '.py':
                return False
        
        return True
    
    async def read_file(self, file_path: str, mode: str = 'text', encoding: str = 'utf-8') -> ToolResult:
        """
        Read a file from the ~/.james directory.
        
        Args:
            file_path: Path to file relative to ~/.james
            mode: Read mode - 'text', 'binary', or 'json'
            encoding: Text encoding for text mode
            
        Returns:
            ToolResult with file content
        """
        start_time = time.time()
        
        try:
            if not self._is_safe_path(file_path):
                return self._create_error_result(f"Unsafe file path: {file_path}")
            
            if mode == 'text':
                content = await self.file_manager.read_text_file(file_path, encoding)
            elif mode == 'binary':
                content = await self.file_manager.read_file(file_path)
            elif mode == 'json':
                content = await self.file_manager.read_json_file(file_path)
            else:
                return self._create_error_result(f"Unsupported read mode: {mode}")
            
            execution_time = time.time() - start_time
            
            return self._create_success_result(
                data={
                    'file_path': file_path,
                    'content': content,
                    'size': len(content) if isinstance(content, (str, bytes)) else None
                },
                metadata={
                    'execution_time': execution_time,
                    'mode': mode,
                    'encoding': encoding if mode == 'text' else None
                }
            )
            
        except FileSystemError as e:
            return self._create_error_result(f"File system error: {e}")
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def list_files(self, directory_path: str = "", pattern: str = "*", 
                        recursive: bool = False) -> ToolResult:
        """
        List files in a directory within ~/.james.
        
        Args:
            directory_path: Directory path relative to ~/.james
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            ToolResult with list of files
        """
        try:
            files = await self.file_manager.list_files(directory_path, pattern, recursive)
            
            return self._create_success_result(
                data={
                    'directory': directory_path or '.',
                    'files': files,
                    'count': len(files)
                },
                metadata={
                    'pattern': pattern,
                    'recursive': recursive
                }
            )
            
        except FileSystemError as e:
            return self._create_error_result(f"File system error: {e}")
        except Exception as e:
            return self._create_error_result(f"Unexpected error: {e}")
    
    async def file_exists(self, file_path: str) -> ToolResult:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            ToolResult with existence status
        """
        try:
            exists = await self.file_manager.file_exists(file_path)
            
            return self._create_success_result(
                data={
                    'file_path': file_path,
                    'exists': exists
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Error checking file existence: {e}")