"""
Database migration and schema management for the Conscious Agent System.

This module handles database schema versioning, migrations, and validation.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Status of a database migration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationError(Exception):
    """Raised when a database migration fails."""
    pass


class DatabaseMigrationManager:
    """
    Manages database schema migrations and versioning.
    
    This class handles:
    - Schema version tracking
    - Migration execution and rollback
    - Schema validation
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the database migration manager.
        
        Args:
            base_dir: Base directory for migrations, defaults to ~/.james/migrations
        """
        self.base_dir = base_dir or Path.home() / ".james" / "migrations"
        self.schema_version_file = self.base_dir / "schema_version.json"
        self.migrations_dir = self.base_dir / "scripts"
        self.current_version = 0
        self.target_version = 0
        self.migration_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """
        Initialize the migration manager.
        
        Creates necessary directories and loads current schema version.
        """
        # Create directories if they don't exist
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.migrations_dir.mkdir(exist_ok=True)
        
        # Load current schema version
        await self._load_schema_version()
        
        # Find available migrations
        available_migrations = await self._find_available_migrations()
        self.target_version = max([0] + [m["version"] for m in available_migrations])
        
        logger.info(f"Database migration manager initialized. Current version: {self.current_version}, "
                   f"Target version: {self.target_version}")
    
    async def _load_schema_version(self) -> None:
        """Load the current schema version from the version file."""
        if not self.schema_version_file.exists():
            # Create initial version file
            version_data = {
                "version": 0,
                "last_updated": datetime.now().isoformat(),
                "history": []
            }
            await self._save_schema_version(version_data)
        
        try:
            with open(self.schema_version_file, "r") as f:
                version_data = json.load(f)
                self.current_version = version_data.get("version", 0)
                self.migration_history = version_data.get("history", [])
                
        except Exception as e:
            logger.error(f"Error loading schema version: {e}")
            # Reset to initial state
            self.current_version = 0
            self.migration_history = []
    
    async def _save_schema_version(self, version_data: Dict[str, Any]) -> None:
        """
        Save the schema version data to the version file.
        
        Args:
            version_data: Schema version data to save
        """
        try:
            with open(self.schema_version_file, "w") as f:
                json.dump(version_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving schema version: {e}")
            raise MigrationError(f"Failed to save schema version: {e}")
    
    async def _find_available_migrations(self) -> List[Dict[str, Any]]:
        """
        Find all available migration scripts.
        
        Returns:
            List of migration metadata
        """
        migrations = []
        
        if not self.migrations_dir.exists():
            return migrations
        
        # Look for migration scripts in format: V{version}__{name}.py
        for script_file in self.migrations_dir.glob("V*__*.py"):
            try:
                # Parse version from filename
                filename = script_file.name
                version_str = filename.split("__")[0].lstrip("V")
                version = int(version_str)
                
                # Extract name from filename
                name = filename.split("__")[1].rstrip(".py")
                
                migrations.append({
                    "version": version,
                    "name": name,
                    "file": str(script_file),
                    "applied": version <= self.current_version
                })
                
            except Exception as e:
                logger.warning(f"Invalid migration script filename: {script_file.name} - {e}")
        
        # Sort by version
        migrations.sort(key=lambda m: m["version"])
        return migrations
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """
        Get the current migration status.
        
        Returns:
            Dictionary with migration status information
        """
        available_migrations = await self._find_available_migrations()
        
        return {
            "current_version": self.current_version,
            "target_version": self.target_version,
            "pending_migrations": [m for m in available_migrations if not m["applied"]],
            "applied_migrations": [m for m in available_migrations if m["applied"]],
            "history": self.migration_history
        }
    
    async def migrate(self, target_version: Optional[int] = None) -> Dict[str, Any]:
        """
        Run database migrations up to the specified version.
        
        Args:
            target_version: Target schema version, defaults to latest available
            
        Returns:
            Migration result with status information
            
        Raises:
            MigrationError: If migration fails
        """
        # Find available migrations
        available_migrations = await self._find_available_migrations()
        
        # Determine target version
        if target_version is None:
            target_version = max([0] + [m["version"] for m in available_migrations])
        
        if target_version < self.current_version:
            return await self.rollback(target_version)
        
        if target_version == self.current_version:
            logger.info(f"Already at target version {target_version}")
            return {
                "status": "success",
                "message": f"Already at target version {target_version}",
                "migrations_applied": 0
            }
        
        # Filter migrations to apply
        migrations_to_apply = [
            m for m in available_migrations 
            if self.current_version < m["version"] <= target_version
        ]
        
        if not migrations_to_apply:
            logger.info(f"No migrations to apply")
            return {
                "status": "success",
                "message": "No migrations to apply",
                "migrations_applied": 0
            }
        
        # Apply migrations in order
        applied_count = 0
        for migration in migrations_to_apply:
            try:
                logger.info(f"Applying migration {migration['version']}: {migration['name']}")
                
                # Record migration start
                migration_record = {
                    "version": migration["version"],
                    "name": migration["name"],
                    "started_at": datetime.now().isoformat(),
                    "status": MigrationStatus.IN_PROGRESS.value
                }
                self.migration_history.append(migration_record)
                await self._update_migration_history()
                
                # Execute migration
                success, message = await self._execute_migration(migration["file"])
                
                if not success:
                    # Update migration record with failure
                    migration_record["status"] = MigrationStatus.FAILED.value
                    migration_record["completed_at"] = datetime.now().isoformat()
                    migration_record["message"] = message
                    await self._update_migration_history()
                    
                    raise MigrationError(f"Migration {migration['version']} failed: {message}")
                
                # Update migration record with success
                migration_record["status"] = MigrationStatus.COMPLETED.value
                migration_record["completed_at"] = datetime.now().isoformat()
                migration_record["message"] = message
                
                # Update current version
                self.current_version = migration["version"]
                await self._update_migration_history()
                
                applied_count += 1
                logger.info(f"Migration {migration['version']} applied successfully")
                
            except Exception as e:
                logger.error(f"Error applying migration {migration['version']}: {e}")
                raise MigrationError(f"Failed to apply migration {migration['version']}: {e}")
        
        logger.info(f"Migration completed. New version: {self.current_version}")
        return {
            "status": "success",
            "message": f"Migration completed. New version: {self.current_version}",
            "migrations_applied": applied_count
        }
    
    async def rollback(self, target_version: int) -> Dict[str, Any]:
        """
        Rollback database migrations to the specified version.
        
        Args:
            target_version: Target schema version to rollback to
            
        Returns:
            Rollback result with status information
            
        Raises:
            MigrationError: If rollback fails
        """
        if target_version >= self.current_version:
            logger.info(f"Target version {target_version} is not lower than current version {self.current_version}")
            return {
                "status": "success",
                "message": f"Already at or below target version {target_version}",
                "migrations_rolled_back": 0
            }
        
        # Find available migrations
        available_migrations = await self._find_available_migrations()
        
        # Filter migrations to rollback (in reverse order)
        migrations_to_rollback = [
            m for m in available_migrations 
            if target_version < m["version"] <= self.current_version
        ]
        migrations_to_rollback.reverse()
        
        if not migrations_to_rollback:
            logger.info(f"No migrations to roll back")
            return {
                "status": "success",
                "message": "No migrations to roll back",
                "migrations_rolled_back": 0
            }
        
        # Roll back migrations in reverse order
        rolled_back_count = 0
        for migration in migrations_to_rollback:
            try:
                logger.info(f"Rolling back migration {migration['version']}: {migration['name']}")
                
                # Record rollback start
                rollback_record = {
                    "version": migration["version"],
                    "name": migration["name"],
                    "started_at": datetime.now().isoformat(),
                    "status": MigrationStatus.IN_PROGRESS.value,
                    "operation": "rollback"
                }
                self.migration_history.append(rollback_record)
                await self._update_migration_history()
                
                # Execute rollback
                success, message = await self._execute_migration_rollback(migration["file"])
                
                if not success:
                    # Update rollback record with failure
                    rollback_record["status"] = MigrationStatus.FAILED.value
                    rollback_record["completed_at"] = datetime.now().isoformat()
                    rollback_record["message"] = message
                    await self._update_migration_history()
                    
                    raise MigrationError(f"Rollback of migration {migration['version']} failed: {message}")
                
                # Update rollback record with success
                rollback_record["status"] = MigrationStatus.ROLLED_BACK.value
                rollback_record["completed_at"] = datetime.now().isoformat()
                rollback_record["message"] = message
                
                # Update current version to previous version
                prev_version = max([0] + [
                    m["version"] for m in available_migrations 
                    if m["version"] < migration["version"]
                ])
                self.current_version = prev_version
                await self._update_migration_history()
                
                rolled_back_count += 1
                logger.info(f"Migration {migration['version']} rolled back successfully")
                
            except Exception as e:
                logger.error(f"Error rolling back migration {migration['version']}: {e}")
                raise MigrationError(f"Failed to roll back migration {migration['version']}: {e}")
        
        logger.info(f"Rollback completed. New version: {self.current_version}")
        return {
            "status": "success",
            "message": f"Rollback completed. New version: {self.current_version}",
            "migrations_rolled_back": rolled_back_count
        }
    
    async def _update_migration_history(self) -> None:
        """Update the migration history in the schema version file."""
        version_data = {
            "version": self.current_version,
            "last_updated": datetime.now().isoformat(),
            "history": self.migration_history
        }
        await self._save_schema_version(version_data)
    
    async def _execute_migration(self, migration_file: str) -> Tuple[bool, str]:
        """
        Execute a migration script.
        
        Args:
            migration_file: Path to migration script
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Import and execute migration
            import importlib.util
            
            # Load module from file
            spec = importlib.util.spec_from_file_location("migration_module", migration_file)
            if spec is None or spec.loader is None:
                return False, f"Failed to load migration script: {migration_file}"
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions
            if not hasattr(module, "migrate"):
                return False, f"Migration script missing 'migrate' function: {migration_file}"
            
            # Execute migration
            result = module.migrate()
            
            # Handle both synchronous and asynchronous migrations
            if asyncio.iscoroutine(result):
                result = await result
            
            return True, f"Migration executed successfully: {result}"
            
        except Exception as e:
            logger.error(f"Error executing migration {migration_file}: {e}")
            return False, f"Migration failed: {str(e)}"
    
    async def _execute_migration_rollback(self, migration_file: str) -> Tuple[bool, str]:
        """
        Execute a migration rollback.
        
        Args:
            migration_file: Path to migration script
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Import and execute rollback
            import importlib.util
            
            # Load module from file
            spec = importlib.util.spec_from_file_location("migration_module", migration_file)
            if spec is None or spec.loader is None:
                return False, f"Failed to load migration script: {migration_file}"
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions
            if not hasattr(module, "rollback"):
                return False, f"Migration script missing 'rollback' function: {migration_file}"
            
            # Execute rollback
            result = module.rollback()
            
            # Handle both synchronous and asynchronous rollbacks
            if asyncio.iscoroutine(result):
                result = await result
            
            return True, f"Rollback executed successfully: {result}"
            
        except Exception as e:
            logger.error(f"Error executing rollback {migration_file}: {e}")
            return False, f"Rollback failed: {str(e)}"
    
    async def validate_schema(self) -> Dict[str, Any]:
        """
        Validate the current database schema.
        
        Returns:
            Validation result with status information
        """
        # This would typically connect to the database and validate the schema
        # For now, we'll just check if we're at the target version
        
        if self.current_version < self.target_version:
            return {
                "valid": False,
                "message": f"Schema is outdated. Current version: {self.current_version}, Target version: {self.target_version}",
                "pending_migrations": self.target_version - self.current_version
            }
        
        return {
            "valid": True,
            "message": f"Schema is up to date. Version: {self.current_version}",
            "pending_migrations": 0
        }
    
    async def create_migration(self, name: str) -> Dict[str, Any]:
        """
        Create a new migration script.
        
        Args:
            name: Migration name (will be used in filename)
            
        Returns:
            Dictionary with migration script information
        """
        # Find available migrations to determine next version
        available_migrations = await self._find_available_migrations()
        next_version = max([0] + [m["version"] for m in available_migrations]) + 1
        
        # Create migration filename
        safe_name = name.replace(" ", "_").lower()
        filename = f"V{next_version}__{safe_name}.py"
        file_path = self.migrations_dir / filename
        
        # Create migration script template
        template = f'''"""
Migration {next_version}: {name}

Created: {datetime.now().isoformat()}
"""

async def migrate():
    """
    Apply the migration.
    
    Returns:
        String with migration result
    """
    # TODO: Implement migration logic
    return "Migration {next_version} applied"

async def rollback():
    """
    Rollback the migration.
    
    Returns:
        String with rollback result
    """
    # TODO: Implement rollback logic
    return "Migration {next_version} rolled back"
'''
        
        try:
            # Create migrations directory if it doesn't exist
            self.migrations_dir.mkdir(exist_ok=True, parents=True)
            
            # Write migration script
            with open(file_path, "w") as f:
                f.write(template)
            
            logger.info(f"Created migration script: {file_path}")
            return {
                "status": "success",
                "version": next_version,
                "name": name,
                "file": str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error creating migration script: {e}")
            raise MigrationError(f"Failed to create migration script: {e}")


# Global migration manager instance
_migration_manager: Optional[DatabaseMigrationManager] = None


def get_migration_manager() -> DatabaseMigrationManager:
    """Get the global migration manager instance."""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = DatabaseMigrationManager()
    return _migration_manager