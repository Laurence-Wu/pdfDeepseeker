"""
Configuration Utilities
Helper functions for configuration management and validation
"""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import json
from functools import lru_cache

def parse_size_string(size_str: str) -> int:
    """
    Parse size string to bytes
    
    Args:
        size_str: Size string like "10MB", "1GB", "512KB"
    
    Returns:
        Size in bytes
    """
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024
    }
    
    size_str = size_str.strip().upper()
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")
    
    # If no unit is specified, assume bytes
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")

def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"

@lru_cache(maxsize=128)
def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable
    
    Args:
        key: Environment variable name
        default: Default value if not set
    
    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default))
    return value.lower() in ('true', '1', 'yes', 'on')

def get_env_list(key: str, delimiter: str = ',', default: Optional[list] = None) -> list:
    """
    Get list from environment variable
    
    Args:
        key: Environment variable name
        delimiter: String delimiter
        default: Default list if not set
    
    Returns:
        List of values
    """
    value = os.getenv(key)
    if not value:
        return default or []
    
    return [item.strip() for item in value.split(delimiter) if item.strip()]

def get_env_dict(key: str, default: Optional[dict] = None) -> dict:
    """
    Get dictionary from JSON environment variable
    
    Args:
        key: Environment variable name
        default: Default dict if not set
    
    Returns:
        Dictionary value
    """
    value = os.getenv(key)
    if not value:
        return default or {}
    
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default or {}

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries
    
    Args:
        *configs: Configuration dictionaries to merge
    
    Returns:
        Merged configuration
    """
    result = {}
    
    for config in configs:
        if not config:
            continue
            
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result

def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> list:
    """
    Validate configuration against a schema
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary defining required fields and types
    
    Returns:
        List of validation errors
    """
    errors = []
    
    for key, requirements in schema.items():
        if isinstance(requirements, dict):
            required = requirements.get('required', False)
            expected_type = requirements.get('type')
            default = requirements.get('default')
            validator = requirements.get('validator')
        else:
            required = False
            expected_type = requirements
            default = None
            validator = None
        
        if key not in config:
            if required:
                errors.append(f"Missing required field: {key}")
            continue
        
        value = config[key]
        
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Field {key} should be {expected_type.__name__}, got {type(value).__name__}")
        
        if validator and not validator(value):
            errors.append(f"Field {key} failed validation")
    
    return errors

def get_config_path(filename: str, config_dir: Optional[Path] = None) -> Path:
    """
    Get configuration file path
    
    Args:
        filename: Configuration filename
        config_dir: Optional config directory override
    
    Returns:
        Full path to configuration file
    """
    if config_dir is None:
        # Try multiple locations
        locations = [
            Path.cwd() / "config" / filename,
            Path(__file__).parent.parent.parent / "config" / filename,
            Path.home() / ".pdf_translator" / filename,
            Path("/etc/pdf_translator") / filename
        ]
        
        for location in locations:
            if location.exists():
                return location
        
        # Default to first location if none exist
        return locations[0]
    
    return config_dir / filename

def load_yaml_config(filename: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        filename: Configuration filename
        config_dir: Optional config directory override
    
    Returns:
        Configuration dictionary
    """
    config_path = get_config_path(filename, config_dir)
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {}

def save_yaml_config(config: Dict[str, Any], filename: str, config_dir: Optional[Path] = None):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        filename: Configuration filename
        config_dir: Optional config directory override
    """
    config_path = get_config_path(filename, config_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

class ConfigWatcher:
    """Watch configuration files for changes"""
    
    def __init__(self, config_files: list, callback=None):
        """
        Initialize configuration watcher
        
        Args:
            config_files: List of configuration files to watch
            callback: Function to call when configuration changes
        """
        self.config_files = config_files
        self.callback = callback
        self._last_modified = {}
        self._update_timestamps()
    
    def _update_timestamps(self):
        """Update file modification timestamps"""
        for file_path in self.config_files:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            if file_path.exists():
                self._last_modified[str(file_path)] = file_path.stat().st_mtime
    
    def check_changes(self) -> bool:
        """
        Check if any configuration files have changed
        
        Returns:
            True if changes detected
        """
        changed = False
        
        for file_path in self.config_files:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            if not file_path.exists():
                continue
            
            current_mtime = file_path.stat().st_mtime
            last_mtime = self._last_modified.get(str(file_path), 0)
            
            if current_mtime > last_mtime:
                changed = True
                self._last_modified[str(file_path)] = current_mtime
                
                if self.callback:
                    self.callback(file_path)
        
        return changed

def create_default_configs(config_dir: Path):
    """
    Create default configuration files if they don't exist
    
    Args:
        config_dir: Configuration directory
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Default main config
    main_config_path = config_dir / "config.yaml"
    if not main_config_path.exists():
        default_config = {
            "pipeline": {
                "name": "PDF Translation Pipeline",
                "version": "2.0.0"
            },
            "translation": {
                "primary_service": {
                    "provider": "openrouter",
                    "model": "${OPENROUTER_MODEL}"
                }
            }
        }
        save_yaml_config(default_config, "config.yaml", config_dir)
    
    # Default prompts
    prompts_path = config_dir / "prompts.yaml"
    if not prompts_path.exists():
        default_prompts = {
            "prompts": {
                "system": {
                    "default": "You are a professional document translator."
                }
            }
        }
        save_yaml_config(default_prompts, "prompts.yaml", config_dir)
    
    # Default VLA models
    vla_path = config_dir / "vla_models.yaml"
    if not vla_path.exists():
        default_vla = {
            "models": {
                "surya": {
                    "repo": "VikParuchuri/surya",
                    "version": "latest"
                }
            }
        }
        save_yaml_config(default_vla, "vla_models.yaml", config_dir)