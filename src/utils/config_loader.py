"""
Configuration Loader Utility
Loads configuration from YAML files with defaults and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML files"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache = {}

    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_name: Name of the config file (without .yaml extension)
            use_cache: Whether to use cached config if available

        Returns:
            Configuration dictionary
        """
        if use_cache and config_name in self._cache:
            return self._cache[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if use_cache:
            self._cache[config_name] = config

        return config

    def get_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.

        Args:
            config_name: Name of the config file
            key_path: Path to the value using dot notation (e.g., "rate_limiting.delay_between_requests")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.load_config(config_name)

        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def clear_cache(self):
        """Clear the configuration cache"""
        self._cache = {}


# Global instance
_global_loader = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """Get or create the global configuration loader"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(config_dir)
    return _global_loader


def load_translation_config() -> Dict[str, Any]:
    """Load translation configuration (convenience function)"""
    loader = get_config_loader()
    return loader.load_config("translation_config")
