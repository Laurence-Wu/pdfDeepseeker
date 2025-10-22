"""
Configuration Management Module
Handles loading and validation of configuration from YAML files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class ConfigurationError(Exception):
    """Configuration related errors"""
    pass

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    user: str = "translator"
    password: str = "secure_password"
    name: str = "pdf_translations"
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "translator"),
            password=os.getenv("DB_PASSWORD", "secure_password"),
            name=os.getenv("DB_NAME", "pdf_translations")
        )

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=int(os.getenv("REDIS_DB", "0"))
        )

@dataclass
class OpenRouterConfig:
    """OpenRouter configuration for Gemini access"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-pro-1.5"
    timeout: int = 60
    fallback_model: str = "anthropic/claude-3-opus"
    secondary_model: str = "meta-llama/llama-3.1-70b"
    
    @classmethod
    def from_env(cls):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENROUTER_API_KEY is required but not set")
        
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL", "google/gemini-pro-1.5"),
            timeout=int(os.getenv("OPENROUTER_TIMEOUT", "60")),
            fallback_model=os.getenv("FALLBACK_MODEL", "anthropic/claude-3-opus"),
            secondary_model=os.getenv("SECONDARY_MODEL", "meta-llama/llama-3.1-70b")
        )

@dataclass
class GPUConfig:
    """GPU configuration"""
    use_gpu: bool = False
    device: str = "cuda:0"
    memory_limit: int = 8192
    
    @classmethod
    def from_env(cls):
        return cls(
            use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
            device=os.getenv("GPU_DEVICE", "cuda:0"),
            memory_limit=int(os.getenv("GPU_MEMORY_LIMIT", "8192"))
        )

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 10
    max_pages_per_job: int = 2000
    batch_size: int = 5
    parallel_pages: int = 10
    cache_ttl: int = 3600
    max_cache_size: str = "10GB"
    
    @classmethod
    def from_env(cls):
        return cls(
            max_workers=int(os.getenv("MAX_WORKERS", "10")),
            max_pages_per_job=int(os.getenv("MAX_PAGES_PER_JOB", "2000")),
            batch_size=int(os.getenv("BATCH_SIZE", "5")),
            parallel_pages=int(os.getenv("PARALLEL_PAGES", "10")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            max_cache_size=os.getenv("MAX_CACHE_SIZE", "10GB")
        )

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    port: int = 9090
    
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            port=int(os.getenv("METRICS_PORT", "9090"))
        )

class ConfigLoader:
    """Loads and manages configuration from YAML files with environment variable substitution"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self._cache = {}
    
    def _substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute environment variables in configuration values"""
        if isinstance(value, str):
            # Check for ${VAR} pattern
            import re
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                env_value = os.getenv(match, "")
                value = value.replace(f"${{{match}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        return value
    
    def load_yaml(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load a YAML configuration file with environment variable substitution"""
        cache_key = filename
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            if use_cache:
                self._cache[cache_key] = config
            
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {filename}: {e}")
    
    def load_main_config(self) -> Dict[str, Any]:
        """Load the main configuration file"""
        return self.load_yaml("../config.yaml")
    
    def load_vla_models(self) -> Dict[str, Any]:
        """Load VLA models configuration"""
        return self.load_yaml("vla_models.yaml")
    
    def load_prompts(self) -> Dict[str, Any]:
        """Load translation prompts configuration"""
        return self.load_yaml("prompts.yaml")
    
    def load_environment_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env = environment or os.getenv("ENVIRONMENT", "development")
        filename = f"environments/{env}.yaml"
        return self.load_yaml(filename)

class Settings:
    """Global settings manager combining all configuration sources"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = self.environment == "development"
        
        # Load configurations
        self.loader = ConfigLoader()
        
        # Load environment variables configurations
        self.database = DatabaseConfig.from_env()
        self.redis = RedisConfig.from_env()
        self.openrouter = OpenRouterConfig.from_env()
        self.gpu = GPUConfig.from_env()
        self.performance = PerformanceConfig.from_env()
        self.monitoring = MonitoringConfig.from_env()
        
        # Load YAML configurations
        try:
            self.main_config = self.loader.load_main_config()
            self.vla_models = self.loader.load_vla_models()
            self.prompts = self.loader.load_prompts()
            self.env_config = self.loader.load_environment_config()
        except ConfigurationError as e:
            print(f"Warning: {e}")
            # Set defaults if files don't exist
            self.main_config = {}
            self.vla_models = {}
            self.prompts = {}
            self.env_config = {}
        
        # Extract key settings
        self.pipeline_config = self.main_config.get("pipeline", {})
        self.translation_config = self.main_config.get("translation", {})
        self.extraction_config = self.main_config.get("extraction", {})
        self.vla_config = self.main_config.get("vla", {})
        self.text_control_config = self.main_config.get("text_control", {})
        self.reconstruction_config = self.main_config.get("reconstruction", {})
        self.limits_config = self.main_config.get("limits", {})
        self.cache_config = self.main_config.get("cache", {})
        
        # Apply environment-specific overrides
        if self.env_config:
            env_settings = self.env_config.get(self.environment, {})
            self.debug = env_settings.get("debug", self.debug)
            self.log_level = env_settings.get("log_level", "INFO")
            self.workers = env_settings.get("workers", self.performance.max_workers)
        else:
            self.log_level = os.getenv("LOG_LEVEL", "INFO")
            self.workers = self.performance.max_workers
        
        self._initialized = True
    
    def get_translation_service_config(self, service: str = "primary") -> Dict[str, Any]:
        """Get translation service configuration"""
        if service == "primary":
            return self.translation_config.get("primary_service", {})
        elif service == "fallback":
            return self.translation_config.get("fallback_service", {})
        return {}
    
    def get_vla_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get VLA model configuration"""
        models = self.vla_models.get("models", {})
        return models.get(model_name, {})
    
    def get_prompt_template(self, prompt_type: str, document_type: Optional[str] = None) -> str:
        """Get prompt template for translation"""
        prompts = self.prompts.get("prompts", {})
        
        if prompt_type == "system":
            return prompts.get("system", {}).get("default", "")
        elif prompt_type == "document" and document_type:
            doc_prompts = prompts.get("document_types", {})
            return doc_prompts.get(document_type, {})
        elif prompt_type == "constraints":
            return prompts.get("constraints", {})
        
        return ""
    
    def get_extraction_settings(self, component: str) -> Dict[str, Any]:
        """Get extraction component settings"""
        return self.extraction_config.get(component, {})
    
    def get_cache_settings(self, cache_type: str) -> Dict[str, Any]:
        """Get cache settings by type"""
        return self.cache_config.get(cache_type, {})
    
    def get_limit(self, limit_name: str) -> Any:
        """Get processing limit value"""
        return self.limits_config.get(limit_name)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        # Check required environment variables
        if not self.openrouter.api_key:
            warnings.append("OPENROUTER_API_KEY is not set")
        
        # Check database connectivity (optional)
        if self.environment == "production":
            if self.database.password == "secure_password":
                warnings.append("Using default database password in production")
        
        # Check file size limits
        max_file_size = self.get_limit("max_file_size")
        if max_file_size and max_file_size > 1024 * 1024 * 1024:  # 1GB
            warnings.append(f"Max file size is very large: {max_file_size / (1024*1024*1024):.1f}GB")
        
        # Check GPU settings
        if self.gpu.use_gpu and not self._check_gpu_available():
            warnings.append("GPU is enabled but may not be available")
        
        return warnings
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available (simplified check)"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port
            },
            "openrouter": {
                "model": self.openrouter.model,
                "base_url": self.openrouter.base_url
            },
            "gpu": {
                "enabled": self.gpu.use_gpu,
                "device": self.gpu.device
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "batch_size": self.performance.batch_size
            }
        }

# Global settings instance
settings = Settings()