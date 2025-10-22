#!/usr/bin/env python3
"""
Configuration CLI Tool
Command-line interface for managing PDF Translation Pipeline configuration
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Optional
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import settings, ConfigLoader, ConfigurationError
from src.utils.config_utils import (
    format_size,
    parse_size_string,
    validate_config_schema,
    create_default_configs
)

def cmd_show(args):
    """Show current configuration"""
    config_type = args.type
    
    if config_type == 'all':
        print("\n=== Environment Variables ===")
        show_env_vars()
        print("\n=== Main Configuration ===")
        show_main_config()
        print("\n=== Active Settings ===")
        show_active_settings()
    elif config_type == 'env':
        show_env_vars()
    elif config_type == 'main':
        show_main_config()
    elif config_type == 'settings':
        show_active_settings()
    elif config_type == 'vla':
        show_vla_config()
    elif config_type == 'prompts':
        show_prompts_config()

def show_env_vars():
    """Display environment variables"""
    env_vars = [
        ('ENVIRONMENT', os.getenv('ENVIRONMENT', 'not set')),
        ('OPENROUTER_API_KEY', '***' if os.getenv('OPENROUTER_API_KEY') else 'not set'),
        ('OPENROUTER_MODEL', os.getenv('OPENROUTER_MODEL', 'not set')),
        ('DB_HOST', os.getenv('DB_HOST', 'not set')),
        ('REDIS_HOST', os.getenv('REDIS_HOST', 'not set')),
        ('USE_GPU', os.getenv('USE_GPU', 'not set')),
        ('MAX_WORKERS', os.getenv('MAX_WORKERS', 'not set')),
        ('LOG_LEVEL', os.getenv('LOG_LEVEL', 'not set'))
    ]
    
    print(tabulate(env_vars, headers=['Variable', 'Value'], tablefmt='grid'))

def show_main_config():
    """Display main configuration"""
    try:
        loader = ConfigLoader()
        config = loader.load_main_config()
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    except ConfigurationError as e:
        print(f"Error loading main configuration: {e}")

def show_active_settings():
    """Display active settings from Settings instance"""
    active = {
        'Environment': settings.environment,
        'Debug Mode': settings.debug,
        'Log Level': settings.log_level,
        'Workers': settings.workers,
        'OpenRouter Model': settings.openrouter.model,
        'GPU Enabled': settings.gpu.use_gpu,
        'Max File Size': format_size(settings.limits_config.get('max_file_size', 0)),
        'Cache TTL': f"{settings.performance.cache_ttl} seconds",
        'Redis Host': f"{settings.redis.host}:{settings.redis.port}",
        'Database': f"{settings.database.host}:{settings.database.port}/{settings.database.name}"
    }
    
    for key, value in active.items():
        print(f"{key:.<30} {value}")

def show_vla_config():
    """Display VLA models configuration"""
    try:
        loader = ConfigLoader()
        config = loader.load_vla_models()
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    except ConfigurationError as e:
        print(f"Error loading VLA configuration: {e}")

def show_prompts_config():
    """Display prompts configuration"""
    try:
        loader = ConfigLoader()
        config = loader.load_prompts()
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    except ConfigurationError as e:
        print(f"Error loading prompts configuration: {e}")

def cmd_validate(args):
    """Validate configuration"""
    print("Validating configuration...")
    
    warnings = settings.validate()
    
    if not warnings:
        print("✓ Configuration is valid")
        return 0
    
    print(f"\n⚠ Found {len(warnings)} warning(s):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    
    return 1

def cmd_set(args):
    """Set configuration value"""
    key = args.key
    value = args.value
    
    # Determine which file to modify based on key prefix
    if key.startswith('ENV.'):
        # Modify .env file
        env_key = key[4:]
        set_env_value(env_key, value)
        print(f"Set {env_key}={value} in .env file")
    else:
        print(f"Setting configuration values not yet implemented for: {key}")

def set_env_value(key: str, value: str):
    """Set environment variable in .env file"""
    env_path = Path.cwd() / '.env'
    
    lines = []
    found = False
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    
    if not found:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, 'w') as f:
        f.writelines(lines)

def cmd_init(args):
    """Initialize configuration files"""
    config_dir = Path.cwd() / 'config'
    
    print(f"Initializing configuration in {config_dir}")
    create_default_configs(config_dir)
    
    # Copy .env.example to .env if it doesn't exist
    env_example = Path.cwd() / '.env.example'
    env_file = Path.cwd() / '.env'
    
    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("Created .env from .env.example")
    
    print("✓ Configuration initialized")

def cmd_check(args):
    """Check configuration requirements"""
    checks = []
    
    # Check OpenRouter API key
    if os.getenv('OPENROUTER_API_KEY'):
        checks.append(('OpenRouter API Key', '✓', 'Configured'))
    else:
        checks.append(('OpenRouter API Key', '✗', 'Not configured (required)'))
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host=settings.redis.host, port=settings.redis.port)
        r.ping()
        checks.append(('Redis', '✓', f'Connected to {settings.redis.host}:{settings.redis.port}'))
    except:
        checks.append(('Redis', '⚠', 'Not available (optional)'))
    
    # Check PostgreSQL
    try:
        import psycopg2
        conn_str = f"host={settings.database.host} port={settings.database.port} dbname={settings.database.name}"
        psycopg2.connect(conn_str)
        checks.append(('PostgreSQL', '✓', f'Connected to {settings.database.name}'))
    except:
        checks.append(('PostgreSQL', '⚠', 'Not available (optional)'))
    
    # Check GPU
    if settings.gpu.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                checks.append(('GPU', '✓', f'Available ({torch.cuda.get_device_name(0)})'))
            else:
                checks.append(('GPU', '✗', 'Enabled but not available'))
        except:
            checks.append(('GPU', '✗', 'PyTorch not installed'))
    else:
        checks.append(('GPU', '-', 'Disabled'))
    
    print("\nConfiguration Status:")
    print(tabulate(checks, headers=['Component', 'Status', 'Details'], tablefmt='grid'))

def cmd_export(args):
    """Export configuration"""
    output = args.output
    format_type = args.format
    
    config = settings.to_dict()
    
    if format_type == 'json':
        content = json.dumps(config, indent=2)
    else:  # yaml
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    if output == '-':
        print(content)
    else:
        with open(output, 'w') as f:
            f.write(content)
        print(f"Configuration exported to {output}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='PDF Translation Pipeline Configuration Manager'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument(
        'type',
        choices=['all', 'env', 'main', 'settings', 'vla', 'prompts'],
        default='all',
        nargs='?',
        help='Configuration type to show'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('key', help='Configuration key (e.g., ENV.OPENROUTER_API_KEY)')
    set_parser.add_argument('value', help='Configuration value')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration files')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check configuration requirements')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument(
        '-o', '--output',
        default='-',
        help='Output file (- for stdout)'
    )
    export_parser.add_argument(
        '-f', '--format',
        choices=['json', 'yaml'],
        default='yaml',
        help='Export format'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'show': cmd_show,
        'validate': cmd_validate,
        'set': cmd_set,
        'init': cmd_init,
        'check': cmd_check,
        'export': cmd_export
    }
    
    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())