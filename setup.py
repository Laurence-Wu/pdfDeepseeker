#!/usr/bin/env python3
"""
PDF Translation Pipeline - Environment Setup Script
Automated setup for development and production environments
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import argparse
import json

class Colors:
    """Terminal colors for output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(message):
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is 3.10+"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required. Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_system_requirements():
    """Check system requirements"""
    print_header("System Requirements Check")
    
    system = platform.system()
    print_info(f"Operating System: {system}")
    
    requirements = {
        "python": check_python_version(),
        "git": shutil.which("git") is not None,
        "pip": shutil.which("pip") or shutil.which("pip3") is not None,
    }
    
    # Optional requirements
    optional = {
        "docker": shutil.which("docker"),
        "redis-server": shutil.which("redis-server"),
        "postgres": shutil.which("postgres") or shutil.which("psql"),
        "nvidia-smi": shutil.which("nvidia-smi"),  # For GPU support
    }
    
    for req, available in requirements.items():
        if available:
            print_success(f"{req} is installed")
        else:
            print_error(f"{req} is NOT installed (required)")
    
    print("\n" + "-"*40 + "\n")
    print_info("Optional components:")
    
    for opt, path in optional.items():
        if path:
            print_success(f"{opt} is installed")
        else:
            print_warning(f"{opt} is not installed (optional)")
    
    return all(requirements.values())

def create_virtual_environment():
    """Create Python virtual environment"""
    print_header("Setting up Virtual Environment")
    
    venv_path = Path.cwd() / "venv"
    
    if venv_path.exists():
        response = input(f"{Colors.WARNING}Virtual environment already exists. Recreate? (y/N): {Colors.ENDC}")
        if response.lower() != 'y':
            print_info("Using existing virtual environment")
            return True
        shutil.rmtree(venv_path)
    
    print_info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_success("Virtual environment created")
        
        # Get activation command based on OS
        if platform.system() == "Windows":
            activate_cmd = "venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print_info(f"To activate, run: {Colors.BOLD}{activate_cmd}{Colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def install_dependencies(dev=False, gpu=False):
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    # Determine pip command
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    if not Path(pip_cmd).exists():
        pip_cmd = "pip3" if shutil.which("pip3") else "pip"
    
    try:
        # Upgrade pip first
        print_info("Upgrading pip...")
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print_success("pip upgraded")
        
        # Install main requirements
        print_info("Installing main dependencies (this may take several minutes)...")
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print_success("Main dependencies installed")
        
        # Install GPU dependencies if requested
        if gpu:
            print_info("Installing GPU dependencies...")
            subprocess.run([pip_cmd, "install", "torch", "torchvision", "--index-url", 
                          "https://download.pytorch.org/whl/cu118"], check=True)
            print_success("GPU dependencies installed")
        
        # Install development dependencies if requested
        if dev:
            print_info("Installing development dependencies...")
            dev_packages = ["pytest", "black", "flake8", "mypy", "pre-commit"]
            subprocess.run([pip_cmd, "install"] + dev_packages, check=True)
            print_success("Development dependencies installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def setup_configuration():
    """Setup configuration files"""
    print_header("Setting up Configuration")
    
    # Create .env from .env.example
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists():
        if not env_file.exists():
            shutil.copy(env_example, env_file)
            print_success("Created .env from .env.example")
            print_warning("Please edit .env and add your OpenRouter API key")
        else:
            print_info(".env file already exists")
    
    # Create necessary directories
    directories = [
        "data/uploads",
        "data/outputs",
        "data/cache",
        "logs",
        "models",
        "tmp"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print_success(f"Created {len(directories)} directories")
    
    return True

def setup_docker_services():
    """Setup Docker services"""
    print_header("Docker Services Setup")
    
    if not shutil.which("docker"):
        print_warning("Docker not installed. Skipping Docker setup.")
        print_info("Install Docker from: https://docs.docker.com/get-docker/")
        return False
    
    try:
        # Check if Docker is running
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        print_success("Docker is running")
        
        response = input(f"{Colors.OKCYAN}Start Docker services (Redis, PostgreSQL)? (y/N): {Colors.ENDC}")
        if response.lower() == 'y':
            print_info("Starting Docker services...")
            subprocess.run(["docker", "compose", "up", "-d", "redis", "postgres"], check=True)
            print_success("Docker services started")
            print_info("Services running on:")
            print_info("  - Redis: localhost:6379")
            print_info("  - PostgreSQL: localhost:5432")
        
        return True
    except subprocess.CalledProcessError:
        print_warning("Docker is not running. Please start Docker first.")
        return False

def download_models():
    """Download required models"""
    print_header("Model Downloads")
    
    print_info("The following models will be downloaded on first use:")
    models = [
        ("Surya (Layout Detection)", "~500MB"),
        ("PaddleOCR (Text Recognition)", "~300MB"),
        ("LaTeX-OCR (Formula Detection)", "~200MB"),
        ("Table Transformer", "~150MB"),
    ]
    
    for model, size in models:
        print(f"  • {model}: {size}")
    
    response = input(f"\n{Colors.OKCYAN}Pre-download models now? (y/N): {Colors.ENDC}")
    if response.lower() == 'y':
        print_info("Creating model download script...")
        # The actual model download would happen here
        print_warning("Model download not yet implemented. Models will download on first use.")
    
    return True

def verify_installation():
    """Verify the installation"""
    print_header("Installation Verification")
    
    checks = []
    
    # Check Python imports
    try:
        import fastapi
        checks.append(("FastAPI", True))
    except ImportError:
        checks.append(("FastAPI", False))
    
    try:
        import pdfplumber
        checks.append(("PDF Processing", True))
    except ImportError:
        checks.append(("PDF Processing", False))
    
    try:
        import redis
        checks.append(("Redis Client", True))
    except ImportError:
        checks.append(("Redis Client", False))
    
    try:
        import yaml
        checks.append(("YAML Support", True))
    except ImportError:
        checks.append(("YAML Support", False))
    
    # Check configuration
    if Path(".env").exists():
        checks.append(("Configuration", True))
    else:
        checks.append(("Configuration", False))
    
    # Print results
    all_passed = True
    for component, status in checks:
        if status:
            print_success(f"{component} ready")
        else:
            print_error(f"{component} not ready")
            all_passed = False
    
    return all_passed

def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print("1. Configure your environment:")
    print(f"   {Colors.OKCYAN}Edit .env and add your OpenRouter API key{Colors.ENDC}")
    print(f"   Get your key from: https://openrouter.ai/\n")
    
    print("2. Activate virtual environment:")
    if platform.system() == "Windows":
        print(f"   {Colors.OKCYAN}venv\\Scripts\\activate{Colors.ENDC}\n")
    else:
        print(f"   {Colors.OKCYAN}source venv/bin/activate{Colors.ENDC}\n")
    
    print("3. Start the API server:")
    print(f"   {Colors.OKCYAN}make api{Colors.ENDC}")
    print("   or")
    print(f"   {Colors.OKCYAN}python -m uvicorn src.api.main:app --reload{Colors.ENDC}\n")
    
    print("4. Start background workers (optional):")
    print(f"   {Colors.OKCYAN}make worker{Colors.ENDC}\n")
    
    print("5. Access the API:")
    print(f"   {Colors.OKCYAN}http://localhost:8000{Colors.ENDC}")
    print(f"   {Colors.OKCYAN}http://localhost:8000/docs{Colors.ENDC} (Interactive API docs)\n")
    
    print("6. Check configuration:")
    print(f"   {Colors.OKCYAN}./scripts/config.sh check{Colors.ENDC}\n")

def main():
    parser = argparse.ArgumentParser(description="PDF Translation Pipeline Setup")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU dependencies")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    
    args = parser.parse_args()
    
    print_header("PDF Translation Pipeline Setup")
    print("This script will set up your development environment\n")
    
    # Run setup steps
    steps = [
        ("System Requirements", check_system_requirements),
        ("Virtual Environment", create_virtual_environment),
        ("Dependencies", lambda: install_dependencies(dev=args.dev, gpu=args.gpu)),
        ("Configuration", setup_configuration),
    ]
    
    if not args.skip_docker:
        steps.append(("Docker Services", setup_docker_services))
    
    if not args.skip_models:
        steps.append(("Model Downloads", download_models))
    
    steps.append(("Verification", verify_installation))
    
    # Execute steps
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"\n{step_name} failed. Please fix the issues and run setup again.")
            sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()