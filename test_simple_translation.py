#!/usr/bin/env python3
"""
Quick test of translation pipeline on simple PDF
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run the demonstration with simple PDF
os.system('python3 demonstration/scripts/demonstrate_reconstruction.py')
