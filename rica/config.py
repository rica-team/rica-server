"""
Configuration module for RiCA.
"""

import os

# Default model name for the transformers adapter
DEFAULT_MODEL_NAME = os.getenv("RICA_DEFAULT_MODEL", "google/gemma3-4b-it")
