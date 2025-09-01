"""
This is the RiCA Connector whose purpose is to connect the RiCA Server with the LLM.
"""

# Expose both names for compatibility: transformer_adapter (alias) and transformers_adapter
from . import transformers_adapter as transformer_adapter
from . import transformers_adapter

__all__ = [
    "transformers_adapter",
]

