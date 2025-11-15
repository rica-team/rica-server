"""
This is the RiCA Connector whose purpose is to connect the RiCA Server with the LLM.
"""

# Import the specific adapter to expose it at the package level
from . import transformers_adapter as transformers_adapter
from . import transformers_adapter as transformer_adapter # alias for compatibility

__all__ = ["transformers_adapter", "transformer_adapter"]
