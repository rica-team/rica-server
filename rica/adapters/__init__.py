"""
This is the RiCA Connector whose purpose is to connect the RiCA Server with the LLM.
"""

# Base adapter class
from .base import ReasoningThreadBase

# ---- Adapters ----

# Transformers Adapter (requires `pip install .[pt]`)
try:
    from . import transformers_adapter as transformer_adapter  # alias for compatibility
    from . import transformers_adapter as transformers_adapter
except ImportError:

    class _MissingAdapter:
        def __getattr__(self, name):
            raise ImportError(
                "The transformers adapter requires 'transformers' and 'torch'. "
                "Please install them with `pip install .[pt]`."
            )

    transformers_adapter = _MissingAdapter()
    transformer_adapter = _MissingAdapter()

# Future adapters (e.g., OpenAI, Anthropic) can be added here in a similar pattern.

__all__ = ["ReasoningThreadBase", "transformers_adapter", "transformer_adapter"]
