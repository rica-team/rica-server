"""
This is the RiCA Connector whose purpose is to connect the RiCA Server with the LLM.
"""

# Import the specific adapter to expose it at the package level
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

__all__ = ["transformers_adapter", "transformer_adapter"]
