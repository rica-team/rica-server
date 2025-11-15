"""
This is the RiCA Connector whose purpose is to connect the RiCA Server with the LLM.
"""

# Expose both names for compatibility: transformer_adapter (alias) and transformers_adapter

def __getattr__(name):
    match name:
        case "transformers_adapter" | "transformer_adapter":
            return __import__(".transformers_adapter")
        case _:
            return ImportError("Adapter {} not found".format(name))
