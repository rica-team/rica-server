from typing import Any

import jsonschema


def validate_tool_input(schema: dict, data: Any) -> bool:
    """Validate tool input against a JSON Schema."""
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


def sanitize_code(code: str, max_length: int = 1000) -> str:
    """Sanitize and limit code input."""
    code = code.strip()
    if len(code) > max_length:
        raise ValueError(f"Code exceeds max length of {max_length}")
    # Prohibit dangerous imports and calls
    dangerous_patterns = ["__import__", "exec", "eval", "compile", "open"]
    for pattern in dangerous_patterns:
        if pattern in code:
            raise ValueError(f"Dangerous pattern '{pattern}' detected")
    return code
