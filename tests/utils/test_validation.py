import pytest
from rica.utils.validation import validate_tool_input, sanitize_code

def test_validate_tool_input_valid():
    """Test valid tool input against a JSON schema."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    data = {"name": "test"}
    assert validate_tool_input(schema, data) is True

def test_validate_tool_input_invalid():
    """Test invalid tool input against a JSON schema."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    data = {"name": 123}
    assert validate_tool_input(schema, data) is False

def test_sanitize_code_valid():
    """Test valid code sanitization."""
    code = "print('hello')"
    sanitized = sanitize_code(code)
    assert sanitized == "print('hello')"

def test_sanitize_code_too_long():
    """Test that code exceeding max length raises an error."""
    code = "a" * 1001
    with pytest.raises(ValueError):
        sanitize_code(code)

def test_sanitize_code_dangerous_pattern():
    """Test that code with dangerous patterns raises an error."""
    with pytest.raises(ValueError):
        sanitize_code("exec('print(1)')")
    with pytest.raises(ValueError):
        sanitize_code("__import__('os').system('ls')")
