import json
import logging
import re
from typing import Any, Tuple
from xml.etree import ElementTree as ET

from rica.exceptions import InvalidRiCAString

logger = logging.getLogger(__name__)


def parse_rica_tag(tag_text: str) -> Tuple[str, str, Any]:
    """
    Parses a single <rica> tag and returns (package, route, content).

    Args:
        tag_text: The raw string of the <rica> tag.

    Returns:
        A tuple containing (package_name, route_name, content).
        Content is a dict or list if JSON, or a string.

    Raises:
        InvalidRiCAString: If the tag is malformed or missing attributes.
    """
    try:
        # Attempt to parse XML
        try:
            root = ET.fromstring(tag_text)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}, tag_text: {tag_text[:200]}")
            # Fallback regex
            package_match = re.search(r'package=["\']([^"\']+)["\']', tag_text)
            route_match = re.search(r'route=["\']([^"\']+)["\']', tag_text)

            if package_match and route_match:
                package_name = package_match.group(1)
                route_name = route_match.group(1)
                content_match = re.search(r">\s*(.*?)\s*<\/rica>", tag_text, re.DOTALL)
                content_str = content_match.group(1) if content_match else ""

                class MockRoot:
                    attrib = {"package": package_name, "route": route_name}
                    text = content_str

                root = MockRoot()
            else:
                raise InvalidRiCAString(f"Invalid XML format: {e}")

        package_name = root.attrib.get("package")
        route_name = root.attrib.get("route")

        if not package_name or not route_name:
            raise InvalidRiCAString("Missing package or route attribute")

        content_str = root.text or ""
        content = json.loads(content_str) if content_str.strip() else {}

        return package_name, route_name, content

    except Exception as e:
        if isinstance(e, InvalidRiCAString):
            raise
        raise InvalidRiCAString(f"Error parsing RiCA tag: {e}") from e
