import logging
from typing import Dict

logger = logging.getLogger(__name__)


class Whiteboard:
    """
    A shared whiteboard for storing context and data across threads.
    """

    def __init__(self):
        self._content: Dict[str, str] = {}
        self._descriptions: Dict[str, str] = {}

    def handle(self, input_data: dict) -> dict:
        """
        Handle whiteboard operations.

        Input Schema:
        {
            "action": "read" | "write" | "append" | "clear",
            "whiteboard_id": "string",
            "content": "string (optional)",
            "description": "string (optional)"
        }
        """
        action = input_data.get("action")
        wb_id = input_data.get("whiteboard_id", "default")
        content = input_data.get("content", "")
        description = input_data.get("description")

        if not action:
            return {"error": "Missing 'action' parameter"}

        if action == "read":
            return {
                "content": self._content.get(wb_id, ""),
                "description": self._descriptions.get(wb_id, ""),
            }

        elif action == "write":
            self._content[wb_id] = content
            if description is not None:
                self._descriptions[wb_id] = description
            return {"status": "success", "message": f"Whiteboard '{wb_id}' updated"}

        elif action == "append":
            current = self._content.get(wb_id, "")
            self._content[wb_id] = current + "\n" + content
            if description is not None:
                self._descriptions[wb_id] = description
            return {"status": "success", "message": f"Appended to whiteboard '{wb_id}'"}

        elif action == "clear":
            self._content.pop(wb_id, None)
            self._descriptions.pop(wb_id, None)
            return {"status": "success", "message": f"Whiteboard '{wb_id}' cleared"}

        elif action == "list":
            return {
                "whiteboards": [
                    {"id": wid, "description": self._descriptions.get(wid, "")}
                    for wid in self._content.keys()
                ]
            }

        else:
            return {"error": f"Unknown action: {action}"}


# Singleton instance
_whiteboard_instance = Whiteboard()


def get_whiteboard():
    return _whiteboard_instance
