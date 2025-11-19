import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages the lifecycle of sub-threads.
    """

    def __init__(self):
        self._threads: Dict[str, Any] = {}  # id -> ReasoningThread instance

    def register_thread(self, thread_id: str, thread_instance: Any):
        self._threads[thread_id] = thread_instance

    def unregister_thread(self, thread_id: str):
        self._threads.pop(thread_id, None)

    def get_thread(self, thread_id: str) -> Optional[Any]:
        return self._threads.get(thread_id)

    def list_threads(self) -> Dict[str, str]:
        return {tid: str(t) for tid, t in self._threads.items()}

    async def spawn(self, parent_thread: Any, input_data: dict) -> dict:
        """
        Spawn a new sub-thread.

        Input:
        {
            "task": "string - initial prompt",
            "model": "string - optional model name",
            "config": "dict - optional generation config"
        }
        """
        task = input_data.get("task")
        if not task:
            return {"error": "Missing 'task' parameter"}

        model_name = input_data.get("model")
        config = input_data.get("config")

        # Create a new thread instance
        # We need to import ReasoningThread dynamically to avoid circular imports
        # or rely on a factory. For now, we assume the parent thread has a factory method.
        if not hasattr(parent_thread, "create_sub_thread"):
            return {"error": "Parent thread does not support spawning sub-threads"}

        try:
            sub_thread = await parent_thread.create_sub_thread(model_name=model_name, config=config)
            thread_id = str(uuid.uuid4())
            self.register_thread(thread_id, sub_thread)

            # Start the thread
            await sub_thread.initialize()
            sub_thread.run()

            # Insert the task
            await sub_thread.insert(task)

            return {"status": "success", "thread_id": thread_id, "message": "Thread spawned"}
        except Exception as e:
            logger.error(f"Failed to spawn thread: {e}", exc_info=True)
            return {"error": f"Failed to spawn thread: {str(e)}"}

    async def kill(self, input_data: dict) -> dict:
        thread_id = input_data.get("thread_id")
        if not thread_id:
            return {"error": "Missing 'thread_id'"}

        thread = self.get_thread(thread_id)
        if not thread:
            return {"error": f"Thread '{thread_id}' not found"}

        await thread.destroy()
        self.unregister_thread(thread_id)
        return {"status": "success", "message": f"Thread '{thread_id}' killed"}

    def list(self, input_data: dict) -> dict:
        return {"threads": self.list_threads()}


# Singleton
_thread_manager = ThreadManager()


def get_thread_manager():
    return _thread_manager
