import asyncio
import json
import re
from typing import Any, Callable, List, Optional

from .. import router

__all__ = ["_ReasoningThreadTemplate"]


class _ReasoningThreadTemplate:
    """
    Base template for reasoning adapters.

    This class provides reusable, framework-agnostic helpers for:
    - Managing a textual context buffer.
    - Registering callbacks via trigger and emitting incremental pieces via _emit.
    - Detecting and executing <rica ...>...</rica> tool calls appended at the tail
      of the context by delegating to router._execute.

    Subclasses should implement model-specific generation and lifecycle methods
    (insert/wait/run/pause/destroy), calling _emit for each new piece of text and
    optionally using _detect_and_execute_tool_tail inside their generation loop.
    """

    def __init__(self, context: str = ""):
        self._context: str = context or ""
        self._callbacks: List[Callable[[str], Any]] = []

    # ---- Lifecycle placeholders (to be implemented by subclasses) ----
    async def insert(self, text: Any):
        """Insert external text into the context (to be implemented by subclass)."""
        raise NotImplementedError

    async def wait(self):
        """Wait for current reasoning/generation to complete (to be implemented)."""
        raise NotImplementedError

    async def destroy(self):
        """Stop and clean up resources (to be implemented)."""
        raise NotImplementedError

    async def run(self):
        """Start or resume reasoning (to be implemented)."""
        raise NotImplementedError

    async def pause(self):
        """Pause reasoning (to be implemented)."""
        raise NotImplementedError

    @property
    def context(self) -> str:
        """Return the current textual context buffer."""
        return self._context

    # ---- Common helpers ----
    @staticmethod
    def _ensure_async(function: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callable into an async function if it is not already async."""
        if asyncio.iscoroutinefunction(function):
            return function

        async def _wrapped(*args, **kwargs):
            return await asyncio.to_thread(function, *args, **kwargs)

        return _wrapped

    def trigger(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register a callback that will be called whenever _emit is invoked.

        The returned wrapper preserves decorator semantics. The callback can be
        a coroutine function or a normal function; normal functions are executed
        in a separate thread via asyncio.to_thread.
        """
        self._callbacks.append(function)

        async def _run_cb(*args, **kwargs):
            fn = self._ensure_async(function)
            return await fn(*args, **kwargs)

        def wrapper(*args, **kwargs):
            asyncio.create_task(_run_cb(*args, **kwargs))

        return wrapper

    async def _emit(self, piece: str):
        """Emit a new piece of text to all registered callbacks."""
        if not piece:
            return
        for cb in list(self._callbacks):
            if asyncio.iscoroutinefunction(cb):
                asyncio.create_task(cb(piece))
            else:
                asyncio.create_task(asyncio.to_thread(cb, piece))

    async def _detect_and_execute_tool_tail(self) -> tuple[bool, Optional[str]]:
        """
        Detect a trailing <rica ...>...</rica> in the current context, execute it
        via router._execute, append the result to the context, and emit it.

        Returns True if a tool call was detected and executed.
        """
        tail = self._context[-400:]
        pattern = r"<rica\s+[^>]*>.*?</rica>$"
        m = re.search(pattern, tail, re.DOTALL)
        if not m:
            return False, None

        # Extract the full last tag from the entire context to ensure complete content
        last = None
        for match in re.finditer(r"<rica\s+[^>]*>.*?</rica>", self._context, re.DOTALL):
            last = match
        if not last:
            return False
        tag_text = self._context[last.start() : last.end()]

        try:
            result = await router._execute(tag_text)
            appended: str
            if hasattr(result, "callback"):
                payload = result.callback
                if isinstance(payload, (dict, list)):
                    appended = json.dumps(payload, ensure_ascii=False)
                else:
                    appended = str(payload)
            else:
                # UUID or other object
                appended = json.dumps({"call_id": str(result)}, ensure_ascii=False)

            self._context += appended
            await self._emit(appended)
            return True, appended
        except Exception as e:
            await self._emit(f"[tool-error]{type(e).__name__}: {e}")
            return False, None