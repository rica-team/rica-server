import asyncio
import functools
import json
import logging
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from uuid import uuid4

from rica.core.application import CallBack, RiCA, Route
from rica.exceptions import (
    ExecutionTimedOut,
    InvalidRiCAString,
    PackageExistError,
    PackageNotFoundError,
    RouteNotFoundError,
    UnexpectedExecutionError,
)
from rica.utils.package_loader import load_app_from_path
from rica.utils.parser import parse_rica_tag

__all__ = ["ReasoningThreadBase"]

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ReasoningThreadBase:
    """
    Base template for reasoning adapters.

    This class provides reusable, framework-agnostic helpers for:
    - Managing a textual context buffer.
    - Managing a collection of RiCA applications.
    - Registering two types of callbacks:
      - @token_generated: Receives every raw token from the model.
      - @trigger: Receives only the payload from a rica.response tool call.
    - Detecting and executing <rica ...>...</rica> tool calls.

    Attributes:
        _apps: A dictionary of installed RiCA applications.
        _context: The textual context buffer.
    """

    def __init__(self, context: str = ""):
        """
        Initializes the reasoning thread.

        Note:
            The instance must be initialized by calling `await instance.initialize()`
            before any other operations.

        Args:
            context: The initial context string.
        """
        self._apps: Dict[str, RiCA] = {}
        self._apps_lock = asyncio.Lock()
        self._context: str = context or ""
        self._response_callbacks: List[Callable[[Any], Any]] = []  # For @trigger
        self._token_callbacks: List[Callable[[str], Any]] = []  # For @token_generated
        self._initialized = False

    async def initialize(self):
        """Initialize the reasoning thread by installing system apps."""
        if self._initialized:
            return
        # Install the virtual 'rica' app for system prompts
        await self.install(RiCA("rica"))
        self._initialized = True

    async def install(self, app: Union[RiCA, str]):
        """
        Installs a RiCA application.

        Args:
            app: An instance of the RiCA class, or a file path (str) to a .py file
                 or .tar.gz archive.

        Raises:
            TypeError: If the 'app' argument is invalid.
            PackageExistError: If an application with the same package name is
                               already installed.
            ImportError: If loading from file fails.
        """
        if isinstance(app, str):
            # Dynamic loading logic
            app_instance = await load_app_from_path(app)
        elif isinstance(app, RiCA):
            app_instance = app
        else:
            raise TypeError("The 'app' argument must be an instance of RiCA or a file path string.")

        async with self._apps_lock:
            if app_instance.package in self._apps:
                raise PackageExistError(
                    f"Application with package '{app_instance.package}' is already installed."
                )
            self._apps[app_instance.package] = app_instance



    async def uninstall(self, package_name: str):
        """
        Uninstalls a RiCA application by its package name.

        Args:
            package_name: The package name of the application to uninstall.

        Raises:
            PackageNotFoundError: If the application with the given package name is not found.
        """
        async with self._apps_lock:
            if package_name not in self._apps:
                raise PackageNotFoundError(f"Application with package '{package_name}' not found.")
            del self._apps[package_name]

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

    def run(self):
        """Start or resume reasoning (to be implemented)."""
        raise NotImplementedError

    def pause(self):
        """Pause reasoning (to be implemented)."""
        raise NotImplementedError

    @property
    def context(self) -> str:
        """Return the current textual context buffer."""
        return self._context

    # ---- Common helpers ----
    def trigger(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register a callback that will be called for `rica.response` tool calls.
        This is for delivering final responses to the user.
        """
        self._response_callbacks.append(function)
        return function

    def token_generated(self, function: Callable[[str], Any]) -> Callable[[str], Any]:
        """
        Register a callback that will be called for every token generated by the model.
        This is for observing the model's "thinking" process.
        """
        self._token_callbacks.append(function)
        return function

    async def _emit_response(self, payload: Any):
        """Emit a final response payload to all @trigger callbacks."""
        if not payload:
            return
        tasks = [
            asyncio.create_task(cb(payload))
            for cb in self._response_callbacks
            if asyncio.iscoroutinefunction(cb)
        ]
        tasks.extend(
            [
                asyncio.create_task(asyncio.to_thread(cb, payload))
                for cb in self._response_callbacks
                if not asyncio.iscoroutinefunction(cb)
            ]
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _emit_token(self, piece: str):
        """Emit a raw generated token to all @token_generated callbacks."""
        if not piece:
            return
        tasks = [
            asyncio.create_task(cb(piece))
            for cb in self._token_callbacks
            if asyncio.iscoroutinefunction(cb)
        ]
        tasks.extend(
            [
                asyncio.create_task(asyncio.to_thread(cb, piece))
                for cb in self._token_callbacks
                if not asyncio.iscoroutinefunction(cb)
            ]
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_tool_call(
        self, app: Route, data: list | dict
    ) -> Union[CallBack, Coroutine[Any, Any, None]]:
        """
        Execute a registered function.

        Args:
            app: The Route object representing the tool to be executed.
            data: The input data for the tool.

        Returns:
            A CallBack object for synchronous tools, or a coroutine for background tools.
        """
        function = app.function
        timeout = app.timeout
        background = app.background

        start_time = asyncio.get_event_loop().time()

        try:
            if not asyncio.iscoroutinefunction(function):
                function = asyncio.to_thread(functools.partial(function, data))
            else:
                function = function(data)

            if background:
                call_id = uuid4()
                loop = asyncio.get_running_loop()
                task = loop.create_task(function, name=str(call_id))

                def on_task_done(t):
                    try:
                        if t.cancelled():
                            logger.warning(f"Task {call_id} was cancelled")
                        elif t.exception():
                            logger.error(f"Task {call_id} failed", exc_info=t.exception())
                    except Exception as e:
                        logger.error(f"Error in task callback: {e}")

                task.add_done_callback(on_task_done)

                if timeout > 0:
                    cancel_handle = loop.call_later(timeout / 1000, task.cancel)
                    task.add_done_callback(lambda _: cancel_handle.cancel())
                return call_id
            else:
                try:
                    result = (
                        await asyncio.wait_for(function, timeout / 1000)
                        if timeout > 0
                        else await function
                    )
                    duration = (asyncio.get_event_loop().time() - start_time) * 1000
                    return CallBack(
                        package=app.route.split("/")[0],
                        route=app.route,
                        call_id=uuid4(),
                        callback=result,
                        duration_ms=duration,
                    )
                except asyncio.TimeoutError as e:
                    logger.error(f"Tool call timed out after {timeout}ms")
                    raise ExecutionTimedOut from e
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}", exc_info=True)
                    raise UnexpectedExecutionError(str(e)) from e
        except Exception as e:
            logger.critical(f"Critical error in _execute_tool_call: {e}", exc_info=True)
            raise

    async def create_sub_thread(
        self, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create a sub-thread. To be implemented by subclasses."""
        raise NotImplementedError

    async def _detect_and_execute_tool_tail(self) -> tuple[bool, Optional[str]]:
        """Detect and execute all <rica ...>...</rica> tags in the context."""
        # Find all tags
        pattern = r"<rica\s+[^>]*>.*?<\/rica>"
        matches = list(re.finditer(pattern, self._context, re.DOTALL | re.IGNORECASE))

        if not matches:
            return False, None

        # Only process tags that haven't been processed yet.
        # Since we append results to the context, we need a way to distinguish processed tags.
        # A simple heuristic is to check if the tag is followed by a result block
        # or if it's at the very end.
        # However, simpler is to assume the model generates tags at the end.
        # But if we have multiple tags, we want to process them all.

        # Strategy: Find the last contiguous block of tags at the end of the context.
        # This is tricky because there might be whitespace.

        # Simplified Strategy:
        # 1. Find all tags.
        # 2. Filter out tags that already have a corresponding result (this is hard without IDs).
        # 3. Actually, the standard flow is:
        #    Model generates -> Stop at </rica> -> We process -> Append result.
        # If the model generates multiple tags in one go (e.g. <rica>A</rica> <rica>B</rica>),
        # the stopping criteria might stop at the first </rica> or the last?
        # The current stopping criteria stops at ANY </rica>.
        # So we likely only have ONE tag at the end.
        # BUT, if the model is fast or we change stopping criteria, we might have multiple.
        # Let's assume we process the LAST tag found.
        # Wait, user wants "simultaneous tool calls".
        # If the model outputs: <rica>A</rica>\n<rica>B</rica>
        # We should execute both.

        # Let's try to find all tags that are NOT followed by a result block (e.g. JSON or UUID).
        # This is getting complex.
        # Alternative: The `_context` grows. We can keep track of the `last_processed_index`.

        # For now, let's stick to the "tail" concept but expand it to "tail block".
        # We look for the last occurrence of a tag.
        # If we want parallel, we need to find ALL tags in the newly generated text.
        # But `_detect_and_execute_tool_tail` is called after generation stops.
        # If generation stops at `</rica>`, we might only have one tag.
        # UNLESS we change stopping criteria to NOT stop, or the model outputs multiple
        # tags quickly.

        # Let's assume the model outputs one tag at a time for now, as per the stopping criteria.
        # To support parallel, the model would need to output multiple tags BEFORE stopping.
        # If we want to support that, we need to adjust `_ToolCallStoppingCriteria` in
        # `transformers_adapter.py`.

        # For this step, let's just make `_detect_and_execute_tool_tail` robust enough
        # to handle multiple tags if they exist.

        # We will process ALL tags found at the end of the string.
        # We iterate backwards? No.

        # Let's just process the last match for now to maintain stability,
        # but if we want parallel, we should process all matches that don't look like
        # they have a result.
        # Given the complexity and the "Demo" nature, let's implement a robust
        # "process all new tags" logic.
        # We can assume "new tags" are those after the last known context length?
        # `ReasoningThread` doesn't track "last known length" explicitly in `base.py`.

        # Let's stick to the original logic but loop through ALL matches found in the "tail".
        # We define "tail" as the text after the last "}" (end of a JSON result) or just the end.

        # Actually, let's just process the LAST match. If the user wants parallel,
        # the model should output `<rica>A</rica><rica>B</rica>` and we should process both.

        combined_result = ""

        # We only process the matches that are at the end of the context.
        # i.e. after the last non-whitespace character that is NOT part of a tag.
        # This is hard.

        # Let's just iterate over all matches and check if they look "unprocessed".
        # An unprocessed tag is usually at the end of the context.

        # Let's try to process the last match only, as per current design.
        # To support parallel, we would need to change the loop in `transformers_adapter`
        # to NOT stop immediately?
        # Or we just process all tags found in the `_context`.
        # But we don't want to re-process old tags.

        # Hack: We can check if the tag is followed by a result.
        # But for now, let's just stick to the last one to avoid breaking things,
        # and maybe the user means "parallel threads" not "parallel tools in one thread".
        # User said "同时调用多工具".

        # OK, let's try to find ALL tags that are at the end.
        # We can iterate from the last match backwards until we find something that is not a tag?

        # Let's simplify: Just find the last match. If there are multiple,
        # the previous loop would have caught them?
        # No, `_run_loop` calls this once per generation cycle.

        # If we want parallel, we need to find ALL tags in the current buffer that are not
        # followed by a result.
        # Let's assume any <rica> tag NOT followed by `{` (start of result) or `call_id` is new.

        matches = list(re.finditer(pattern, self._context, re.DOTALL | re.IGNORECASE))
        if not matches:
            return False, None

        # Filter matches that are likely already processed
        candidates = []
        for m in matches:
            end_pos = m.end()
            # Check text after this match
            following_text = self._context[end_pos:].strip()
            # If following text starts with {, it's likely a result.
            if not following_text or (
                not following_text.startswith("{") and not following_text.startswith("[")
            ):
                candidates.append(m)

        if not candidates:
            return False, None

        # Execute all candidates in parallel
        tasks = []
        for match in candidates:
            tag_text = match.group(0)
            tasks.append(self._parse_and_execute(tag_text))

        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                self._context += res
                await self._emit_token(res)
                combined_result += res

        return True, combined_result

    async def _parse_and_execute(self, tag_text: str) -> str:
        """Helper to parse and execute a single tag."""
        try:
            package_name, route_name, content = parse_rica_tag(tag_text)

            async with self._apps_lock:
                app_instance = self._apps.get(package_name)
                if not app_instance:
                    raise PackageNotFoundError(f"Package '{package_name}' not found")

                application = app_instance.find_route(route_name)
                if not application:
                    raise RouteNotFoundError(f"Route '{route_name}' not found")

            # Special handling for rica/response
            if package_name == "rica" and route_name == "/response":
                await self._emit_response(content)
                return ""  # No result appended for response

            result = await self._execute_tool_call(application, content)

            if isinstance(result, CallBack):
                payload = result.callback
                appended = (
                    json.dumps(payload, ensure_ascii=False)
                    if isinstance(payload, (dict, list))
                    else str(payload)
                )
            else:  # UUID
                appended = json.dumps({"call_id": str(result)}, ensure_ascii=False)

            return appended

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return f"[tool-error]{type(e).__name__}: {e}"
