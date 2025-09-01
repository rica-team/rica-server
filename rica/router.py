"""
Router utilities for RiCA Server.

This module decodes <rica ...>...</rica> tool-call strings, locates the
registered package in the provided RiCA application instance, and executes the
corresponding function.
"""
import asyncio
import json
from uuid import uuid4, UUID
from typing import Callable, Any

from .server import RiCA, CallBack, Application
from .exceptions import *

__all__ = ["execute_tool_call"]


async def _call(app: Application, data: list | dict) -> UUID | CallBack:
    """
    Execute a registered function either in background or foreground.

    - If background is True, schedule the call on the current event loop and return a UUID.
    - If background is False, await completion and return a CallBack wrapping the result.
    """
    function = app.function
    timeout = app.timeout
    background = app.background

    # Ensure the function is awaitable
    if not asyncio.iscoroutinefunction(function):
        # functools.partial is used to pass args to the thread
        import functools
        function = asyncio.to_thread(functools.partial(function, data))
    else:
        function = function(data)

    if background:
        call_id = uuid4()
        loop = asyncio.get_running_loop()
        task = loop.create_task(function, name=str(call_id))
        if timeout > 0:
            # Use a handle to cancel the cancellation task if the main task finishes early
            cancel_handle = loop.call_later(timeout / 1000, task.cancel)
            task.add_done_callback(lambda _: cancel_handle.cancel())
        return call_id
    else:
        try:
            if timeout > 0:
                result = await asyncio.wait_for(function, timeout / 1000)
            else:
                result = await function
            return CallBack(package=str(uuid4()), call_id=uuid4(), callback=result)
        except asyncio.TimeoutError as e:
            raise ExecutionTimedOut from e
        except Exception as e:
            raise UnexpectedExecutionError(str(e)) from e


def _decode(app: RiCA, input_: str) -> tuple[Application, list | dict]:
    """Parse a <rica ...>...</rica> string and return the target application and its data."""
    if not isinstance(input_, str):
        raise InvalidRiCAString("Input must be a string")

    input_ = input_.strip()
    if not (input_.startswith("<rica") and input_.endswith("</rica>")):
        raise InvalidRiCAString("Invalid RiCA format: must start with <rica and end with </rica>")

    try:
        # Find the end of the opening tag to separate attributes from content
        tag_end_pos = input_.find(">")
        if tag_end_pos == -1:
            raise InvalidRiCAString("Invalid RiCA format: missing closing '>' for opening tag")

        header = input_[:tag_end_pos]

        # Extract package name from attributes
        pkg_start = header.find('package="') + 9
        pkg_end = header.find('"', pkg_start)
        if pkg_start < 9 or pkg_end == -1:
            raise InvalidRiCAString("Missing or malformed package attribute")
        package_name = header[pkg_start:pkg_end]

        # Extract data content
        data_start = tag_end_pos + 1
        data_end = input_.rfind("</rica>")
        if data_end == -1:
            # This case should be caught by the initial endswith check, but for safety
            raise InvalidRiCAString("Invalid RiCA format: missing closing </rica> tag")
        data_str = input_[data_start:data_end]

        data = json.loads(data_str) if data_str else {}

        application = app.find_endpoint(package_name)
        if not application:
            raise PackageNotFoundError(f"Package '{package_name}' not found")

        return application, data

    except json.JSONDecodeError as e:
        raise InvalidRiCAString("Invalid JSON data in RiCA tag") from e
    except Exception as e:
        raise InvalidRiCAString(f"Failed to decode RiCA format: {e}") from e


async def execute_tool_call(app: RiCA, tool_string: str) -> UUID | CallBack:
    """
    Decode and execute a tool-call string using the provided RiCA application.

    Args:
        app: The RiCA instance containing the tool definitions.
        tool_string: The full tool call string, e.g., '<rica package="...">...</rica>'.

    Returns:
        A UUID for background tasks or a CallBack object for foreground tasks.
    """
    application, data = _decode(app, tool_string)
    return await _call(application, data)
