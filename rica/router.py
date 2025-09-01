"""
Router utilities for RiCA Server.

This module decodes <rica ...>...</rica> tool-call strings, locates the
registered package in the configured RiCA application, and executes the
corresponding function either in the foreground (awaited) or background
(asyncio Task with optional timeout). It returns a UUID for background calls
or a CallBack object for foreground calls.
"""
import asyncio
import json
from uuid import uuid4, UUID
from typing import Callable, Any

from .server import RiCA, CallBack
from .exceptions import *

__all__ = ["preset"]

try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

_interactive_tasks:list[asyncio.Task] = []
_deactivate_tasks:list[asyncio.Task] = []

_application:RiCA = RiCA()

class _Router:
    """Holds the configured RiCA application for routing tool calls."""

    def __init__(self):
        self._application: RiCA = RiCA()
        self._interactive_tasks: list[asyncio.Task] = []
        self._deactivate_tasks: list[asyncio.Task] = []

    async def configure(self, application: RiCA):
        """Configure the router with a RiCA application instance."""
        global _application
        self._application = application
        _application = application

Router = _Router()
preset = Router.configure

async def _call(function: Callable[..., Any], data: list | dict, timeout: int, background: bool) -> UUID | CallBack:
    """Execute a registered function either in background or foreground.

    - If background is True, schedule the call on the global event loop and return a UUID.
    - If background is False, await completion and return a CallBack wrapping the result.
    """
    function = function if asyncio.iscoroutinefunction(function) else asyncio.to_thread(function)
    if background:
        call_id = uuid4()
        task = _loop.create_task(function(data), name=str(call_id))
        if timeout > 0:
            _loop.call_later(timeout / 1000, task.cancel, tuple())
        return call_id
    else:
        try:
            if timeout > 0:
                result = await asyncio.wait_for(function(data), timeout / 1000)
            else:
                result = await function(data)
            return CallBack(package=str(uuid4()), call_id=uuid4(), callback=result)
        except asyncio.TimeoutError:
            raise ExecutionTimedOut
        except Exception as e:
            raise UnexpectedExecutionError(str(e))

async def _decode(input_: str) -> tuple[Callable[..., Any], list | dict, int, bool]:
    """Parse a <rica ...>...</rica> string and return the target callable and options."""
    if not isinstance(input_, str):
        raise InvalidRiCAString("Input must be a string")

    input_ = input_.strip()
    if not (input_.startswith("<rica") and input_.endswith("</rica>")):
        raise InvalidRiCAString("Invalid Rica format")

    try:
        pkg_start = input_.find('package="') + 9
        pkg_end = input_.find('"', pkg_start)
        if pkg_start < 9 or pkg_end == -1:
            raise InvalidRiCAString("Missing package name")
        package_name = input_[pkg_start:pkg_end]

        data_start = input_.find(">") + 1
        data_end = input_.rfind("</rica>")
        if data_start == 0 or data_end == -1:
            raise InvalidRiCAString("Invalid Rica format")
        data_str = input_[data_start:data_end]

        data = json.loads(data_str)

        application = next((app for app in _application.endpoints if app.package == package_name), None)
        if not application:
            raise PackageNotFoundError(f"Package {package_name} not found")

        return application.function, data, application.timeout, application.background

    except json.JSONDecodeError:
        raise InvalidRiCAString("Invalid JSON data")
    except Exception as e:
        raise InvalidRiCAString(f"Failed to decode Rica format: {str(e)}")


async def _execute(input_: str) -> UUID | CallBack:
    """Decode and execute a tool-call string, returning a UUID or CallBack."""
    _call_tuple = await _decode(input_)
    return await _call(*_call_tuple)


async def _deactivate(package: str) -> None:
    """Deactivate a package (placeholder for future implementation)."""
    ...
