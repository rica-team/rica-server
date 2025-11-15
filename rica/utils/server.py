"""
Server module for RiCA.

Defines the RiCA class, route registration utilities, and simple Application
and CallBack containers. Ensures packages are validated and that externally
provided functions are wrapped into async callables when registered.
"""

import asyncio
import functools
from typing import Any, Callable, Optional
from uuid import UUID

from rica.exceptions import PackageInvalidError, RouteExistError

__all__ = ["RiCA", "Application", "CallBack"]


def _package_checker(package: str) -> bool:
    """Check whether a package name is syntactically valid."""
    if not package or len(package) > 256:
        return False

    # Allow 'rica' as a special case for the virtual system app
    if package == "rica":
        return True

    segments = package.split(".")
    if len(segments) < 2:
        return False

    for segment in segments:
        if not segment:
            return False
        if not segment[0].isalpha():
            return False
        if not all(c.isalnum() or c == "_" for c in segment):
            return False

    return True


class Application:
    """A registered tool endpoint description."""

    def __init__(self, route: str, function: Callable[..., Any], background: bool, timeout: int):
        self.route: str = route
        self.function: Callable[..., Any] = function
        self.background: bool = background
        self.timeout: int = timeout


class CallBack:
    """Encapsulates the result of a synchronous tool call."""

    def __init__(self, package: str, call_id: UUID, callback: str | dict | list):
        self.package: str = package
        self.call_id: UUID = call_id
        self.callback: str | dict | list = callback


class RiCA:
    """Reasoning Interface for Connector Applications.

    A RiCA instance represents an "application" with a unique package name.
    Functions can be exposed as endpoints by decorating them with the .route() method.
    """

    def __init__(self, package: str, description: str = ""):
        if not _package_checker(package):
            raise PackageInvalidError(f"Package name '{package}' is invalid.")
        self.package = package
        self.description = description
        self.routes: list[Application] = []

    def find_route(self, route_path: str) -> Optional[Application]:
        """Finds a registered application endpoint by its route path."""
        for route in self.routes:
            if route.route == route_path:
                return route
        return None

    def route(
        self, route_path: str, background: bool = True, timeout: int = -1
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a new endpoint (route) dynamically.

        Non-async functions are wrapped with `asyncio.to_thread` so they behave as async callables.
        """
        if self.find_route(route_path):
            raise RouteExistError(
                f"Route '{route_path}' already exists in package '{self.package}'."
            )

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            self.routes.append(Application(route_path, function, background, timeout))

            if asyncio.iscoroutinefunction(function):

                @functools.wraps(function)
                async def wrapper(*args, **kwargs):
                    return await function(*args, **kwargs)

                return wrapper
            else:

                @functools.wraps(function)
                async def wrapper(*args, **kwargs):
                    return await asyncio.to_thread(function, *args, **kwargs)

                return wrapper

        return decorator
