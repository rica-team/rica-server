"""
Server module for RiCA.

Defines the RiCA class, route registration utilities, and simple Route
and CallBack containers. Ensures packages are validated and that externally
provided functions are wrapped into async callables when registered.
"""

import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional, Union
from uuid import UUID

from ..exceptions import PackageInvalidError, RouteExistError

__all__ = ["RiCA", "Route", "CallBack"]


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


class Route:
    """
    A registered tool endpoint description.

    Attributes:
        route: The path of the route (e.g., "/my_tool").
        function: The callable function associated with the route.
        background: A boolean indicating if the tool should run in the background.
        timeout: The timeout for the tool in milliseconds.
    """

    def __init__(self, route: str, function: Callable[..., Any], background: bool, timeout: int):
        self.route: str = route
        self.function: Callable[..., Any] = function
        self.background: bool = background
        self.timeout: int = timeout


@dataclass
class CallBack:
    """
    Encapsulates the result of a tool call.

    Attributes:
        package: The package name of the tool.
        route: The route that was called.
        call_id: A unique identifier for the tool call.
        callback: The result of the tool call.
        status: The status of the tool call ("success", "error", "timeout").
        error: The exception object if the status is "error".
        duration_ms: The execution time of the tool in milliseconds.
    """
    package: str
    route: str
    call_id: UUID
    callback: Union[str, dict, list]
    status: str = "success"
    error: Optional[Exception] = None
    duration_ms: Optional[float] = None


class RiCA:
    """
    Reasoning Interface for Connector Applications.

    A RiCA instance represents an "application" with a unique package name.
    Functions can be exposed as endpoints by decorating them with the .route() method.

    Attributes:
        package: The unique package name for the application.
        description: A description of the application.
        routes: A list of registered routes.
    """

    def __init__(self, package: str, description: str = ""):
        if not _package_checker(package):
            raise PackageInvalidError(f"Package name '{package}' is invalid.")
        self.package: str = package
        self.description: str = description
        self.routes: List[Route] = []

    def find_route(self, route_path: str) -> Optional[Route]:
        """Finds a registered application endpoint by its route path."""
        for route in self.routes:
            if route.route == route_path:
                return route
        return None

    def route(
        self, route_path: str, background: bool = True, timeout: int = -1
    ) -> Callable[[Callable[..., Any]], Callable[..., Coroutine[Any, Any, Any]]]:
        """
        Register a new endpoint (route) dynamically.

        Non-async functions are wrapped with `asyncio.to_thread` so they behave as async callables.

        Args:
            route_path: The path for the new endpoint.
            background: Whether the function should run in the background.
            timeout: The timeout for the function in milliseconds.

        Returns:
            A decorator that registers the function as a route.
        """
        if self.find_route(route_path):
            raise RouteExistError(
                f"Route '{route_path}' already exists in package '{self.package}'."
            )

        def decorator(function: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
            self.routes.append(Route(route_path, function, background, timeout))

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
