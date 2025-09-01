"""Server module for RiCA.

Defines the RiCA class, route registration utilities, and simple Application
and CallBack containers. Ensures packages are validated and that externally
provided functions are wrapped into async callables when registered.
"""
import asyncio
import functools
from uuid import UUID
from typing import Callable, Any

from .exceptions import *


__all__ = ["RiCA", "Application", "CallBack"]


def _package_checker(package: str) -> bool:
    """Check whether a package name is syntactically valid."""
    if not package or len(package) > 256:
        return False

    segments = package.split(".")
    if len(segments) < 2:
        return False

    for segment in segments:
        if not segment:
            return False
        if not segment[0].isalpha():
            return False
        if not all(c.isalnum() or c == '_' for c in segment):
            return False

    return True


class Application:
    """A registered tool endpoint description."""

    def __init__(self, package: str, function: Callable[..., Any], background: bool, timeout: int):
        self.package: str = package
        self.function: Callable[..., Any] = function
        self.background: bool = background
        self.timeout: int = timeout


class CallBack:
    """Encapsulates the result of a synchronous tool call."""

    def __init__(self, package: str, call_id: UUID, callback: str | dict | list):
        self.package: str = package
        self.call_id: UUID = call_id
        self.callback: str | dict | list = callback


def _lock(package: str, background: bool = False, timeout: int = -1):
    """Decorator to mark a method as a tool endpoint with routing metadata."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            func,
            "_rica_route_meta",
            {
                "package": package,
                "background": background,
                "timeout": timeout,
            },
        )
        return func

    return decorator


class RiCA:
    """Reasoning Interface for Connector Applications.

    Instances scan their class dict for methods decorated with @_lock on init
    and register them as endpoints. New endpoints can be added dynamically via
    register/new.
    """

    def __init__(self):
        self.endpoints: list[Application] = []

        for name, obj in self.__class__.__dict__.items():
            meta = getattr(obj, "_rica_route_meta", None)
            if not meta:
                continue

            package = meta["package"]
            background = meta["background"]
            timeout = meta["timeout"]

            bound_method = getattr(self, name)
            self.endpoints.append(Application(package, bound_method, background, timeout))

    def register(self, package: str, background: bool = True, timeout: int = -1) -> Callable[
        [Callable[..., Any]], Callable[..., Any]]:
        """Register a new endpoint dynamically.

        Non-async functions are wrapped with asyncio.to_thread so they behave as async callables.
        """

        def init_checks():
            if not _package_checker(package):
                raise PackageInvalidError("Package name {} is invalid.".format(package))

            if package in [application.package for application in self.endpoints]:
                raise PackageExistError("Package {} already exists.".format(package))

        init_checks()

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            self.endpoints.append(Application(package, function, background, timeout))

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

    def new(self, package: str, background: bool = True, timeout: int = -1) -> Callable[
        [Callable[..., Any]], Callable[..., Any]]:
        """Alias of register for convenience."""
        return self.register(package, background, timeout)

    async def append(self, server: "RiCA"):
        """Append endpoints from another RiCA instance, checking duplicates."""
        backup = self.endpoints.copy()
        for application in server.endpoints:
            if hasattr(application.function, "_rica_route_meta"):
                continue
            if application.package in [app.package for app in self.endpoints]:
                self.endpoints = backup
                raise PackageExistError(f"Package {application.package} already exists")
            self.endpoints.append(application)

    @staticmethod
    async def concat(*servers: "RiCA") -> "RiCA":
        """Concatenate endpoints from multiple RiCA instances with duplicate detection."""
        result = RiCA()
        package_sources = {}

        for i, server in enumerate(servers):
            for app in server.endpoints:
                if hasattr(app.function, "_rica_route_meta"):
                    continue

                if app.package in package_sources:
                    prev_server = package_sources[app.package]
                    raise PackageDuplicateError(
                        f"Package '{app.package}' is duplicated between "
                        f"server #{prev_server} and server #{i}"
                    )
                package_sources[app.package] = i

        for server in servers:
            await result.append(server)

        return result

    @_lock("rica.response", background=False, timeout=-1)
    async def _rica_response(self):
        """
        Tool to respond to user with formatted content. Currently only plain text is supported.
        input: [<objects>]
        output: {"status": "success"}
        objects:
            {"type": "text", "content": "text content"}
        """
        ...

    @_lock("rica.userinput")
    async def _rica_userinput(self):
        """
        IMPORTANT: This package cannot be called. It's only for Insert A User Input.
        output: [Plain Text] <User Inserted Text>
        """
        ...
