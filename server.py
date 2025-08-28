import asyncio
import functools
from uuid import UUID
from typing import Callable, Any

from exceptions import *


async def package_checker(package: str) -> bool:
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
    def __init__(self, package: str, function: Callable[..., Any], background: bool, timeout: int):
        self.package: str = package
        self.function: Callable[..., Any] = function
        self.background: bool = background
        self.timeout: int = timeout


class CallBack:
    def __init__(self, package: str, call_id: UUID, callback: str | dict | list):
        self.package: str = package
        self.call_id: UUID = call_id
        self.callback: str | dict | list = callback


def route(package: str, background: bool = False, timeout: int = -1):
    """
    用于类内部方法的声明式路由装饰器。
    仅在定义期标记元数据，实际注册在 RiCA.__init__ 中完成。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_rica_route_meta", {
            "package": package,
            "background": background,
            "timeout": timeout,
        })
        return func

    return decorator


class RiCA:
    def __init__(self):
        self.endpoints: list[Application] = []

        # 扫描类中被 @route 标记的方法并注册
        for name, obj in self.__class__.__dict__.items():
            # 仅处理函数（未绑定）
            meta = getattr(obj, "_rica_route_meta", None)
            if not meta:
                continue

            package = meta["package"]
            background = meta["background"]
            timeout = meta["timeout"]

            # 校验包名与重复
            if not asyncio.run(package_checker(package)):
                raise PackageInvalidError
            if package in [application.package for application in self.endpoints]:
                raise PackageExistError

            # 绑定成实例方法后注册
            bound_method = getattr(self, name)
            self.endpoints.append(Application(package, bound_method, background, timeout))

    def new(self, package: str, background: bool = False, timeout: int = -1) -> Callable[
        [Callable[..., Any]], Callable[..., Any]]:
        async def init_checks():
            if not await package_checker(package):
                raise PackageInvalidError("Package name {} is invalid.".format(package))

            if package in [application.package for application in self.endpoints]:
                raise PackageExistError("Package {} already exists.".format(package))

        asyncio.create_task(init_checks())

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

    async def append(self, server: "RiCA"):
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
        result = RiCA()
        package_sources = {}

        # Check for duplicates first
        for i, server in enumerate(servers):
            for app in server.endpoints:
                # Skip @route decorated methods
                if hasattr(app.function, "_rica_route_meta"):
                    continue

                if app.package in package_sources:
                    prev_server = package_sources[app.package]
                    raise PackageDuplicateError(
                        f"Package '{app.package}' is duplicated between "
                        f"server #{prev_server} and server #{i}"
                    )
                package_sources[app.package] = i

        # If no duplicates found, merge all servers
        for server in servers:
            await result.append(server)

        return result

    @route("rica.threading.new", background=True, timeout=-1)
    def _rica_threading_new(self):
        ...

    @route("rica.task.wait", background=False, timeout=-1)
    def _rica_task_wait(self):
        ...
