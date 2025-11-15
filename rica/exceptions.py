"""Custom exceptions for RiCA Server."""


class PackageInvalidError(ValueError):
    """Raised when a package name is syntactically invalid."""


class PackageExistError(ValueError):
    """Raised when attempting to register a package that already exists."""


class RouteExistError(ValueError):
    """Raised when attempting to register a route that already exists in a package."""


class RouteNotFoundError(ValueError):
    """Raised when a requested route cannot be found in the package."""


class PackageNotFoundError(ValueError):
    """Raised when a requested package cannot be found in the registry."""


class ExecutionTimedOut(Exception):
    """Raised when a routed tool call times out."""


class UnexpectedExecutionError(Exception):
    """Raised when a routed tool call fails unexpectedly."""


class InvalidRiCAString(ValueError):
    """Raised when an input tool-call string is malformed or invalid."""


class AdapterDependenciesImportError(ImportError):
    """Raised when optional adapter dependencies are missing."""
