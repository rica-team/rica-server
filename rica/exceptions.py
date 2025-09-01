"""Custom exceptions for RiCA Server."""


class PackageInvalidError(ValueError):
    """Raised when a package name is syntactically invalid."""


class PackageExistError(ValueError):
    """Raised when attempting to register a package that already exists."""


class PackageNotFoundError(ValueError):
    """Raised when a requested package cannot be found in the registry."""


class PackageDuplicateError(ValueError):
    """Raised when duplicate packages are detected while concatenating servers."""


class ExecutionTimedOut(Exception):
    """Raised when a routed tool call times out."""


class UnexpectedExecutionError(Exception):
    """Raised when a routed tool call fails unexpectedly."""


class InvalidRiCAString(ValueError):
    """Raised when an input tool-call string is malformed or invalid."""


class AdapterDependenciesImportError(ImportError):
    """Raised when optional adapter dependencies are missing."""
