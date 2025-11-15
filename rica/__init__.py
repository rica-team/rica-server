"""
Copyright(c)MINIOpenSource 2025

RiCA Server for Python
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .core.application import CallBack, RiCA, Route
from .core.executor import Executor

__all__ = ["RiCA", "Route", "CallBack", "Executor"]

try:
    _DISTRIBUTION_NAME = "rica-server"
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    # Fallback for when the package is not installed
    __version__ = "0.0.0-dev"
