"""
Copyright(c)MINIOpenSource 2025

RiCA Server for Python
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from rica.adapters.base import ReasoningThreadBase

from .core.application import CallBack, RiCA, Route

__all__ = ["RiCA", "Route", "CallBack", "ReasoningThreadBase"]

try:
    _DISTRIBUTION_NAME = "rica-server"
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    # Fallback for when the package is not installed
    __version__ = "0.0.1-dev2"
