"""
Copyright(c)MINIOpenSource 2025

RiCA Server for Python
"""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from server import *
import connector

__all__ = ["RiCA", "Application", "CallBack", "connector"]

_DISTRIBUTION_NAME = "rica-server"

def _read_version_fallback() -> str:
    try:
        import tomllib
        root = Path(__file__).resolve().parent.parent
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as f:
                data = tomllib.load(f)
                v = data.get("project", {}).get("version")
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return "unspecified"

try:
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = _read_version_fallback()
