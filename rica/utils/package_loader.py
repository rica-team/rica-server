import importlib.util
import os
import sys
import tarfile
import tempfile

from rica.core.application import RiCA


async def load_app_from_path(path: str) -> RiCA:
    """
    Helper to load RiCA app from a file path.

    Args:
        path: Path to a .py file or .tar.gz/.tar archive.

    Returns:
        The loaded RiCA application instance.

    Raises:
        ImportError: If loading fails or no RiCA instance is found.
    """
    if path.endswith(".tar.gz") or path.endswith(".tar"):
        # Handle archive
        temp_dir = tempfile.mkdtemp()
        try:
            with tarfile.open(path, "r:*") as tar:
                tar.extractall(path=temp_dir)

            # Look for a python package or module in the extracted content
            # We assume there's an __init__.py or a .py file
            # Simple heuristic: add temp_dir to sys.path and try to import
            sys.path.insert(0, temp_dir)

            # Find what to import. We look for the first directory with __init__.py
            # or the first .py file. This is a bit ambiguous. Let's assume the
            # archive contains a folder with the package name OR the archive content
            # IS the package.

            # Let's try to find a .py file or a directory
            found_module = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".py"):
                        # If it's __init__.py, the package is the parent dir name
                        if file == "__init__.py":
                            found_module = os.path.basename(root)
                            break
                        else:
                            # It's a module file
                            found_module = file[:-3]
                            break
                if found_module:
                    break

            if not found_module:
                raise ImportError("Could not find a valid Python module or package in the archive.")

            try:
                module = importlib.import_module(found_module)
            finally:
                # Clean up sys.path? Maybe not if we want to keep it loaded.
                # But for this session it might be fine.
                # Ideally we should remove it to avoid pollution, but we need the module code.
                pass

        except Exception as e:
            raise ImportError(f"Failed to load app from archive '{path}': {e}")

    elif path.endswith(".py"):
        # Handle single .py file
        module_name = os.path.basename(path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from '{path}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        raise ImportError(f"Unsupported file type: {path}")

    # Find RiCA instance in module
    # 1. Look for 'app' variable
    if hasattr(module, "app") and isinstance(module.app, RiCA):
        return module.app

    # 2. Look for any RiCA instance
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, RiCA):
            return attr

    raise ImportError(f"No 'RiCA' instance found in module loaded from '{path}'.")
