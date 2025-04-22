"""
SuperLU module
__init__.py under sparse_numba/sparse_superlu
"""
import os
import sys
import platform
import logging
import ctypes
import ctypes.util

# Get logger
logger = logging.getLogger("sparse_numba")

# Track SuperLU availability
_HAS_SUPERLU = False

# Determine platform
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_LINUX = PLATFORM == 'Linux'
IS_MACOS = PLATFORM == 'Darwin'


# Setup library paths
def initialize_superlu():
    global _HAS_SUPERLU
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if IS_WINDOWS:
        # Windows library handling
        dll_paths = [
            os.path.join(package_dir, "vendor", "superlu", "bin"),
            os.path.join(package_dir, "vendor", "openblas", "bin"),
            os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "superlu", "bin"),
            os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "openblas", "bin"),
        ]

        # Add paths to DLL search path
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                logger.info(f"Found SuperLU DLL directory: {dll_path}")

                # Set DLL directory on newer Python versions (3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(dll_path)
                    except Exception as e:
                        logger.warning(f"Failed to add DLL directory {dll_path}: {e}")

                # Also update PATH for all Python versions
                if dll_path not in os.environ['PATH']:
                    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

    elif IS_LINUX:
        # Linux library handling
        lib_paths = [
            os.path.join(package_dir, "vendor", "superlu", "lib"),
            os.path.join(package_dir, "vendor", "openblas", "lib"),
        ]

        # Add to LD_LIBRARY_PATH if they exist
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                logger.info(f"Found SuperLU library directory: {lib_path}")
                if 'LD_LIBRARY_PATH' in os.environ:
                    if lib_path not in os.environ['LD_LIBRARY_PATH']:
                        os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['LD_LIBRARY_PATH']
                else:
                    os.environ['LD_LIBRARY_PATH'] = lib_path

    elif IS_MACOS:
        # macOS library handling
        lib_paths = [
            os.path.join(package_dir, "vendor", "superlu", "lib"),
            os.path.join(package_dir, "vendor", "openblas", "lib"),
        ]

        # Add to DYLD_LIBRARY_PATH if they exist
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                logger.info(f"Found SuperLU library directory: {lib_path}")
                if 'DYLD_LIBRARY_PATH' in os.environ:
                    if lib_path not in os.environ['DYLD_LIBRARY_PATH']:
                        os.environ['DYLD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['DYLD_LIBRARY_PATH']
                else:
                    os.environ['DYLD_LIBRARY_PATH'] = lib_path

    # Check if SuperLU is available
    # Try a more reliable detection method - see if we can import the Cython wrapper
    try:
        # First try importing the Cython extension module directly
        from . import cy_superlu_wrapper
        _HAS_SUPERLU = True
        logger.info("SuperLU libraries found. Module should work correctly.")
    except ImportError as e:
        # If that fails, check if we can find the library using ctypes
        logger.debug(f"Error importing cy_superlu_wrapper: {e}")
        superlu_lib = ctypes.util.find_library("superlu")
        if superlu_lib:
            _HAS_SUPERLU = True
            logger.info("SuperLU libraries found via ctypes. Module should work correctly.")
        else:
            # Otherwise, issue the warning
            _HAS_SUPERLU = False
            logger.warning(
                "\n" + "=" * 80 + "\n" +
                "SuperLU libraries were not found in system paths! SuperLU functionality may not be available.\n" +
                "However, bundled libraries may still work.\n" +
                "=" * 80
            )

            # Try one more approach - dynamic import to see if it actually works
            try:
                # Create a temporary module object to test if import would work
                import importlib.util
                spec = importlib.util.find_spec("sparse_numba.sparse_superlu.cy_superlu_wrapper")
                if spec is not None:
                    _HAS_SUPERLU = True
                    logger.info("SuperLU extension module found via spec. Module should work correctly.")
            except Exception:
                pass


# Initialize SuperLU during module import
initialize_superlu()


# Lazy import to avoid circular dependencies
def __getattr__(name):
    if name == 'superlu_numba_interface':
        if _HAS_SUPERLU:
            from . import superlu_numba_interface
            return superlu_numba_interface
        else:
            raise ImportError("SuperLU libraries are not available")
    raise AttributeError(f"module 'sparse_numba.sparse_superlu' has no attribute '{name}'")


__all__ = ['superlu_numba_interface', 'is_available']


def is_available():
    """Check if SuperLU is available"""
    return _HAS_SUPERLU


__author__ = 'Tianqi Hong'