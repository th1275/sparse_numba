"""
UMFPACK module
__init__.py under sparse_numba/sparse_umfpack
"""
import os
import sys
import platform
import logging
import ctypes
import ctypes.util
import importlib

# Get logger
logger = logging.getLogger("sparse_numba")

# Track UMFPACK availability
_HAS_UMFPACK = False

# Determine platform
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_LINUX = PLATFORM == 'Linux'
IS_MACOS = PLATFORM == 'Darwin'


# Setup library paths
def initialize_umfpack():
    global _HAS_UMFPACK
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if IS_WINDOWS:
        # Windows library handling
        dll_paths = [
            os.path.join(package_dir, "vendor", "suitesparse", "bin"),
            os.path.join(package_dir, "vendor", "openblas", "bin"),
            os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "suitesparse", "bin"),
            os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "openblas", "bin"),
        ]

        # Check system path locations
        suitesparse_sys_env = os.environ.get('SUITESPARSE_BIN', '')
        if suitesparse_sys_env:
            dll_paths.append(suitesparse_sys_env)

        # Track if we found SuiteSparse DLLs
        found_suitesparse = False

        # Add paths to DLL search path
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                logger.info(f"Found DLL directory: {dll_path}")

                # Check for UMFPACK DLLs
                import glob
                umfpack_pattern = os.path.join(dll_path, "*umfpack*.dll")
                if glob.glob(umfpack_pattern):
                    found_suitesparse = True
                    logger.info(f"Found SuiteSparse DLLs in: {dll_path}")

                # Set DLL directory on newer Python versions (3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(dll_path)
                    except Exception as e:
                        logger.warning(f"Failed to add DLL directory {dll_path}: {e}")

                # Also update PATH for all Python versions
                if dll_path not in os.environ['PATH']:
                    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

        _HAS_UMFPACK = found_suitesparse

    elif IS_LINUX:
        # Linux library handling
        lib_paths = [
            os.path.join(package_dir, "vendor", "suitesparse", "lib"),
            os.path.join(package_dir, "vendor", "openblas", "lib"),
        ]

        # Add to LD_LIBRARY_PATH if they exist
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                logger.info(f"Found library directory: {lib_path}")
                if 'LD_LIBRARY_PATH' in os.environ:
                    if lib_path not in os.environ['LD_LIBRARY_PATH']:
                        os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['LD_LIBRARY_PATH']
                else:
                    os.environ['LD_LIBRARY_PATH'] = lib_path

        # Check if UMFPACK library is available
        umfpack_lib = ctypes.util.find_library("umfpack")
        _HAS_UMFPACK = umfpack_lib is not None

    elif IS_MACOS:
        # macOS library handling
        lib_paths = [
            os.path.join(package_dir, "vendor", "suitesparse", "lib"),
            os.path.join(package_dir, "vendor", "openblas", "lib"),
        ]

        # Add to DYLD_LIBRARY_PATH if they exist
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                logger.info(f"Found library directory: {lib_path}")
                if 'DYLD_LIBRARY_PATH' in os.environ:
                    if lib_path not in os.environ['DYLD_LIBRARY_PATH']:
                        os.environ['DYLD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['DYLD_LIBRARY_PATH']
                else:
                    os.environ['DYLD_LIBRARY_PATH'] = lib_path

        # Check if UMFPACK library is available
        umfpack_lib = ctypes.util.find_library("umfpack")
        _HAS_UMFPACK = umfpack_lib is not None

    # Log the availability status
    if _HAS_UMFPACK:
        logger.info("UMFPACK libraries found. Module should work correctly.")
    else:
        logger.warning(
            "\n" + "=" * 80 + "\n" +
            "UMFPACK libraries were not found! UMFPACK functionality will not be available.\n" +
            "Please install SuiteSparse via your package manager or ensure the libraries are in the search path.\n" +
            "=" * 80
        )


# Initialize UMFPACK during module import
initialize_umfpack()


# Lazy import to avoid circular dependencies
def __getattr__(name):
    if name == 'umfpack_numba_interface':
        if _HAS_UMFPACK:
            # from . import umfpack_numba_interface
            # return umfpack_numba_interface
            module = importlib.import_module('.umfpack_numba_interface', package='sparse_numba.sparse_umfpack')
            return module
        else:
            raise ImportError("UMFPACK libraries are not available")
    # elif name == 'cy_umfpack_wrapper':
    #     if _HAS_UMFPACK:
    #         module = importlib.import_module('.cy_umfpack_wrapper', package='sparse_numba.sparse_umfpack')
    #         return module
    #     else:
    #         raise ImportError("cy_umfpack_wrapper libraries are not available")
    else:
        raise AttributeError(f"module 'sparse_numba.sparse_umfpack' has no attribute '{name}'")

from . import cy_umfpack_wrapper

__all__ = ['umfpack_numba_interface', 'cy_umfpack_wrapper', 'is_available']


def is_available():
    """Check if UMFPACK is available"""
    return _HAS_UMFPACK


__author__ = 'Tianqi Hong'