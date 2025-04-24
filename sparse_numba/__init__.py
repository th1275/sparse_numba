"""
Sparse Numba - Fast sparse solver with Numba support
__init__.py under sparse_numba
"""

import os
import sys
import platform
import logging
import ctypes
import ctypes.util
import importlib

# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sparse_numba")

# Import submodules - we'll import these lazily to avoid circular imports
from .conversion import matrix_conversion_numba


# Track SuperLU availability
_HAS_SUPERLU = False
# Track UMFPACK availability
_HAS_UMFPACK = False

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
        from .sparse_superlu import cy_superlu_wrapper
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
# Initialize SuperLU during module import
initialize_superlu()


__all__ = [
    'matrix_conversion_numba',
    'is_slu_available',
    'is_umf_available'
]

def is_slu_available():
    """Check if SuperLU is available"""
    return _HAS_SUPERLU

def is_umf_available():
    """Check if umfpack is available"""
    return _HAS_UMFPACK

# Variables to track availability of solvers
# has_umfpack = False
# has_superlu = False



__author__ = "Tianqi Hong"