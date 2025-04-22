"""
Sparse Numba - Fast sparse solver with Numba support
__init__.py under sparse_numba
"""

import os
import sys
import platform
import logging
import importlib
import glob
import ctypes
import ctypes.util  # Explicitly import util submodule


# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sparse_numba")

# Determine platform
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_LINUX = PLATFORM == 'Linux'
IS_MACOS = PLATFORM == 'Darwin'

# Add library paths to system for dynamic loading
package_dir = os.path.dirname(os.path.abspath(__file__))

# Track library availability
_HAS_UMFPACK = False
_HAS_SUPERLU = True  # Assume SuperLU is always available as per requirement

if IS_WINDOWS:
    # Windows-specific DLL handling
    dll_paths = [
        # Check vendor directory in development mode
        os.path.join(package_dir, "vendor", "suitesparse", "bin"),
        os.path.join(package_dir, "vendor", "openblas", "bin"),
        os.path.join(package_dir, "vendor", "superlu", "bin"),
        # Check for wheel installation paths
        os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "suitesparse", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "openblas", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "superlu", "bin"),
    ]

    # Check system path locations
    suitesparse_sys_env = os.environ.get('SUITESPARSE_BIN', '')
    if suitesparse_sys_env:
        dll_paths.append(suitesparse_sys_env)

    # For debugging: log all paths we're checking
    logger.debug("Checking for DLLs in the following paths:")

    # Track if we found any DLL directories
    found_dll_paths = []
    found_suitesparse = False

    # Add all existing paths to the system PATH
    for dll_path in dll_paths:
        if os.path.exists(dll_path):
            logger.info(f"Found DLL directory: {dll_path}")
            found_dll_paths.append(dll_path)

            # Check if this path contains SuiteSparse DLLs
            path_parts = os.path.normpath(dll_path).split(os.sep)
            if path_parts and path_parts[-1] == "bin" and len(path_parts) > 1 and path_parts[-2] == "suitesparse":
                # Check for umfpack DLLs in this directory
                umfpack_pattern = os.path.join(dll_path, "*umfpack*.dll")
                if glob.glob(umfpack_pattern):
                    found_suitesparse = True
                    logger.info(f"Found SuiteSparse DLLs in: {dll_path}")

            # Set DLL directory on newer Python versions (3.8+)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(dll_path)
                    logger.debug(f"Added DLL directory using os.add_dll_directory: {dll_path}")
                except Exception as e:
                    logger.warning(f"Failed to add DLL directory {dll_path}: {e}")

            # Also update PATH for all Python versions
            if dll_path not in os.environ['PATH']:
                os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
                logger.debug(f"Added to PATH: {dll_path}")

    if not found_suitesparse:
        parent_dir = os.path.dirname(package_dir)
        logger.warning(
            "\n" + "=" * 80 + "\n" +
            "SuiteSparse DLLs were not found! This package requires SuiteSparse to work correctly.\n" +
            "Please install SuiteSparse and make sure the DLLs are in your system PATH, or:\n" +
            "1. Download SuiteSparse and place the DLLs in one of these locations:\n" +
            f"   - {os.path.join(package_dir, 'vendor', 'suitesparse', 'bin')}\n" +
            f"   - {os.path.join(parent_dir, 'vendor', 'suitesparse', 'bin')}\n" +
            "2. Set the SUITESPARSE_BIN environment variable to the directory containing SuiteSparse DLLs\n" +
            "=" * 80
        )
    else:
        _HAS_UMFPACK = True
        logger.info("SuiteSparse DLLs found. Package should work correctly.")

    # Log the current PATH for debugging
    logger.debug(f"Updated PATH: {os.environ['PATH']}")
elif IS_LINUX:
    # Linux-specific library handling
    # On Linux, libraries are usually found via LD_LIBRARY_PATH or in standard system locations
    # We can add vendored libraries if they exist
    lib_paths = [
        os.path.join(package_dir, "vendor", "suitesparse", "lib"),
        os.path.join(package_dir, "vendor", "openblas", "lib"),
        os.path.join(package_dir, "vendor", "superlu", "lib"),
    ]

    # Add vendored libraries to LD_LIBRARY_PATH if they exist
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            logger.info(f"Found library directory: {lib_path}")
            if 'LD_LIBRARY_PATH' in os.environ:
                if lib_path not in os.environ['LD_LIBRARY_PATH']:
                    os.environ['LD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['LD_LIBRARY_PATH']
            else:
                os.environ['LD_LIBRARY_PATH'] = lib_path
            logger.debug(f"Added to LD_LIBRARY_PATH: {lib_path}")

    # Check if SuiteSparse libraries are available
    # This is a simple heuristic - we're checking common system locations
    umfpack_lib = ctypes.util.find_library("umfpack")
    if not umfpack_lib:
        logger.warning(
            "\n" + "=" * 80 + "\n" +
            "SuiteSparse libraries were not found! This package requires SuiteSparse to work correctly.\n" +
            "Please install SuiteSparse via your package manager. For example:\n" +
            "  - Debian/Ubuntu: sudo apt-get install libsuitesparse-dev\n" +
            "  - Fedora: sudo dnf install suitesparse-devel\n" +
            "  - Arch Linux: sudo pacman -S suitesparse\n" +
            "=" * 80
        )
    else:
        _HAS_UMFPACK = True
        logger.info("SuiteSparse libraries found. Package should work correctly.")

elif IS_MACOS:
    # macOS-specific library handling
    # On macOS, libraries are usually found via DYLD_LIBRARY_PATH or in standard system locations
    lib_paths = [
        os.path.join(package_dir, "vendor", "suitesparse", "lib"),
        os.path.join(package_dir, "vendor", "openblas", "lib"),
        os.path.join(package_dir, "vendor", "superlu", "lib"),
    ]

    # Add vendored libraries to DYLD_LIBRARY_PATH if they exist
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            logger.info(f"Found library directory: {lib_path}")
            if 'DYLD_LIBRARY_PATH' in os.environ:
                if lib_path not in os.environ['DYLD_LIBRARY_PATH']:
                    os.environ['DYLD_LIBRARY_PATH'] = lib_path + os.pathsep + os.environ['DYLD_LIBRARY_PATH']
            else:
                os.environ['DYLD_LIBRARY_PATH'] = lib_path
            logger.debug(f"Added to DYLD_LIBRARY_PATH: {lib_path}")

    # Check if SuiteSparse libraries are available through Homebrew or MacPorts
    umfpack_lib = ctypes.util.find_library("umfpack")
    if not umfpack_lib:
        logger.warning(
            "\n" + "=" * 80 + "\n" +
            "SuiteSparse libraries were not found! This package requires SuiteSparse to work correctly.\n" +
            "Please install SuiteSparse via Homebrew or MacPorts:\n" +
            "  - Homebrew: brew install suite-sparse\n" +
            "  - MacPorts: sudo port install SuiteSparse\n" +
            "=" * 80
        )
    else:
        _HAS_UMFPACK = True
        logger.info("SuiteSparse libraries found. Package should work correctly.")

# Define function stubs for missing dependencies
class MissingDependencyError:
    def __init__(self, name, library_name):
        self.name = name
        self.library_name = library_name

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"The function '{self.name}' requires {self.library_name} libraries which are not available. "
            f"Please install {self.library_name} and try again."
        )

# Import conversion functions (should work regardless of solver availability)
try:
    from .conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc
except ImportError as e:
    logger.warning(f"Failed to import conversion functions: {e}")

    # Define placeholder functions
    def convert_coo_to_csc(*args, **kwargs):
        raise ImportError("Conversion functions are not available.")

    def convert_csr_to_csc(*args, **kwargs):
        raise ImportError("Conversion functions are not available.")

# Define placeholder functions for UMFPACK and SuperLU
umfpack_solve_csc = None
umfpack_solve_coo = None
umfpack_solve_csr = None
superlu_solve_csc = None
superlu_solve_coo = None
superlu_solve_csr = None

# Import UMFPACK functions at the end to avoid circular imports
if _HAS_UMFPACK:
    try:
        # Import the module without creating circular dependencies
        from .sparse_umfpack import umfpack_numba_interface

        # Now assign the functions
        umfpack_solve_csc = umfpack_numba_interface.umfpack_solve_csc
        umfpack_solve_coo = umfpack_numba_interface.umfpack_solve_coo
        umfpack_solve_csr = umfpack_numba_interface.umfpack_solve_csr

        logger.info("Successfully imported UMFPACK functions")
    except ImportError as e:
        logger.warning(f"Failed to import UMFPACK functions: {e}")
        _HAS_UMFPACK = False

# If imports failed or UMFPACK not available, define placeholder functions
if not umfpack_solve_csc:
    umfpack_solve_csc = MissingDependencyError("umfpack_solve_csc", "UMFPACK")
    umfpack_solve_coo = MissingDependencyError("umfpack_solve_coo", "UMFPACK")
    umfpack_solve_csr = MissingDependencyError("umfpack_solve_csr", "UMFPACK")

# Import SuperLU functions at the end to avoid circular imports
try:
    # Import the module without creating circular dependencies
    from .sparse_superlu import superlu_numba_interface

    # Now assign the functions
    superlu_solve_csc = superlu_numba_interface.superlu_solve_csc
    superlu_solve_coo = superlu_numba_interface.superlu_solve_coo
    superlu_solve_csr = superlu_numba_interface.superlu_solve_csr

    logger.info("Successfully imported SuperLU functions")
except ImportError as e:
    logger.warning(f"Failed to import SuperLU functions: {e}")

# If imports failed, define placeholder functions
if not superlu_solve_csc:
    superlu_solve_csc = MissingDependencyError("superlu_solve_csc", "SuperLU")
    superlu_solve_coo = MissingDependencyError("superlu_solve_coo", "SuperLU")
    superlu_solve_csr = MissingDependencyError("superlu_solve_csr", "SuperLU")

# Define what should be exported
__all__ = [
    'umfpack_solve_csc',
    'umfpack_solve_coo',
    'umfpack_solve_csr',
    'superlu_solve_csc',
    'superlu_solve_coo',
    'superlu_solve_csr',
    'convert_coo_to_csc',
    'convert_csr_to_csc',
]

# Try importing benchmark modules, but don't fail if not available
# try:
#     from . import benchmark_single_slu
#     from . import benchmark_parallel_slu
#     from . import benchmark_single_umf
#     from . import benchmark_parallel_umf
#     __all__.extend([
#         'benchmark_single_slu',
#         'benchmark_single_umf',
#         'benchmark_parallel_slu',
#         'benchmark_parallel_umf',
#     ])
# except ImportError:
#     logger.debug("Benchmark modules not available")

__author__ = "Tianqi Hong"