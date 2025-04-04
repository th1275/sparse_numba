"""
Sparse Numba UMFPACK - Fast UMFPACK sparse solver with Numba support
__init__.py under sparse_numba.sparse_numba
"""

import os
import sys
import site
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sparse_numba")

# Add DLL directory to PATH for Windows
if sys.platform.startswith('win'):
    # Try to find the right location for the DLLs
    package_dir = Path(__file__).parent.absolute()
    dll_paths = [
        # Check vendor directory in development mode
        package_dir / "vendor" / "suitesparse" / "bin",
        package_dir / "vendor" / "openblas" / "bin",
        # Check next to the package in site-packages (wheel installation)
        # Check at site-packages level (sibling to sparse_numba)
        # package_dir.parent / "vendor" / "suitesparse" / "bin",
        # package_dir.parent / "vendor" / "openblas" / "bin",
    ]
    # Check system path locations
    suitesparse_sys_env = os.environ.get('SUITESPARSE_BIN', '')
    if suitesparse_sys_env:
        dll_paths.append(Path(suitesparse_sys_env))

    # For debugging: log all paths we're checking
    logger.debug("Checking for DLLs in the following paths:")

    # Track if we found any DLL directories
    found_dll_paths = []
    found_suitesparse = False

    # Add all existing paths to the system PATH
    for dll_path in dll_paths:
        if dll_path.exists():
            logger.info(f"Found DLL directory: {dll_path}")
            found_dll_paths.append(dll_path)

            # Check if this path contains SuiteSparse DLLs
            if dll_path.name == "bin" and dll_path.parent.name == "suitesparse":
                if any(dll_file.name.startswith(('libumfpack', 'umfpack'))
                       for dll_file in dll_path.glob('*.dll')):
                    found_suitesparse = True
                    logger.info(f"Found SuiteSparse DLLs in: {dll_path}")

            # Set DLL directory on newer Python versions (3.8+)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(str(dll_path))
                    logger.debug(f"Added DLL directory using os.add_dll_directory: {dll_path}")
                except Exception as e:
                    logger.warning(f"Failed to add DLL directory {dll_path}: {e}")

            # Also update PATH for all Python versions
            if str(dll_path) not in os.environ['PATH']:
                os.environ['PATH'] = str(dll_path) + os.pathsep + os.environ['PATH']
                logger.debug(f"Added to PATH: {dll_path}")

    if not found_suitesparse:
        logger.warning(
            "\n" + "=" * 80 + "\n" +
            "SuiteSparse DLLs were not found! This package requires SuiteSparse to work correctly.\n" +
            "Please install SuiteSparse and make sure the DLLs are in your system PATH, or:\n" +
            "1. Download SuiteSparse and place the DLLs in one of these locations:\n" +
            f"   - {package_dir}/vendor/suitesparse/bin\n" +
            f"   - {package_dir.parent}/vendor/suitesparse/bin\n" +
            "2. Set the SUITESPARSE_BIN environment variable to the directory containing SuiteSparse DLLs\n" +
            "=" * 80
        )
    else:
        logger.info("SuiteSparse DLLs found. Package should work correctly.")

        # Log the current PATH for debugging
    logger.debug(f"Updated PATH: {os.environ['PATH']}")


# Import UMFPACK solver
try:
    # Export main functions
    from sparse_numba.sparse_umfpack.umfpack_numba_interface import (
        umfpack_solve_csc,
        umfpack_solve_coo,
        umfpack_solve_csr
    )

    # Export conversion functions
    from sparse_numba.conversion import convert_coo_to_csc, convert_csr_to_csc
    import sparse_numba.benchmark_single_umf, sparse_numba.benchmark_parallel_umf, sparse_numba.test_import

    __all__ = [
        'umfpack_solve_csc',
        'umfpack_solve_coo',
        'umfpack_solve_csr',
        'convert_coo_to_csc',
        'convert_csr_to_csc',
        'benchmark_single_umf',
        'benchmark_parallel_umf',
        'test_import'
    ]

except ImportError as e:
    logger.error(f"Failed to import sparse_numba components: {e}")
    # Re-raise the exception
    raise

__author__ = "Tianqi Hong"