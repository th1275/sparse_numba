"""
Diagnostic script to help identify DLL loading issues with sparse_numba.

Save this as diagnose_dll.py and run it in your environment where you're trying to use sparse_numba:
python diagnose_dll.py
"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: dll_diagnosis.py

import os
import sys
import ctypes
import logging
import site
from pathlib import Path
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('sparse_numba_diagnostics')

def check_python_version():
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    if sys.platform.startswith('win'):
        if hasattr(os, 'add_dll_directory'):
            logger.info("os.add_dll_directory() is available (Python 3.8+)")
        else:
            logger.warning("os.add_dll_directory() is NOT available (Python < 3.8)")

def check_dll_dirs():
    logger.info("\nChecking for DLL directories:")

    try:
        import sparse_numba
        package_dir = Path(sparse_numba.__file__).parent.absolute()
        logger.info(f"sparse_numba package directory: {package_dir}")

        # Get site-packages directories
        site_packages = site.getsitepackages()
        logger.info(f"Site-packages directories: {site_packages}")

        # Define possible DLL locations based on the correct venv structure
        dll_paths = [
            # Inside the package
            package_dir / "vendor" / "suitesparse" / "bin",
            package_dir / "vendor" / "openblas" / "bin",

            # At site-packages level
            package_dir.parent / "vendor" / "suitesparse" / "bin",
            package_dir.parent / "vendor" / "openblas" / "bin",

            # Python root paths
            Path(sys.prefix) / "vendor" / "suitesparse" / "bin",
            Path(sys.prefix) / "vendor" / "openblas" / "bin",

            # Lib folder
            Path(sys.prefix) / "Lib" / "vendor" / "suitesparse" / "bin",
            Path(sys.prefix) / "Lib" / "vendor" / "openblas" / "bin",

            # Scripts folder
            Path(sys.prefix) / "Scripts" / "vendor" / "suitesparse" / "bin",
            Path(sys.prefix) / "Scripts" / "vendor" / "openblas" / "bin",
        ]

        for path in dll_paths:
            if path.exists():
                logger.info(f"✓ Found DLL directory: {path}")
                logger.info(f"  Contents: {[f.name for f in path.glob('*.dll')]}")
            else:
                logger.info(f"✗ Not found: {path}")

    except ImportError as e:
        logger.error(f"Could not import sparse_numba: {e}")

def check_path_environment():
    logger.info("\nChecking PATH environment variable:")
    paths = os.environ.get('PATH', '').split(os.pathsep)
    dll_related_paths = [p for p in paths if 'vendor' in p.lower() or 'sparse' in p.lower()]

    if dll_related_paths:
        logger.info("Found related paths in PATH:")
        for path in dll_related_paths:
            logger.info(f"  {path}")
    else:
        logger.warning("No sparse_numba related paths found in PATH environment variable")

def try_load_dll_manually():
    logger.info("\nAttempting to manually load DLLs:")

    # List of DLLs to check
    dll_names = ["libumfpack.dll", "libcholmod.dll", "libamd.dll", "libcolamd.dll",
                "libcamd.dll", "libccolamd.dll", "libsuitesparseconfig.dll", "libopenblas.dll"]

    for dll_name in dll_names:
        try:
            dll = ctypes.CDLL(dll_name)
            logger.info(f"✓ Successfully loaded {dll_name}")
        except Exception as e:
            logger.warning(f"✗ Failed to load {dll_name}: {e}")

def check_import_extension():
    logger.info("\nChecking extension import:")
    try:
        # First just import the package
        import sparse_numba
        logger.info("✓ Successfully imported sparse_numba package")

        # Try to import the Cython extension
        try:
            # Try direct import
            import sparse_numba.sparse_umfpack.cy_umfpack_wrapper
            logger.info("✓ Successfully imported cy_umfpack_wrapper")
        except ImportError as e:
            logger.error(f"✗ Failed to import cy_umfpack_wrapper: {e}")

            # Try to locate the extension file
            import sparse_numba.sparse_umfpack as su
            ext_dir = Path(su.__file__).parent
            logger.info(f"sparse_umfpack directory: {ext_dir}")

            # List files in the directory
            files = list(ext_dir.glob("*"))
            logger.info(f"Files in sparse_umfpack directory: {[f.name for f in files]}")

            # Check if extension exists
            ext_files = list(ext_dir.glob("cy_umfpack_wrapper*"))
            if ext_files:
                logger.info(f"Found extension files: {[f.name for f in ext_files]}")
            else:
                logger.error("No extension files found!")

    except ImportError as e:
        logger.error(f"Failed to import sparse_numba: {e}")

def try_basic_functionality():
    logger.info("\nTrying basic sparse_numba functionality:")
    try:
        import numpy as np
        from scipy.sparse import coo_matrix
        import sparse_numba

        # Create a simple sparse matrix
        row = np.array([0, 0, 1, 3, 1, 0, 0])
        col = np.array([0, 2, 1, 3, 1, 0, 0])
        data = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

        A = coo_matrix((data, (row, col)), shape=(4, 4))
        b = np.array([1, 2, 3, 4], dtype=np.float64)

        logger.info("Created test matrix and vector")

        # Try importing and running the solver
        try:
            from sparse_numba import umfpack_solve_coo
            logger.info("Successfully imported umfpack_solve_coo")

            try:
                x = umfpack_solve_coo(A, b)
                logger.info(f"✓ Successfully solved system! Result: {x}")
            except Exception as e:
                logger.error(f"✗ Failed to solve system: {e}")

        except ImportError as e:
            logger.error(f"✗ Failed to import solver: {e}")

    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")

def write_patch_script():
    """Create a patch script that can be run to fix DLL loading issues"""
    script_content = """import os
import sys
import site
from pathlib import Path

# This script patches the environment to help with DLL loading for sparse_numba
print("Patching environment for sparse_numba DLLs...")

# Try to locate the package
try:
    import sparse_numba
    package_dir = Path(sparse_numba.__file__).parent.absolute()
    print(f"Found sparse_numba at: {package_dir}")
    
    # Check for DLLs in common locations
    dll_locations = [
        package_dir / "vendor" / "suitesparse" / "bin",
        package_dir / "vendor" / "openblas" / "bin",
        package_dir.parent / "vendor" / "suitesparse" / "bin",
        package_dir.parent / "vendor" / "openblas" / "bin",
        Path(sys.prefix) / "vendor" / "suitesparse" / "bin",
        Path(sys.prefix) / "vendor" / "openblas" / "bin",
    ]
    
    found_locations = []
    for loc in dll_locations:
        if loc.exists() and list(loc.glob("*.dll")):
            print(f"Found DLLs at: {loc}")
            found_locations.append(loc)
    
    if found_locations:
        # Add to PATH
        for loc in found_locations:
            if str(loc) not in os.environ.get('PATH', ''):
                os.environ['PATH'] = str(loc) + os.pathsep + os.environ.get('PATH', '')
                print(f"Added to PATH: {loc}")
        
        # For Python 3.8+, add DLL directories
        if hasattr(os, 'add_dll_directory'):
            for loc in found_locations:
                try:
                    os.add_dll_directory(str(loc))
                    print(f"Added DLL directory: {loc}")
                except Exception as e:
                    print(f"Error adding DLL directory {loc}: {e}")
        
        print("Environment patched successfully.")
        print("Now try importing and using sparse_numba again.")
    else:
        print("No DLL directories found. The installation may be corrupted.")
        print("Try reinstalling the package with:")
        print("pip uninstall sparse_numba")
        print("pip install sparse_numba")
except ImportError:
    print("Could not import sparse_numba. Please make sure it's installed.")
"""

    patch_file = "patch_sparse_numba.py"
    with open(patch_file, "w") as f:
        f.write(script_content)

    logger.info(f"\nCreated patch script at {patch_file}")
    logger.info("You can run this script before importing sparse_numba to fix DLL loading issues:")
    logger.info("  python patch_sparse_numba.py")

if __name__ == "__main__":
    logger.info("=== sparse_numba DLL Diagnostic Tool ===")
    check_python_version()
    check_dll_dirs()
    check_path_environment()
    try_load_dll_manually()
    check_import_extension()
    try_basic_functionality()
    write_patch_script()
    logger.info("\nDiagnostic complete!")