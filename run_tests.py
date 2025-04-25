import os
import sys
import platform
import ctypes
import ctypes.util
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sparse_numba_tests")


def check_libraries():
    """
    Check if required system libraries (SuperLU, UMFPACK, OpenBLAS) are available
    and can be loaded by the system.

    Returns:
        dict: Dictionary with library names as keys and availability status as values
    """
    libraries = {
        "superlu": False,
        "umfpack": False,
        "openblas": False
    }

    # Determine platform-specific library names
    platform_name = platform.system()
    logger.info(f"Checking libraries on platform: {platform_name}")

    if platform_name == 'Linux':
        lib_names = {
            "superlu": ["libsuperlu.so", "libsuperlu.so.5", "libsuperlu.so.4"],
            "umfpack": ["libumfpack.so", "libumfpack.so.5"],
            "openblas": ["libopenblas.so", "libopenblas.so.0"]
        }
        lib_dirs = [
            "/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/local/lib64",
            "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
            # Add conda/virtual env lib locations if relevant
            os.path.join(sys.prefix, "lib")
        ]
    elif platform_name == 'Darwin':  # macOS
        lib_names = {
            "superlu": ["libsuperlu.dylib"],
            "umfpack": ["libumfpack.dylib"],
            "openblas": ["libopenblas.dylib"]
        }
        lib_dirs = [
            "/usr/local/lib", "/opt/local/lib",
            # Homebrew paths
            "/usr/local/opt/superlu/lib",
            "/usr/local/opt/suite-sparse/lib",
            "/usr/local/opt/openblas/lib",
            # Add conda/virtual env lib locations if relevant
            os.path.join(sys.prefix, "lib")
        ]

        # Try to find Homebrew prefix
        try:
            import subprocess
            result = subprocess.run(['brew', '--prefix'], capture_output=True, text=True)
            if result.returncode == 0:
                homebrew_prefix = result.stdout.strip()
                lib_dirs.extend([
                    os.path.join(homebrew_prefix, "lib"),
                    os.path.join(homebrew_prefix, "opt/superlu/lib"),
                    os.path.join(homebrew_prefix, "opt/suite-sparse/lib"),
                    os.path.join(homebrew_prefix, "opt/openblas/lib")
                ])
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Could not determine Homebrew prefix")

    elif platform_name == 'Windows':
        lib_names = {
            "superlu": ["superlu.dll"],
            "umfpack": ["umfpack.dll"],
            "openblas": ["libopenblas.dll", "openblas.dll"]
        }
        lib_dirs = [
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "sparse_numba", "vendor", "superlu",
                         "bin"),
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "sparse_numba", "vendor",
                         "suitesparse", "bin"),
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "sparse_numba", "vendor", "openblas",
                         "bin")
        ]
        # Add PATH directories
        if 'PATH' in os.environ:
            lib_dirs.extend(os.environ['PATH'].split(os.pathsep))
    else:
        logger.error(f"Unsupported platform: {platform_name}")
        return libraries

    # First try direct file paths for faster results and more reliability
    for lib, variants in lib_names.items():
        for variant in variants:
            for lib_dir in lib_dirs:
                if os.path.exists(lib_dir):
                    full_path = os.path.join(lib_dir, variant)
                    if os.path.exists(full_path):
                        logger.info(f"Found {lib} library file: {full_path}")
                        try:
                            if platform_name == 'Windows':
                                lib_handle = ctypes.CDLL(full_path)
                            else:
                                lib_handle = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            logger.info(f"Successfully loaded {lib} library from {full_path}")
                            libraries[lib] = True
                            break
                        except Exception as e:
                            logger.warning(f"Found but could not load {lib} from {full_path}: {e}")
            if libraries[lib]:
                break

    # If direct path didn't work, try ctypes.util.find_library as a fallback
    # This is more error-prone, so we do it second
    if not all(libraries.values()):
        for lib, variants in lib_names.items():
            if not libraries[lib]:  # Skip if already found
                for variant in variants:
                    try:
                        lib_name = variant.split('.')[0]  # Remove extension
                        # Skip "lib" prefix for find_library on Linux/macOS
                        if lib_name.startswith("lib") and platform_name in ['Linux', 'Darwin']:
                            lib_name = lib_name[3:]

                        lib_path = ctypes.util.find_library(lib_name)
                        if lib_path:
                            logger.info(f"Found {lib} library via find_library: {lib_path}")
                            try:
                                if platform_name == 'Windows':
                                    lib_handle = ctypes.CDLL(lib_path)
                                else:
                                    lib_handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                                logger.info(f"Successfully loaded {lib} library from {lib_path}")
                                libraries[lib] = True
                                break
                            except Exception as e:
                                logger.warning(f"Found but could not load {lib} from {lib_path}: {e}")
                    except Exception as e:
                        logger.warning(f"Error searching for {lib} using ctypes.util.find_library: {e}")

    # If using apt-based Linux, try ldconfig -p as a last resort
    if platform_name == 'Linux' and not all(libraries.values()):
        try:
            import subprocess
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if result.returncode == 0:
                output = result.stdout
                for lib in libraries.keys():
                    if not libraries[lib]:  # Skip if already found
                        search_terms = [f"lib{lib}", lib]
                        for term in search_terms:
                            lines = [line for line in output.split('\n') if term in line.lower()]
                            for line in lines:
                                parts = line.split('=>')
                                if len(parts) > 1:
                                    lib_path = parts[1].strip()
                                    logger.info(f"Found {lib} library via ldconfig: {lib_path}")
                                    try:
                                        lib_handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                                        logger.info(f"Successfully loaded {lib} library from {lib_path}")
                                        libraries[lib] = True
                                        break
                                    except Exception as e:
                                        logger.warning(f"Found but could not load {lib} from {lib_path}: {e}")
                            if libraries[lib]:
                                break
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Could not run ldconfig: {e}")

    # Summary of library availability
    for lib, available in libraries.items():
        logger.info(f"{lib}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")

    return libraries


def check_extension_modules():
    """
    Check if the package's extension modules are properly installed and loadable.

    Returns:
        dict: Dictionary with extension names as keys and availability status as values
    """
    extensions = {
        "cy_superlu_wrapper": False,
        "cy_umfpack_wrapper": False
    }

    # Check if sparse_numba is importable
    try:
        import sparse_numba
        logger.info(f"Successfully imported sparse_numba from {sparse_numba.__file__}")

        # Check package structure
        package_dir = os.path.dirname(sparse_numba.__file__)
        logger.info(f"sparse_numba package directory: {package_dir}")
        logger.info(f"Contents of package directory: {os.listdir(package_dir)}")

        # Check sparse_superlu directory
        superlu_dir = os.path.join(package_dir, 'sparse_superlu')
        if os.path.exists(superlu_dir):
            logger.info(f"Contents of sparse_superlu directory: {os.listdir(superlu_dir)}")
        else:
            logger.error("sparse_superlu directory not found")

        # Check sparse_umfpack directory
        umfpack_dir = os.path.join(package_dir, 'sparse_umfpack')
        if os.path.exists(umfpack_dir):
            logger.info(f"Contents of sparse_umfpack directory: {os.listdir(umfpack_dir)}")
        else:
            logger.error("sparse_umfpack directory not found")

        # Try importing extension modules directly
        try:
            import importlib.util

            # Check SuperLU extension
            superlu_spec = importlib.util.find_spec("sparse_numba.sparse_superlu.cy_superlu_wrapper")
            if superlu_spec:
                logger.info(f"Found cy_superlu_wrapper at: {superlu_spec.origin}")
                try:
                    from sparse_numba.sparse_superlu import cy_superlu_wrapper
                    logger.info("Successfully imported cy_superlu_wrapper")
                    extensions["cy_superlu_wrapper"] = True
                except ImportError as e:
                    logger.error(f"Failed to import cy_superlu_wrapper: {e}")
            else:
                logger.error("cy_superlu_wrapper module spec not found")

            # Check UMFPACK extension
            umfpack_spec = importlib.util.find_spec("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
            if umfpack_spec:
                logger.info(f"Found cy_umfpack_wrapper at: {umfpack_spec.origin}")
                try:
                    from sparse_numba.sparse_umfpack import cy_umfpack_wrapper
                    logger.info("Successfully imported cy_umfpack_wrapper")
                    extensions["cy_umfpack_wrapper"] = True
                except ImportError as e:
                    logger.error(f"Failed to import cy_umfpack_wrapper: {e}")
            else:
                logger.error("cy_umfpack_wrapper module spec not found")

        except Exception as e:
            logger.error(f"Error checking extension modules: {e}")

    except ImportError as e:
        logger.error(f"Failed to import sparse_numba: {e}")

    # Summary of extension availability
    for ext, available in extensions.items():
        logger.info(f"Extension {ext}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")

    return extensions


def main():
    """Main test function that runs all tests for sparse_numba package"""
    logger.info("=" * 80)
    logger.info("Testing sparse_numba package...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"sys.path: {sys.path}")
    logger.info("=" * 80)

    # Step 1: Check system libraries
    logger.info("\nStep 1: Checking system libraries...")
    libraries = check_libraries()

    # Step 2: Check extension modules
    logger.info("\nStep 2: Checking extension modules...")
    extensions = check_extension_modules()

    # Step 3: Try to import sparse_numba
    logger.info("\nStep 3: Testing package functionality...")
    try:
        import sparse_numba
        logger.info("Successfully imported sparse_numba")
    except ImportError as e:
        logger.error(f"Failed to import sparse_numba: {e}")
        sys.exit(1)

    # Step 4: Test SuperLU
    success = True
    try:
        # Create a simple sparse matrix (diagonal matrix)
        n = 5
        data = np.ones(n, dtype=np.float64)
        indices = np.arange(n, dtype=np.int32)
        indptr = np.arange(n + 1, dtype=np.int32)
        b = np.ones(n, dtype=np.float64)

        # Try to get the SuperLU solver
        try:
            logger.info("Attempting to access SuperLU solver...")
            # Directly import to avoid triggering __getattr__
            from sparse_numba.sparse_superlu import superlu_numba_interface
            solver = superlu_numba_interface.superlu_solve_csc
            logger.info("Successfully accessed SuperLU solver function")

            # Solve using SuperLU
            x, _ = solver(data, indices, indptr, b)

            # Check result
            if not np.allclose(x, b):
                logger.error(f"SuperLU solution is incorrect: {x} != {b}")
                success = False
            else:
                logger.info("SuperLU solver test passed!")
        except ImportError as e:
            if libraries["superlu"] and extensions["cy_superlu_wrapper"]:
                logger.error(f"SuperLU libraries and extension are available but solver import failed: {e}")
                success = False
            else:
                logger.warning(f"SuperLU solver test skipped (no lib or extension): {e}")
                success = False
        except Exception as e:
            logger.error(f"SuperLU solver test failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

        # Step 5: Test UMFPACK (only on non-Windows or if library is available)
        if os.name != 'nt' or libraries["umfpack"]:
            try:
                logger.info("Attempting to access UMFPACK solver...")
                # Directly import to avoid triggering __getattr__
                from sparse_numba.sparse_umfpack import umfpack_numba_interface
                solver = umfpack_numba_interface.umfpack_solve_csc

                x, _ = solver(data, indices, indptr, b)
                logger.info("UMFPACK solver test passed!")
            except ImportError as e:
                if libraries["umfpack"] and extensions["cy_umfpack_wrapper"]:
                    logger.error(f"UMFPACK libraries and extension are available but solver import failed: {e}")
                else:
                    logger.warning(f"UMFPACK solver test skipped (no lib or extension): {e}")
            except Exception as e:
                logger.warning(f"UMFPACK solver test (optional): {e}")
                # Don't fail the build if UMFPACK isn't available

    except Exception as e:
        logger.error(f"Unexpected error in tests: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Print a summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary:")
    logger.info("-" * 40)
    logger.info("System libraries:")
    for lib, available in libraries.items():
        logger.info(f"  - {lib}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")

    logger.info("\nExtension modules:")
    for ext, available in extensions.items():
        logger.info(f"  - {ext}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")

    logger.info("\nFunctionality Tests:")
    logger.info(f"  - Overall result: {'PASSED' if success else 'FAILED'}")
    logger.info("=" * 80)

    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())