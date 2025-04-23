# dll_diagnosis.py
import os
import sys
import importlib.util
import importlib.machinery


def print_separator():
    print("=" * 80)


def find_module_file(module_name):
    """Find a module's file path using all available methods."""
    print(f"Looking for module: {module_name}")

    # Method 1: Find spec
    spec = importlib.util.find_spec(module_name)
    if spec:
        print(f"  Found via find_spec: {spec.origin}")
        return spec.origin

    # Method 2: Try finder
    try:
        path = importlib.machinery.PathFinder.find_spec(module_name)
        if path and path.origin:
            print(f"  Found via PathFinder: {path.origin}")
            return path.origin
    except (ImportError, AttributeError):
        pass

    # Method 3: Check if already imported
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if hasattr(module, '__file__'):
            print(f"  Found in sys.modules: {module.__file__}")
            return module.__file__

    print(f"  Module not found!")
    return None


def is_extension_module(filepath):
    """Check if a file is a Python extension module."""
    return filepath and (filepath.endswith('.so') or filepath.endswith('.pyd') or '.cpython-' in filepath)


def main():
    print_separator()
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"Current Directory: {os.getcwd()}")
    print_separator()

    # Try to import sparse_numba
    try:
        import sparse_numba
        print(f"sparse_numba package found at: {sparse_numba.__file__}")
        package_dir = os.path.dirname(sparse_numba.__file__)

        # List all files in package directory recursively
        print("Package structure:")
        for root, dirs, files in os.walk(package_dir):
            relpath = os.path.relpath(root, package_dir)
            if relpath == '.':
                relpath = ''
            print(f"Directory: {relpath}")

            for file in files:
                if file.endswith('.py') or is_extension_module(file):
                    print(f"  - {file}")

        # Look for extension modules
        print_separator()
        print("Looking for extension modules:")
        ext_modules = [
            'sparse_numba.sparse_superlu.cy_superlu_wrapper',
            'sparse_numba.sparse_umfpack.cy_umfpack_wrapper',
        ]

        for ext in ext_modules:
            find_module_file(ext)

        # Try direct imports
        print_separator()
        print("Attempting direct imports:")

        try:
            from sparse_numba.sparse_superlu import cy_superlu_wrapper
            print("✅ Successfully imported sparse_numba.sparse_superlu.cy_superlu_wrapper")
        except ImportError as e:
            print(f"❌ Failed to import sparse_numba.sparse_superlu.cy_superlu_wrapper: {e}")

        try:
            from sparse_numba.sparse_umfpack import cy_umfpack_wrapper
            print("✅ Successfully imported sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
        except ImportError as e:
            print(f"❌ Failed to import sparse_numba.sparse_umfpack.cy_umfpack_wrapper: {e}")

    except ImportError as e:
        print(f"Failed to import sparse_numba: {e}")

    print_separator()
    print("SYSTEM PATH:")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
    print_separator()


if __name__ == "__main__":
    main()