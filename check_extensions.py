# Save as check_extensions.py
import os
import sys
import subprocess


def print_separator():
    print("=" * 70)


print_separator()
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print_separator()

# Check if sparse_numba is installed
try:
    import sparse_numba

    print(f"sparse_numba installed at: {sparse_numba.__file__}")
    package_dir = os.path.dirname(sparse_numba.__file__)

    # List all files in the package recursively
    print("\nPackage files:")
    for root, dirs, files in os.walk(package_dir):
        rel_path = os.path.relpath(root, package_dir)
        print(f"\nDirectory: {rel_path}")
        for file in files:
            print(f"  - {file}")

    # Check if extensions exist on disk
    superlu_dir = os.path.join(package_dir, "sparse_superlu")
    umfpack_dir = os.path.join(package_dir, "sparse_umfpack")

    print("\nSearching for extension modules:")
    extensions_found = False

    for directory in [superlu_dir, umfpack_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.startswith("cy_") and (file.endswith(".so") or ".cpython-" in file):
                    extensions_found = True
                    print(f"Found extension: {os.path.join(directory, file)}")

    if not extensions_found:
        print("No extension modules found in package directories!")

    # Try pip show to see metadata
    print_separator()
    print("Pip package info:")
    subprocess.run([sys.executable, "-m", "pip", "show", "sparse_numba"])

except ImportError as e:
    print(f"Error importing sparse_numba: {e}")