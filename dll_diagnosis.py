import os
import sys
import importlib.util

print("Extension Diagnosis Tool")
print("-----------------------")
print(f"Python: {sys.version}")
print(f"System platform: {sys.platform}")

# Check if package is installed
try:
    import sparse_numba
    print(f"sparse_numba package found at: {sparse_numba.__file__}")
    package_dir = os.path.dirname(sparse_numba.__file__)

    # List contents
    print(f"Package directory contents: {os.listdir(package_dir)}")

    # Check compiled extensions
    extensions = []
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.so') or file.endswith('.pyd'):
                extensions.append(os.path.join(root, file))

    if extensions:
        print(f"Found {len(extensions)} compiled extensions:")
        for ext in extensions:
            print(f"  - {ext}")
    else:
        print("No compiled extensions found!")

except ImportError as e:
    print(f"Could not import sparse_numba: {e}")

# Try to directly find extensions
for module_name in ['sparse_numba.sparse_superlu.cy_superlu_wrapper',
                   'sparse_numba.sparse_umfpack.cy_umfpack_wrapper']:
    spec = importlib.util.find_spec(module_name)
    if spec:
        print(f"Found module spec for {module_name} at {spec.origin}")
    else:
        print(f"Module spec for {module_name} not found")

print("Diagnosis complete")