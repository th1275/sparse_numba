# Installation Guide for sparse_numba

`sparse_numba` is a Python package providing efficient sparse matrix solvers with Numba support. It interfaces with UMFPACK and SuperLU libraries to provide fast sparse linear system solutions.

## System Requirements

### Windows (rebuild)
- Python 3.8 or higher
- MinGW-w64 with GCC (for compilation)
- Required DLLs are bundled with the package (Except SuiteSparse)
- Prebuilt wheel is available through pip install sparse_numba

### Linux
- Python 3.8 or higher
- GCC compiler
- SuiteSparse (UMFPACK), SuperLU, and OpenBLAS development libraries

### macOS
- Python 3.8 or higher
- Clang compiler (included with Xcode Command Line Tools)
- SuiteSparse (UMFPACK), SuperLU, and OpenBLAS libraries

## Installing Dependencies

### Windows

The Windows package includes the necessary DLLs, but you'll need MinGW for compilation:

```bash
# Install MinGW using Chocolatey
choco install mingw -y
```

### Ubuntu/Debian Linux

```bash
# Install required libraries
sudo apt-get update
sudo apt-get install -y libsuitesparse-dev libopenblas-dev libsuperlu-dev
```

### Fedora/RHEL/CentOS Linux

```bash
# Install required libraries
sudo dnf install -y suitesparse-devel openblas-devel superlu-devel
```

### Arch Linux

```bash
# Install required libraries
sudo pacman -S suitesparse openblas superlu
```

### macOS

```bash
# Install using Homebrew
brew install suite-sparse openblas superlu
```

## Installation

### From PyPI (recommended)

```bash
pip install sparse_numba
```

### From Source

```bash
git clone https://github.com/th1275/sparse_numba.git
cd sparse_numba
pip install -e .
```

## Verifying Installation

After installation, you can verify that the package is working correctly by running:

```python
import sparse_numba
# If no errors occur, the package is installed correctly
```

To run a more comprehensive test:

```python
from sparse_numba.sparse_umfpack.test import test_example_umfpack
test_example_umfpack.run_test()
```

## Troubleshooting

### Windows

If you encounter issues with missing DLLs, you may need to ensure that the DLLs are properly installed in the package directory or add them to your PATH.

### Linux

If you get an error like `libumfpack.so: cannot open shared object file`:

1. Make sure SuiteSparse is installed
2. Update your library cache: `sudo ldconfig`
3. Check if the library is found: `ldconfig -p | grep umfpack`

### macOS

If you get an error like `Library not loaded: libumfpack.dylib`:

1. Make sure SuiteSparse is installed via Homebrew
2. Check if the dylib is found: `otool -L /path/to/sparse_numba/sparse_umfpack/cy_umfpack_wrapper.*.so`

## Support

If you encounter any issues with the installation or use of sparse_numba, please open an issue on the GitHub repository: https://github.com/th1275/sparse_numba/issues
However, I currently only have access to the Linux and MacOS systems.
