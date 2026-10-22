"""
Sparse Numba - Fast sparse solver with Numba support
__init__.py under sparse_numba

Import does path setup only, silently: the library
directories are discovered relative to the package; nothing is imported
from a backend and nothing is logged above DEBUG. The public names are
bound lazily on first access, so an absent backend costs nothing at import
and its first call returns BACKEND_ABSENT; is_slu_available(),
is_umf_available() and backends() answer without logging, and
require(backend) raises with the install hint.
"""

import os
import sys
import platform
import logging
import importlib

# Logging stays at DEBUG; the package configures no handler of its own
logger = logging.getLogger("sparse_numba")
logger.addHandler(logging.NullHandler())

# Determine platform
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_LINUX = PLATFORM == 'Linux'
IS_MACOS = PLATFORM == 'Darwin'

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PACKAGE_DIR)

_SUPERLU_MODULE = "sparse_numba.sparse_superlu.cy_superlu_wrapper"
_UMFPACK_MODULE = "sparse_numba.sparse_umfpack.cy_umfpack_wrapper"


def _prepend_env_path(variable, path):
    current = os.environ.get(variable, "")
    if path not in current.split(os.pathsep):
        os.environ[variable] = path + os.pathsep + current if current else path


def _add_library_dirs(windows_dirs, posix_dirs):
    if IS_WINDOWS:
        for dll_path in windows_dirs:
            if os.path.exists(dll_path):
                logger.debug("library directory: %s", dll_path)
                # Set DLL directory on newer Python versions (3.8+)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(dll_path)
                    except OSError as e:
                        logger.debug("add_dll_directory(%s) failed: %s", dll_path, e)
                # Also update PATH for all Python versions
                _prepend_env_path('PATH', dll_path)
    elif IS_LINUX or IS_MACOS:
        variable = 'LD_LIBRARY_PATH' if IS_LINUX else 'DYLD_LIBRARY_PATH'
        for lib_path in posix_dirs:
            if os.path.exists(lib_path):
                logger.debug("library directory: %s", lib_path)
                _prepend_env_path(variable, lib_path)


def _vendor_dirs(name):
    """The search list of one vendored library: this package's own
    vendor/<name> (wherever it is installed: site-packages, --target, a
    user site; added at 0.2.0), then 0.1.11's list -- the source tree's
    vendor/<name> and sys.prefix's site-packages/sparse_numba/vendor/<name>."""
    windows = [
        os.path.join(_PACKAGE_DIR, "vendor", name, "bin"),
        os.path.join(_PACKAGE_DIR, "vendor", "openblas", "bin"),
        os.path.join(_ROOT_DIR, "vendor", name, "bin"),
        os.path.join(_ROOT_DIR, "vendor", "openblas", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", name, "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "sparse_numba", "vendor", "openblas", "bin"),
    ]
    posix = [
        os.path.join(_ROOT_DIR, "vendor", name, "lib"),
        os.path.join(_ROOT_DIR, "vendor", "openblas", "lib"),
    ]
    return windows, posix


# Setup library paths
def initialize_superlu():
    """The SuperLU library directories (path setup only, silent)."""
    _add_library_dirs(*_vendor_dirs("superlu"))


# Setup library paths
def initialize_umfpack():
    """The SuiteSparse library directories, and SUITESPARSE_BIN on Windows
    (path setup only, silent)."""
    windows, posix = _vendor_dirs("suitesparse")
    suitesparse_sys_env = os.environ.get('SUITESPARSE_BIN', '')
    if suitesparse_sys_env:
        windows.append(suitesparse_sys_env)
    _add_library_dirs(windows, posix)


# Initialize UMFPACK during module import
initialize_umfpack()
# Initialize SuperLU during module import
initialize_superlu()


_AVAILABLE = {}


def _available(module):
    if module not in _AVAILABLE:
        try:
            importlib.import_module(module)
            _AVAILABLE[module] = True
        except ImportError as e:
            logger.debug("%s did not import: %s", module, e)
            _AVAILABLE[module] = False
    return _AVAILABLE[module]


def is_slu_available():
    """Check if SuperLU is available"""
    return _available(_SUPERLU_MODULE)


def is_umf_available():
    """Check if umfpack is available: the interface extension imports and
    the user's UMFPACK library is found (UMFPACK is interface-only: the
    library is the user's, loaded at runtime)"""
    if not _available(_UMFPACK_MODULE):
        return False
    from sparse_numba._symbols import umfpack_library
    return umfpack_library()[0] == 0


# The lazily bound names: name -> the module that defines it
_SUPERLU_INTERFACE = "sparse_numba.sparse_superlu.superlu_numba_interface"
_UMFPACK_INTERFACE = "sparse_numba.sparse_umfpack.umfpack_numba_interface"
_CONVERSION = "sparse_numba.conversion.matrix_conversion_numba"
_API = "sparse_numba.api"

_LAZY = {}
for _verb in ("solve_csc", "solve_coo", "solve_csr", "factorize_csc", "factorize_coo",
              "factorize_csr", "solve_factored", "free_factors"):
    _LAZY["superlu_" + _verb] = _SUPERLU_INTERFACE
    _LAZY["umfpack_" + _verb] = _UMFPACK_INTERFACE
for _name in ("convert_coo_to_csc", "convert_csr_to_csc", "convert_coo_to_csr",
              "sparse_matvec_csr", "validate_sparse_matrix", "check_matrix_properties",
              "validate_csc_matrix"):
    _LAZY[_name] = _CONVERSION
for _name in ("factorize", "solve_into", "solve_many_into", "refactorize", "free", "solve",
              "handle_info", "Factors", "require", "backends", "is_eigen_available",
              "SUPERLU", "UMFPACK", "EIGEN_LU", "EIGEN_LDLT",
              "NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A"):
    _LAZY[_name] = _API
for _name in ("explain", "SparseNumbaError", "SymbolConflictError"):
    _LAZY[_name] = "sparse_numba.errors"
_LAZY["cache_key"] = "sparse_numba._symbols"
del _verb, _name

# Submodules bound by name
_SUBMODULES = {
    "matrix_conversion_numba": _CONVERSION,
    "symbols": "sparse_numba.symbols",
}


def __getattr__(name):
    if name in _SUBMODULES:
        value = importlib.import_module(_SUBMODULES[name])
    elif name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
    else:
        raise AttributeError(f"module 'sparse_numba' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))


__all__ = [
    'matrix_conversion_numba',
    'is_slu_available',
    'is_umf_available',
    'cache_key',
    # Pre-factorization API (SuperLU)
    'superlu_factorize_csc', 'superlu_factorize_coo', 'superlu_factorize_csr',
    'superlu_solve_factored', 'superlu_free_factors',
    # Pre-factorization API (UMFPACK)
    'umfpack_factorize_csc', 'umfpack_factorize_coo', 'umfpack_factorize_csr',
    'umfpack_solve_factored', 'umfpack_free_factors',
    # Sparse utilities
    'convert_coo_to_csr', 'sparse_matvec_csr',
    # The backend-neutral API (added at 0.2.0)
    'factorize', 'solve_into', 'solve_many_into', 'refactorize', 'free', 'solve',
    'handle_info', 'Factors', 'explain', 'SparseNumbaError', 'SymbolConflictError',
    'symbols', 'is_eigen_available', 'backends', 'require',
    'SUPERLU', 'UMFPACK', 'EIGEN_LU', 'EIGEN_LDLT',
    'NATURAL', 'COLAMD', 'AMD', 'MMD_AT_PLUS_A',
]


__version__ = "0.2.0"
__author__ = "Tianqi Hong"
