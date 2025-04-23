"""
Sparse Numba - Fast sparse solver with Numba support
__init__.py under sparse_numba
"""

import logging
import importlib

# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sparse_numba")

# Import submodules - we'll import these lazily to avoid circular imports
from .conversion import matrix_conversion_numba

# Define public API
# __all__ = [
#     'matrix_conversion_numba',
#     'superlu_numba_interface',
#     'umfpack_numba_interface',
#     'has_umfpack',
#     'has_superlu'
# ]

__all__ = [
    'matrix_conversion_numba',
    'sparse_superlu',
    'sparse_umfpack'
]

# Variables to track availability of solvers
# has_umfpack = False
# has_superlu = False


# Lazy imports
def __getattr__(name):
    """Lazy import of submodules to avoid circular dependencies"""
    # global has_umfpack, has_superlu

    # if name == 'sparse_superlu':
    #     # from .sparse_superlu import superlu_numba_interface
    #     from . import sparse_superlu
    #     # has_superlu = True
    #     return sparse_superlu #superlu_numba_interface
    # elif name == 'sparse_umfpack':
    #     # from .sparse_umfpack import umfpack_numba_interface
    #     from . import sparse_umfpack
    #     # has_umfpack = True
    #     return sparse_umfpack #umfpack_numba_interface
    # else:
    #     raise AttributeError(f"module 'sparse_numba' has no attribute '{name}'")
    if name == 'sparse_superlu':
        module = importlib.import_module('.sparse_superlu', package='sparse_numba')
        return module
    elif name == 'sparse_umfpack':
        module = importlib.import_module('.sparse_umfpack', package='sparse_numba')
        return module
    else:
        raise AttributeError(f"module 'sparse_numba' has no attribute '{name}'")


__author__ = "Tianqi Hong"