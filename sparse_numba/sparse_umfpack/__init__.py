"""
UMFPACK module
__init__.py under sparse_numba/sparse_umfpack
"""

# Import from conversion directory
from ..conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc

# Import modules directly without creating circular references
from . import cy_umfpack_wrapper
from . import umfpack_numba_interface
from . import test

__all__ = [
    'umfpack_numba_interface',
    'cy_umfpack_wrapper',
    'convert_coo_to_csc',
    'convert_csr_to_csc',
    'test'
]

__author__ = 'Tianqi Hong'