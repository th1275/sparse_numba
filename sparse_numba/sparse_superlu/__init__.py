"""
SuperLU module
__init__.py under sparse_numba/sparse_superlu
"""
# Import from conversion directory
# from ..conversion.matrix_conversion_numba import convert_coo_to_csc, convert_csr_to_csc

# Import modules directly without creating circular references
from . import cy_superlu_wrapper
from . import superlu_numba_interface
# from . import test

__all__ = [
    'superlu_numba_interface',
    'cy_superlu_wrapper',
    # 'convert_coo_to_csc',
    # 'convert_csr_to_csc',
    # 'test'
]

__author__ = 'Tianqi Hong'