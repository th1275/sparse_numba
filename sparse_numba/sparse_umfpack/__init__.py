"""
UMFPACK module
__init__.py under sparse_numba/sparse_umfpack
"""
import os
import sys
import logging

logger = logging.getLogger("sparse_numba.sparse_umfpack")

# Explicitly make the extension importable from this package
try:
    # This line tells Python that the cy_umfpack_wrapper is part of this package
    from . import cy_umfpack_wrapper
    logger.info("UMFPACK extension found and imported")
except ImportError as e:
    logger.warning(f"UMFPACK extension import failed: {e}")

__author__ = 'Tianqi Hong'