"""
SuperLU module
__init__.py under sparse_numba/sparse_superlu
"""
import os
import sys
import logging

logger = logging.getLogger("sparse_numba.sparse_superlu")

# Explicitly make the extension importable from this package
# This is the critical part that was missing
try:
    # This line tells Python that the cy_superlu_wrapper is part of this package
    from . import cy_superlu_wrapper
    logger.info("SuperLU extension found and imported")
except ImportError as e:
    logger.warning(f"SuperLU extension import failed: {e}")

__author__ = 'Tianqi Hong'