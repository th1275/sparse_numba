"""
Python Interface with Eigen's sparse solvers

The ExternalFunction bindings of the Eigen extension: EIGEN_LU =
Eigen::SparseLU at NATURAL / COLAMD / AMD, EIGEN_LDLT =
Eigen::SimplicialLDLT (Lower, AMDOrdering) for a symmetric pattern, both
reached through sparse_numba.api (factorize(..., EIGEN_LU or EIGEN_LDLT,
ordering), solve_into, solve_many_into, refactorize, free, solve,
handle_info). There are no 0.1.11 names for Eigen. The Eigen 3.4.0 headers
are compiled in (third_party/eigen-3.4.0/, -DEIGEN_MPL2_ONLY,
-DEIGEN_DONT_PARALLELIZE); nothing is loaded at run time.

THE CODES CONTRACT (the info of every call): 0 success; a negative code a
refusal or a backend error, by name in sparse_numba.errors (explain(info)
says it in words); a positive k SINGULAR (Eigen reports 1: its
factorization met a zero pivot). `info != 0` is failure. EIGEN_LDLT refuses
a pattern that is not symmetric (BAD_SHAPE) and every ordering but AMD
(ORDERING_UNSUPPORTED); EIGEN_LU refuses MMD_AT_PLUS_A.
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: eigen_numba_interface.py

from numba import types

from sparse_numba import _symbols

_MODULE = "sparse_numba.sparse_eigen.cy_eigen_wrapper"

# Every row of this backend registered at import under its stable name,
# each address resolved by module name; an absent extension's rows at
# stubs returning BACKEND_ABSENT
_PRESENT = _symbols.register_backend(_MODULE)


def _function(name):
    return types.ExternalFunction(name, _symbols.declared(name))


c_lu_factorize = _function("sparse_numba_eigen_lu_factorize_ordered")
c_ldlt_factorize = _function("sparse_numba_eigen_ldlt_factorize_ordered")
c_refactorize = _function("sparse_numba_eigen_refactorize")
c_solve_into = _function("sparse_numba_eigen_solve_into")
c_handle_info = _function("sparse_numba_eigen_handle_info")
c_free = _function("sparse_numba_eigen_free")


def is_present():
    """True when the Eigen extension imported (header-only: nothing else to find)."""
    return _PRESENT
