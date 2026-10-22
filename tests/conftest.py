"""
UMFPACK is interface-only: the library is the user's SuiteSparse, loaded at
the first call. Where it is not found (SPARSE_NUMBA_UMFPACK_LIBRARY and
SUITESPARSE_BIN unset, nothing on the loader's path -- CI's case), every
cell whose name names UMFPACK or SuiteSparse is skipped by name, except the
cells written for the library's absence and the ones that read only the
declarations; and the cells named in NEEDS_LIBRARY, which use UMFPACK
without naming it.
"""

import re

import pytest

# cells that name UMFPACK and pass without the user's library
RUN_WITHOUT_LIBRARY = (
    "test_sn2_contract.py::test_import_silent_with_umfpack_hidden",
    "test_sn3_backends.py::test_umfpack_library_not_found",
    "test_sn3_backends.py::test_umfpack_library_bad_env_path",
    "test_sn1_cache.py::test_declared_signature[umfpack]",
    "test_sn2_contract.py::test_kept_signatures[umfpack]",
    "test_sn3_backends.py::test_codes_contract_stated[sparse_numba.sparse_umfpack.umfpack_numba_interface]",
)

# cells that use the library without naming UMFPACK
NEEDS_LIBRARY = (
    "test_sn2_refusals.py::test_handle_wrong_backend_kept_names",
)

_NAMES_UMFPACK = re.compile(r"umf|suitesparse", re.IGNORECASE)


def pytest_collection_modifyitems(config, items):
    import sparse_numba
    if sparse_numba.is_umf_available():
        return
    skip = pytest.mark.skip(reason="UMFPACK: the user's library is not found "
                                   "(SPARSE_NUMBA_UMFPACK_LIBRARY / SUITESPARSE_BIN unset)")
    for item in items:
        name = item.nodeid.replace("\\", "/").rsplit("/", 1)[-1]
        if name in NEEDS_LIBRARY or (_NAMES_UMFPACK.search(name)
                                     and name not in RUN_WITHOUT_LIBRARY):
            item.add_marker(skip)
