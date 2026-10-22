"""
Refusal cells (the validated boundary, the codes, the handle registry):
every code of sparse_numba.errors produced by one input, in-process, the
interpreter alive after every one. The Eigen LDLT row (BAD_SHAPE on an
unsymmetric pattern) is in test_sn4_eigen.py. ALLOC_FAILED has no cell: no
input reaches it short of exhausting the memory.

The kept-name guard cells run the 0.1.11 names against the C registry (a
double free returns HANDLE_FREED; at 0.1.11 it ended the interpreter).
"""

import importlib

import numpy as np
import pytest

import _sn1_common as common
import _sn2_common as c2



def _api():
    import sparse_numba.api as api
    return api


def _kept(backend):
    return importlib.import_module(common.BACKENDS[backend])


@pytest.fixture
def system():
    return c2.m100()


# ------------------------------------------------------------- the boundary

@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_bad_dtype(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    assert api.factorize(data.astype(np.float32), indices, indptr, n, be, order) \
        == (0, api.BAD_DTYPE)
    assert api.factorize(data, indices.astype(np.int64), indptr, n, be, order) \
        == (0, api.BAD_DTYPE)
    assert api.factorize(data, indices, indptr.astype(np.int64), n, be, order) \
        == (0, api.BAD_DTYPE)
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    try:
        x = np.empty(n)
        assert api.solve_into(handle, b.astype(np.float32), x) == api.BAD_DTYPE
        assert api.solve_into(handle, b, np.empty(n, dtype=np.int64)) == api.BAD_DTYPE
        assert api.refactorize(handle, data.astype(np.float32)) == api.BAD_DTYPE
        assert api.solve_many_into(handle, np.ones((n, 2), dtype=np.float32),
                                   np.empty((n, 2))) == api.BAD_DTYPE
    finally:
        assert api.free(handle) == api.OK


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_bad_shape(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    assert api.factorize(data.reshape(-1, 1), indices, indptr, n, be, order) \
        == (0, api.BAD_SHAPE)
    assert api.factorize(np.append(data, 1.0), indices, indptr, n, be, order) \
        == (0, api.BAD_SHAPE)
    assert api.factorize(data, indices, indptr[:1], 0, be, order) == (0, api.BAD_SHAPE)
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    try:
        assert api.solve_into(handle, b[:-1].copy(), np.empty(n - 1)) == api.BAD_SHAPE
        assert api.solve_into(handle, b, np.empty(n + 1)) == api.BAD_SHAPE
        wide = np.empty(2 * n)
        assert api.solve_into(handle, b, wide[::2]) == api.BAD_SHAPE
        assert api.refactorize(handle, data[:-1].copy()) == api.BAD_SHAPE
        assert api.solve_many_into(handle, np.ones((n, 2)), np.empty((n, 3))) \
            == api.BAD_SHAPE
        assert api.solve_many_into(handle, np.ones(n), np.empty(n)) == api.BAD_SHAPE
        x = np.empty(n)
        assert api.solve_into(handle, b, x) == api.OK
    finally:
        assert api.free(handle) == api.OK


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_bad_indptr(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    p = indptr.copy()
    p[0] = 1
    assert api.factorize(data, indices, p, n, be, order) == (0, api.BAD_INDPTR)
    p = indptr.copy()
    p[n] = p[n] - 1
    assert api.factorize(data, indices, p, n, be, order) == (0, api.BAD_INDPTR)
    p = indptr.copy()
    p[2], p[3] = p[3], p[2] - 1
    assert api.factorize(data, indices, p, n, be, order) == (0, api.BAD_INDPTR)


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_bad_index(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    for bad in (n, -1):
        i = indices.copy()
        i[3] = bad
        assert api.factorize(data, i, indptr, n, be, order) == (0, api.BAD_INDEX), bad


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_not_square(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    assert api.factorize(data, indices, indptr, n - 1, be, order) == (0, api.NOT_SQUARE)
    assert api.factorize(data, indices, indptr, n + 1, be, order) == (0, api.NOT_SQUARE)


# ------------------------------------------------------------- the handle

@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_handle_invalid(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    x = np.empty(n)
    for h in (0, -1, 12345):
        assert api.free(h) == api.HANDLE_INVALID, h
        assert api.solve_into(h, b, x) == api.HANDLE_INVALID, h
        assert api.refactorize(h, data) == api.HANDLE_INVALID, h
        assert api.handle_info(h)[-1] == api.HANDLE_INVALID, h
    # a generation never handed out on a live slot word
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    forged = handle + (1000 << 32)
    assert api.solve_into(forged, b, x) == api.HANDLE_INVALID
    assert api.free(forged) == api.HANDLE_INVALID
    assert api.free(handle) == api.OK


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_handle_freed(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK
    assert api.free(handle) == api.OK
    x = np.empty(n)
    assert api.free(handle) == api.HANDLE_FREED                     # double free
    assert api.solve_into(handle, b, x) == api.HANDLE_FREED         # use after free
    assert api.solve_many_into(handle, np.ones((n, 2)), np.empty((n, 2))) \
        == api.HANDLE_FREED
    assert api.refactorize(handle, data) == api.HANDLE_FREED
    assert api.handle_info(handle)[-1] == api.HANDLE_FREED
    # the slot is reused by the next factorize; the stale handle stays refused
    h2, info = api.factorize(data, indices, indptr, n, be, order)
    assert info == api.OK and h2 != handle
    assert api.solve_into(handle, b, x) == api.HANDLE_FREED
    assert api.free(h2) == api.OK


def test_handle_wrong_backend_kept_names(system):
    c2.require_backend("superlu")
    c2.require_backend("umfpack")
    slu, umf = _kept("superlu"), _kept("umfpack")
    data, indices, indptr, n, b = system
    h_slu, info = slu.superlu_factorize_csc(data, indices, indptr)
    assert info == 0
    h_umf, info = umf.umfpack_factorize_csc(data, indices, indptr)
    assert info == 0
    x, info = slu.superlu_solve_factored(h_umf, b)
    assert info == -12                                  # HANDLE_WRONG_BACKEND
    x, info = umf.umfpack_solve_factored(h_slu, b)
    assert info == -12
    assert slu.superlu_free_factors(h_umf) == -12
    assert umf.umfpack_free_factors(h_slu) == -12
    assert slu.superlu_free_factors(h_slu) == 0
    assert umf.umfpack_free_factors(h_umf) == 0


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_handle_guards_kept_names(backend, system):
    c2.require_backend(backend)
    mod = _kept(backend)
    factorize = getattr(mod, backend + "_factorize_csc")
    solve_factored = getattr(mod, backend + "_solve_factored")
    free_factors = getattr(mod, backend + "_free_factors")
    data, indices, indptr, n, b = system
    assert free_factors(0) == -10                       # HANDLE_INVALID
    assert free_factors(12345) == -10
    handle, info = factorize(data, indices, indptr)
    assert info == 0
    assert free_factors(handle) == 0
    assert free_factors(handle) == -11                  # HANDLE_FREED, double free
    x, info = solve_factored(handle, b)                 # use after free
    assert info == -11


# ------------------------------------------------------- backend and ordering

def test_backend_absent(system):
    """An unknown backend id; the Eigen kinds are present
    (tests/test_sn4_eigen.py)."""
    api = _api()
    data, indices, indptr, n, b = system
    assert api.factorize(data, indices, indptr, n, 99, api.NATURAL) \
        == (0, api.BACKEND_ABSENT)
    x, info = api.solve(data, indices, indptr, n, b, 99, api.COLAMD)
    assert info == api.BACKEND_ABSENT


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_ordering_unsupported(backend, system):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n, b = system
    # SuperLU COLAMD / MMD_AT_PLUS_A and UMFPACK NATURAL / COLAMD are open
    # (tests/test_sn3_backends.py); what stays refused (SuperLU has no AMD):
    refused = {"superlu": (api.AMD, 99),
               "umfpack": (api.MMD_AT_PLUS_A, 99)}[backend]
    for ordering in refused:
        assert api.factorize(data, indices, indptr, n, be, ordering) \
            == (0, api.ORDERING_UNSUPPORTED), ordering


# ------------------------------------------------------------- the backend

def _singular_3x3():
    """diag(1, 1, 0) with the (2, 2) entry an explicit zero: U(3, 3) = 0."""
    data = np.array([1.0, 1.0, 0.0])
    indices = np.array([0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    return data, indices, indptr, 3


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_singular(backend):
    c2.require_backend(backend)
    api = _api()
    be, order = c2.api_ids(api, backend)
    data, indices, indptr, n = _singular_3x3()
    handle, info = api.factorize(data, indices, indptr, n, be, order)
    assert handle == 0
    if backend == "superlu":
        assert info == 3                                # SINGULAR = k, dgstrf's info
    else:
        assert 1 <= info <= n                           # SINGULAR = k, U's zero diagonal
    assert "3" in api.explain(3)


def test_backend_error_umfpack():
    """A duplicate row index in a column: UMFPACK's invalid_matrix status."""
    c2.require_backend("umfpack")
    api = _api()
    data = np.array([1.0, 1.0, 1.0, 1.0])
    indices = np.array([0, 0, 1, 2], dtype=np.int32)
    indptr = np.array([0, 2, 3, 4], dtype=np.int32)
    assert api.factorize(data, indices, indptr, 3, api.UMFPACK, api.AMD) \
        == (0, api.BACKEND_ERROR)


# --------------------------------------------------- the kept names, no garbage

@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_kept_names_refuse_wrong_dtype(backend, system, capfd):
    c2.require_backend(backend)
    mod = _kept(backend)
    data, indices, indptr, n, b = system
    handle, info = getattr(mod, backend + "_factorize_csc")(
        data.astype(np.float32), indices, indptr)
    assert (handle, info) == (0, -1)                    # BAD_DTYPE
    x, info = getattr(mod, backend + "_solve_csc")(data, indices, indptr, b.astype(np.float32))
    assert info == -1
    p = indptr.copy()
    p[0] = 1
    handle, info = getattr(mod, backend + "_factorize_csc")(data, indices, p)
    assert (handle, info) == (0, -3)                    # BAD_INDPTR, nothing printed
    out, err = capfd.readouterr()
    assert out == "" and err == ""
