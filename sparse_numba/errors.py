"""
Typed errors of sparse_numba: the status codes, explain(info),
SparseNumbaError and SymbolConflictError

THE CODES CONTRACT (the info of every call): 0 success; a negative code a
refusal or a backend error, by name in sparse_numba.errors (explain(info)
says it in words); a positive k SINGULAR, the zero pivot's 1-based index.
`info != 0` is failure, as at 0.1.11.
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: errors.py

# The status codes of sparse_numba, mirroring sparse_numba_codes.h. A
# positive k is SINGULAR: the zero pivot's 1-based index
OK = 0
BAD_DTYPE = -1
BAD_SHAPE = -2
BAD_INDPTR = -3
BAD_INDEX = -4
NOT_SQUARE = -5
HANDLE_INVALID = -10
HANDLE_FREED = -11
HANDLE_WRONG_BACKEND = -12
BACKEND_ABSENT = -20
ORDERING_UNSUPPORTED = -21
ALLOC_FAILED = -30
BACKEND_ERROR = -31


class SymbolConflictError(RuntimeError):
    """An LLVM symbol name is already registered at a different address.

    llvmlite's add_symbol overwrites silently, so sparse_numba reads the name
    before registering it and refuses a name another module claimed.
    """


_TEXT = {
    OK: "OK: success",
    BAD_DTYPE: "BAD_DTYPE: an array of the wrong dtype (data, b, out float64; "
               "indices, indptr int32)",
    BAD_SHAPE: "BAD_SHAPE: an array of the wrong rank, length or layout (1-D, "
               "C-contiguous, len n; B and OUT (n, k)), or n <= 0",
    BAD_INDPTR: "BAD_INDPTR: indptr[0] != 0, indptr[n] != nnz, or indptr decreasing",
    BAD_INDEX: "BAD_INDEX: a row index outside [0, n)",
    NOT_SQUARE: "NOT_SQUARE: len(indptr) != n + 1",
    HANDLE_INVALID: "HANDLE_INVALID: zero, negative, or never handed out",
    HANDLE_FREED: "HANDLE_FREED: the handle was freed (a double free or a use after free)",
    HANDLE_WRONG_BACKEND: "HANDLE_WRONG_BACKEND: the handle belongs to another backend",
    BACKEND_ABSENT: "BACKEND_ABSENT: the backend is not available in this process",
    ORDERING_UNSUPPORTED: "ORDERING_UNSUPPORTED: the backend does not offer this ordering",
    ALLOC_FAILED: "ALLOC_FAILED: an allocation failed",
    BACKEND_ERROR: "BACKEND_ERROR: the backend reported an error (its own status in "
                   "handle_info's last_status)",
}


def explain(info):
    """The status code info in words (python-level)."""
    info = int(info)
    if info > 0:
        return f"SINGULAR: a zero pivot at the 1-based index {info}"
    return _TEXT.get(info, f"UNKNOWN: status {info}")


class SparseNumbaError(RuntimeError):
    """A status code raised by a python-level convenience wrapper; the njit
    functions return the code and never raise."""

    def __init__(self, info, detail=None):
        self.info = int(info)
        text = explain(self.info)
        super().__init__(text if detail is None else f"{text} ({detail})")
