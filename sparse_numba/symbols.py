"""
The external symbols of sparse_numba, public: TABLE =
(symbol name, cdef api name, module, numba signature string) rows, one per
external function, the names stable within a major version;
declared(name) = the ExternalFunction signature, for callers' tests
"""


#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: symbols.py

from sparse_numba._symbols import TABLE, declared

__all__ = ["TABLE", "declared"]
