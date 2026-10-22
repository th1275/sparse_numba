"""
Contract cells (the codes, the symbol table, the KEPT names, the silent
import).

- (d) every TABLE row registered in llvmlite after import; for every row
  (the declared signature against the .pyx `cdef api` declaration, the
  registered address against get_cython_function_address); a conflicting
  pre-registration refused with SymbolConflictError (in a child); no
  platform filename in any package module;
- (e) `import sparse_numba` in a child with the UMFPACK DLLs hidden: empty
  stdout and stderr, UMFPACK reported absent, its first call
  BACKEND_ABSENT; no `print(` in the njit sources; no `printf` outside
  `#ifdef SPARSE_NUMBA_DEBUG` in the C sources;
- the codes header mirrored by errors.py and the enums by sparse_numba.api;
- the KEPT names' positional signatures and functype_* as ctypes
  data; `from sparse_numba import *` binding what __all__ lists.

No cell carries an xfail marker.
"""

import ctypes
import importlib
import json
import os
import re
import subprocess
import sys

import pytest

import _sn1_common as common
import _sn2_common as c2


PKG = os.path.join(common.REPO_DIR, "sparse_numba")
CODES_HEADER = os.path.join(PKG, "sparse_numba_codes.h")
C_SOURCES = [
    CODES_HEADER,
    os.path.join(PKG, "sparse_superlu", "superlu_wrapper.c"),
    os.path.join(PKG, "sparse_superlu", "superlu_wrapper.h"),
    os.path.join(PKG, "sparse_umfpack", "umfpack_wrapper.c"),
    os.path.join(PKG, "sparse_umfpack", "umfpack_wrapper.h"),
]
NJIT_SOURCES = [
    os.path.join(PKG, "api.py"),
    os.path.join(PKG, "sparse_superlu", "superlu_numba_interface.py"),
    os.path.join(PKG, "sparse_umfpack", "umfpack_numba_interface.py"),
]
PYX = {
    "superlu": os.path.join(PKG, "sparse_superlu", "cy_superlu_wrapper.pyx"),
    "umfpack": os.path.join(PKG, "sparse_umfpack", "cy_umfpack_wrapper.pyx"),
}
EIGEN_MODULE = "sparse_numba.sparse_eigen.cy_eigen_wrapper"
PYX_BY_MODULE = {
    c2.CY_MODULE["superlu"]: PYX["superlu"],
    c2.CY_MODULE["umfpack"]: PYX["umfpack"],
    EIGEN_MODULE: os.path.join(PKG, "sparse_eigen", "cy_eigen_wrapper.pyx"),
}
IMPORT_CHILD = os.path.join(common.TESTS_DIR, "sn2_import_child.py")


def _read(path):
    with open(path, "r", encoding="ascii") as f:
        return f.read()


def _child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = common.IMPORT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ------------------------------------------------------- (d) the symbol table

_PYX_API = re.compile(r"cdef\s+api\s+(\w+)\s+(\w+)\s*\(([^)]*)\)", re.S)


def _pyx_declarations(path):
    decls = {}
    for ret, name, args in _PYX_API.findall(_read(path)):
        types_ = []
        for arg in args.split(","):
            arg = " ".join(arg.split())
            ptr = "*" in arg
            ctype = arg.replace("*", " ").split()[0]
            types_.append(ctype + "*" if ptr else ctype)
        decls[name] = (ret, types_)
    return decls


def _numba_signature(ret, args):
    from numba import types
    ctypes_map = {"int": types.intc, "int64_t": types.int64}
    conv = lambda t: types.voidptr if t.endswith("*") else ctypes_map[t]
    return conv(ret)(*[conv(a) for a in args])


def test_table_registered_after_import():
    """Every TABLE row, the API rows among them, registered by the interface
    modules' import."""
    import llvmlite.binding as llb
    from sparse_numba import _symbols
    for backend in c2.BACKEND_NAMES:
        c2.require_backend(backend)
        importlib.import_module(common.BACKENDS[backend])
    names = [r[0] for r in _symbols.TABLE]
    for verb in ("factorize_ordered", "refactorize", "solve_into", "handle_info"):
        for backend in c2.BACKEND_NAMES:
            assert f"sparse_numba_{backend}_{verb}" in names
    # the SuperLU and UMFPACK rows; the Eigen rows: tests/test_sn4_eigen.py
    assert len([r for r in _symbols.TABLE if r[2] != EIGEN_MODULE]) == 16
    for name in names:
        assert llb.address_of_symbol(name) is not None, name


def test_r5_every_row():
    """sparse_numba.symbols: each declared signature equal to the .pyx's
    `cdef api` declaration; each registered address the module's capsule."""
    import llvmlite.binding as llb
    from numba.extending import get_cython_function_address
    import sparse_numba.api  # noqa: F401  registers every row
    from sparse_numba import symbols
    # per extension module (the Eigen module's: tests/test_sn4_eigen.py)
    decls = {module: _pyx_declarations(path) for module, path in PYX_BY_MODULE.items()
             if os.path.exists(path)}
    for module in sorted({r[2] for r in symbols.TABLE}):
        rows = [r for r in symbols.TABLE if r[2] == module]
        assert sorted(r[1] for r in rows) == sorted(decls[module]), module
    for symbol, cdef_name, module, sig_text in symbols.TABLE:
        declared = symbols.declared(symbol)
        assert declared == _numba_signature(*decls[module][cdef_name]), symbol
        assert str(declared) == sig_text
        assert llb.address_of_symbol(symbol) is not None, symbol
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        assert llb.address_of_symbol(symbol) == get_cython_function_address(module, cdef_name)


_CONFLICT_CHILD = r"""
import ctypes, sys
import llvmlite.binding as llb
keep = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64)(lambda h: 0)
llb.add_symbol("sparse_numba_superlu_free_factors", ctypes.cast(keep, ctypes.c_void_p).value)
from sparse_numba.errors import SymbolConflictError
try:
    import sparse_numba.sparse_superlu.superlu_numba_interface
except SymbolConflictError as e:
    sys.stdout.write("SymbolConflictError " + str(e))
    sys.exit(0)
sys.exit(3)
"""


def test_conflicting_preregistration_refused():
    """A downstream caller's shape (its own symbol registered first): a
    name another module claimed is refused at import, never overwritten."""
    c2.require_backend("superlu")
    proc = subprocess.run([sys.executable, "-c", _CONFLICT_CHILD], cwd=common.IMPORT_ROOT,
                          env=_child_env(), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr[-4000:])
    assert proc.stdout.startswith("SymbolConflictError")
    assert "sparse_numba_superlu_free_factors" in proc.stdout


_PLATFORM_NAME = re.compile(r"cp3\d+-|win_amd64|\.pyd\b|\.dll\b|\.dylib\b|\.so\b")


def test_no_platform_filenames():
    hits = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in ("test", "__pycache__", "vendor")]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                for k, line in enumerate(_read(path).splitlines(), 1):
                    if _PLATFORM_NAME.search(line):
                        hits.append(f"{os.path.relpath(path, PKG)}:{k}: {line.strip()}")
    assert hits == []


# ------------------------------------------------------- the codes, mirrored

_DEFINE = re.compile(r"^#define\s+SN_(\w+)\s+\(?(-?\d+)\)?\s*$", re.M)


def _header_defines():
    return {k: int(v) for k, v in _DEFINE.findall(_read(CODES_HEADER))}


_CODES = ("OK", "BAD_DTYPE", "BAD_SHAPE", "BAD_INDPTR", "BAD_INDEX", "NOT_SQUARE",
          "HANDLE_INVALID", "HANDLE_FREED", "HANDLE_WRONG_BACKEND", "BACKEND_ABSENT",
          "ORDERING_UNSUPPORTED", "ALLOC_FAILED", "BACKEND_ERROR")


def test_codes_header_mirrored_by_errors():
    from sparse_numba import errors
    defines = _header_defines()
    expected = dict(OK=0, BAD_DTYPE=-1, BAD_SHAPE=-2, BAD_INDPTR=-3, BAD_INDEX=-4,
                    NOT_SQUARE=-5, HANDLE_INVALID=-10, HANDLE_FREED=-11,
                    HANDLE_WRONG_BACKEND=-12, BACKEND_ABSENT=-20,
                    ORDERING_UNSUPPORTED=-21, ALLOC_FAILED=-30, BACKEND_ERROR=-31)
    for name in _CODES:
        assert defines[name] == expected[name] == getattr(errors, name), name


def test_enums_mirrored_by_api():
    import sparse_numba.api as api
    defines = _header_defines()
    for name in ("SUPERLU", "UMFPACK", "EIGEN_LU", "EIGEN_LDLT"):
        assert defines["BACKEND_" + name] == getattr(api, name), name
    for name in ("NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A"):
        assert defines["ORDERING_" + name] == getattr(api, name), name


# ------------------------------------------------------- (e) no print, no banner

def test_import_silent_with_umfpack_hidden(tmp_path):
    if sys.platform != "win32":
        pytest.skip("the hidden-DLL child is the Windows layout")
    out = tmp_path / "import_child.json"
    proc = subprocess.run([sys.executable, IMPORT_CHILD, str(out)], cwd=common.IMPORT_ROOT,
                          env=_child_env(), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-4000:])
    assert proc.stdout == ""
    assert proc.stderr == ""
    with open(out, "r", encoding="ascii") as f:
        facts = json.load(f)
    # the extension module itself imports (UMFPACK interface-only): its
    # library is what is hidden (tests/test_sn3_backends.py)
    assert facts["is_umf_available"] is False
    assert facts["is_slu_available"] is True
    # UMFPACK absent, SuperLU present (the Eigen kinds are 3 and 4)
    assert 1 in facts["backends"] and 2 not in facts["backends"]
    assert facts["umfpack_factorize_info"] == -20
    assert facts["kept_umfpack_factorize_info"] == -20
    assert facts["superlu_factorize_info"] == 0


def test_no_print_in_njit_sources():
    hits = []
    for path in NJIT_SOURCES:
        for k, line in enumerate(_read(path).splitlines(), 1):
            if "print(" in line:
                hits.append(f"{os.path.relpath(path, PKG)}:{k}: {line.strip()}")
    assert hits == []


def test_no_printf_outside_debug():
    debug_if = re.compile(r"#\s*(ifdef\s+SPARSE_NUMBA_DEBUG\b|"
                          r"if\s+defined\s*\(\s*SPARSE_NUMBA_DEBUG\s*\))")
    hits = []
    for path in C_SOURCES:
        stack = []      # one entry per open #if: True while the branch is debug-only
        for k, line in enumerate(_read(path).splitlines(), 1):
            s = line.strip()
            if re.match(r"#\s*if", s):
                stack.append(bool(debug_if.match(s)))
                continue
            if re.match(r"#\s*(else|elif)\b", s):
                if stack:
                    stack[-1] = False
                continue
            if re.match(r"#\s*endif\b", s):
                if stack:
                    stack.pop()
                continue
            if "printf" in line and not any(stack):
                hits.append(f"{os.path.relpath(path, PKG)}:{k}: {s}")
    assert hits == []


# ------------------------------------------------------- the KEPT surface

KEPT_SIGNATURES = {
    "solve_csc": ("csc_data", "csc_indices", "csc_indptr", "b"),
    "solve_coo": ("row_indices", "col_indices", "data", "shape", "b"),
    "solve_csr": ("csr_data", "csr_indices", "csr_indptr", "b"),
    "factorize_csc": ("csc_data", "csc_indices", "csc_indptr"),
    "factorize_coo": ("row_indices", "col_indices", "data", "shape"),
    "factorize_csr": ("csr_data", "csr_indices", "csr_indptr"),
    "solve_factored": ("handle", "b"),
    "free_factors": ("handle",),
}


@pytest.mark.parametrize("backend", c2.BACKEND_NAMES)
def test_kept_signatures(backend):
    """Every KEPT name with its 0.1.11 positional signature; the
    functype_* names ctypes CFUNCTYPE data."""
    c2.require_backend(backend)
    mod = importlib.import_module(common.BACKENDS[backend])
    for name, params in KEPT_SIGNATURES.items():
        fn = getattr(mod, f"{backend}_{name}")
        code = fn.py_func.__code__
        assert code.co_varnames[:code.co_argcount] == params, name
        assert fn.py_func.__defaults__ is None, name
    for name in ("functype", "functype_factorize", "functype_solve_factored", "functype_free"):
        t = getattr(mod, name)
        assert isinstance(t, type) and issubclass(t, ctypes._CFuncPtr), name
    assert mod.functype_solve_factored._argtypes_ == (
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    assert mod.functype_solve_factored._restype_ == ctypes.c_int


def test_star_import_binds_all():
    import sparse_numba
    namespace = {}
    exec("from sparse_numba import *", namespace)
    for name in sparse_numba.__all__:
        assert name in namespace, name
    for name in ("factorize", "solve_into", "solve_many_into", "refactorize", "free",
                 "solve", "handle_info", "Factors", "explain", "SparseNumbaError",
                 "SymbolConflictError", "cache_key", "symbols", "is_eigen_available",
                 "backends", "require", "SUPERLU", "UMFPACK", "EIGEN_LU", "EIGEN_LDLT",
                 "NATURAL", "COLAMD", "AMD", "MMD_AT_PLUS_A",
                 "superlu_factorize_csc", "superlu_solve_factored", "superlu_free_factors",
                 "umfpack_factorize_csc", "umfpack_free_factors",
                 "convert_coo_to_csr", "sparse_matvec_csr",
                 "is_slu_available", "is_umf_available"):
        assert name in sparse_numba.__all__, name
