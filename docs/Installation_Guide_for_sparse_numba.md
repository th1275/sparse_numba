# Installation guide for sparse_numba 0.2.0

sparse_numba solves sparse linear systems from inside `@njit(nogil=True)`
code with SuperLU, Eigen (SparseLU, SimplicialLDLT) and a user-supplied
UMFPACK. This guide covers installing a wheel, adding UMFPACK, building
from source and checking the result. The [user guide](USER_GUIDE.md)
covers using it.

## 1 Requirements

- CPython 3.10, 3.11, 3.12 or 3.13
- numba >= 0.61 and numpy >= 1.24 (pip installs them)
- Python 3.9 or older: stay on sparse_numba 0.1.11

## 2 Install a wheel

```bash
pip install sparse-numba
```

| platform | wheel | SuperLU / OpenBLAS | Eigen | UMFPACK |
|---|---|---|---|---|
| Windows x86_64 | yes | in the wheel | compiled in | the user's SuiteSparse (section 3) |
| Linux x86_64 (manylinux) | yes | the system's: `sudo apt-get install libsuperlu-dev libopenblas-dev` (Debian / Ubuntu) | compiled in | the user's SuiteSparse |
| macOS arm64 | yes | Homebrew's: `brew install superlu openblas` | compiled in | the user's SuiteSparse |
| macOS x86_64 (Intel) | no: build from source (section 4) | Homebrew's | compiled in | the user's SuiteSparse |

SuperLU and the two Eigen backends work without anything else. UMFPACK is
optional.

## 3 UMFPACK: bring your own SuiteSparse

No SuiteSparse binary ships with sparse_numba (UMFPACK and CHOLMOD are
GPL / LGPL). The UMFPACK backend loads your library at its first call, once
per process:

1. `SPARSE_NUMBA_UMFPACK_LIBRARY=/path/to/libumfpack.{dll,so,dylib}`: only
   that path is tried;
2. otherwise the platform's names through the platform's loader:
   `libumfpack.dll` / `umfpack.dll` on Windows (sparse_numba adds the
   directory in `SUITESPARSE_BIN` when it is set), `libumfpack.so.6` /
   `.so.5` / `.so` on Linux, `libumfpack.6.dylib` / `.5.dylib` / `.dylib`
   on macOS.

Getting a library:

- Debian / Ubuntu: `sudo apt-get install libsuitesparse-dev`
- macOS: `brew install suite-sparse`
- Windows: MSYS2's `mingw-w64-x86_64-suitesparse`, then
  `set SUITESPARSE_BIN=C:\msys64\mingw64\bin` (or a SuiteSparse build of
  your own, named by `SPARSE_NUMBA_UMFPACK_LIBRARY`)

Set the variables before the first UMFPACK call. When no library is found,
every UMFPACK call returns `BACKEND_ABSENT` (-20) and the other backends
are unaffected.

## 4 Build from source

From a checkout or the sdist:

```bash
git clone https://github.com/th1275/sparse_numba.git
cd sparse_numba
pip install build
python -m build --wheel
pip install dist/sparse_numba-0.2.0-<TAGS>.whl
```

What the build needs:

- Windows: MinGW-w64 gcc (x86_64, posix threads, seh) on PATH; setup.py
  selects the `mingw32` compiler itself. SuperLU, OpenBLAS and their
  headers are in the repository's `vendor/`; Eigen 3.4.0 is in
  `third_party/eigen-3.4.0`. The UMFPACK interface needs SuiteSparse's
  headers (`umfpack.h` and the ones it includes) copied into
  `vendor/suitesparse/include`; without them set
  `SPARSE_NUMBA_SKIP_UMFPACK=1` and the UMFPACK extension is not built.
- Linux: gcc, and the development packages
  `libsuperlu-dev libopenblas-dev libsuitesparse-dev` (Debian / Ubuntu
  names).
- macOS: the Xcode command line tools and
  `brew install superlu openblas suite-sparse`.

Build switches, read by setup.py at build time:

| variable | effect |
|---|---|
| `SPARSE_NUMBA_SKIP_UMFPACK=1` | the UMFPACK interface extension is not built |
| `SPARSE_NUMBA_DEBUG=1` | the extensions are compiled without `-DNDEBUG` (asserts on); development only |
| `SPARSE_NUMBA_EXTRA_CFLAGS="..."` | appended to every extension's compile flags; development only |

## 5 Check the installation

```python
import sparse_numba as sn

print(sn.__version__)           # 0.2.0
print(sn.backends())            # e.g. (1, 3, 4): SUPERLU, EIGEN_LU, EIGEN_LDLT; 2 = UMFPACK when found
sn.require(sn.SUPERLU)          # raises SparseNumbaError with an install hint if absent
print(sn.cache_key())           # the version, the extensions and the libraries in one string
```

Then run a script from [examples/](../examples/README.md), for example
`python examples/superlu_walk.py`.

## 6 Troubleshooting

- `backends()` leaves out SUPERLU on Linux or macOS: the system's SuperLU
  or OpenBLAS is missing; install the packages of section 2 (on Linux,
  `ldconfig -p | grep -e superlu -e openblas` shows what the loader finds).
- `backends()` leaves out UMFPACK: no library was found (section 3);
  `sn.require(sn.UMFPACK)` raises with the install hint.
- A build on Windows fails with `slu_ddefs.h` or `umfpack.h` not found:
  build from a checkout or the sdist (both carry `vendor/superlu`), and
  add SuiteSparse's headers or set `SPARSE_NUMBA_SKIP_UMFPACK=1`.
- On a read-only install numba cannot write its cache beside the package:
  set `NUMBA_CACHE_DIR` to a writable directory.

Issues: https://github.com/th1275/sparse_numba/issues
