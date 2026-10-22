"""
Wheel cells (the test data copies; the wheel's contents and licences):

- the vehicles' files the cells read are in tests/data/ (copies of the
  comparison vehicles), kept byte for byte (tests/data/.gitattributes),
  each sha256 equal to tests/data/SHA256SUMS and, where the source tree
  carries the vehicles, to theirs;
- THE WHEEL'S CONTENTS (SPARSE_NUMBA_WHEEL names the .whl; skipped when
  unset): no __pycache__, *.pyc, *.nbi or *.nbc; no tests; the version
  0.2.0; on Windows the vendored DLLs are the SuperLU / OpenBLAS link chain
  (objdump's import lists: libsuperlu-7 -> libopenblas -> libgfortran-5,
  libgomp-1, libgcc_s_seh-1 -> libquadmath-0, libwinpthread-1; the Eigen
  extension -> libwinpthread-1), libsuperlu_fortran-7 and every SuiteSparse
  DLL absent, the licence file of each DLL beside it; on every platform no
  SuiteSparse library file, and Eigen's COPYING.MPL2 and COPYING.README
  beside the Eigen extension (byte-equal to third_party/eigen-3.4.0's where
  the source tree carries it);
- THE LIBRARIES A REPAIR BUNDLES (sparse_numba.libs/, auditwheel's, on the
  Linux wheels): each one maps to its folder of licence texts,
  sparse_numba/third_party_licenses/linux/<name>/ (copied from the
  manylinux image's rpm licence and COPYING / LICENSE files by
  build_wheels.yml), which holds at least one text besides SOURCE.txt, and
  the GCC runtime's named files; a library with no mapping fails; no
  sparse_numba.libs (the macOS and Windows wheels) passes.
"""

import hashlib
import os
import re
import zipfile

import pytest

import _sn1_common as common

VERSION = "0.2.0"

# the DLL -> the licence files beside it in the wheel (docs/LICENSE_AUDIT.md)
WINDOWS_DLLS = {
    "superlu/bin/libsuperlu-7.dll": ("LICENSE.superlu.txt",),
    "superlu/bin/libgcc_s_seh-1.dll": ("LICENSE.gcc-runtime.COPYING3",
                                       "LICENSE.gcc-runtime.COPYING.RUNTIME"),
    "superlu/bin/libgfortran-5.dll": ("LICENSE.gcc-runtime.COPYING3",
                                      "LICENSE.gcc-runtime.COPYING.RUNTIME"),
    "superlu/bin/libgomp-1.dll": ("LICENSE.gcc-runtime.COPYING3",
                                  "LICENSE.gcc-runtime.COPYING.RUNTIME"),
    "superlu/bin/libquadmath-0.dll": ("LICENSE.libquadmath.COPYING.LIB",),
    "superlu/bin/libwinpthread-1.dll": ("LICENSE.winpthreads.COPYING",),
    "openblas/bin/libopenblas.dll": ("LICENSE.openblas", "LICENSE.openblas-lapack"),
}

# a library in sparse_numba.libs/ (auditwheel's name: <lib>-<hash>.so...) ->
# its folder of licence texts and the files that folder must hold besides
# one text at least (docs/LICENSE_AUDIT.md, the Linux rows)
LINUX_LICENCES = "sparse_numba/third_party_licenses/linux/"
BUNDLED_LIBRARIES = (
    (re.compile(r"libsuperlu[-.]"), "superlu", ()),
    (re.compile(r"libopenblas"), "openblas", ()),
    (re.compile(r"lib[st]atlas[-.]"), "atlas", ()),
    (re.compile(r"libgfortran[-.]"), "libgfortran", ("COPYING3", "COPYING.RUNTIME")),
    (re.compile(r"libquadmath[-.]"), "libquadmath", ("COPYING.LIB",)),
)


# a SuiteSparse library file by its name: lib<name>.<ext> or lib<name>-<hash>...
SUITESPARSE_LIBRARY = re.compile(r"(lib)?(umfpack|cholmod|amd|camd|colamd|ccolamd|"
                                 r"suitesparseconfig|klu|btf|spqr|cxsparse|ldl)[.-]")


def _sums(path):
    sums = {}
    with open(path, "r", encoding="ascii") as f:
        for ln in f.read().splitlines():
            if ln.strip() and not ln.startswith("#"):
                digest, name = ln.split(None, 1)
                sums[name.lstrip("*")] = digest
    return sums


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_test_data_copies():
    sums = _sums(os.path.join(common.DATA_DIR, "SHA256SUMS"))
    assert sorted(sums) == ["n207/S.mtx", "n207/b_s.mtx", "n207/x_C5.txt",
                            "n69/G.mtx", "n69/b.mtx"]
    for name, digest in sums.items():
        assert _sha256(os.path.join(common.DATA_DIR, *name.split("/"))) == digest, name
    records = os.path.join(common.REPO_DIR, "records", "vehicles", "solve_gap_0", "SHA256SUMS")
    if os.path.exists(records):
        theirs = _sums(records)
        for name, digest in sums.items():
            assert theirs[name] == digest, name


def _wheel():
    path = os.environ.get("SPARSE_NUMBA_WHEEL")
    if not path:
        pytest.skip("SPARSE_NUMBA_WHEEL unset: no wheel to read")
    with zipfile.ZipFile(path) as z:
        return os.path.basename(path), z.namelist(), {
            n: z.read(n) for n in z.namelist()
            if n.endswith("METADATA") or n.startswith("sparse_numba/sparse_eigen/COPYING.")}


def test_wheel_contents():
    name, names, metadata = _wheel()
    assert name.startswith("sparse_numba-%s-" % VERSION), name
    bad = [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".pyo", ".nbi", ".nbc"))]
    assert bad == [], bad
    assert not [n for n in names if n.startswith("tests/")]
    meta = [v for k, v in metadata.items() if k.endswith(".dist-info/METADATA")]
    assert len(meta) == 1 and re.search(rb"^Version: 0\.2\.0\r?$", meta[0], re.M)
    for ext in ("sparse_superlu/cy_superlu_wrapper", "sparse_umfpack/cy_umfpack_wrapper",
                "sparse_eigen/cy_eigen_wrapper"):
        assert [n for n in names if n.startswith("sparse_numba/%s." % ext)], ext
    # no SuiteSparse library on any platform (UMFPACK is the user's library;
    # a repair step bundling one would put it here)
    bundled = [n for n in names if SUITESPARSE_LIBRARY.match(n.rsplit("/", 1)[-1].lower())]
    assert bundled == [], bundled
    # Eigen is compiled in: its licence text beside the Eigen extension
    third_party = os.path.join(common.REPO_DIR, "third_party", "eigen-3.4.0")
    for licence in ("COPYING.MPL2", "COPYING.README"):
        entry = "sparse_numba/sparse_eigen/" + licence
        assert entry in names, entry
        if os.path.isdir(third_party):
            assert (hashlib.sha256(metadata[entry]).hexdigest()
                    == _sha256(os.path.join(third_party, licence))), entry
    if "win_amd64" not in name:
        pytest.skip("the vendored DLL set is the Windows wheel's")
    dlls = sorted(n[len("sparse_numba/vendor/"):] for n in names
                  if n.startswith("sparse_numba/vendor/") and n.lower().endswith(".dll"))
    assert dlls == sorted(WINDOWS_DLLS), dlls
    for dll, licences in WINDOWS_DLLS.items():
        folder = "sparse_numba/vendor/" + dll.rsplit("/", 1)[0] + "/"
        for licence in licences:
            assert folder + licence in names, (dll, licence)
    assert not [n for n in names if n.lower().endswith(".dll")
                and ("suitesparse" in n.lower() or "umfpack" in n.lower())]


def test_wheel_bundled_libraries_licensed():
    path = os.environ.get("SPARSE_NUMBA_WHEEL")
    if not path:
        pytest.skip("SPARSE_NUMBA_WHEEL unset: no wheel to read")
    with zipfile.ZipFile(path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist() if not i.filename.endswith("/")}
    libs = sorted(n for n in sizes if n.startswith("sparse_numba.libs/"))
    missing = []
    for lib in libs:
        base = lib.rsplit("/", 1)[-1]
        found = [(folder, required) for pattern, folder, required in BUNDLED_LIBRARIES
                 if pattern.match(base)]
        if not found:
            missing.append((lib, "no licence mapping"))
            continue
        folder, required = found[0]
        prefix = LINUX_LICENCES + folder + "/"
        texts = sorted(n[len(prefix):] for n in sizes
                       if n.startswith(prefix) and "/" not in n[len(prefix):] and sizes[n] > 0)
        if not [t for t in texts if t != "SOURCE.txt"]:
            missing.append((lib, prefix + ": no licence text"))
        for name in required:
            if name not in texts:
                missing.append((lib, prefix + name + " absent"))
    assert missing == [], missing
