# LICENSE_AUDIT -- the components sparse_numba ships or compiles in

Opened for 0.2.0 (2026-10-21) over the vendor/ tree and the package's
LICENSE; re-checked against a built wheel's actual contents (2026-10-22).
Read on the build machine only (no network). ASCII, LF.

THE RULE: BSD- / MIT- / MPL-compatible components, and runtime libraries
under a runtime exception, may be vendored, each with its licence file
beside it in the wheel; COPYLEFT components (GPL / LGPL: UMFPACK, CHOLMOD
and their kin) are INTERFACE-ONLY, the user supplying the library. The one
exception the maintainer made is libquadmath (R5, L1). A component with no
licence file on the build machine is a row that says so.

HOW EACH ROW WAS READ (<MSYS2> = the build machine's MSYS2 installation
root). Every vendored DLL under vendor/*/bin/ is
byte-identical (sha256) to the file of the same name in
<MSYS2>/mingw64/bin/, so its package and version are read from the
msys2 pacman database (<MSYS2>/var/lib/pacman/local/<pkg>/desc and
files; "declared" below = that desc's %LICENSE% field). The licence
files were looked for in <MSYS2>/ucrt64/share/licenses/ first, then
<MSYS2>/mingw64/share/licenses/ and <MSYS2>/mingw64/share/doc/;
where both prefixes hold a file, the two sha256 are equal. A sha256
below is the file's on the build machine; the copy beside the binaries (column
"beside") is the same bytes (LF; equal to the committed blob).
"Imports" are read by `objdump -p` (DLL Name), the system DLLs left out.

THE IMPORT CHAIN AT 0.2.0 (Windows, objdump -p of the vendored DLLs and
the three pyd): cy_superlu_wrapper -> libsuperlu-7 ->
libopenblas -> libgcc_s_seh-1, libgfortran-5 (-> libquadmath-0,
libwinpthread-1), libgomp-1 (-> libwinpthread-1); cy_eigen_wrapper ->
libwinpthread-1 (the C++ runtime linked statically); cy_umfpack_wrapper ->
none (the user's UMFPACK loaded at run time). These seven DLLs are the
vendored set; libsuperlu_fortran-7 left (L3).

THE IMPORT CHAIN (Windows, the 0.1.11 tree, the in-place build; kept as
read):
cy_superlu_wrapper.pyd -> libsuperlu-7.dll -> libopenblas.dll ->
libgcc_s_seh-1.dll, libgfortran-5.dll (-> libquadmath-0.dll),
libgomp-1.dll; libwinpthread-1.dll under the gcc runtime.
libsuperlu_fortran-7.dll is shipped and imported by nothing above.
cy_umfpack_wrapper.pyd -> libumfpack.dll -> libamd, libcholmod,
libopenblas, libsuitesparseconfig (link-time at 0.1.11; at 0.2.0 loaded
at run time by name, the user's library).

--------------------------------------------------------------------------
R1  sparse_numba itself
    version:  0.2.0 (setup.py, setup.cfg, sparse_numba.__version__; the
              audit opened on 0.1.11)
    licence:  BSD 3-Clause, read from LICENSE (repo root)
              sha256
      881d04a58243b1d247ce7f39a2c03991151af480ac29106f16b665269dbc62c7
    sits:     the package (sdist and wheel; MANIFEST.in includes it; the
              0.2.0 wheel carries it as sparse_numba-0.2.0.dist-info/
              licenses/LICENSE, the same sha256)
    status:   BSD; the package's own licence.

R2  SuperLU
    version:  7.0.0 (mingw-w64-x86_64-superlu-7.0.0-1)
    files:    vendor/superlu/bin/libsuperlu-7.dll (tracked; in the
              Windows wheel); libsuperlu_fortran-7.dll REMOVED at 0.2.0
              (L3: imported by nothing) with its import libraries;
              vendor/superlu/lib/*.a (tracked; link time, not shipped),
              vendor/superlu/include/*.h (tracked; compiled-in headers)
    declared: spdx:BSD-3-Clause
    licence:  <MSYS2>/mingw64/share/licenses/superlu/LICENSE.txt
              sha256
      6b2a6d93c1e1df35200fa5a575fd0ec0ccb222d3fd41229ecf2c7687f583f1b3
    beside:   vendor/superlu/bin/LICENSE.superlu.txt
    sits:     vendored binary + compiled-in headers
    status:   BSD; vendorable under the rule; the licence file beside
              it in vendor/ and, from 0.2.0, in the wheel (setup.py's
              build_ext copies LICENSE.* with the DLLs; package_data
              carries them).

R3  OpenBLAS (with its bundled reference LAPACK)
    version:  0.3.29 (mingw-w64-x86_64-openblas-0.3.29-1)
    files:    vendor/openblas/bin/libopenblas.dll (tracked; in the
              Windows wheel); vendor/openblas/include/, lib/
              (gitignored, on the build machine; link time)
    declared: spdx:BSD-3-Clause
    licence:  <MSYS2>/mingw64/share/licenses/openblas/LICENSE
              sha256
      190b5a9c8d9723fe958ad33916bd7346d96fab3c5ea90832bb02d854f620fcff
              <MSYS2>/mingw64/share/licenses/openblas/LICENSE-lapack
              sha256
      978539245c405775bf347b1256c36df4daf8b69a64b099e94b1d53742201621e
    beside:   vendor/openblas/bin/LICENSE.openblas,
              vendor/openblas/bin/LICENSE.openblas-lapack
    sits:     vendored binary
    status:   BSD (both files); vendorable under the rule.

R4  GCC runtime: libgcc_s_seh-1.dll, libgomp-1.dll
    version:  GCC 14.2.0 (mingw-w64-x86_64-gcc-libs-14.2.0-2)
    files:    vendor/superlu/bin/ (tracked; in the Windows wheel)
    declared: spdx:GPL-3.0-or-later WITH GCC-exception-3.1 AND
              LGPL-2.1-or-later (the package as a whole)
    licence:  <MSYS2>/ucrt64/share/licenses/gcc-libs/README (= the
              mingw64 one) sha256

      33400d067fa3e2bd670a0868dcdb60c9486e8f559b675cf9ff1ff55d4e0d06ab
              -- it names libgcc, libstdc++, libgomp and libatomic as
              GPL-3.0-or-later with GCC-exception-3.1;
              .../gcc-libs/COPYING.RUNTIME sha256

      9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74
              .../gcc-libs/COPYING3 sha256

      8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903
    beside:   vendor/superlu/bin/LICENSE.gcc-libs.README,
              LICENSE.gcc-runtime.COPYING.RUNTIME,
              LICENSE.gcc-runtime.COPYING3
    sits:     vendored binary (imported by libopenblas.dll)
    status:   a runtime library under a runtime exception (the README
              read); the rule's vendorable class.

R5  GCC runtime: libquadmath-0.dll
    version:  GCC 14.2.0 (mingw-w64-x86_64-gcc-libs-14.2.0-2)
    files:    vendor/superlu/bin/libquadmath-0.dll (tracked; in the
              Windows wheel)
    declared: as R4 (the package as a whole)
    licence:  the same README: "The libquadmath library is covered by
              LGPL-2.1-or-later"; .../gcc-libs/COPYING.LIB (LGPL 2.1)
              sha256
      a9bdde5616ecdd1e980b44f360600ee8783b1f99b8cc83a2beb163a0a390e861
    beside:   vendor/superlu/bin/LICENSE.libquadmath.COPYING.LIB
    sits:     vendored binary (imported by libgfortran-5.dll, which
              libopenblas.dll imports)
    status:   LGPL-2.1-or-later, no runtime exception named in the
              README; the rule places LGPL as interface-only; it is
              vendored by the maintainer's exception L1 below
              (unmodified, replaceable, its LGPL text beside it).

R6  GCC runtime: libgfortran-5.dll
    version:  GCC 14.2.0 (mingw-w64-x86_64-gcc-libgfortran-14.2.0-2)
    files:    vendor/superlu/bin/libgfortran-5.dll (tracked; in the
              Windows wheel)
    declared: spdx:GPL-3.0-or-later (no exception in the field)
    licence:  <MSYS2>/mingw64/share/licenses/gcc-libgfortran/
              COPYING.RUNTIME sha256

      9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74
              (the only file the package installs; equal to R4's)
    beside:   vendor/superlu/bin/LICENSE.gcc-runtime.COPYING.RUNTIME
              (shared with R4), LICENSE.gcc-runtime.COPYING3
    sits:     vendored binary (imported by libopenblas.dll)
    status:   the declared field says GPL-3.0-or-later; the file
              installed is the GCC runtime library exception. The two
              differ in what they name; read with R4 (L2 below).

R7  mingw-w64 winpthreads: libwinpthread-1.dll
    version:  12.0.0.r509.g079e6092b
              (mingw-w64-x86_64-libwinpthread-git-12.0.0.r509.g079e6092b-1)
    files:    vendor/superlu/bin/libwinpthread-1.dll (tracked; in the
              Windows wheel)
    declared: spdx:MIT AND BSD-3-Clause-Clear
    licence:  <MSYS2>/ucrt64/share/licenses/libwinpthread/COPYING
              (= the mingw64 one) sha256

      63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177
    beside:   vendor/superlu/bin/LICENSE.winpthreads.COPYING
    sits:     vendored binary (imported by the gcc runtime)
    status:   MIT / BSD; vendorable under the rule.

R8  SuiteSparse (UMFPACK, CHOLMOD, AMD, COLAMD, CAMD, CCOLAMD,
    SuiteSparse_config linked at 0.1.11; the rest of the set installed)
    version:  7.8.3 (mingw-w64-x86_64-suitesparse-7.8.3-1)
    files:    vendor/suitesparse/bin/*.dll (21 DLLs of the package plus
              10 runtime DLLs, R9), include/, lib/ -- all gitignored, on
              the build machine only; NOT in any wheel (setup.py run()
              copies the OpenBLAS and SuperLU DLLs only; MANIFEST.in
              excludes vendor/suitesparse/bin/*.dll)
    declared: spdx:LGPL-3.0-or-later AND BSD-3-Clause AND
              LGPL-2.1-or-later AND GPL-2.0-or-later AND
              LGPL-2.0-or-later AND Apache-2.0 AND GPL-3.0-or-later
    licence:  NO LICENCE FILE INSTALLED (no share/licenses/suitesparse
              in ucrt64 or mingw64, no share/doc/suitesparse; the
              package's file list names none)
    sits:     link-time dependency of cy_umfpack_wrapper at 0.1.11;
              INTERFACE-ONLY at 0.2.0 (loaded at run time by name, the
              user supplying SuiteSparse; the extension is compiled
              against the headers only)
    status:   copyleft (GPL / LGPL in the declared field; which
              library carries which is not installed with it); the
              rule's interface-only class; not shipped.

R9  SuiteSparse's dependencies present in vendor/suitesparse/bin/
    (gitignored, on the build machine only, not in any wheel):
    libgmp-10.dll, libgmpxx-4.dll -- GMP 6.3.0
      (mingw-w64-x86_64-gmp-6.3.0-2), declared LGPL3 / GPL;
      NO LICENCE FILE INSTALLED (no share/licenses/gmp, no
      share/doc/gmp in mingw64)
    libmpfr-6.dll -- MPFR 4.2.1 (mingw-w64-x86_64-mpfr-4.2.1-2),
      declared spdx:LGPL-3.0-or-later;
      <MSYS2>/mingw64/share/doc/mpfr/COPYING sha256
      3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986,
      COPYING.LESSER sha256
      e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118
    libstdc++-6.dll, libatomic-1.dll -- GCC 14.2.0 gcc-libs, as R4
    (and copies of R4-R7's DLLs, byte-identical)
    sits:     used by SPEX / GraphBLAS of the SuiteSparse set; not
              linked by sparse_numba
    status:   leave with SuiteSparse; no copy beside them
              (the folder is gitignored).

R10 Eigen
    version:  3.4.0 (eigen-3.4.0.tar.gz, sha256
      8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72,
              the release tarball, its sha256 checked before
              extraction; third_party/eigen-3.4.0/SHA256SUMS lists every
              copied file)
    files:    third_party/eigen-3.4.0/Eigen/ (337 headers, tracked;
              compiled into cy_eigen_wrapper at build, no binary of its
              own; unsupported/ not copied), README.md
    licence:  MPL-2.0, read from third_party/eigen-3.4.0/COPYING.MPL2
              sha256
      fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85
              COPYING.README (Eigen "is primarily MPL2 licensed"; some
              files carry third-party code under BSD or LGPL terms;
              EIGEN_MPL2_ONLY makes an LGPL header a compile error) sha256
      c83230b770f17ef1386ea1fd3681271dd98aa93646bdbfb5bff3a1b7050fff9d
              the tarball's other licence files, copied beside it:
              COPYING.BSD sha256
      51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb
              COPYING.LGPL sha256
      dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551
              COPYING.GPL sha256
      8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903
              COPYING.MINPACK sha256
      c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0
              COPYING.APACHE sha256
      03379001a7b12a2ec997a25554247d985270b353c10d5bafee9ac8d6519820b7
    beside:   third_party/eigen-3.4.0/COPYING.* (the files above)
    sits:     compiled-in headers (the Eigen shim, sparse_numba/
              sparse_eigen/, built with -DEIGEN_MPL2_ONLY and
              -DEIGEN_DONT_PARALLELIZE; the build passes with the
              MPL2-only guard on: no non-MPL2 header is included).
              Headers it includes: Eigen/SparseCore, OrderingMethods,
              SparseLU, SparseCholesky and what they include.
    in wheel: sparse_numba/sparse_eigen/COPYING.MPL2 and COPYING.README
              beside the Eigen extension (setup.py's package_data, copied
              from third_party/eigen-3.4.0 at the build; the same sha256
              as above), from 0.2.0 (L4)
    status:   MPL-2.0 (BSD-licensed parts possible under the guard);
              the rule's MPL-compatible class, compiled in; its licence
              text beside the extension in the wheel; the README's
              licence table names MPL-2.0 and where the source form is
              (eigen.tuxfamily.org; third_party/eigen-3.4.0 in the
              repository).

--------------------------------------------------------------------------
--------------------------------------------------------------------------
THE RE-CHECK ON A WHEEL'S OWN LISTING (2026-10-22): a locally built
release-candidate wheel sparse_numba-0.2.0-cp311-cp311-win_amd64.whl
(sha256 c366d619c38362357044a0a6b883c3a42502d279348c388c8bcda95bd1311ce6,
48 entries; tests/test_sn6_wheel.py reads the same facts from any wheel).
Each vendored file, its sha256 in the wheel, and the licence beside it:

| wheel path (sparse_numba/vendor/...) | sha256 (first 16) | row | licence file(s) beside, sha256 (first 16) |
|---|---|---|---|
| superlu/bin/libsuperlu-7.dll | 701547fe2d6ec233 | R2 | LICENSE.superlu.txt 6b2a6d93c1e1df35 |
| openblas/bin/libopenblas.dll | 62bef63a0c88a80e | R3 | LICENSE.openblas 190b5a9c8d9723fe, LICENSE.openblas-lapack 978539245c405775 |
| superlu/bin/libgcc_s_seh-1.dll | 00194fb242d0d7a7 | R4 | LICENSE.gcc-libs.README 33400d067fa3e2bd, LICENSE.gcc-runtime.COPYING.RUNTIME 9d6b43ce4d8de0c8, LICENSE.gcc-runtime.COPYING3 8ceb4b9ee5adedde |
| superlu/bin/libgomp-1.dll | 8f7a0296efe00423 | R4 | as above |
| superlu/bin/libquadmath-0.dll | 7773a6fee763a5c1 | R5 | LICENSE.libquadmath.COPYING.LIB a9bdde5616ecdd1e |
| superlu/bin/libgfortran-5.dll | 234bf85b84005639 | R6 | LICENSE.gcc-runtime.COPYING.RUNTIME, COPYING3 (as R4) |
| superlu/bin/libwinpthread-1.dll | f00016d8c4c727ee | R7 | LICENSE.winpthreads.COPYING 63263614cdd29f2f |

Every licence file in the wheel is byte-equal (sha256) to the one under
vendor/*/bin/ and to the sha256 of its row above. No SuiteSparse file,
no libsuperlu_fortran-7, no __pycache__ / *.nbi / *.nbc, no tests. The
three extensions (cy_superlu_wrapper, cy_umfpack_wrapper,
cy_eigen_wrapper) are sparse_numba's own (R1); the Eigen one has Eigen
compiled in (R10). The 0.2.0 release wheels add Eigen's COPYING.MPL2 and
COPYING.README beside the Eigen extension (L4): a wheel rebuilt with them
(sha256 7d3ea761c17794e65f8ddf73c63edd3ef2a620a97b3343ebcdeadf3f6f97db05,
50 entries) differs from the one above by those two entries (sha256
fab3dd6bdab226f1... and c83230b770f17ef1..., equal to third_party's), its
text files and RECORD; its three extensions and every vendored file are the
same bytes. The release's own wheel listing is re-read before the tag.

LINUX (2026-10-22, 0.2.0): read on CI's cp311 Linux wheel of PR #1's run
at 4c9d9ad (sparse_numba-0.2.0-cp311-cp311-manylinux_2_27_x86_64.
manylinux_2_28_x86_64.whl, sha256 45523b486266c969...). The wheels are
built in the manylinux_2_28 image (AlmaLinux 8) against its packages
(build_wheels.yml's CIBW_BEFORE_ALL_LINUX: openblas-devel and
suitesparse-devel from powertools, SuperLU-devel from EPEL 8), and
auditwheel's repair copies every non-policy library the extensions load
into sparse_numba.libs/ (renamed <lib>-<hash>). That wheel carried five,
and no licence text for any of them (only dist-info/licenses/LICENSE and
Eigen's COPYING files); the version is read from the file name:

| sparse_numba.libs/ file | size | row | library | licence | texts in the wheel (sparse_numba/third_party_licenses/linux/) |
|---|---|---|---|---|---|
| libsuperlu-17b63ae5.so.5.1 | 6.3 MB | RL1 | SuperLU 5.x (EPEL 8) | BSD 3-Clause | superlu/ |
| libopenblas-r0-11edc3fa.3.15.so | 37.4 MB | RL2 | OpenBLAS 0.3.15 (AlmaLinux 8) | BSD 3-Clause | openblas/ |
| libsatlas-6c1f674f.so.3.10 | 16.9 MB | RL3 | ATLAS 3.10 (EPEL's SuperLU links it) | BSD-style | atlas/ |
| libgfortran-e2a3c254.so.5.0.0 | 8.4 MB | RL4 | GCC runtime, libgfortran (AlmaLinux 8) | GPL-3.0+ with the GCC Runtime Library Exception (as R6 / L2) | libgfortran/ (COPYING3, COPYING.RUNTIME) |
| libquadmath-80a2a415.so.0.0.0 | 2.1 MB | RL5 | GCC runtime, libquadmath (AlmaLinux 8) | LGPL-2.1+ (L1) | libquadmath/ (COPYING.LIB) |

From 0.2.0 the texts are copied at build time from the image itself: for
each library, the rpm package owning it and the installed packages built
from the same source package (GCC's licence files are libgcc's, from the
same gcc source package as libgfortran and libquadmath; OpenBLAS's
LICENSE is the base openblas package's), their rpm %license files and
their COPYING / LICENSE documentation files, into the folder named
above, with SOURCE.txt naming the library, its package and the packages
the texts came from (setup.py's Linux package_data puts the folder in
the wheel). tests/test_sn6_wheel.py, run on every Linux and macOS test
job against its wheel, maps each sparse_numba.libs file to its folder
and fails a library with no mapping, a folder with no text besides
SOURCE.txt, or a GCC runtime folder without the files named above. The
texts' names and sha256 are read from the first CI wheel that carries
them (not from this machine: no manylinux image here). The rule holds:
RL1 - RL4 are BSD-compatible or runtimes under the exception; RL5 is
read under L1 (L5). None is a SuiteSparse library (CI fails a wheel that
carries one).

MACOS: delocate is skipped; the wheels load Homebrew's superlu / openblas
at run time and bundle nothing (no sparse_numba.libs; the cell above
passes on an absent folder). Not covered here either: the
Cython-generated C and numpy's headers used at build.

THE DECISIONS (the maintainer's)
L1  libquadmath-0.dll (R5) is LGPL-2.1-or-later and is imported by
    libgfortran-5.dll (OpenBLAS's gfortran LAPACK). KEPT as a runtime the
    user may replace: unmodified, dynamically loaded, its LGPL text
    beside it (LGPL 2.1 permits shipping a replaceable DLL) -- the one
    exception to the rule's text. The alternative, an OpenBLAS build
    without the gfortran LAPACK, would rebuild the vendored set.
L2  libgfortran-5.dll (R6): the pacman field says GPL-3.0-or-later, the
    only file installed is the GCC runtime library exception. Read with
    R4 as a runtime under the exception (the file read).
L3  libsuperlu_fortran-7.dll was shipped and imported by nothing in the
    chain above: REMOVED at 0.2.0 with its import libraries (no licence
    change; R2's licence).
L4  Eigen's licence text in the wheel: the extension compiles Eigen in;
    COPYING.MPL2 and COPYING.README ship beside it from 0.2.0.
L5  The Linux wheels' libquadmath (RL5) is read under L1, the same
    exception: AlmaLinux 8's binary, its code unmodified (auditwheel's
    repair changes only its file name and its ELF soname / needed / rpath
    entries), dynamically loaded and replaceable, its LGPL text in the
    wheel (third_party_licenses/linux/libquadmath/). The Linux licence
    texts ship from 0.2.0; 0.1.11's Linux wheels carried none.
