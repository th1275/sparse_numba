"""
Release-build cells (the debug switch, the version, EIGEN_LDLT's SPD
check):

- the flag list: -DNDEBUG on every extension's compile line by default;
  SPARSE_NUMBA_DEBUG set at the build leaves it out (the asserts on, never
  in a wheel); MinGW links -Wl,--no-insert-timestamp -- read from setup.py's
  own configuration print (`setup.py --version`, which builds nothing);
- the version 0.2.0: setup.py, setup.cfg, sparse_numba.__version__ and
  cache_key()'s sparse_numba= are one string; cache_key() equal to the
  release build's comparison record where the tree carries it;
- EIGEN_LDLT refuses an indefinite input: SINGULAR when a D entry <= 0
  (diag(1, -1) factors with Eigen's Success and is refused), at factorize
  and at refactorize.
"""

import configparser
import os
import platform
import re
import subprocess
import sys

import numpy as np
import pytest

import _sn1_common as common

VERSION = "0.2.0"


def _setup_print(debug):
    """setup.py's configuration print with SPARSE_NUMBA_DEBUG unset / set."""
    env = dict(os.environ)
    env.pop("SPARSE_NUMBA_DEBUG", None)
    env.pop("SPARSE_NUMBA_EXTRA_CFLAGS", None)
    if debug:
        env["SPARSE_NUMBA_DEBUG"] = "1"
    out = subprocess.run([sys.executable, "setup.py", "--version"], cwd=common.REPO_DIR,
                         env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _list_after(text, label):
    m = re.search(r"^%s: (\[.*\])$" % re.escape(label), text, re.M)
    assert m, text[-2000:]
    return eval(m.group(1), {})


def test_ndebug_flag_list_with_and_without_the_switch():
    release = _setup_print(debug=False)
    debug = _setup_print(debug=True)
    flags = _list_after(release, "Extra compile args")
    flags_debug = _list_after(debug, "Extra compile args")
    assert "-DNDEBUG" in flags, flags
    assert "-DNDEBUG" not in flags_debug, flags_debug
    assert [f for f in flags if f != "-DNDEBUG"] == flags_debug
    assert "Debug build (SPARSE_NUMBA_DEBUG, no -DNDEBUG): False" in release
    assert "Debug build (SPARSE_NUMBA_DEBUG, no -DNDEBUG): True" in debug
    if platform.system() == "Windows":
        assert "-Wl,--no-insert-timestamp" in _list_after(release, "Extra link args")
    assert release.strip().splitlines()[-1] == VERSION
    # every platform's list carries it (setup.py's text: Windows, Linux, macOS)
    setup_text = open(os.path.join(common.REPO_DIR, "setup.py"), encoding="ascii").read()
    assert len(re.findall(r"extra_compile_args (\+= NDEBUG|= \[\"-O3\", \"-fPIC\"\] \+ NDEBUG)",
                          setup_text)) == 3


def test_built_flags_carry_ndebug():
    """The in-place build is the release default: -DNDEBUG in the flags
    compiled into both shims that name them."""
    from sparse_numba import cache_key
    key = cache_key()
    for token in ("superlu", "eigen"):
        flags = re.search(r"(^|;)%s=[^;]*?,flags=([^;]*)" % token, key).group(2)
        assert "-DNDEBUG" in flags.split(), key


def test_version_020():
    import sparse_numba
    assert sparse_numba.__version__ == VERSION
    assert sparse_numba.cache_key().startswith("sparse_numba=%s;" % VERSION)
    setup_text = open(os.path.join(common.REPO_DIR, "setup.py"), encoding="ascii").read()
    assert re.search(r'^\s+version="%s",$' % re.escape(VERSION), setup_text, re.M)
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(common.REPO_DIR, "setup.cfg"), encoding="ascii")
    assert cfg["metadata"]["version"] == VERSION
    assert cfg["options"]["python_requires"] == ">=3.10"
    assert 'python_requires=">=3.10"' in setup_text
    assert cfg["options"]["install_requires"].split() == ["numpy>=1.24", "numba>=0.61"]
    for minor in (10, 11, 12, 13):
        assert "Programming Language :: Python :: 3.%d" % minor in cfg["metadata"]["classifiers"]
        assert '"Programming Language :: Python :: 3.%d"' % minor in setup_text


# the comparison record of the release build (its environment.cache_key);
# present in the development tree only
RECORD_JSON = os.path.join(common.REPO_DIR, "records", "compare_full_cb0617e.json")


def test_cache_key_equals_record():
    """cache_key() with the version read from sparse_numba.__version__ is the
    string the release build's comparison record names, field by field; the
    umfpack= field is the user's library: equal when it is found, "absent"
    when it is not."""
    import json
    import sparse_numba
    if not os.path.exists(RECORD_JSON):
        pytest.skip("the comparison record is not in this tree")
    with open(RECORD_JSON, encoding="utf-8") as f:
        expected = json.load(f)["environment"]["cache_key"]
    got = dict(part.split("=", 1) for part in sparse_numba.cache_key().split(";"))
    want = dict(part.split("=", 1) for part in expected.split(";"))
    assert list(got) == list(want)
    if sparse_numba.is_umf_available():
        assert got == want
    else:
        assert got.pop("umfpack") == "absent"
        want.pop("umfpack")
        assert got == want


def _diag(values):
    n = len(values)
    return (np.ascontiguousarray(values, dtype=np.float64), np.arange(n, dtype=np.int32),
            np.arange(n + 1, dtype=np.int32), n)


def test_ldlt_indefinite_singular():
    """diag(1, -1): a symmetric pattern, nonzero pivots, Eigen's Success,
    a D entry < 0 -> SINGULAR (= not SPD here), the handle not made."""
    import sparse_numba.api as api
    data, indices, indptr, n = _diag([1.0, -1.0])
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, api.AMD)
    assert handle == 0
    assert info == 1                                    # SINGULAR = k > 0 (Eigen: 1)
    x, info = api.solve(data, indices, indptr, n, np.ones(n), api.EIGEN_LDLT, api.AMD)
    assert info == 1
    # EIGEN_LU factors the same matrix
    h, info = api.factorize(data, indices, indptr, n, api.EIGEN_LU, api.AMD)
    assert info == api.OK
    assert api.free(h) == api.OK


def test_ldlt_refactorize_to_indefinite_singular():
    """SPD at factorize, indefinite at refactorize -> SINGULAR, and a solve
    on the handle returns it; SPD values again -> OK."""
    import sparse_numba.api as api
    data, indices, indptr, n = _diag([2.0, 3.0, 4.0])
    handle, info = api.factorize(data, indices, indptr, n, api.EIGEN_LDLT, api.AMD)
    assert info == api.OK
    try:
        assert api.refactorize(handle, np.array([2.0, -3.0, 4.0])) == 1
        out = np.zeros(n)
        assert api.solve_into(handle, np.ones(n), out) == 1
        assert api.refactorize(handle, data) == api.OK
        assert api.solve_into(handle, np.ones(n), out) == api.OK
        assert np.allclose(out, 1.0 / data, rtol=0, atol=1e-15)
    finally:
        assert api.free(handle) == api.OK
