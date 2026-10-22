"""
The build-knob cell (SPARSE_NUMBA_EXTRA_CFLAGS):
setup.py appends SPARSE_NUMBA_EXTRA_CFLAGS (shlex-split; unset = nothing)
to the one compile-flag list; with the variable unset at the build, the
flags= text in cache_key() is the shared flags with -DNDEBUG (the release
default, SPARSE_NUMBA_DEBUG unset at the build).
"""

import os
import platform
import re
import struct

import pytest

import _sn1_common as common


def _flags(key, token):
    return re.search(r"(^|;)%s=[^;]*?,flags=([^;]*)" % token, key).group(2)


def test_extra_cflags_knob_unset_keeps_flags():
    setup_text = open(os.path.join(common.REPO_DIR, "setup.py"),
                      encoding="utf-8").read()
    assert re.search(r'shlex\.split\(\s*os\.environ\.get\("SPARSE_NUMBA_EXTRA_CFLAGS", ""\)\)',
                     setup_text)
    if platform.system() == "Windows":
        base = "-O3" + (" -DMS_WIN64" if struct.calcsize("P") == 8 else "") + " -DNDEBUG"
    elif platform.system() == "Linux":
        base = "-O3 -fPIC -DNDEBUG"
    else:
        pytest.skip("the flag text is written for Windows and Linux only")
    from sparse_numba import cache_key
    key = cache_key()
    assert _flags(key, "superlu") == base, key
    assert _flags(key, "eigen") == (base + " -std=c++17 -DEIGEN_MPL2_ONLY"
                                    " -DEIGEN_DONT_PARALLELIZE"), key
