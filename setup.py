import os
import shlex
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel
import numpy as np
import distutils.ccompiler
from distutils.errors import CompileError

# Detect if we're in a CI environment
CI_BUILD = os.environ.get('CI', '') == 'true' or os.environ.get('GITHUB_ACTIONS', '') == 'true'

# THE DEBUG SWITCH: every extension is compiled with
# -DNDEBUG on every platform (the release default); SPARSE_NUMBA_DEBUG set
# to a non-empty value other than 0 / false / no at the build leaves it out,
# so the asserts (Eigen's eigen_assert among them) stay on for development.
# Never set for a wheel.
DEBUG_BUILD = os.environ.get('SPARSE_NUMBA_DEBUG', '').strip().lower() not in ('', '0', 'false', 'no')
NDEBUG = [] if DEBUG_BUILD else ["-DNDEBUG"]

# Determine the platform
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_LINUX = PLATFORM == 'Linux'
IS_MACOS = PLATFORM == 'Darwin'

# Force MinGW compiler on Windows
if IS_WINDOWS:
    # Save the original compiler selection function
    original_compiler = distutils.ccompiler.get_default_compiler


    # Override with a function that always returns 'mingw32'
    def force_mingw():
        return 'mingw32'


    # Replace the function
    distutils.ccompiler.get_default_compiler = force_mingw

# Define platform-specific settings
if IS_WINDOWS:
    include_dirs = [
        np.get_include(),
        "sparse_numba/sparse_umfpack",
        "sparse_numba/sparse_superlu",
        "vendor/suitesparse/include",
        "vendor/superlu/include",
        "vendor/openblas/include"
    ]
    library_dirs = [
        "vendor/suitesparse/lib",
        "vendor/superlu/lib",
        "vendor/openblas/lib",
        "vendor/openblas/bin",  # MinGW can link against DLLs directly
    ]
    # Libraries needed for Windows
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas"
    ]
    superlu_libraries = ["superlu", "openblas"]
    # Use MS_WIN64 to help pyconfig.h detect 64-bit correctly with MinGW.
    # Don't pass -DSIZEOF_VOID_P directly - it conflicts with pyconfig.h's own definition.
    import struct
    extra_compile_args = ["-O3"]
    if struct.calcsize("P") == 8:
        extra_compile_args.append("-DMS_WIN64")
    extra_compile_args += NDEBUG
    # MinGW: no link timestamp in the PE header, so a rebuild of unchanged
    # source gives the same extension bytes (and the same cache_key())
    extra_link_args = ["-Wl,--no-insert-timestamp"]
elif IS_LINUX:
    # On Linux, we'll use system libraries if available
    # Try to locate SuperLU headers dynamically
    superlu_include_dirs = [
        "/usr/include/superlu",
        "/usr/local/include/superlu",
        "/usr/include/suitesparse/superlu",
        "/usr/include/SuperLU",
        "/tmp/superlu_include",  # Our custom location from the GitHub workflow
        "/usr/include/x86_64-linux-gnu/superlu",  # Debian/Ubuntu specific location
    ]

    # Check if each directory exists
    valid_superlu_dirs = []
    for dir_path in superlu_include_dirs:
        if os.path.exists(dir_path):
            valid_superlu_dirs.append(dir_path)
            print(f"Found valid SuperLU include dir: {dir_path}")

    # Base include directories
    include_dirs = [
        np.get_include(),
        "sparse_numba/sparse_umfpack",
        "sparse_numba/sparse_superlu",
        "/usr/include/suitesparse",  # Standard location on most Linux distros
        "/usr/local/include/suitesparse",  # Possible alternate location
    ]

    # Add valid SuperLU directories
    include_dirs.extend(valid_superlu_dirs)

    # If no valid SuperLU directories found, try to find headers manually
    if not valid_superlu_dirs and not CI_BUILD:  # Skip search in CI to avoid delays
        import subprocess

        try:
            # Find SuperLU headers
            find_cmd = "find /usr -name slu_ddefs.h 2>/dev/null || echo ''"
            print(f"Running command: {find_cmd}")
            superlu_path = subprocess.check_output(find_cmd, shell=True).decode().strip()
            if superlu_path:
                superlu_dir = os.path.dirname(superlu_path)
                print(f"Found SuperLU headers at: {superlu_dir}")
                include_dirs.append(superlu_dir)
        except Exception as e:
            print(f"Error finding SuperLU headers: {e}")

    # Define library directories
    library_dirs = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/usr/lib/x86_64-linux-gnu"
    ]

    # Libraries needed for Linux
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas"
    ]
    superlu_libraries = ["superlu", "openblas"]

    # Don't add "blas" library in CI builds as it sometimes causes issues
    if not CI_BUILD:
        umfpack_libraries.append("blas")
        superlu_libraries.append("blas")

    extra_compile_args = ["-O3", "-fPIC"] + NDEBUG
    extra_link_args = []
elif IS_MACOS:
    # On macOS, check both system locations and Homebrew/MacPorts locations
    homebrew_prefix = os.popen("brew --prefix 2>/dev/null || echo ''").read().strip()
    if not homebrew_prefix:
        homebrew_prefix = "/usr/local"  # Default Homebrew location

    # Find Homebrew OpenBLAS location
    openblas_prefix = os.environ.get('OPENBLAS_PREFIX')
    if not openblas_prefix:
        openblas_prefix = os.popen("brew --prefix openblas 2>/dev/null || echo ''").read().strip()
    if not openblas_prefix:
        openblas_prefix = "/usr/local/opt/openblas"  # Default location

    print(f"Using OpenBLAS prefix: {openblas_prefix}")

    # Find Homebrew SuperLU location
    superlu_prefix = os.popen("brew --prefix superlu 2>/dev/null || echo ''").read().strip()
    if not superlu_prefix:
        superlu_prefix = "/usr/local/opt/superlu"  # Default location

    # Find Homebrew SuiteSparse location
    suitesparse_prefix = os.popen("brew --prefix suite-sparse 2>/dev/null || echo ''").read().strip()
    if not suitesparse_prefix:
        suitesparse_prefix = "/usr/local/opt/suite-sparse"  # Default location

    # Check for home directory symlinks (from GitHub workflow)
    home_libs = os.environ.get('HOME_LIBS')
    home_openblas_lib = None
    if home_libs:
        home_openblas_lib = os.path.join(home_libs, "openblas", "lib")
        if os.path.exists(home_openblas_lib):
            print(f"Found home directory OpenBLAS lib: {home_openblas_lib}")

    include_dirs = [
        np.get_include(),
        "sparse_numba/sparse_umfpack",
        "sparse_numba/sparse_superlu",
        f"{suitesparse_prefix}/include/suitesparse",
        f"{superlu_prefix}/include/superlu",
        f"{superlu_prefix}/include",  # Newer Homebrew might put headers here
        "/usr/local/include/suitesparse",
        "/opt/local/include/suitesparse",  # MacPorts
        "/usr/local/include/superlu",
        "/opt/local/include/superlu",
        f"{openblas_prefix}/include"
    ]

    library_dirs = [
        f"{homebrew_prefix}/lib",
        f"{openblas_prefix}/lib",
        f"{superlu_prefix}/lib",
        f"{suitesparse_prefix}/lib",
        "/usr/local/lib",
        "/usr/local/opt/openblas/lib",
        "/opt/local/lib"  # MacPorts
    ]

    # Add home directory symlinks if available
    if home_openblas_lib:
        library_dirs.insert(0, home_openblas_lib)  # Add with high priority

    # Libraries needed for macOS
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas"
    ]
    superlu_libraries = ["superlu", "openblas"]

    # For macOS, ensure we're building for the right architecture
    extra_compile_args = ["-O3", "-fPIC"] + NDEBUG
    extra_link_args = [f"-L{openblas_prefix}/lib", "-lopenblas"]

    if home_openblas_lib:
        extra_link_args.insert(0, f"-L{home_openblas_lib}")  # Add with high priority

    # Skip architecture flags in CI environments
    if not CI_BUILD:
        if platform.machine() == 'arm64':
            extra_compile_args.append("-arch arm64")
            extra_link_args.append("-arch arm64")
        else:
            extra_compile_args.append("-arch x86_64")
            extra_link_args.append("-arch x86_64")
else:
    raise RuntimeError(f"Unsupported platform: {PLATFORM}")

# The compile flags: one list for every extension. Its
# text is compiled into each shim's build facts as SN_BUILD_FLAGS
# (sparse_numba_registry.h) and so named by cache_key(). SPARSE_NUMBA_EXTRA_
# CFLAGS (shlex-split; unset = nothing) is appended for a comparison build
# (e.g. -march=native); the wheel never sets it
compile_flags = list(extra_compile_args) + shlex.split(
    os.environ.get("SPARSE_NUMBA_EXTRA_CFLAGS", ""))


def with_build_flags(flags):
    """flags plus -DSN_BUILD_FLAGS="<their text>" (no shell: the compiler
    gets the argument as one string)."""
    text = " ".join(flags).replace("\\", "\\\\").replace('"', '\\"')
    return list(flags) + ['-DSN_BUILD_FLAGS="%s"' % text]


# Print configuration for debugging
print("\n=== Build Configuration ===")
print(f"Platform: {PLATFORM}")
print(f"Include directories: {include_dirs}")
print(f"Library directories: {library_dirs}")
print(f"UMFPACK libraries: {umfpack_libraries}")
print(f"SuperLU libraries: {superlu_libraries}")
print(f"Extra compile args: {compile_flags}")
print(f"Extra link args: {extra_link_args}")
print(f"Debug build (SPARSE_NUMBA_DEBUG, no -DNDEBUG): {DEBUG_BUILD}")
print(f"CI Build: {CI_BUILD}")
print("===========================\n")

# Define the extension modules
extensions = []

# UMFPACK extension (requires SuiteSparse headers/libs to be installed)
SKIP_UMFPACK = os.environ.get('SPARSE_NUMBA_SKIP_UMFPACK', '').lower() in ('1', 'true', 'yes')
if not SKIP_UMFPACK:
    extensions.append(
        Extension(
            "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
            sources=[
                "sparse_numba/sparse_umfpack/cy_umfpack_wrapper.pyx",
                "sparse_numba/sparse_umfpack/umfpack_wrapper.c"
            ],
            include_dirs=include_dirs,
            # interface-only: no SuiteSparse at link
            # time; the user's library is loaded at runtime by name (dlopen
            # lives in libdl on glibc before 2.34)
            libraries=["dl"] if IS_LINUX else [],
            library_dirs=[],
            extra_compile_args=with_build_flags(compile_flags),
            extra_link_args=extra_link_args,
        )
    )
else:
    print("Skipping UMFPACK extension (SPARSE_NUMBA_SKIP_UMFPACK is set)")

# SuperLU extension (always built - vendor libraries bundled)
extensions.append(
    Extension(
        "sparse_numba.sparse_superlu.cy_superlu_wrapper",
        sources=[
            "sparse_numba/sparse_superlu/cy_superlu_wrapper.pyx",
            "sparse_numba/sparse_superlu/superlu_wrapper.c"
        ],
        include_dirs=include_dirs,
        libraries=superlu_libraries,
        library_dirs=library_dirs,
        extra_compile_args=with_build_flags(compile_flags),
        extra_link_args=extra_link_args,
    )
)

# Eigen extension (always built): the Eigen 3.4.0
# headers compiled in from third_party/eigen-3.4.0 (pinned by its
# SHA256SUMS), C++17, the shared compile flags plus -DEIGEN_MPL2_ONLY (a
# non-MPL2 Eigen header is a build error) and
# -DEIGEN_DONT_PARALLELIZE; no library to link (no BLAS)
eigen_compile_flags = compile_flags + ["-std=c++17", "-DEIGEN_MPL2_ONLY",
                                       "-DEIGEN_DONT_PARALLELIZE"]
if IS_WINDOWS:
    # MinGW: the C++ runtime linked in, so the pyd imports no libstdc++ /
    # libgcc DLL
    eigen_link_args = extra_link_args + ["-static-libstdc++", "-static-libgcc"]
else:
    # Linux / macOS (written, not built on the box): the platform's C++
    # runtime, the architecture flags only
    eigen_link_args = [arg for arg in extra_link_args if arg.startswith("-arch")]
extensions.append(
    Extension(
        "sparse_numba.sparse_eigen.cy_eigen_wrapper",
        sources=[
            "sparse_numba/sparse_eigen/cy_eigen_wrapper.pyx",
            "sparse_numba/sparse_eigen/eigen_wrapper.cpp"
        ],
        include_dirs=include_dirs + ["sparse_numba/sparse_eigen", "third_party/eigen-3.4.0"],
        language="c++",
        extra_compile_args=with_build_flags(eigen_compile_flags),
        extra_link_args=eigen_link_args,
    )
)


# Customize the build process
# Eigen's licence files, beside the Eigen extension in the wheel
EIGEN_LICENCE_FILES = ("COPYING.MPL2", "COPYING.README")


class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        # Add NumPy include directory
        self.include_dirs.append(np.get_include())

        # Force MinGW compiler on Windows
        if IS_WINDOWS:
            self.compiler = 'mingw32'

    def build_extensions(self):
        # Print debugging info for include paths and libraries
        print("\n--- Build Extension Configuration ---")
        print(f"self.include_dirs: {self.include_dirs}")
        print(f"library_dirs: {library_dirs}")
        print(f"UMFPACK libraries: {umfpack_libraries}")
        print(f"SuperLU libraries: {superlu_libraries}")
        print(f"Extra compile args: {extra_compile_args}")
        print(f"Extra link args: {extra_link_args}")

        # List all SuperLU headers in include paths
        for include_dir in self.include_dirs:
            if os.path.exists(include_dir):
                slu_headers = []
                try:
                    for f in os.listdir(include_dir):
                        if f.startswith('slu_') and f.endswith('.h'):
                            slu_headers.append(f)
                    if slu_headers:
                        print(f"SuperLU headers in {include_dir}: {', '.join(slu_headers)}")
                except Exception as e:
                    print(f"Error checking {include_dir}: {e}")

        print("--------------------------------\n")

        # Ensure MinGW is being used on Windows
        if platform.system() == 'Windows':
            if self.compiler.compiler_type != 'mingw32':
                raise CompileError(
                    "This package must be compiled with MinGW (GCC) on Windows. "
                    "Your DLLs were compiled with GCC, and mixing compilers can cause memory issues."
                )

        build_ext.build_extensions(self)

    def run(self):
        print("\n" + "=" * 80)
        print("BUILDING EXTENSIONS")
        print(f"Extensions to build: {[ext.name for ext in self.extensions]}")
        print(f"Build directory: {self.build_lib}")
        print(f"Include directories: {self.include_dirs}")
        print("=" * 80 + "\n")
        build_ext.run(self)

        print("\n" + "=" * 80)
        print("EXTENSION BUILD COMPLETED")
        print("Checking for built extensions:")
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(ext_path):
                print(f"[OK] {ext.name} successfully built at {ext_path}")
            else:
                print(f"[FAIL] {ext.name} FAILED TO BUILD (expected at {ext_path})")
        print("=" * 80 + "\n")

        # Eigen is compiled into its extension: its licence text (MPL-2.0)
        # goes beside the extension in the wheel, copied byte for byte from
        # third_party/eigen-3.4.0 (not into the source tree at --inplace)
        if not self.inplace:
            eigen_dir = os.path.dirname(self.get_ext_fullpath("sparse_numba.sparse_eigen.cy_eigen_wrapper"))
            os.makedirs(eigen_dir, exist_ok=True)
            import shutil
            for licence in EIGEN_LICENCE_FILES:
                shutil.copyfile(os.path.join("third_party", "eigen-3.4.0", licence),
                                os.path.join(eigen_dir, licence))
                print(f"Copied the Eigen licence {licence} to {eigen_dir}")

        # Get extension paths for all platforms
        ext_path_umfpack = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
        ext_path_superlu = self.get_ext_fullpath("sparse_numba.sparse_superlu.cy_superlu_wrapper")
        package_dir = os.path.dirname(os.path.dirname(ext_path_umfpack))

        # Print for debugging
        print(f"Extension umfpack path: {ext_path_umfpack}")
        print(f"Extension superlu path: {ext_path_superlu}")
        print(f"Package directory: {package_dir}")

        # Handle Windows-specific tasks
        if platform.system() == 'Windows':
            # Create the vendor directories in the build
            suitesparse_target_dir = os.path.join(package_dir, "vendor", "suitesparse", "bin")
            openblas_target_dir = os.path.join(package_dir, "vendor", "openblas", "bin")
            superlu_target_dir = os.path.join(package_dir, "vendor", "superlu", "bin")
            os.makedirs(suitesparse_target_dir, exist_ok=True)
            os.makedirs(superlu_target_dir, exist_ok=True)
            os.makedirs(openblas_target_dir, exist_ok=True)

            # Source directories
            suitesparse_bin_dir = os.path.join("vendor", "suitesparse", "bin")
            superlu_bin_dir = os.path.join("vendor", "superlu", "bin")
            openblas_bin_dir = os.path.join("vendor", "openblas", "bin")

            # Verify source directories exist
            print(f"SuiteSparse bin dir exists: {os.path.exists(suitesparse_bin_dir)}")
            print(f"SuperLU bin dir exists: {os.path.exists(superlu_bin_dir)}")
            print(f"OpenBLAS bin dir exists: {os.path.exists(openblas_bin_dir)}")

            import shutil

            # Copy the OpenBLAS DLL and its licence files (the licence
            # beside each vendored DLL)
            if os.path.exists(openblas_bin_dir):
                for dll_file in os.listdir(openblas_bin_dir):
                    if dll_file.endswith('.dll') or dll_file.startswith('LICENSE.'):
                        dest_path = os.path.join(openblas_target_dir, dll_file)
                        shutil.copy(
                            os.path.join(openblas_bin_dir, dll_file),
                            dest_path
                        )
                        print(f"Copied OpenBLAS DLL: {dll_file} to {dest_path}")

            # Copy the SuperLU link chain's DLLs and their licence files
            if os.path.exists(superlu_bin_dir):
                for dll_file in os.listdir(superlu_bin_dir):
                    if dll_file.endswith('.dll') or dll_file.startswith('LICENSE.'):
                        dest_path = os.path.join(superlu_target_dir, dll_file)
                        shutil.copy(
                            os.path.join(superlu_bin_dir, dll_file),
                            dest_path
                        )
                        print(f"Copied SuperLU DLL: {dll_file} to {dest_path}")

        # Handle Linux and macOS - ensure extension modules have the right permissions
        elif platform.system() in ('Linux', 'Darwin'):
            # The extensions are already in the right place, we just need to ensure they have
            # the right permissions
            try:
                import stat

                # Set executable permissions for SuperLU extension
                if os.path.exists(ext_path_superlu):
                    print(f"Setting executable permissions on SuperLU extension: {ext_path_superlu}")
                    os.chmod(ext_path_superlu, os.stat(ext_path_superlu).st_mode | stat.S_IEXEC)
                else:
                    print(f"WARNING: SuperLU extension not found at {ext_path_superlu}")

                # Set executable permissions for UMFPACK extension
                if os.path.exists(ext_path_umfpack):
                    print(f"Setting executable permissions on UMFPACK extension: {ext_path_umfpack}")
                    os.chmod(ext_path_umfpack, os.stat(ext_path_umfpack).st_mode | stat.S_IEXEC)
                else:
                    print(f"WARNING: UMFPACK extension not found at {ext_path_umfpack}")

                # Verify the extensions exist in their directories
                superlu_dir = os.path.dirname(ext_path_superlu)
                umfpack_dir = os.path.dirname(ext_path_umfpack)

                print(f"Contents of SuperLU directory: {os.listdir(superlu_dir)}")
                print(f"Contents of UMFPACK directory: {os.listdir(umfpack_dir)}")

            except Exception as e:
                print(f"Error setting permissions on extensions: {e}")


# Define platform-specific package data
package_data = {
    'sparse_numba': []
}

if IS_WINDOWS:
    package_data['sparse_numba'] = [
        'vendor/superlu/bin/*.dll',
        'vendor/superlu/bin/LICENSE.*',
        'vendor/openblas/bin/*.dll',
        'vendor/openblas/bin/LICENSE.*',
        'sparse_superlu/*.pyd',
        'sparse_umfpack/*.pyd',
        'sparse_eigen/*.pyd',
        'sparse_superlu/test/*.py',
        'sparse_umfpack/test/*.py',
        'conversion/*.py'
    ]
elif IS_LINUX:
    package_data['sparse_numba'] = [
        'sparse_superlu/*.so',
        'sparse_superlu/cy_superlu_wrapper*.so',  # More specific pattern
        'sparse_umfpack/*.so',
        'sparse_umfpack/cy_umfpack_wrapper*.so',  # More specific pattern
        'sparse_eigen/cy_eigen_wrapper*.so',
        'sparse_superlu/test/*.py',
        'sparse_umfpack/test/*.py',
        'conversion/*.py'
    ]
elif IS_MACOS:
    package_data['sparse_numba'] = [
        'sparse_superlu/*.so',
        'sparse_superlu/*.dylib',
        'sparse_umfpack/*.so',
        'sparse_umfpack/*.dylib',
        'sparse_eigen/*.so',
        'sparse_superlu/test/*.py',
        'sparse_umfpack/test/*.py',
        'conversion/*.py'
    ]

# Setup configuration
packages=[
        'sparse_numba',
        'sparse_numba.conversion',
        'sparse_numba.sparse_superlu',
        'sparse_numba.sparse_superlu.test',
        'sparse_numba.sparse_umfpack',
        'sparse_numba.sparse_umfpack.test',
        'sparse_numba.sparse_eigen',
    ]

setup(
    name="sparse_numba",
    version="0.2.0",
    description="Customized sparse solver with Numba support",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Tianqi Hong",
    author_email="tianqi.hong@uga.edu",
    url="https://github.com/th1275/sparse_numba",
    # packages=find_packages(),
    packages=packages,
    ext_modules=extensions,
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    package_data=package_data,
    python_requires=">=3.10",
    # numba 0.61 is the first numba with Python 3.13 and needs 3.10; numpy
    # at numba 0.61's floor (pip takes a numpy with wheels for the running
    # Python: 1.26 is the first with 3.12 wheels, 2.1 the first with 3.13's)
    install_requires=[
        "numpy>=1.24",
        "numba>=0.61",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    include_package_data=True,
    zip_safe=False,  # This is important for extensions
)