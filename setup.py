import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel
import numpy as np
import distutils.ccompiler
from distutils.errors import CompileError


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
        "vendor/openblas/lib"
    ]
    # Libraries needed for Windows
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas"
    ]
    superlu_libraries = ["superlu", "openblas"]
    extra_compile_args = ["-O3"]
    extra_link_args = []
elif IS_LINUX:
    # On Linux, we'll use system libraries if available
    include_dirs = [
        np.get_include(),
        "sparse_numba/sparse_umfpack",
        "sparse_numba/sparse_superlu",
        "/usr/include/suitesparse",  # Standard location on most Linux distros
        "/usr/local/include/suitesparse",  # Possible alternate location
        "/usr/include/superlu",
        "/usr/local/include/superlu"
    ]
    library_dirs = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64"
    ]
    # Libraries needed for Linux
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas", "blas"
    ]
    superlu_libraries = ["superlu", "openblas", "blas"]
    extra_compile_args = ["-O3", "-fPIC"]
    extra_link_args = []
elif IS_MACOS:
    # On macOS, we check both system locations and Homebrew/MacPorts locations
    homebrew_prefix = os.popen("brew --prefix 2>/dev/null || echo ''").read().strip()
    if not homebrew_prefix:
        homebrew_prefix = "/usr/local"  # Default Homebrew location

    include_dirs = [
        np.get_include(),
        "sparse_numba/sparse_umfpack",
        "sparse_numba/sparse_superlu",
        f"{homebrew_prefix}/include/suitesparse",
        f"{homebrew_prefix}/include/superlu",
        "/usr/local/include/suitesparse",
        "/opt/local/include/suitesparse",  # MacPorts
        "/usr/local/include/superlu",
        "/opt/local/include/superlu"
    ]
    library_dirs = [
        f"{homebrew_prefix}/lib",
        "/usr/local/lib",
        "/opt/local/lib"  # MacPorts
    ]
    # Libraries needed for macOS
    umfpack_libraries = [
        "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
        "suitesparseconfig", "openblas"
    ]
    superlu_libraries = ["superlu", "openblas"]
    # For macOS, ensure we're building for the right architecture
    extra_compile_args = ["-O3", "-fPIC"]
    # Handle Apple Silicon vs Intel Mac
    if platform.machine() == 'arm64':
        extra_compile_args.append("-arch arm64")
        extra_link_args = ["-arch arm64"]
    else:
        extra_compile_args.append("-arch x86_64")
        extra_link_args = ["-arch x86_64"]
else:
    raise RuntimeError(f"Unsupported platform: {PLATFORM}")

# Define the extension modules
extensions = [
    Extension(
        "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
        sources=[
            "sparse_numba/sparse_umfpack/cy_umfpack_wrapper.pyx",
            "sparse_numba/sparse_umfpack/umfpack_wrapper.c"
        ],
        include_dirs=include_dirs,
        libraries=umfpack_libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=True,  # for multiple python version
    ),
    Extension(
        "sparse_numba.sparse_superlu.cy_superlu_wrapper",
        sources=[
            "sparse_numba/sparse_superlu/cy_superlu_wrapper.pyx",
            "sparse_numba/sparse_superlu/superlu_wrapper.c"
        ],
        include_dirs=include_dirs,
        libraries=superlu_libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=True, # for multiple python version
    )
]


# Customize the build process
class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        # Add NumPy include directory
        self.include_dirs.append(np.get_include())

        # Force MinGW compiler on Windows
        if IS_WINDOWS:
            self.compiler = 'mingw32'

    def build_extensions(self):
        # Ensure MinGW is being used
        if platform.system() == 'Windows':
            if self.compiler.compiler_type != 'mingw32':
                raise CompileError(
                    "This package must be compiled with MinGW (GCC) on Windows. "
                    "Your DLLs were compiled with GCC, and mixing compilers can cause memory issues."
                )

        build_ext.build_extensions(self)

    def run(self):
        build_ext.run(self)

        # After building, copy DLLs to the package directory
        if platform.system() == 'Windows':
            ext_path_umfpack = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
            ext_path_superlu = self.get_ext_fullpath("sparse_numba.sparse_superlu.cy_superlu_wrapper")
            # package_dir = os.path.dirname(ext_path)
            package_dir = os.path.dirname(os.path.dirname(ext_path_umfpack))  # package_dir is sparse_numba

            # Print for debugging
            print(f"Extension umfpack path: {ext_path_umfpack}")
            print(f"Extension superlu path: {ext_path_superlu}")
            print(f"Package directory: {package_dir}")

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

            # # Copy SuiteSparse DLLs
            # if os.path.exists(suitesparse_bin_dir):
            #     for dll_file in os.listdir(suitesparse_bin_dir):
            #         if dll_file.endswith('.dll'):
            #             dest_path = os.path.join(suitesparse_target_dir, dll_file)
            #             shutil.copy(
            #                 os.path.join(suitesparse_bin_dir, dll_file),
            #                 dest_path
            #             )
            #             print(f"Copied SuiteSparse DLL: {dll_file} to {dest_path}")

            # Copy OpenBLAS DLLs
            if os.path.exists(openblas_bin_dir):
                for dll_file in os.listdir(openblas_bin_dir):
                    if dll_file.endswith('.dll'):
                        dest_path = os.path.join(openblas_target_dir, dll_file)
                        shutil.copy(
                            os.path.join(openblas_bin_dir, dll_file),
                            dest_path
                        )
                        print(f"Copied OpenBLAS DLL: {dll_file} to {dest_path}")

            # Copy SuperLU DLLs
            if os.path.exists(superlu_bin_dir):
                for dll_file in os.listdir(superlu_bin_dir):
                    if dll_file.endswith('.dll'):
                        dest_path = os.path.join(superlu_target_dir, dll_file)
                        shutil.copy(
                            os.path.join(superlu_bin_dir, dll_file),
                            dest_path
                        )
                        print(f"Copied SuperLU DLL: {dll_file} to {dest_path}")


# Define a custom wheel command
class BDistWheelABI3(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # Mark us as not tied to a specific Python API version
        if not self.py_limited_api:
            self.py_limited_api = "cp38"  # This sets minimum Python version to 3.8


# Define platform-specific package data
package_data = {
    'sparse_numba': []
}

if IS_WINDOWS:
    package_data['sparse_numba'] = [
        'vendor/superlu/bin/*.dll',
        'vendor/openblas/bin/*.dll'
    ]
elif IS_LINUX:
    # No need to include system libraries on Linux
    pass
elif IS_MACOS:
    # For macOS, we might include dylibs if we're bundling them
    # package_data['sparse_numba'] = [
    #     'vendor/superlu/lib/*.dylib',
    #     'vendor/openblas/lib/*.dylib'
    # ]
    pass


setup(
    name="sparse_numba",
    version="0.1.8",
    description="Customized sparse solver with Numba support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tianqi Hong",
    author_email="tianqi.hong@uga.edu",
    url="https://github.com/th1275/sparse_numba",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={
        'build_ext': CustomBuildExt,
        'bdist_wheel': BDistWheelABI3
    },
    package_data=package_data,
    # data_files=data_files,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.60.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    # Include DLLs in multiple locations with include_package_data
    include_package_data=False,
)