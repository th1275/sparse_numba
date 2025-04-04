import os
import sys
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import distutils.ccompiler
from distutils.errors import CompileError

# Force MinGW compiler on Windows
if platform.system() == 'Windows':
    # Save the original compiler selection function
    original_compiler = distutils.ccompiler.get_default_compiler


    # Override with a function that always returns 'mingw32'
    def force_mingw():
        return 'mingw32'


    # Replace the function
    distutils.ccompiler.get_default_compiler = force_mingw

# Define the extension modules - only UMFPACK for now
extensions = [
    Extension(
        "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
        sources=[
            "sparse_numba/sparse_umfpack/cy_umfpack_wrapper.pyx",
            "sparse_numba/sparse_umfpack/umfpack_wrapper.c"
        ],
        include_dirs=[
            np.get_include(),
            "sparse_numba/sparse_umfpack",
            "vendor/suitesparse/include",
            "vendor/openblas/include"  # Fixed missing comma here
        ],
        libraries=[
            "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
            "suitesparseconfig", "openblas"
        ],
        library_dirs=[
            "vendor/suitesparse/lib",
            "vendor/openblas/lib"
        ],
        extra_compile_args=["-O3"],  # GCC optimization flag
    )
]


# Customize the build process
class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        # Add NumPy include directory
        self.include_dirs.append(np.get_include())

        # Force MinGW compiler on Windows
        if platform.system() == 'Windows':
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
            ext_path = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
            # package_dir = os.path.dirname(ext_path)
            package_dir = os.path.dirname(os.path.dirname(ext_path))  # package_dir is sparse_numba

            # Print for debugging
            print(f"Extension path: {ext_path}")
            print(f"Package directory: {package_dir}")

            # Create the vendor directories in the build
            suitesparse_target_dir = os.path.join(package_dir, "vendor", "suitesparse", "bin")
            openblas_target_dir = os.path.join(package_dir, "vendor", "openblas", "bin")
            os.makedirs(suitesparse_target_dir, exist_ok=True)
            os.makedirs(openblas_target_dir, exist_ok=True)

            # Source directories
            suitesparse_bin_dir = os.path.join("vendor", "suitesparse", "bin")
            openblas_bin_dir = os.path.join("vendor", "openblas", "bin")

            # Verify source directories exist
            print(f"SuiteSparse bin dir exists: {os.path.exists(suitesparse_bin_dir)}")
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


# Package data to include DLLs but not lib and include files
package_data = {
    'sparse_numba': [
        # 'vendor/suitesparse/bin/*.dll',
        'vendor/openblas/bin/*.dll'
    ],
}

# # Define data files
# data_files = []
# if platform.system() == 'Windows':
#     # Source directories
#     suitesparse_bin_dir = os.path.join("vendor", "suitesparse", "bin")
#     openblas_bin_dir = os.path.join("vendor", "openblas", "bin")
#
#     # Check if directories exist and add DLLs (only include in sparse_numba package)
#     if os.path.exists(suitesparse_bin_dir):
#         data_files.append(
#             ('sparse_numba/vendor/suitesparse/bin',
#              [os.path.join(suitesparse_bin_dir, f) for f in os.listdir(suitesparse_bin_dir) if f.endswith('.dll')])
#         )
#
#     if os.path.exists(openblas_bin_dir):
#         data_files.append(
#             ('sparse_numba/vendor/openblas/bin',
#              [os.path.join(openblas_bin_dir, f) for f in os.listdir(openblas_bin_dir) if f.endswith('.dll')])
#         )

    # # Check if directories exist and add DLLs
    # if os.path.exists(suitesparse_bin_dir):
    #     data_files.extend([
    #         ('sparse_numba/vendor/suitesparse/bin',
    #          [os.path.join(suitesparse_bin_dir, f) for f in os.listdir(suitesparse_bin_dir) if f.endswith('.dll')]),
    #         ('vendor/suitesparse/bin',
    #          [os.path.join(suitesparse_bin_dir, f) for f in os.listdir(suitesparse_bin_dir) if f.endswith('.dll')])
    #     ])
    #
    # if os.path.exists(openblas_bin_dir):
    #     data_files.extend([
    #         ('sparse_numba/vendor/openblas/bin',
    #          [os.path.join(openblas_bin_dir, f) for f in os.listdir(openblas_bin_dir) if f.endswith('.dll')]),
    #         ('vendor/openblas/bin',
    #          [os.path.join(openblas_bin_dir, f) for f in os.listdir(openblas_bin_dir) if f.endswith('.dll')])
    #     ])

setup(
    name="sparse_numba",
    version="0.1.5",
    description="Customized sparse solver with Numba support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tianqi Hong",
    author_email="tianqi.hong@uga.edu",
    url="https://github.com/th1275/sparse_numba",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={'build_ext': CustomBuildExt},
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
    ],
    # Include DLLs in multiple locations with include_package_data
    include_package_data=False,
)