# import os
# import sys
# import platform
# from setuptools import setup, find_packages, Extension
# from setuptools.command.build_ext import build_ext
# from setuptools.command.bdist_wheel import bdist_wheel
# import numpy as np
# import distutils.ccompiler
# from distutils.errors import CompileError
#
#
# # Determine the platform
# PLATFORM = platform.system()
# IS_WINDOWS = PLATFORM == 'Windows'
# IS_LINUX = PLATFORM == 'Linux'
# IS_MACOS = PLATFORM == 'Darwin'
#
# # Force MinGW compiler on Windows
# if IS_WINDOWS:
#     # Save the original compiler selection function
#     original_compiler = distutils.ccompiler.get_default_compiler
#
#     # Override with a function that always returns 'mingw32'
#     def force_mingw():
#         return 'mingw32'
#
#     # Replace the function
#     distutils.ccompiler.get_default_compiler = force_mingw
#
# # Define platform-specific settings
# if IS_WINDOWS:
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         "vendor/suitesparse/include",
#         "vendor/superlu/include",
#         "vendor/openblas/include"
#     ]
#     library_dirs = [
#         "vendor/suitesparse/lib",
#         "vendor/superlu/lib",
#         "vendor/openblas/lib"
#     ]
#     # Libraries needed for Windows
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas"
#     ]
#     superlu_libraries = ["superlu", "openblas"]
#     extra_compile_args = ["-O3"]
#     extra_link_args = []
# elif IS_LINUX:
#     # On Linux, we'll use system libraries if available
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         "/usr/include/suitesparse",  # Standard location on most Linux distros
#         "/usr/local/include/suitesparse",  # Possible alternate location
#         "/usr/include/superlu",
#         "/usr/local/include/superlu"
#     ]
#     library_dirs = [
#         "/usr/lib",
#         "/usr/lib64",
#         "/usr/local/lib",
#         "/usr/local/lib64"
#     ]
#     # Libraries needed for Linux
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas", "blas"
#     ]
#     superlu_libraries = ["superlu", "openblas", "blas"]
#     extra_compile_args = ["-O3", "-fPIC"]
#     extra_link_args = []
# elif IS_MACOS:
#     # On macOS, we check both system locations and Homebrew/MacPorts locations
#     homebrew_prefix = os.popen("brew --prefix 2>/dev/null || echo ''").read().strip()
#     if not homebrew_prefix:
#         homebrew_prefix = "/usr/local"  # Default Homebrew location
#
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         f"{homebrew_prefix}/include/suitesparse",
#         f"{homebrew_prefix}/include/superlu",
#         "/usr/local/include/suitesparse",
#         "/opt/local/include/suitesparse",  # MacPorts
#         "/usr/local/include/superlu",
#         "/opt/local/include/superlu"
#     ]
#     library_dirs = [
#         f"{homebrew_prefix}/lib",
#         "/usr/local/lib",
#         "/opt/local/lib"  # MacPorts
#     ]
#     # Libraries needed for macOS
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas"
#     ]
#     superlu_libraries = ["superlu", "openblas"]
#     # For macOS, ensure we're building for the right architecture
#     extra_compile_args = ["-O3", "-fPIC"]
#     extra_link_args = []
#
#     # Check if we're in a CI environment
#     ci_build = os.environ.get('CI', '') == 'true' or os.environ.get('GITHUB_ACTIONS', '') == 'true'
#
#     # Only add architecture flags if not in CI
#     if not ci_build:
#         if platform.machine() == 'arm64':
#             extra_compile_args.append("-arch arm64")
#             extra_link_args = ["-arch arm64"]
#         else:
#             extra_compile_args.append("-arch x86_64")
#             extra_link_args = ["-arch x86_64"]
# else:
#     raise RuntimeError(f"Unsupported platform: {PLATFORM}")
#
# # Define the extension modules
# extensions = [
#     Extension(
#         "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
#         sources=[
#             "sparse_numba/sparse_umfpack/cy_umfpack_wrapper.pyx",
#             "sparse_numba/sparse_umfpack/umfpack_wrapper.c"
#         ],
#         include_dirs=include_dirs,
#         libraries=umfpack_libraries,
#         library_dirs=library_dirs,
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#         # py_limited_api=True,  # for multiple python version
#     ),
#     Extension(
#         "sparse_numba.sparse_superlu.cy_superlu_wrapper",
#         sources=[
#             "sparse_numba/sparse_superlu/cy_superlu_wrapper.pyx",
#             "sparse_numba/sparse_superlu/superlu_wrapper.c"
#         ],
#         include_dirs=include_dirs,
#         libraries=superlu_libraries,
#         library_dirs=library_dirs,
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#         # py_limited_api=True, # for multiple python version
#     )
# ]
#
#
# # Customize the build process
# class CustomBuildExt(build_ext):
#     def finalize_options(self):
#         build_ext.finalize_options(self)
#
#         # Add NumPy include directory
#         self.include_dirs.append(np.get_include())
#
#         # Force MinGW compiler on Windows
#         if IS_WINDOWS:
#             self.compiler = 'mingw32'
#
#     def build_extensions(self):
#         # Ensure MinGW is being used
#         if platform.system() == 'Windows':
#             if self.compiler.compiler_type != 'mingw32':
#                 raise CompileError(
#                     "This package must be compiled with MinGW (GCC) on Windows. "
#                     "Your DLLs were compiled with GCC, and mixing compilers can cause memory issues."
#                 )
#
#         build_ext.build_extensions(self)
#
#     def run(self):
#         build_ext.run(self)
#
#         # After building, copy DLLs to the package directory
#         if platform.system() == 'Windows':
#             ext_path_umfpack = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
#             ext_path_superlu = self.get_ext_fullpath("sparse_numba.sparse_superlu.cy_superlu_wrapper")
#             # package_dir = os.path.dirname(ext_path)
#             package_dir = os.path.dirname(os.path.dirname(ext_path_umfpack))  # package_dir is sparse_numba
#
#             # Print for debugging
#             print(f"Extension umfpack path: {ext_path_umfpack}")
#             print(f"Extension superlu path: {ext_path_superlu}")
#             print(f"Package directory: {package_dir}")
#
#             # Create the vendor directories in the build
#             suitesparse_target_dir = os.path.join(package_dir, "vendor", "suitesparse", "bin")
#             openblas_target_dir = os.path.join(package_dir, "vendor", "openblas", "bin")
#             superlu_target_dir = os.path.join(package_dir, "vendor", "superlu", "bin")
#             os.makedirs(suitesparse_target_dir, exist_ok=True)
#             os.makedirs(superlu_target_dir, exist_ok=True)
#             os.makedirs(openblas_target_dir, exist_ok=True)
#
#             # Source directories
#             suitesparse_bin_dir = os.path.join("vendor", "suitesparse", "bin")
#             superlu_bin_dir = os.path.join("vendor", "superlu", "bin")
#             openblas_bin_dir = os.path.join("vendor", "openblas", "bin")
#
#             # Verify source directories exist
#             print(f"SuiteSparse bin dir exists: {os.path.exists(suitesparse_bin_dir)}")
#             print(f"SuperLU bin dir exists: {os.path.exists(superlu_bin_dir)}")
#             print(f"OpenBLAS bin dir exists: {os.path.exists(openblas_bin_dir)}")
#
#             import shutil
#
#             # # Copy SuiteSparse DLLs
#             # if os.path.exists(suitesparse_bin_dir):
#             #     for dll_file in os.listdir(suitesparse_bin_dir):
#             #         if dll_file.endswith('.dll'):
#             #             dest_path = os.path.join(suitesparse_target_dir, dll_file)
#             #             shutil.copy(
#             #                 os.path.join(suitesparse_bin_dir, dll_file),
#             #                 dest_path
#             #             )
#             #             print(f"Copied SuiteSparse DLL: {dll_file} to {dest_path}")
#
#             # Copy OpenBLAS DLLs
#             if os.path.exists(openblas_bin_dir):
#                 for dll_file in os.listdir(openblas_bin_dir):
#                     if dll_file.endswith('.dll'):
#                         dest_path = os.path.join(openblas_target_dir, dll_file)
#                         shutil.copy(
#                             os.path.join(openblas_bin_dir, dll_file),
#                             dest_path
#                         )
#                         print(f"Copied OpenBLAS DLL: {dll_file} to {dest_path}")
#
#             # Copy SuperLU DLLs
#             if os.path.exists(superlu_bin_dir):
#                 for dll_file in os.listdir(superlu_bin_dir):
#                     if dll_file.endswith('.dll'):
#                         dest_path = os.path.join(superlu_target_dir, dll_file)
#                         shutil.copy(
#                             os.path.join(superlu_bin_dir, dll_file),
#                             dest_path
#                         )
#                         print(f"Copied SuperLU DLL: {dll_file} to {dest_path}")
#
#
# # Define a custom wheel command
# class BDistWheelABI3(bdist_wheel):
#     def finalize_options(self):
#         super().finalize_options()
#         # Mark us as not tied to a specific Python API version
#         if not self.py_limited_api:
#             self.py_limited_api = "cp38"  # This sets minimum Python version to 3.8
#
#
# # Define platform-specific package data
# package_data = {
#     'sparse_numba': []
# }
#
# if IS_WINDOWS:
#     package_data['sparse_numba'] = [
#         'vendor/superlu/bin/*.dll',
#         'vendor/openblas/bin/*.dll'
#     ]
# elif IS_LINUX:
#     # No need to include system libraries on Linux
#     pass
# elif IS_MACOS:
#     # For macOS, we might include dylibs if we're bundling them
#     # package_data['sparse_numba'] = [
#     #     'vendor/superlu/lib/*.dylib',
#     #     'vendor/openblas/lib/*.dylib'
#     # ]
#     pass
#
#
# setup(
#     name="sparse_numba",
#     version="0.1.9",
#     description="Customized sparse solver with Numba support",
#     long_description=open("README.md").read(),
#     long_description_content_type="text/markdown",
#     author="Tianqi Hong",
#     author_email="tianqi.hong@uga.edu",
#     url="https://github.com/th1275/sparse_numba",
#     packages=find_packages(),
#     ext_modules=extensions,
#     cmdclass={
#         'build_ext': CustomBuildExt,
#         # 'bdist_wheel': BDistWheelABI3
#     },
#     package_data=package_data,
#     # data_files=data_files,
#     python_requires=">=3.8",
#     install_requires=[
#         "numpy>=1.13.3",
#         "numba>=0.58.0",
#     ],
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: BSD License",
#         "Operating System :: Microsoft :: Windows",
#         "Operating System :: POSIX :: Linux",
#         "Operating System :: MacOS :: MacOS X",
#     ],
#     # Include DLLs in multiple locations with include_package_data
#     include_package_data=False,
# )

# import os
# import sys
# import platform
# from setuptools import setup, find_packages, Extension
# from setuptools.command.build_ext import build_ext
# from setuptools.command.bdist_wheel import bdist_wheel
# import numpy as np
# import distutils.ccompiler
# from distutils.errors import CompileError
#
# # CI environment detection
# CI_BUILD = os.environ.get('CI', '') == 'true' or os.environ.get('GITHUB_ACTIONS', '') == 'true'
#
# # Determine the platform
# PLATFORM = platform.system()
# IS_WINDOWS = PLATFORM == 'Windows'
# IS_LINUX = PLATFORM == 'Linux'
# IS_MACOS = PLATFORM == 'Darwin'
#
# # Force MinGW compiler on Windows
# if IS_WINDOWS:
#     # Save the original compiler selection function
#     original_compiler = distutils.ccompiler.get_default_compiler
#
#
#     # Override with a function that always returns 'mingw32'
#     def force_mingw():
#         return 'mingw32'
#
#
#     # Replace the function
#     distutils.ccompiler.get_default_compiler = force_mingw
#
# # Define platform-specific settings
# if IS_WINDOWS:
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         "vendor/suitesparse/include",
#         "vendor/superlu/include",
#         "vendor/openblas/include"
#     ]
#     library_dirs = [
#         "vendor/suitesparse/lib",
#         "vendor/superlu/lib",
#         "vendor/openblas/lib"
#     ]
#     # Libraries needed for Windows
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas"
#     ]
#     superlu_libraries = ["superlu", "openblas"]
#     extra_compile_args = ["-O3"]
#     extra_link_args = []
# elif IS_LINUX:
#     # On Linux, we'll use system libraries if available
#     # Try to locate SuperLU headers dynamically
#     superlu_include_dirs = [
#         "/usr/include/superlu",
#         "/usr/local/include/superlu",
#         "/usr/include/suitesparse/superlu",
#         "/usr/include/SuperLU",
#         "/tmp/superlu_include"  # Our custom location from the GitHub workflow
#     ]
#
#     # Check if each directory exists
#     valid_superlu_dirs = []
#     for dir_path in superlu_include_dirs:
#         if os.path.exists(dir_path):
#             valid_superlu_dirs.append(dir_path)
#             print(f"Found valid SuperLU include dir: {dir_path}")
#
#     # Base include directories
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         "/usr/include/suitesparse",  # Standard location on most Linux distros
#         "/usr/local/include/suitesparse",  # Possible alternate location
#     ]
#
#     # Add valid SuperLU directories
#     include_dirs.extend(valid_superlu_dirs)
#
#     # If no valid SuperLU directories found, try to find headers manually
#     if not valid_superlu_dirs:
#         import subprocess
#
#         try:
#             # Find SuperLU headers
#             find_cmd = "find /usr -name slu_ddefs.h 2>/dev/null || echo ''"
#             print(f"Running command: {find_cmd}")
#             superlu_path = subprocess.check_output(find_cmd, shell=True).decode().strip()
#             if superlu_path:
#                 superlu_dir = os.path.dirname(superlu_path)
#                 print(f"Found SuperLU headers at: {superlu_dir}")
#                 include_dirs.append(superlu_dir)
#         except Exception as e:
#             print(f"Error finding SuperLU headers: {e}")
#
#     # Define library directories
#     library_dirs = [
#         "/usr/lib",
#         "/usr/lib64",
#         "/usr/local/lib",
#         "/usr/local/lib64",
#         "/usr/lib/x86_64-linux-gnu"
#     ]
#
#     # Libraries needed for Linux
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas"
#     ]
#     superlu_libraries = ["superlu", "openblas"]
#
#     extra_compile_args = ["-O3", "-fPIC"]
#     extra_link_args = []
# elif IS_MACOS:
#     # On macOS, check both system locations and Homebrew/MacPorts locations
#     homebrew_prefix = os.popen("brew --prefix 2>/dev/null || echo ''").read().strip()
#     if not homebrew_prefix:
#         homebrew_prefix = "/usr/local"  # Default Homebrew location
#
#     # Find Homebrew OpenBLAS location
#     openblas_prefix = os.environ.get('OPENBLAS_PREFIX')
#     if not openblas_prefix:
#         openblas_prefix = os.popen("brew --prefix openblas 2>/dev/null || echo ''").read().strip()
#     if not openblas_prefix:
#         openblas_prefix = "/usr/local/opt/openblas"  # Default location
#
#     print(f"Using OpenBLAS prefix: {openblas_prefix}")
#
#     # Find Homebrew SuperLU location
#     superlu_prefix = os.popen("brew --prefix superlu 2>/dev/null || echo ''").read().strip()
#     if not superlu_prefix:
#         superlu_prefix = "/usr/local/opt/superlu"  # Default location
#
#     # Find Homebrew SuiteSparse location
#     suitesparse_prefix = os.popen("brew --prefix suite-sparse 2>/dev/null || echo ''").read().strip()
#     if not suitesparse_prefix:
#         suitesparse_prefix = "/usr/local/opt/suite-sparse"  # Default location
#
#     # Check for home directory symlinks (from GitHub workflow)
#     home_libs = os.environ.get('HOME_LIBS')
#     home_openblas_lib = None
#     if home_libs:
#         home_openblas_lib = os.path.join(home_libs, "openblas", "lib")
#         if os.path.exists(home_openblas_lib):
#             print(f"Found home directory OpenBLAS lib: {home_openblas_lib}")
#
#     include_dirs = [
#         np.get_include(),
#         "sparse_numba/sparse_umfpack",
#         "sparse_numba/sparse_superlu",
#         f"{suitesparse_prefix}/include/suitesparse",
#         f"{superlu_prefix}/include/superlu",
#         f"{superlu_prefix}/include",  # Newer Homebrew might put headers here
#         "/usr/local/include/suitesparse",
#         "/opt/local/include/suitesparse",  # MacPorts
#         "/usr/local/include/superlu",
#         "/opt/local/include/superlu",
#         f"{openblas_prefix}/include"
#     ]
#
#     library_dirs = [
#         f"{homebrew_prefix}/lib",
#         f"{openblas_prefix}/lib",
#         f"{superlu_prefix}/lib",
#         f"{suitesparse_prefix}/lib",
#         "/usr/local/lib",
#         "/usr/local/opt/openblas/lib",
#         "/opt/local/lib"  # MacPorts
#     ]
#
#     # Add home directory symlinks if available
#     if home_openblas_lib:
#         library_dirs.insert(0, home_openblas_lib)  # Add with high priority
#
#     # Libraries needed for macOS
#     umfpack_libraries = [
#         "umfpack", "cholmod", "amd", "colamd", "camd", "ccolamd",
#         "suitesparseconfig", "openblas"
#     ]
#     superlu_libraries = ["superlu", "openblas"]
#
#     # For macOS, ensure we're building for the right architecture
#     extra_compile_args = ["-O3", "-fPIC"]
#     extra_link_args = [f"-L{openblas_prefix}/lib", "-lopenblas"]
#
#     if home_openblas_lib:
#         extra_link_args.insert(0, f"-L{home_openblas_lib}")  # Add with high priority
#
#     # Skip architecture flags in CI environments
#     if not CI_BUILD:
#         if platform.machine() == 'arm64':
#             extra_compile_args.append("-arch arm64")
#             extra_link_args.append("-arch arm64")
#         else:
#             extra_compile_args.append("-arch x86_64")
#             extra_link_args.append("-arch x86_64")
# else:
#     raise RuntimeError(f"Unsupported platform: {PLATFORM}")
#
# # Print configuration for debugging
# print("\n=== Build Configuration ===")
# print(f"Platform: {PLATFORM}")
# print(f"Include directories: {include_dirs}")
# print(f"Library directories: {library_dirs}")
# print(f"UMFPACK libraries: {umfpack_libraries}")
# print(f"SuperLU libraries: {superlu_libraries}")
# print(f"Extra compile args: {extra_compile_args}")
# print(f"Extra link args: {extra_link_args}")
# print(f"CI Build: {CI_BUILD}")
# print("===========================\n")
#
# # Define the extension modules
# extensions = [
#     Extension(
#         "sparse_numba.sparse_umfpack.cy_umfpack_wrapper",
#         sources=[
#             "sparse_numba/sparse_umfpack/cy_umfpack_wrapper.pyx",
#             "sparse_numba/sparse_umfpack/umfpack_wrapper.c"
#         ],
#         include_dirs=include_dirs,
#         libraries=umfpack_libraries,
#         library_dirs=library_dirs,
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#     ),
#     Extension(
#         "sparse_numba.sparse_superlu.cy_superlu_wrapper",
#         sources=[
#             "sparse_numba/sparse_superlu/cy_superlu_wrapper.pyx",
#             "sparse_numba/sparse_superlu/superlu_wrapper.c"
#         ],
#         include_dirs=include_dirs,
#         libraries=superlu_libraries,
#         library_dirs=library_dirs,
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#     )
# ]
#
#
# # Customize the build process
# class CustomBuildExt(build_ext):
#     def finalize_options(self):
#         build_ext.finalize_options(self)
#
#         # Add NumPy include directory
#         self.include_dirs.append(np.get_include())
#
#         # Force MinGW compiler on Windows
#         if IS_WINDOWS:
#             self.compiler = 'mingw32'
#
#     def build_extensions(self):
#         # Print debugging info for include paths and libraries
#         print("\n--- Build Extension Configuration ---")
#         print(f"self.include_dirs: {self.include_dirs}")
#         print(f"library_dirs: {library_dirs}")
#         print(f"UMFPACK libraries: {umfpack_libraries}")
#         print(f"SuperLU libraries: {superlu_libraries}")
#         print(f"Extra compile args: {extra_compile_args}")
#         print(f"Extra link args: {extra_link_args}")
#
#         # List all SuperLU headers in include paths
#         for include_dir in self.include_dirs:
#             if os.path.exists(include_dir):
#                 slu_headers = []
#                 try:
#                     for f in os.listdir(include_dir):
#                         if f.startswith('slu_') and f.endswith('.h'):
#                             slu_headers.append(f)
#                     if slu_headers:
#                         print(f"SuperLU headers in {include_dir}: {', '.join(slu_headers)}")
#                 except Exception as e:
#                     print(f"Error checking {include_dir}: {e}")
#
#         print("--------------------------------\n")
#
#         # Ensure MinGW is being used on Windows
#         if platform.system() == 'Windows':
#             if self.compiler.compiler_type != 'mingw32':
#                 raise CompileError(
#                     "This package must be compiled with MinGW (GCC) on Windows. "
#                     "Your DLLs were compiled with GCC, and mixing compilers can cause memory issues."
#                 )
#
#         build_ext.build_extensions(self)
#
#     def run(self):
#         build_ext.run(self)
#
#         # After building, copy DLLs to the package directory
#         if platform.system() == 'Windows':
#             ext_path_umfpack = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
#             ext_path_superlu = self.get_ext_fullpath("sparse_numba.sparse_superlu.cy_superlu_wrapper")
#             package_dir = os.path.dirname(os.path.dirname(ext_path_umfpack))
#
#             # Print for debugging
#             print(f"Extension umfpack path: {ext_path_umfpack}")
#             print(f"Extension superlu path: {ext_path_superlu}")
#             print(f"Package directory: {package_dir}")
#
#             # Create the vendor directories in the build
#             suitesparse_target_dir = os.path.join(package_dir, "vendor", "suitesparse", "bin")
#             openblas_target_dir = os.path.join(package_dir, "vendor", "openblas", "bin")
#             superlu_target_dir = os.path.join(package_dir, "vendor", "superlu", "bin")
#             os.makedirs(suitesparse_target_dir, exist_ok=True)
#             os.makedirs(superlu_target_dir, exist_ok=True)
#             os.makedirs(openblas_target_dir, exist_ok=True)
#
#             # Source directories
#             suitesparse_bin_dir = os.path.join("vendor", "suitesparse", "bin")
#             superlu_bin_dir = os.path.join("vendor", "superlu", "bin")
#             openblas_bin_dir = os.path.join("vendor", "openblas", "bin")
#
#             # Verify source directories exist
#             print(f"SuiteSparse bin dir exists: {os.path.exists(suitesparse_bin_dir)}")
#             print(f"SuperLU bin dir exists: {os.path.exists(superlu_bin_dir)}")
#             print(f"OpenBLAS bin dir exists: {os.path.exists(openblas_bin_dir)}")
#
#             import shutil
#
#             # Copy OpenBLAS DLLs
#             if os.path.exists(openblas_bin_dir):
#                 for dll_file in os.listdir(openblas_bin_dir):
#                     if dll_file.endswith('.dll'):
#                         dest_path = os.path.join(openblas_target_dir, dll_file)
#                         shutil.copy(
#                             os.path.join(openblas_bin_dir, dll_file),
#                             dest_path
#                         )
#                         print(f"Copied OpenBLAS DLL: {dll_file} to {dest_path}")
#
#             # Copy SuperLU DLLs
#             if os.path.exists(superlu_bin_dir):
#                 for dll_file in os.listdir(superlu_bin_dir):
#                     if dll_file.endswith('.dll'):
#                         dest_path = os.path.join(superlu_target_dir, dll_file)
#                         shutil.copy(
#                             os.path.join(superlu_bin_dir, dll_file),
#                             dest_path
#                         )
#                         print(f"Copied SuperLU DLL: {dll_file} to {dest_path}")
#
#
# # Define platform-specific package data
# package_data = {
#     'sparse_numba': []
# }
#
# if IS_WINDOWS:
#     package_data['sparse_numba'] = [
#         'vendor/superlu/bin/*.dll',
#         'vendor/openblas/bin/*.dll'
#     ]
# elif IS_LINUX:
#     # No need to include system libraries on Linux
#     pass
# elif IS_MACOS:
#     # For macOS, we might include dylibs if we're bundling them
#     pass
#
# setup(
#     name="sparse_numba",
#     version="0.1.9",
#     description="Customized sparse solver with Numba support",
#     long_description=open("README.md").read(),
#     long_description_content_type="text/markdown",
#     author="Tianqi Hong",
#     author_email="tianqi.hong@uga.edu",
#     url="https://github.com/th1275/sparse_numba",
#     packages=find_packages(),
#     ext_modules=extensions,
#     cmdclass={
#         'build_ext': CustomBuildExt,
#     },
#     package_data=package_data,
#     python_requires=">=3.8",
#     install_requires=[
#         "numpy>=1.13.3",
#         "numba>=0.58.0",
#     ],
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: BSD License",
#         "Operating System :: Microsoft :: Windows",
#         "Operating System :: POSIX :: Linux",
#         "Operating System :: MacOS :: MacOS X",
#     ],
#     include_package_data=False,
# )

import os
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

    extra_compile_args = ["-O3", "-fPIC"]
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
    extra_compile_args = ["-O3", "-fPIC"]
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

# Print configuration for debugging
print("\n=== Build Configuration ===")
print(f"Platform: {PLATFORM}")
print(f"Include directories: {include_dirs}")
print(f"Library directories: {library_dirs}")
print(f"UMFPACK libraries: {umfpack_libraries}")
print(f"SuperLU libraries: {superlu_libraries}")
print(f"Extra compile args: {extra_compile_args}")
print(f"Extra link args: {extra_link_args}")
print(f"CI Build: {CI_BUILD}")
print("===========================\n")

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
                print(f"✅ {ext.name} successfully built at {ext_path}")
            else:
                print(f"❌ {ext.name} FAILED TO BUILD (expected at {ext_path})")
        print("=" * 80 + "\n")

        # After building, copy DLLs to the package directory
        if platform.system() == 'Windows':
            ext_path_umfpack = self.get_ext_fullpath("sparse_numba.sparse_umfpack.cy_umfpack_wrapper")
            ext_path_superlu = self.get_ext_fullpath("sparse_numba.sparse_superlu.cy_superlu_wrapper")
            package_dir = os.path.dirname(os.path.dirname(ext_path_umfpack))

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
    pass

# Setup configuration
setup(
    name="sparse_numba",
    version="0.1.10",
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
    },
    package_data=package_data,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.13.3",
        "numba>=0.58.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    include_package_data=False,
)