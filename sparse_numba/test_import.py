"""
Package Installation Test File
"""

#  [sparse_numba] (C)2025-2025 Tianqi Hong
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the BSD License.
#
#  File name: test_import.py

import sys
print("Python path:", sys.path)

def run_test():
   try:
      import sparse_numba
      print("sparse_numba imported successfully")
      try:
         import sparse_numba.sparse_superlu
         print("sparse_numba.sparse_superlu imported successfully")
         import sparse_numba.sparse_superlu.cy_superlu_wrapper
         print("cy_superlu_wrapper imported successfully")
      except Exception as e:
         print(f"Error: {e}")

      try:
         import sparse_numba.sparse_umfpack
         print("sparse_numba.sparse_umfpack imported successfully")
         import sparse_numba.sparse_umfpack.cy_umfpack_wrapper
         print("cy_umfpack_wrapper imported successfully")
      except Exception as e:
         print(f"Error: {e}")

   except Exception as e:
      print(f"Error: {e}")