cmake_minimum_required(VERSION 3.27)

project(mlx_sparse_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)
option(MLX_SPARSE_BUILD_METAL "Build the mlx_sparse Metal library" ON)
option(
  MLX_SPARSE_ENABLE_ACCELERATE
  "Detect and link Apple's Accelerate framework for future sparse solver support"
  OFF)
set(MLX_SPARSE_HAS_METAL 0)
set(MLX_SPARSE_HAS_ACCELERATE 0)
set(MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 0)

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if(DEFINED ENV{VIRTUAL_ENV})
  set(Python_EXECUTABLE
      "$ENV{VIRTUAL_ENV}/bin/python"
      CACHE FILEPATH "Python executable from active virtualenv" FORCE)
endif()
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR)
endif()
find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_DIR}" NO_DEFAULT_PATH)

if(NOT DEFINED MLX_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_DIR)
endif()
find_package(MLX CONFIG REQUIRED PATHS "${MLX_DIR}" NO_DEFAULT_PATH)

add_library(mlx_sparse_native)

target_sources(
  mlx_sparse_native
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src/common/common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common/cpu_parallel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/accelerate/adapter/csc_adapter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/accelerate/errors/status.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/accelerate/factorization/factorization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/accelerate/solve/solve.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/arnoldi/arnoldi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/cg/cg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/cholesky/cholesky.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/eigs/eigs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/eigsh/eigsh.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/gmres/gmres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/inner_product/inner_product.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/lanczos/lanczos.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/lu/lu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/minres/minres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/permute_vector/permute_vector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/svds/svds.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/linalg/triangular_solve/triangular_solve.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/diagonal/diagonal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/chebyshev/chebyshev.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/exact/exact.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/gmres/gmres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/ic0/ic0.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/ilu0/ilu0.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/minres/minres.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/pcg/chebyshev.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/pcg/ic0.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/pcg/pcg.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsr/coo_tocsr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsc/coo_tocsc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_col_norms/coo_col_norms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_col_sums/coo_col_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_diagonal/coo_diagonal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_batched_matmul/coo_batched_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmat/coo_matmat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmul/coo_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_row_norms/coo_row_norms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_row_sums/coo_row_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_trace/coo_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_batched_matmul/csc_batched_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_col_norms/csc_col_norms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_col_sums/csc_col_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_diagonal/csc_diagonal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmat/csc_matmat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul/csc_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul_transpose/csc_matmul_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matvec/csc_matvec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matvec_transpose/csc_matvec_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_row_norms/csc_row_norms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_row_sums/csc_row_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_sort_indices/csc_sort_indices.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_sum_duplicates/csc_sum_duplicates.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_tocsr/csc_tocsr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_todense/csc_todense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_trace/csc_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_batched_matmul/csr_batched_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_batched_matvec/csr_batched_matvec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_col_sums/csr_col_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_diagonal/csr_diagonal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmat/csr_matmat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul/csr_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_transpose/csr_matmul_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec/csr_matvec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_transpose/csr_matvec_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_row_norms/csr_row_norms.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_row_sums/csr_row_sums.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sort_indices/csr_sort_indices.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sum_duplicates/csr_sum_duplicates.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_todense/csr_todense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_trace/csr_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_transpose/csr_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_tocsc/csr_tocsc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/fromdense/fromdense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/identity_like/identity_like.cpp)

target_include_directories(
  mlx_sparse_native
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src)

target_link_libraries(mlx_sparse_native PUBLIC mlx)

if(MLX_SPARSE_ENABLE_ACCELERATE)
  if(NOT APPLE)
    message(FATAL_ERROR
      "MLX_SPARSE_ENABLE_ACCELERATE=ON requires an Apple platform with the "
      "Accelerate framework.")
  endif()

  find_library(MLX_SPARSE_ACCELERATE_FRAMEWORK Accelerate)
  if(NOT MLX_SPARSE_ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR
      "MLX_SPARSE_ENABLE_ACCELERATE=ON but the Accelerate framework was not found.")
  endif()

  set(MLX_SPARSE_HAS_ACCELERATE 1)
  set(MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 1)
  target_link_libraries(
    mlx_sparse_native PUBLIC "${MLX_SPARSE_ACCELERATE_FRAMEWORK}")
  message(
    STATUS
      "Apple Accelerate framework enabled: ${MLX_SPARSE_ACCELERATE_FRAMEWORK}")
endif()

set(MLX_SPARSE_METAL_FIND_RESULT 1)
if(APPLE AND MLX_SPARSE_BUILD_METAL)
  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    RESULT_VARIABLE MLX_SPARSE_METAL_FIND_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)
endif()

if(APPLE AND MLX_BUILD_METAL AND MLX_SPARSE_BUILD_METAL AND MLX_SPARSE_METAL_FIND_RESULT EQUAL 0)
  set(MLX_SPARSE_HAS_METAL 1)
  mlx_build_metallib(
    TARGET mlx_sparse_metallib
    TITLE mlx_sparse
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/arnoldi/arnoldi.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/cg/cg.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/inner_product/inner_product.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/lanczos/lanczos.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/minres/minres.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/permute_vector/permute_vector.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/svds/svds.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/triangular_solve/triangular_solve.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/diagonal/diagonal.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/chebyshev/chebyshev.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/gmres/gmres.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/minres/minres.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/pcg/chebyshev.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/preconditioners/pcg/pcg.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsr/coo_tocsr.metal
	  ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsc/coo_tocsc.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_col_norms/coo_col_norms.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_col_sums/coo_col_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_diagonal/coo_diagonal.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_batched_matmul/coo_batched_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmat/coo_matmat.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmul/coo_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_row_norms/coo_row_norms.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_row_sums/coo_row_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_trace/coo_trace.metal
	  ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_batched_matmul/csc_batched_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_col_norms/csc_col_norms.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_col_sums/csc_col_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_diagonal/csc_diagonal.metal
	  ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmat/csc_matmat.metal
	  ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul/csc_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matmul_transpose/csc_matmul_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matvec/csc_matvec.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_matvec_transpose/csc_matvec_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_row_norms/csc_row_norms.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_row_sums/csc_row_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_sort_indices/csc_sort_indices.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_sum_duplicates/csc_sum_duplicates.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_tocsr/csc_tocsr.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_todense/csc_todense.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csc_trace/csc_trace.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_batched_matmul/csr_batched_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_batched_matvec/csr_batched_matvec.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_col_sums/csr_col_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_diagonal/csr_diagonal.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmat/csr_matmat.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul/csr_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_transpose/csr_matmul_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec/csr_matvec.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_transpose/csr_matvec_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_row_norms/csr_row_norms.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_row_sums/csr_row_sums.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sort_indices/csr_sort_indices.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sum_duplicates/csr_sum_duplicates.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_todense/csr_todense.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_trace/csr_trace.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_transpose/csr_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_tocsc/csr_tocsc.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/fromdense/fromdense.metal
    DEPS
      ${CMAKE_CURRENT_LIST_DIR}/src/common/metal_common.h
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/common/metal_common.h
    INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}
      ${CMAKE_CURRENT_LIST_DIR}/src
      ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_sparse_native mlx_sparse_metallib)
else()
  message(STATUS "Skipping mlx_sparse.metallib build. Metal compiler is unavailable.")
endif()

target_compile_definitions(
  mlx_sparse_native
  PUBLIC
    MLX_SPARSE_HAS_CPU=1
    MLX_SPARSE_HAS_METAL=${MLX_SPARSE_HAS_METAL}
    MLX_SPARSE_HAS_ACCELERATE=${MLX_SPARSE_HAS_ACCELERATE}
    MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK=${MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK}
    MLX_SPARSE_HAS_CUDA=0
    MLX_SPARSE_HAS_ROCM=0)

nanobind_add_module(
  _ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/src/bindings.cpp)

target_link_libraries(_ext PRIVATE mlx_sparse_native)

if(BUILD_SHARED_LIBS)
  if(APPLE)
    target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
  elseif(UNIX)
    target_link_options(_ext PRIVATE "-Wl,-rpath,$ORIGIN")
  endif()
endif()
