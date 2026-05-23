cmake_minimum_required(VERSION 3.27)

project(mlx_sparse_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)
option(MLX_SPARSE_BUILD_METAL "Build the mlx_sparse Metal library" ON)

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
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsr/coo_tocsr.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmat/csr_matmat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul/csr_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_transpose/csr_matmul_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec/csr_matvec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_transpose/csr_matvec_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sort_indices/csr_sort_indices.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sum_duplicates/csr_sum_duplicates.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_todense/csr_todense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_transpose/csr_transpose.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/fromdense/fromdense.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sparse/identity_like/identity_like.cpp)

target_include_directories(
  mlx_sparse_native
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/src)

target_link_libraries(mlx_sparse_native PUBLIC mlx)

execute_process(
  COMMAND xcrun -sdk macosx -find metal
  RESULT_VARIABLE MLX_SPARSE_METAL_FIND_RESULT
  OUTPUT_QUIET
  ERROR_QUIET)

if(MLX_BUILD_METAL AND MLX_SPARSE_BUILD_METAL AND MLX_SPARSE_METAL_FIND_RESULT EQUAL 0)
  mlx_build_metallib(
    TARGET mlx_sparse_metallib
    TITLE mlx_sparse
    SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/arnoldi/arnoldi.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/cg/cg.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/inner_product/inner_product.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/lanczos/lanczos.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/permute_vector/permute_vector.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/linalg/triangular_solve/triangular_solve.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/coo_tocsr/coo_tocsr.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul/csr_matmul.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matmul_transpose/csr_matmul_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec/csr_matvec.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_matvec_transpose/csr_matvec_transpose.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sort_indices/csr_sort_indices.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_sum_duplicates/csr_sum_duplicates.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_todense/csr_todense.metal
      ${CMAKE_CURRENT_LIST_DIR}/src/sparse/csr_transpose/csr_transpose.metal
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
  message(STATUS "Skipping mlx_sparse.metallib build; Metal compiler is unavailable.")
endif()

nanobind_add_module(
  _ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/src/bindings.cpp)

target_link_libraries(_ext PRIVATE mlx_sparse_native)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
