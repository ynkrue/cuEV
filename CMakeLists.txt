cmake_minimum_required(VERSION 3.25)
project(cuEV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# =============================================================================
# Options
# =============================================================================
option(CUEV_ENABLE_MP    "Build distributed multi-GPU solver" OFF)
option(CUEV_ENABLE_TESTS "Build tests"                        ON)
option(CUEV_ENABLE_BENCH "Build benchmarks"                   ON)
option(CUEV_BUILD_DOCS   "Build Doxygen documentation"        OFF)

# =============================================================================
# Dependencies
# =============================================================================
find_package(CUDAToolkit REQUIRED)

if(CUEV_ENABLE_MP)
  find_package(MPI REQUIRED)

  # nvhpc libraries - set NCCL_ROOT / CUBLASMP_ROOT / CUSOLVERMP_ROOT
  find_library(NCCL_LIB       nccl       REQUIRED)
  find_library(CUBLASMP_LIB   cublasmp   REQUIRED)
  find_library(CUSOLVERMP_LIB cusolverMp REQUIRED)
  find_path(NCCL_INC       nccl.h       REQUIRED)
  find_path(CUBLASMP_INC   cublasmp.h   REQUIRED)
  find_path(CUSOLVERMP_INC cusolverMp.h REQUIRED)

  add_library(NCCL::NCCL             INTERFACE IMPORTED)
  target_link_libraries(NCCL::NCCL           INTERFACE "${NCCL_LIB}")
  target_include_directories(NCCL::NCCL     INTERFACE "${NCCL_INC}")

  add_library(cuBLASMp::cuBLASMp     INTERFACE IMPORTED)
  target_link_libraries(cuBLASMp::cuBLASMp   INTERFACE "${CUBLASMP_LIB}")
  target_include_directories(cuBLASMp::cuBLASMp INTERFACE "${CUBLASMP_INC}")

  add_library(cuSOLVERMp::cuSOLVERMp INTERFACE IMPORTED)
  target_link_libraries(cuSOLVERMp::cuSOLVERMp INTERFACE "${CUSOLVERMP_LIB}")
  target_include_directories(cuSOLVERMp::cuSOLVERMp INTERFACE "${CUSOLVERMP_INC}")
endif()

# =============================================================================
# cuev — single-GPU library
# =============================================================================
if(NOT CUEV_ENABLE_MP)
  add_library(cuev SHARED
    src/cuda/handle.cu
    src/cuda/cublas.cu
    src/cuda/cusolver.cu
    # src/cuda/dbbr.cu
    # src/cuda/bulge.cu
    # src/cuda/backtransform.cu
    src/cuda/solver.cu
  )
  target_include_directories(cuev PUBLIC include)
  target_link_libraries(cuev PUBLIC CUDA::cublas CUDA::cusolver)
  set_target_properties(cuev PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)
endif()

# =============================================================================
# cuev_mp — distributed library
# =============================================================================
if(CUEV_ENABLE_MP)
  add_library(cuev_mp SHARED
    src/mp/handle_mp.cu
    src/mp/cublas_mp.cu
    src/mp/cusolver_mp.cu
    src/mp/dbbr_mp.cu
    src/mp/bulge_mp.cu
    src/mp/backtransform_mp.cu
    src/mp/solver_mp.cu
  )
  target_include_directories(cuev_mp PUBLIC include)
  target_compile_definitions(cuev_mp PUBLIC CUEV_ENABLE_MP)
  target_link_libraries(cuev_mp PUBLIC
    CUDA::cublas CUDA::cusolver
    MPI::MPI_CXX NCCL::NCCL
    CUBLASMp::CUBLASMp CUSOLVERMp::CUSOLVERMp)
  set_target_properties(cuev_mp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(CUEV_ENABLE_BENCH)
  if(NOT CUEV_ENABLE_MP)
    add_executable(cuBench bench/bench.cpp)
    target_link_libraries(cuBench PRIVATE cuev CUDA::cudart CUDA::cublas CUDA::cusolver)
  else()
    add_executable(cuBenchMp   bench/bench_mp.cpp)
    target_link_libraries(cuBenchMp   PRIVATE cuev_mp CUDA::cudart MPI::MPI_CXX)
    add_executable(cuBenchCuev bench/bench_cuev.cpp)
    target_link_libraries(cuBenchCuev PRIVATE cuev_mp CUDA::cudart MPI::MPI_CXX)
  endif()
endif()

# =============================================================================
# Tests
# =============================================================================
# if(CUEV_ENABLE_TESTS)
#   if(NOT CUEV_ENABLE_MP)
#     add_executable(cuTest
#       test/cuda/test_solver.cu
#     )
#     target_link_libraries(cuTest PRIVATE cuev CUDA::cublas CUDA::cusolver)
#     set_target_properties(cuTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   else()
#     add_executable(cuTestMp
#       test/mp/test_mp.cpp
#     )
#     target_link_libraries(cuTestMp PRIVATE cuev_mp CUDA::cudart MPI::MPI_CXX)
#     set_target_properties(cuTestMp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   endif()
# endif()

# =============================================================================
# Doxygen
# =============================================================================
if(CUEV_BUILD_DOCS)
  find_package(Doxygen QUIET OPTIONAL_COMPONENTS dot)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME        "cuEV")
    set(DOXYGEN_PROJECT_BRIEF       "CUDA symmetric dense eigensolver")
    set(DOXYGEN_EXTRACT_ALL         YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_OUTPUT_DIRECTORY    "${CMAKE_BINARY_DIR}/docs")
    doxygen_add_docs(docs include src)
  endif()
endif()

# =============================================================================
# Format
# =============================================================================
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE FORMAT_SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/bench/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
    COMMENT "Running clang-format")
endif()
