cmake_minimum_required(VERSION 3.25)
project(cuEV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project Find modules (NCCL, cuBLASMp, cuSOLVERMp)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# =============================================================================
# CPM — package manager
# =============================================================================
set(CPM_VERSION 0.40.2)
set(CPM_FILE "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_VERSION}.cmake")
if(NOT EXISTS "${CPM_FILE}")
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake"
    "${CPM_FILE}"
  )
endif()
include("${CPM_FILE}")

# =============================================================================
# Options
# =============================================================================
option(CUEV_ENABLE_MP "Build distributed multi-GPU solver" OFF)
option(CUEV_ENABLE_TESTS "Build tests" ON)
option(CUEV_ENABLE_BENCH "Build benchmarks" ON)

# =============================================================================
# Dependencies
# =============================================================================
if (CUEV_ENABLE_TESTS)
  CPMAddPackage("gh:google/googletest#v1.14.0")
endif()

find_package(CUDAToolkit REQUIRED)

if(CUEV_ENABLE_MP)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
  find_package(CUBLASMp REQUIRED)
  find_package(CUSOLVERMp REQUIRED)
  find_package(NVSHMEM REQUIRED)
  find_package(CAL REQUIRED)
  find_package(UCC REQUIRED)
endif()

# =============================================================================
# cuEV single-GPU library
# =============================================================================
add_library(cuev SHARED
  src/cuda/cublas.cu
  src/cuda/cusolver.cu
  src/cuda/qdwh.cu
  src/cuda/sdc.cu
  src/cuda/solver.cu
)
target_include_directories(cuev PUBLIC include)
target_link_libraries(cuev PUBLIC CUDA::cublas CUDA::cusolver)
set_target_properties(cuev PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON
)

# =============================================================================
# cuEV distributed library  (CUEV_ENABLE_MP=ON)
# =============================================================================
if(CUEV_ENABLE_MP)
  add_library(cuev_mp SHARED
    src/mp/comm.cu
  )
  target_include_directories(cuev_mp PUBLIC include)
  target_compile_definitions(cuev_mp PUBLIC CUEV_ENABLE_MP)
  target_link_libraries(cuev_mp PUBLIC
    CUDA::cublas CUDA::cusolver
    MPI::MPI_CXX NCCL::NCCL
    cuBLASMp::cuBLASMp cuSOLVERMp::cuSOLVERMp
    NVSHMEM::NVSHMEM UCC::UCC
    CAL::CAL
  )
  set_target_properties(cuev_mp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
  )

  if (CUEV_ENABLE_BENCH)
    add_executable(cuBenchMp bench/bench_mp.cpp)
    target_link_libraries(cuBenchMp PRIVATE cuev_mp CUDA::cudart MPI::MPI_CXX)
  endif()
endif()

# =============================================================================
# Benchmark
# =============================================================================
if (CUEV_ENABLE_BENCH)
  add_executable(cuBench bench/bench.cpp)
  target_link_libraries(cuBench PRIVATE cuev CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# =============================================================================
# Tests
# =============================================================================

if (CUEV_ENABLE_TESTS)
  enable_testing()
  include(GoogleTest)

  # Unit / integration test suite
  add_executable(cuTest
  test/test_qdwh.cu
  test/test_sdc.cu
  test/test_solver.cu
  )
  target_link_libraries(cuTest PRIVATE cuev GTest::gtest_main CUDA::cublas CUDA::cusolver)
  set_target_properties(cuTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  gtest_discover_tests(cuTest)
endif()

# =============================================================================
# Format (clang-format)
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
    COMMENT "Running clang-format"
  )
endif()

# =============================================================================
# Doxygen  (-DCUEV_BUILD_DOCS=ON)
# =============================================================================
option(CUEV_BUILD_DOCS "Build Doxygen documentation" OFF)
if(CUEV_BUILD_DOCS)
  find_package(Doxygen QUIET OPTIONAL_COMPONENTS dot)
  if(DOXYGEN_FOUND AND DOXYGEN_VERSION)
    set(DOXYGEN_PROJECT_NAME        "cuEV")
    set(DOXYGEN_PROJECT_BRIEF       "CUDA symmetric dense eigensolver")
    set(DOXYGEN_EXTRACT_ALL         YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_OUTPUT_DIRECTORY    "${CMAKE_BINARY_DIR}/docs")
    doxygen_add_docs(docs include src)
  endif()
endif()
