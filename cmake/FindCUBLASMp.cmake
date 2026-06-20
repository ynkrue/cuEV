# FindCUBLASMp.cmake — cuBLASMp (distributed multi-GPU/multi-node cuBLAS).
#
# Override with  -DCUBLASMp_ROOT=/path  or env CUBLASMp_ROOT. Ships inside the
# NVHPC math_libs tree (…/math_libs/<ver>/targets/<arch>/{include,lib}).
#
# Provides:  CUBLASMp_FOUND, CUBLASMp_INCLUDE_DIR, CUBLASMp_LIBRARY,
#            target  cuBLASMp::cuBLASMp.
#

# CUEV_MP_ROOT: shared hint for the whole nvhpc tree (e.g. .../25.7); the
# math_libs suffixes below reach cuBLASMp from there. Per-package CUBLASMp_ROOT
# (cache or env) is still honored and takes precedence.
find_path(CUBLASMp_INCLUDE_DIR
  NAMES cublasmp.h
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES include math_libs/include libcublasmp/13/include
)

find_library(CUBLASMp_LIBRARY
  NAMES cublasmp
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES lib lib64 math_libs/lib64 math_libs/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUBLASMp
  REQUIRED_VARS CUBLASMp_LIBRARY CUBLASMp_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "Set -DCUBLASMp_ROOT=/path (dir containing include/cublasmp.h)."
)

if(CUBLASMp_FOUND AND NOT TARGET cuBLASMp::cuBLASMp)
  add_library(cuBLASMp::cuBLASMp UNKNOWN IMPORTED)
  set_target_properties(cuBLASMp::cuBLASMp PROPERTIES
    IMPORTED_LOCATION "${CUBLASMp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUBLASMp_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(CUBLASMp_INCLUDE_DIR CUBLASMp_LIBRARY)
