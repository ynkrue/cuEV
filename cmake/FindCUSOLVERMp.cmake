# FindCUSOLVERMp.cmake — cuSOLVERMp (distributed multi-GPU/multi-node cuSOLVER).
#
# Override with  -DCUSOLVERMp_ROOT=/path  or env CUSOLVERMp_ROOT.
#
# Provides:  CUSOLVERMp_FOUND, CUSOLVERMp_INCLUDE_DIR, CUSOLVERMp_LIBRARY,
#            target  cuSOLVERMp::cuSOLVERMp.

# CUEV_MP_ROOT: shared hint for the whole nvhpc tree (e.g. .../25.7); the
# math_libs suffixes below reach cuSOLVERMp from there. Per-package
# CUSOLVERMp_ROOT (cache or env) is still honored and takes precedence.
find_path(CUSOLVERMp_INCLUDE_DIR
  NAMES cusolverMp.h
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES include math_libs/include
)

find_library(CUSOLVERMp_LIBRARY
  NAMES cusolverMp
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES lib lib64 math_libs/lib64 math_libs/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUSOLVERMp
  REQUIRED_VARS CUSOLVERMp_LIBRARY CUSOLVERMp_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "Set -DCUSOLVERMp_ROOT=/path (dir containing include/cusolverMp.h)."
)

if(CUSOLVERMp_FOUND AND NOT TARGET cuSOLVERMp::cuSOLVERMp)
  add_library(cuSOLVERMp::cuSOLVERMp UNKNOWN IMPORTED)
  set_target_properties(cuSOLVERMp::cuSOLVERMp PROPERTIES
    IMPORTED_LOCATION "${CUSOLVERMp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUSOLVERMp_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(CUSOLVERMp_INCLUDE_DIR CUSOLVERMp_LIBRARY)
