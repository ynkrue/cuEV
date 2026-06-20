# FindNCCL.cmake — locate the NVIDIA Collective Communications Library.
#
# Autodetects from standard prefixes, CMAKE_PREFIX_PATH, and
# the user override  -DNCCL_ROOT=/path  or env NCCL_ROOT.
#
# Provides:
#   NCCL_FOUND, NCCL_INCLUDE_DIR, NCCL_LIBRARY
#   imported target  NCCL::NCCL

# CUEV_MP_ROOT: one shared hint for the whole nvhpc tree (e.g. .../25.7),
# consulted by every distributed dependency (NCCL, cuBLASMp, cuSOLVERMp). The
# nvhpc-style suffixes below reach NCCL under its comm_libs subtree. The
# per-package NCCL_ROOT (cache or env) is still honored and takes precedence.
find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES include nccl/include comm_libs/nccl/include
)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES lib lib64 nccl/lib comm_libs/nccl/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "Set -DNCCL_ROOT=/path/to/nccl (dir containing include/nccl.h)."
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL UNKNOWN IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
