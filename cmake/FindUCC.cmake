# FindUCC.cmake — Unified Collective Communication.
#
# CAL (libcal, required by cuBLASMp/cuSOLVERMp grids) uses UCC as its collective
# backend. In the nvhpc tree UCC ships inside HPC-X under a *versioned* dir
# (comm_libs/<ver>/hpcx/hpcx-*/ucc/lib) with no stable symlink, so we glob for
# the lib directory before searching. Override with -DUCC_ROOT=/path.
#
# Provides:  UCC_FOUND, UCC_LIBRARY, target  UCC::UCC

# Discover candidate ucc/lib dirs under the nvhpc tree (newest hpcx last).
file(GLOB _ucc_hint_dirs
  "${CUEV_MP_ROOT}/comm_libs/hpcx/hpcx-*/ucc/lib"
  "${CUEV_MP_ROOT}/comm_libs/*/hpcx/hpcx-*/ucc/lib"
  "$ENV{CUEV_MP_ROOT}/comm_libs/hpcx/hpcx-*/ucc/lib"
  "$ENV{CUEV_MP_ROOT}/comm_libs/*/hpcx/hpcx-*/ucc/lib"
)
if(_ucc_hint_dirs)
  list(SORT _ucc_hint_dirs)
  list(REVERSE _ucc_hint_dirs)
endif()

find_library(UCC_LIBRARY
  NAMES ucc
  HINTS ${UCC_ROOT} ENV UCC_ROOT ${_ucc_hint_dirs}
  PATH_SUFFIXES lib lib64 ucc/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UCC
  REQUIRED_VARS UCC_LIBRARY
  REASON_FAILURE_MESSAGE "Set -DUCC_ROOT=/path (dir containing lib/libucc.so)."
)

if(UCC_FOUND AND NOT TARGET UCC::UCC)
  add_library(UCC::UCC UNKNOWN IMPORTED)
  set_target_properties(UCC::UCC PROPERTIES IMPORTED_LOCATION "${UCC_LIBRARY}")
endif()

mark_as_advanced(UCC_LIBRARY)
