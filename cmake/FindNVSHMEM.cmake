# FindNVSHMEM.cmake — NVSHMEM host library.
#
# cuBLASMp uses NVSHMEM for its device-side communication. Override
# with -DNVSHMEM_ROOT=/path, or rely on the shared -DCUEV_MP_ROOT
# (nvhpc tree, stable comm_libs/nvshmem symlink).
#
# Provides:  NVSHMEM_FOUND, NVSHMEM_LIBRARY, target  NVSHMEM::NVSHMEM

find_library(NVSHMEM_LIBRARY
  NAMES nvshmem_host
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES lib lib64 comm_libs/nvshmem/lib nvshmem/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
  REQUIRED_VARS NVSHMEM_LIBRARY
  REASON_FAILURE_MESSAGE "Set -DNVSHMEM_ROOT=/path (dir containing lib/libnvshmem_host.so)."
)

if(NVSHMEM_FOUND AND NOT TARGET NVSHMEM::NVSHMEM)
  add_library(NVSHMEM::NVSHMEM UNKNOWN IMPORTED)
  set_target_properties(NVSHMEM::NVSHMEM PROPERTIES IMPORTED_LOCATION "${NVSHMEM_LIBRARY}")
endif()

mark_as_advanced(NVSHMEM_LIBRARY)
