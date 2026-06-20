# FindCAL.cmake — Communication Abstraction Library.
#
# CAL (libcal) bootstraps cuBLASMp's process grid via MPI all-gather callbacks.
# Override with -DCAL_ROOT=/path.
#
# Provides:  CAL_FOUND, CAL_LIBRARY, target  CAL::CAL

find_library(CAL_LIBRARY
  NAMES cal
  HINTS ${CUEV_MP_ROOT} ENV CUEV_MP_ROOT
  PATH_SUFFIXES lib lib64 math_libs/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CAL
  REQUIRED_VARS CAL_LIBRARY
  REASON_FAILURE_MESSAGE "Set -DCAL_ROOT=/path (dir containing lib/libcal.so)."
)

if(CAL_FOUND AND NOT TARGET CAL::CAL)
  add_library(CAL::CAL UNKNOWN IMPORTED)
  set_target_properties(CAL::CAL PROPERTIES IMPORTED_LOCATION "${CAL_LIBRARY}")
endif()

mark_as_advanced(CAL_LIBRARY)
