# Try to find the CUBHIP library and headers.
#  CUBHIP_FOUND        - system has CUB
#  CUBHIP_INCLUDE_DIRS - the CUB include directory

find_path(CUB_INCLUDE_DIR
	NAMES cub/cub.cuh
	DOC "The directory where CUB includes reside"
)

set(CUBHIP_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUBHIP
	FOUND_VAR CUBHIP_FOUND
	REQUIRED_VARS CUB_INCLUDE_DIR
)

mark_as_advanced(CUBHIP_FOUND)
