# set the output destination
set(SIFTGPU_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/SiftGPU/bin/libsiftgpu.a)
set(USAC_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/USAC/gccRelease/libUSAC.a)
# create a custom target called build_siftgpu that is part of ALL
# and will run each time you type make 
add_custom_target(build_siftgpu ALL 
                   COMMAND make
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/siftgpu
                   COMMENT "Original siftgpu makefile target")
add_custom_target(build_usac ALL
                   COMMAND make
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/USAC
                   COMMENT "Original USAC makefile target")                

# now create an imported static target
add_library(siftgpu STATIC IMPORTED)
add_library(USAC STATIC IMPORTED)
# Import target "siftgpu" for configuration ""
set_property(TARGET siftgpu APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_property(TARGET USAC APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(siftgpu PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${SIFTGPU_LIBRARY}")

set_target_properties(USAC PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${USAC_LIBRARY}")

# now you can use siftgpu as if it were a regular cmake built target in your project
add_dependencies(siftgpu build_siftgpu)
add_dependencies(USAC build_usac)


# note, this will only work on linux/unix platforms, also it does building
# in the source tree which is also sort of bad style and keeps out of source 
# builds from working.  
