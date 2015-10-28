FIND_PACKAGE(OpenMVG REQUIRED)
INCLUDE_DIRECTORIES(${OPENMVG_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/thirdparty")
add_library(imretFuncs imretFuncs.hpp) 
TARGET_LINK_LIBRARIES(imretFuncs ${OPENMVG_LIBRARIES})
target_link_libraries(imretFuncs siftgpu)



