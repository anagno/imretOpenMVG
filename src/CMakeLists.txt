cmake_minimum_required(VERSION 2.6)
project(MasterThesis C CXX)

IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--no-as-needed -ldl -D_GLIBCXX_USE_CXX11_ABI=0")
add_definitions(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

include_directories ("${PROJECT_BINARY_DIR}")

include_directories("${PROJECT_SOURCE_DIR}/imret")

add_subdirectory (imret)

FIND_PACKAGE(OpenMVG REQUIRED)
INCLUDE_DIRECTORIES(${OPENMVG_INCLUDE_DIRS})

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/thirdparty")
add_library(mainwindow  ${PROJECT_SOURCE_DIR}/mainwindow.cpp)
target_link_libraries (mainwindow Qt5::Widgets)
target_link_libraries (mainwindow imretFuncs)

ADD_EXECUTABLE(main main.cpp)
#target_link_libraries(main imretFuncs)
target_link_libraries (main mainwindow)
TARGET_LINK_LIBRARIES(main ${OPENMVG_LIBRARIES})








