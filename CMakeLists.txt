cmake_minimum_required(VERSION 3.18)
project(TRTCorridorKey VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

#-------------------------------------------------------------------------------
# Nuke NDK
#-------------------------------------------------------------------------------
if(NOT DEFINED NUKE_ROOT)
    if(EXISTS "/opt/Nuke17.0v1")
        set(NUKE_ROOT "/opt/Nuke17.0v1")
    elseif(EXISTS "/usr/local/Nuke17.0v1")
        set(NUKE_ROOT "/usr/local/Nuke17.0v1")
    else()
        message(FATAL_ERROR "NUKE_ROOT not set and Nuke not found")
    endif()
endif()

message(STATUS "Nuke root: ${NUKE_ROOT}")

if(NOT EXISTS "${NUKE_ROOT}/include/DDImage/Iop.h")
    message(FATAL_ERROR "Invalid NUKE_ROOT: ${NUKE_ROOT} - DDImage headers not found")
endif()

set(NUKE_INCLUDE_DIRS "${NUKE_ROOT}/include")

find_library(DDIMAGE_LIBRARY
    NAMES DDImage
    PATHS "${NUKE_ROOT}"
    NO_DEFAULT_PATH
    REQUIRED
)

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")

#-------------------------------------------------------------------------------
# TensorRT
#-------------------------------------------------------------------------------
set(TENSORRT_ROOT "/opt/TensorRT-10.15.1.29" CACHE PATH "TensorRT installation root")

if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found in ${TENSORRT_ROOT}/include")
endif()

message(STATUS "TensorRT found: ${TENSORRT_ROOT}")

find_library(NVINFER_LIBRARY
    NAMES nvinfer
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

#-------------------------------------------------------------------------------
# Source files
#-------------------------------------------------------------------------------
set(SOURCES
    TRTCorridorKey.cpp
)

#-------------------------------------------------------------------------------
# Build plugin
#-------------------------------------------------------------------------------
add_library(TRTCorridorKey SHARED ${SOURCES})

target_include_directories(TRTCorridorKey PRIVATE
    ${NUKE_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

target_link_libraries(TRTCorridorKey PRIVATE
    ${DDIMAGE_LIBRARY}
    ${NVINFER_LIBRARY}
    CUDA::cudart
)

#-------------------------------------------------------------------------------
# Linux-specific settings
#-------------------------------------------------------------------------------
set_target_properties(TRTCorridorKey PROPERTIES
    SUFFIX ".so"
    PREFIX ""
)

target_compile_options(TRTCorridorKey PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -fvisibility=hidden
    -O3
)

# RPATH
set_target_properties(TRTCorridorKey PROPERTIES
    INSTALL_RPATH "$ORIGIN:${TENSORRT_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Uncomment the following if needed for compatibility
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------
install(TARGETS TRTCorridorKey
    LIBRARY DESTINATION .
)

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========================================")
message(STATUS "TRTCorridorKey Configuration Summary")
message(STATUS "========================================")
message(STATUS "  Nuke:          ${NUKE_ROOT}")
message(STATUS "  CUDA:          ${CUDAToolkit_VERSION}")
message(STATUS "  TensorRT:      ${TENSORRT_ROOT}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
message(STATUS "")
