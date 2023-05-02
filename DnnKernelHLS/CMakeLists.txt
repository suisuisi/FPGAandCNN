cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
enable_testing()

# Project
set(PROJECT_NAME dnn-kernel)
project(${PROJECT_NAME} LANGUAGES C CXX)


# Default to Debug build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Project settings
set(DNNK_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
set(DNNK_CXX_FLAGS "-g" "-Wall" "-Wno-uninitialized" "-Wno-unused-function" "-Wno-unknown-pragmas")

# Vivado HLS
set(VIVADO_HLS_ROOT "/opt/Xilinx/VivadoHLS/2019.1" CACHE STRING "Path to Vivado HLS root directory")
set(VHLS_INCLUDE_DIRS ${VIVADO_HLS_ROOT}/include)
set(XILINX_XRT "/opt/xilinx/xrt")

# Target board
set(TARGET_BOARD "u200" CACHE STRING "Select target Alveo board (available: \"u200\", \"u250\", \"u280\", \"u50\")")
if (${TARGET_BOARD} STREQUAL "u200")
  set(CHIP_PART "xcu200-fsgd2104-2-e")
  set(VITIS_PLATFORM "/opt/xilinx/platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm")
elseif (${TARGET_BOARD} STREQUAL "u250")
  set(CHIP_PART "xcu250-figd2104-2L-e")
  set(VITIS_PLATFORM "/opt/xilinx/platforms/xilinx_u250_xdma_201830_2/xilinx_u250_xdma_201830_2.xpfm")
elseif (${TARGET_BOARD} STREQUAL "u280")
  set(CHIP_PART "xcu280-fsvh2892-2L-e")
  set(VITIS_PLATFORM "/opt/xilinx/platforms/xilinx_u280-es1_xdma_201910_1/xilinx_u280-es1_xdma_201910_1.xpfm")
elseif (${TARGET_BOARD} STREQUAL "u50")
  set(CHIP_PART "xcu50-fsvh2104-2-e")
  set(VITIS_PLATFORM "/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_201920_3/xilinx_u50_gen3x16_xdma_201920_3.xpfm")
else()
  message(FATAL_ERROR "Unknown TARGET_BOARD value \"${TARGET_BOARD}\"")
endif()

# thirdpartys
include(ExternalProject)
include(FetchContent)

## libtorch
set(LIBTORCH_LOCAL_PATH "filefile://://${CMAKE_SOURCE_DIR}/thirdparty/libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip")
FetchContent_Declare(
  libtorch
  URL ${LIBTORCH_LOCAL_PATH} https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
)
FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  FetchContent_Populate(libtorch)
endif()

list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
set(TORCH_LIBRARY_DIRS ${libtorch_SOURCE_DIR}/lib)
find_package(Torch REQUIRED)

## googletest
set(GTEST_PREFIX ${PROJECT_BINARY_DIR}/thirdparty/googletest)
set(GTEST_INSTALL ${GTEST_PREFIX}/install)
set(GTEST_INCLUDE_DIRS ${GTEST_INSTALL}/include)
set(GTEST_LIBRARY_DIRS ${GTEST_INSTALL}/lib)
set(GTEST_LIBRARIES ${GTEST_INSTALL}/lib/libgtest.a)
set(GTEST_LOCAL_PATH "filefile://://${CMAKE_SOURCE_DIR}/thirdparty/release-1.10.0.zip")

ExternalProject_Add(
  googletest
  PREFIX ${GTEST_PREFIX}
  URL ${GTEST_LOCAL_PATH} https://github.com/google/googletest/archive/release-1.10.0.zip
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL} -DCMAKE_BUILD_TYPE=Release
  BUILD_BYPRODUCTS ${GTEST_LIB}
)

# Logging
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Path to Vivado HLS: ${VIVADO_HLS_ROOT}")

# tests
add_subdirectory(tests)
add_subdirectory(host)
