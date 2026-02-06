# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)

# Avoid in-source builds (prevent build artifacts from cluttering the source directory)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source."
  )
endif()

# Get version number from file
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON_STRING)
string(JSON PACKAGE_VERSION GET ${VERSION_JSON_STRING} package-version)
string(REGEX MATCH "(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*" BASE_VERSION ${PACKAGE_VERSION})

project(
  fusilli
  VERSION ${BASE_VERSION}
  DESCRIPTION "Fusilli: A C++ Graph API and Frontend backed by IREE code-generated kernels"
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
  add_compile_options(/bigobj)
endif()

# Includes
include(CTest)
include(FetchContent)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(FusilliBuildUtils)
include(FusilliTestUtils)

# Build options
option(FUSILLI_BUILD_TESTS "Builds C++ tests and samples" ON)
option(FUSILLI_BUILD_BENCHMARKS "Builds C++ benchmarks" ON)
option(FUSILLI_CODE_COVERAGE "Enable code coverage for tests" OFF)
option(FUSILLI_ENABLE_LOGGING "Enable logging for tests and samples" OFF)
option(FUSILLI_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(FUSILLI_SYSTEMS_AMDGPU  "Builds for AMD GPU systems" OFF)

message(STATUS "Fusilli supported systems:")
if(FUSILLI_SYSTEMS_AMDGPU)
  message(STATUS "  - AMD GPU")
endif()
message(STATUS "  - Host")

################################################################################
# IREE Dependency
#
# Fusilli takes a source dependency on the IREE runtime for interfacing with its
# C-API. The runtime is structured as a modular set of library components designed
# to be (statically) linked into applications directly and compiled with LTO style
# optimizations.
#
# To point to a local IREE source directory, simply specify:
#
#     -DIREE_SOURCE_DIR=</path/to/iree/source>
#
# When not specified, Fusilli will FetchContent the IREE sources from GitHub
# at the tag specified by `IREE_GIT_TAG`.
#
# Fusilli shall not take a source dependency on the IREE compiler, which is quite
# heavy and can be slow to build. It is left to the consumer of Fusilli to either
# bring a prebuilt `libIREECompile.so` via `find_package` for C-API interfacing,
# or side-load a python-installed `iree-compile` binary via `find_program` for
# CLI integration.
#
################################################################################

# Version pins for dependencies.
# Prefer to keep the IREE_GIT_TAG synced with the python-installed IREE version.
# At a minimum, the compiler from those packages must be compatible with the
# runtime at this source ref.
set(IREE_GIT_TAG "iree-3.10.0rc20260114")
set(IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Set IREE build flags.
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
# Disable missing submodules error because we are only building the runtime.
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
# Only enable local_sync/local_task/hip drivers for now.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)
# Enable HAL HIP driver for AMD GPU systems.
if(FUSILLI_SYSTEMS_AMDGPU)
  set(IREE_HAL_DRIVER_HIP ON)
  set(IREE_HIP_TEST_TARGET_CHIP "" CACHE STRING "Enable ROCm testing on specific architecture (e.g. gfx942)")
endif()

if(IREE_SOURCE_DIR)
  message(STATUS "Using existing IREE sources: ${IREE_SOURCE_DIR}")
  add_subdirectory(${IREE_SOURCE_DIR} fusilli_iree SYSTEM EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${IREE_GIT_TAG}")
  set(IREE_SUBMODULES "third_party/benchmark third_party/flatcc third_party/hip-build-deps")
  FetchContent_Declare(
    fusilli_iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG "${IREE_GIT_TAG}"
    GIT_SUBMODULES ${IREE_SUBMODULES}
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(fusilli_iree)
  if(NOT fusilli_iree_POPULATED)
    FetchContent_MakeAvailable(fusilli_iree)
  endif()
endif()

# Defines variables CMAKE_INSTALL_<dir> for common install locations.
include(GNUInstallDirs)

# Create header-only library.
add_library(libfusilli INTERFACE)

# Consumers of exported/installed target use fusilli::fusilli.
set_target_properties(libfusilli PROPERTIES EXPORT_NAME fusilli)

# Create alias for consistent naming in consuming projects building locally (via
# FetchContent) and against an installed version (via find_package).
add_library(fusilli::fusilli ALIAS libfusilli)

# Enable clang-tidy.
fusilli_enable_clang_tidy(libfusilli)

# INTERFACE libraries don't install/export header files added via
# `target_include_directories`. Using a file set property ensures headers are
# available in the installed/exported package.
file(GLOB_RECURSE FUSILLI_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)
target_sources(libfusilli INTERFACE
  FILE_SET fusilli_headers TYPE HEADERS # file sets require lowercase name
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
  FILES ${FUSILLI_HEADERS}
)

# Statically link IREERuntime
target_link_libraries(libfusilli INTERFACE iree_runtime_unified)

# Build Type - Release/Debug
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release build" FORCE)
endif()
if(FUSILLI_CODE_COVERAGE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug build" FORCE)
endif()

# Enable AMD GPU build if requested
if(FUSILLI_SYSTEMS_AMDGPU)
  # Check for Linux AMD GPU device node - `/dev/kfd`
  if(NOT EXISTS "/dev/kfd")
    message(WARNING "No AMD GPU detected (`/dev/kfd` not found) but FUSILLI_SYSTEMS_AMDGPU is set; Only proceed if cross-compiling for AMD GPU systems.")
  endif()
  target_compile_definitions(libfusilli INTERFACE FUSILLI_ENABLE_AMDGPU)
endif()

# Both tests and benchmarks depend on iree-compile, Catch2, and libutils
if(FUSILLI_BUILD_TESTS OR FUSILLI_BUILD_BENCHMARKS)
  # Side-load iree-compile program
  fusilli_find_program(iree-compile INSTALL_INSTRUCTIONS "Please install iree-compile (e.g., pip install iree-base-compiler).")

  # Find prebuilt Catch2 library or fetch it if not found
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found on system: Fetching from GitHub")
    FetchContent_Declare(
      Catch2
      QUIET
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.11.0
      # SYSTEM marks Catch2 headers as system includes, suppressing compiler
      # warnings and clang-tidy diagnostics from third-party code.
      SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  # Add library for tests/utils.h
  add_library(libutils INTERFACE)
  target_include_directories(libutils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(libutils INTERFACE
    libfusilli
    Catch2::Catch2WithMain
  )
  # Enable clang-tidy.
  fusilli_enable_clang_tidy(libutils)
endif()

# Build tests and samples
if(FUSILLI_BUILD_TESTS)
  message(STATUS "Building Fusilli tests and samples")

  # Add tests and samples sub directories
  add_subdirectory(tests)
  add_subdirectory(samples)

  # Activate CTest integration
  enable_testing()
endif()

# Build benchmarks
if(FUSILLI_BUILD_BENCHMARKS)
  message(STATUS "Building Fusilli benchmarks")

  # Get CLI11 (argparse library for C++)
  FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.1
  )
  FetchContent_MakeAvailable(cli11_proj)

  # Add benchmarks sub directory
  add_subdirectory(benchmarks)
endif()

# Add libfusilli target to export set
install(TARGETS libfusilli
  EXPORT FusilliTargets
  FILE_SET fusilli_headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create FusilliTargets.cmake in the install tree from FusilliTargets export set
install(EXPORT FusilliTargets
  FILE FusilliTargets.cmake
  NAMESPACE fusilli::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Create FusilliTargets.cmake in the build tree from FusilliTargets export set,
# allowing for other projects to find_package directly from the build directory
# without installing.
export(TARGETS libfusilli
  FILE ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliTargets.cmake
  NAMESPACE fusilli::
)

# Generate config file in build directory.
#
# configure_package_config_file uses INSTALL_DESTINATION to calculate relative
# paths within the generated config file. But it does NOT actually install the
# file, you must use install(FILES...) for that. INSTALL_DESTINATION must match
# the DESTINATION in the install(FILES...) command below, but CMake doesn't
# verify that for you.
#
# We're generating FusilliConfig.cmake in the build tree, so no extra work is
# needed to make it consumable via find_package directly from the build
# directory, i.e. we don't need an `export(FILE...` call.
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake/templates/FusilliConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Write FusilliConfig.cmake file to install tree
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)
