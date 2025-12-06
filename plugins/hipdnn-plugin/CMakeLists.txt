# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

project(fusilli-plugin
  VERSION 0.1.0
  DESCRIPTION "Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation."
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(FetchContent)

# Global constants
set(FUSILLI_PLUGIN_NAME fusilli_plugin)
set(FUSILLI_PLUGIN_ENGINE_ID 1001)
set(FUSILLI_PLUGIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib/${HIPDNN_PLUGIN_ENGINE_SUBDIR}")

# Useful compile warnings
list(PREPEND FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
    -Wno-gnu-statement-expression            # Disable gnu error for statement expression, this will need to change on Windows.
    -Wswitch-default                         # Warn if a switch statement does not have a default case
)

# Dependencies
set(HIP_PLATFORM "amd")
find_package(hip CONFIG REQUIRED)
find_package(IREERuntime CONFIG REQUIRED)
find_package(hipdnn_sdk CONFIG REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(Fusilli CONFIG REQUIRED)
# Fetch GTest if system install isn't available
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  message(STATUS "GTest not found on system: Fetching from GitHub")
  FetchContent_Declare(
    googletest
    QUIET
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

# Includes
include_directories(include)

# Plugin definition
add_library(${FUSILLI_PLUGIN_NAME} SHARED
    src/fusilli_plugin.cpp
)
target_compile_options(${FUSILLI_PLUGIN_NAME} PRIVATE ${FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS})
target_link_libraries(${FUSILLI_PLUGIN_NAME} PRIVATE hipdnn_sdk hip::host fusilli::fusilli)
target_compile_definitions(${FUSILLI_PLUGIN_NAME} PRIVATE
  FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
  FUSILLI_PLUGIN_ENGINE_ID=${FUSILLI_PLUGIN_ENGINE_ID}
)
set_target_properties(${FUSILLI_PLUGIN_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    LIBRARY_OUTPUT_DIRECTORY "${FUSILLI_PLUGIN_DIR}"
)

# Tests
enable_testing()
add_subdirectory(test)
