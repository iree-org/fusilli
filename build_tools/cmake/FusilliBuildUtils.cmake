# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find an external tool and create an imported executable target.
#
# Usage:
#   fusilli_find_program(<tool-name> [REQUIRED] [ERROR_MESSAGE <message>])
#
# Options:
#   REQUIRED        - Fail with FATAL_ERROR if tool not found
#   ERROR_MESSAGE   - Custom error message to display if tool not found
macro(fusilli_find_program TOOL_NAME)
  cmake_parse_arguments(
    ARG                    # prefix
    "REQUIRED"             # options
    "ERROR_MESSAGE"        # one value keywords
    ""                     # multi-value keywords
    ${ARGN}                # extra arguments
  )

  # Replace hyphens in tool name with underscores and convert to uppercase.
  # Cache variables can be set through the shell, where hyphens are invalid in variable names.
  string(REPLACE "-" "_" _TOOL_VAR_NAME "${TOOL_NAME}")
  # Yes, TOUPPER argument order is - in fact - the opposite of REPLACE.
  #   string(REPLACE <match> <replace> <output_variable> <input>)
  #   string(TOUPPER <input> <output_variable>)
  string(TOUPPER "${_TOOL_VAR_NAME}" _TOOL_VAR_NAME)
  set(_FULL_VAR_NAME "FUSILLI_EXTERNAL_${_TOOL_VAR_NAME}")

  # Find the tool if not already set.
  if(NOT ${_FULL_VAR_NAME})
    find_program(${_FULL_VAR_NAME} NAMES ${TOOL_NAME})
  endif()

  # When find_program(VAR ...) fails it sets VAR value (aka ${VAR}) to
  # VAR-NOTFOUND.
  if(NOT ${${_FULL_VAR_NAME}} MATCHES "-NOTFOUND$")
    message(STATUS "Using ${TOOL_NAME}: ${${_FULL_VAR_NAME}}")
    add_executable(${TOOL_NAME} IMPORTED GLOBAL)
    set_target_properties(${TOOL_NAME} PROPERTIES IMPORTED_LOCATION "${${_FULL_VAR_NAME}}")
  else()
    if(ARG_REQUIRED)
      message(FATAL_ERROR "Could not find '${TOOL_NAME}' in PATH. ${ARG_ERROR_MESSAGE}")
    else()
      message(WARNING
        "${TOOL_NAME} not on PATH during compilation. At runtime, ${TOOL_NAME} must be available "
        "on PATH or provided through ${_FULL_VAR_NAME} environment variable.")
    endif()
  endif()
endmacro()
