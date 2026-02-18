# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Enable clang-tidy for a specific target.
#
# This function enables clang-tidy analysis only for project-owned targets,
# ensuring that third-party dependencies (Catch2, IREE, CLI11) are never
# analyzed.
#
# Usage:
#   fusilli_enable_clang_tidy(<target-name>)
#
# This should only be called for targets owned by this repository, not for
# imported targets or third-party dependencies.
function(fusilli_enable_clang_tidy target)
  if(NOT FUSILLI_ENABLE_CLANG_TIDY)
    return()
  endif()

  # Ensure clang-tidy is available
  fusilli_find_program(clang-tidy INSTALL_INSTRUCTIONS
    "Please install clang-tidy (e.g., apt install clang-tidy).")

  # Value must be quoted: set_target_properties uses paired argument parsing
  # (PROP VALUE PROP VALUE ...) so each unquoted line becomes a separate
  # argument, and only the first is used as the property value.
  set_target_properties(${target} PROPERTIES
    CXX_CLANG_TIDY
      "clang-tidy;-warnings-as-errors=*;--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
  )
endfunction()


# Find an external tool needed for the fusilli build, and create an imported
# executable target for said tool.
#
# Usage:
#   fusilli_find_program(<tool-name> [QUIET] [INSTALL_INSTRUCTIONS <message>])
#
# fusilli_find_program first checks the FUSILLI_EXTERNAL_<TOOL_NAME> cache
# variable (with <tool-name> converted to ALL CAPS and hyphens replaced with
# underscores), then falls back to find_program.
#
# Options:
#   QUIET - Do not error if the program is not found
#   INSTALL_INSTRUCTIONS - Instructions to install tool if not found
macro(fusilli_find_program TOOL_NAME)
  cmake_parse_arguments(
    ARG                     # prefix
    "QUIET"                 # options
    "INSTALL_INSTRUCTIONS"  # one value keywords
    ""                      # multi-value keywords
    ${ARGN}                 # extra arguments
  )

  # Replace hyphens in tool name with underscores and convert to uppercase.
  # Cache variables can be set through the shell, where hyphens are invalid in
  # variable names.
  string(REPLACE "-" "_" _TOOL_VAR_NAME "${TOOL_NAME}")
  # Yes, TOUPPER argument order is - in fact - the opposite of REPLACE.
  #   string(REPLACE <match> <replace> <output_variable> <input>)
  #   string(TOUPPER <input> <output_variable>)
  string(TOUPPER "${_TOOL_VAR_NAME}" _TOOL_VAR_NAME)
  set(_FULL_VAR_NAME "FUSILLI_EXTERNAL_${_TOOL_VAR_NAME}")

  # Find the tool if not already set.
  if(NOT ${_FULL_VAR_NAME})
    find_program(${_FULL_VAR_NAME} NAMES ${TOOL_NAME})
    # find_program will only set ${_FULL_VAR_NAME} if the program was found.
    if(NOT ${_FULL_VAR_NAME})
      if(NOT ARG_QUIET)
        message(FATAL_ERROR "Could not find '${TOOL_NAME}' in PATH. ${ARG_INSTALL_INSTRUCTIONS}")
      endif()
      return()
    endif()
  endif()

  # Create an imported executable for the tool (only if it doesn't already exist)
  if(NOT TARGET ${TOOL_NAME})
    message(STATUS "Using ${TOOL_NAME}: ${${_FULL_VAR_NAME}}")
    add_executable(${TOOL_NAME} IMPORTED GLOBAL)
    set_target_properties(${TOOL_NAME} PROPERTIES IMPORTED_LOCATION "${${_FULL_VAR_NAME}}")
  endif()
endmacro()
