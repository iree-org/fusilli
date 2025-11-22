// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
#define FUSILLI_SUPPORT_EXTERNAL_TOOLS_H

#include "fusilli/support/external_tools.h.inc" // generated

#include <cstdlib>
#include <string>
#include <string_view>

namespace fusilli {

inline std::string getIreeCompilePath() {
  constexpr std::string_view path = IREE_COMPILE_PATH;

  // CMake's find_program(VAR ...) populates VAR (aka ${VAR}) as VAR-NOTFOUND.
  if constexpr (path.find("-NOTFOUND") == std::string_view::npos) {
    return std::string(IREE_COMPILE_PATH);
  }

  // Check environment variable (matches CMake variable name).
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_IREE_COMPILE");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let the shell will search for it.
  return "iree-compile";
}

inline std::string getRocmAgentEnumeratorPath() {
  constexpr std::string_view path = ROCM_AGENT_ENUMERATOR_PATH;

  // CMake's find_program(VAR ...) populates VAR (aka ${VAR}) as VAR-NOTFOUND.
  if constexpr (path.find("-NOTFOUND") == std::string_view::npos) {
    return std::string(ROCM_AGENT_ENUMERATOR_PATH);
  }

  // Check environment variable (matches CMake variable name).
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_ROCM_AGENT_ENUMERATOR");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let the shell will search for it.
  return std::string("rocm_agent_enumerator");
}

} // namespace fusilli

#undef IREE_COMPILE_PATH
#undef ROCM_AGENT_ENUMERATOR_PATH

#endif // FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
