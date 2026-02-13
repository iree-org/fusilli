// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for finding required external programs at
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
#define FUSILLI_SUPPORT_EXTERNAL_TOOLS_H

#include "fusilli/support/python_utils.h"
#include "fusilli/support/target_platform.h"
#include <cstdlib>
#include <string>

namespace fusilli {

inline std::string getIreeCompilePath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_IREE_COMPILE");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let the shell search for it.
  return std::string("iree-compile");
}

inline std::string getRocmAgentEnumeratorPath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_ROCM_AGENT_ENUMERATOR");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

// Let shell search for it.
#if defined(FUSILLI_PLATFORM_WINDOWS)
  return std::string("rocm-sdk targets");
#else
  return std::string("rocm_agent_enumerator");
#endif
}

inline std::string getAmdSmiPath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_AMD_SMI");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let shell search for it.
  return std::string("amd-smi");
}

inline std::string getIreeCompilerLibPath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_IREE_COMPILER_LIB");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Try to find libIREECompiler.so in Python site-packages.
  auto libPath = findIreeCompilerLib();
  if (libPath.has_value()) {
    return *libPath;
  }

  // Fallback: let the system search for it (may be in LD_LIBRARY_PATH).
#if defined(FUSILLI_PLATFORM_WINDOWS)
  return std::string("IREECompiler.dll");
#else
  return std::string("libIREECompiler.so");
#endif
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
