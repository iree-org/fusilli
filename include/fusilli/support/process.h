// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains a cross-platform process utility for executing shell
// commands and capturing their output.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_PROCESS_H
#define FUSILLI_SUPPORT_PROCESS_H

#include "fusilli/support/target_platform.h"

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>

namespace fusilli {

// Custom deleter for FILE* from popen
struct PopenDeleter {
  void operator()(FILE *fp) const {
    if (fp)
#ifdef FUSILLI_PLATFORM_WINDOWS
      _pclose(fp);
#else
      pclose(fp);
#endif
  }
};

// Executes a shell command and returns the output as a string.
inline std::optional<std::string> execCommand(const std::string &cmd) {
  std::array<char, 128> buffer;
  std::string result;

#ifdef FUSILLI_PLATFORM_WINDOWS
  FILE *opened = _popen(cmd.c_str(), "r");
#elif FUSILLI_PLATFORM_LINUX
  FILE *opened = popen(cmd.c_str(), "r");
#else
#error "Unsupported platform"
#endif
  std::unique_ptr<FILE, PopenDeleter> pipe(opened);
  if (!pipe)
    return std::nullopt;

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    result += buffer.data();

  return result;
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_PROCESS_H
