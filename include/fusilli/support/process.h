// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains a cross-platform memory stream utility that provides
// a FILE* interface backed by memory. On Linux, it uses open_memstream.
// On Windows, it uses a pure C++ implementation with temporary files.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_PROCESS_H
#define FUSILLI_SUPPORT_PROCESS_H

#include "fusilli/support/target_platform.h"

#include <array>

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
#else
  FILE *opened = popen(cmd.c_str(), "r");
#endif
  std::unique_ptr<FILE, PopenDeleter> pipe(opened);
  if (!pipe)
    return std::nullopt;

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    result += buffer.data();

  return result;
}

#endif // FUSILLI_SUPPORT_PROCESS_H
