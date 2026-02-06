// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for interacting with Python environments.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_PYTHON_UTILS_H
#define FUSILLI_SUPPORT_PYTHON_UTILS_H

#include "fusilli/support/process.h"
#include "fusilli/support/target_platform.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace fusilli {

// Gets the list of Python site-packages directories.
// Returns a vector of paths where Python packages are installed.
inline std::vector<std::string> getPythonSitePackages() {
  std::vector<std::string> sitePaths;

  // Try to get site-packages from Python.
#ifdef FUSILLI_PLATFORM_WINDOWS
  const char *pythonCmd =
      "python -c \"import site; print('\\n'.join(site.getsitepackages()))\" "
      "2> NUL";
#elif defined(FUSILLI_PLATFORM_LINUX)
  const char *pythonCmd =
      "python3 -c \"import site; print('\\n'.join(site.getsitepackages()))\" "
      "2>/dev/null";
#else
#error "unknown platform"
#endif

  auto output = execCommand(pythonCmd);
  if (!output.has_value() || output->empty())
    return sitePaths;

  // Parse the output - one path per line.
  // Handle both \n and \r for cross-platform compatibility.
  std::string path;
  for (char c : *output) {
    if (c == '\n' || c == '\r') {
      if (!path.empty()) {
        sitePaths.push_back(path);
        path.clear();
      }
    } else {
      path += c;
    }
  }
  if (!path.empty())
    sitePaths.push_back(path);

  return sitePaths;
}

// Searches for a file in Python site-packages directories.
// Returns the full path to the file if found, otherwise std::nullopt.
inline std::optional<std::string>
findInSitePackages(const std::string &relativePathPattern) {
  auto sitePaths = getPythonSitePackages();

  for (const auto &sitePath : sitePaths) {
    std::filesystem::path fullPath =
        std::filesystem::path(sitePath) / relativePathPattern;

    if (std::filesystem::exists(fullPath))
      return fullPath.string();
  }

  return std::nullopt;
}

// Searches for the IREE compiler library in Python site-packages.
// Specifically looks in the iree/compiler/_mlir_libs/ subdirectory
// where it's typically installed by pip.
inline std::optional<std::string> findIreeCompilerLib() {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  static std::optional<std::string> libPath;
  if (libPath.has_value())
    return libPath;
#ifdef FUSILLI_PLATFORM_WINDOWS
  const char *libRelPath = "iree\\compiler\\_mlir_libs\\IREECompiler.dll";
  const char *libRelPathUnd = "iree_compiler\\_mlir_libs\\IREECompiler.dll";
#elif defined(FUSILLI_PLATFORM_LINUX)
  const char *libRelPath = "iree/compiler/_mlir_libs/libIREECompiler.so";
  const char *libRelPathUnd = "iree_compiler/_mlir_libs/libIREECompiler.so";
#else
#error "unknown platform"
#endif

  // Try the standard pip install location
  libPath = findInSitePackages(libRelPath);
  if (libPath.has_value())
    return libPath;

  // Try alternative locations
  libPath = findInSitePackages(libRelPathUnd);
  return libPath;
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_PYTHON_UTILS_H
