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

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace fusilli {

// Custom deleter for FILE* from popen
struct PopenDeleter {
  void operator()(FILE *fp) const {
    if (fp)
      pclose(fp);
  }
};

// Executes a shell command and returns the output as a string.
inline std::optional<std::string> execCommand(const std::string &cmd) {
  std::array<char, 128> buffer;
  std::string result;

  std::unique_ptr<FILE, PopenDeleter> pipe(popen(cmd.c_str(), "r"));
  if (!pipe)
    return std::nullopt;

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    result += buffer.data();

  return result;
}

// Gets the list of Python site-packages directories.
// Returns a vector of paths where Python packages are installed.
inline std::vector<std::string> getPythonSitePackages() {
  std::vector<std::string> sitePaths;

  // Try to get site-packages from Python
  const char *pythonCmd =
      "python3 -c \"import site; print('\\n'.join(site.getsitepackages()))\" "
      "2>/dev/null";

  auto output = execCommand(pythonCmd);
  if (!output.has_value() || output->empty())
    return sitePaths;

  // Parse the output - one path per line
  std::string path;
  for (char c : *output) {
    if (c == '\n') {
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

// Searches for libIREECompiler.so in Python site-packages.
// Specifically looks in the iree/compiler/_mlir_libs/ subdirectory
// where it's typically installed by pip.
inline std::optional<std::string> findIreeCompilerLib() {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  static std::optional<std::string> libPath;
  if (libPath.has_value())
    return libPath;

  // Try the standard pip install location
  libPath = findInSitePackages("iree/compiler/_mlir_libs/libIREECompiler.so");
  if (libPath.has_value())
    return libPath;

  // Try alternative locations
  libPath = findInSitePackages("iree_compiler/_mlir_libs/libIREECompiler.so");
  return libPath;
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_PYTHON_UTILS_H
