// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains implementations of cache file handling functions for
// generated artifacts.
//
//===----------------------------------------------------------------------===//

#include "fusilli/support/cache.h"

#include "fusilli/support/logging.h"
#include "fusilli/support/target_platform.h" // IWYU pragma: keep

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

#if defined(FUSILLI_PLATFORM_WINDOWS)
#include <KnownFolders.h>
#include <shlobj.h>
#include <windows.h>
#endif

namespace fusilli {

ErrorOr<CacheFile> CacheFile::create(const std::string &graphName,
                                     const std::string &fileName, bool remove) {
  std::filesystem::path path = CacheFile::getPath(graphName, fileName);
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating Cache file");
  FUSILLI_LOG_ENDL(path);

  // Create directory: ${HOME}/.cache/fusilli/<graphName>
  std::filesystem::path cacheDir = path.parent_path();
  std::error_code ec;
  std::filesystem::create_directories(cacheDir, ec);
  FUSILLI_RETURN_ERROR_IF(ec, ErrorCode::FileSystemFailure,
                          "Failed to create cache directory: " +
                              cacheDir.string() + " - " + ec.message());

  // Create file: ${HOME}/.cache/fusilli/<graphName>/<fileName>
  std::ofstream file(path);
  FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                          "Failed to create file: " + path.string());
  file.close();

  return ok(CacheFile(path, remove));
}

ErrorOr<CacheFile> CacheFile::open(const std::string &graphName,
                                   const std::string &fileName) {
  std::filesystem::path path = CacheFile::getPath(graphName, fileName);

  // Check if the file exists.
  FUSILLI_RETURN_ERROR_IF(!std::filesystem::exists(path),
                          ErrorCode::FileSystemFailure,
                          "File does not exist: " + path.string());

  return ok(CacheFile(path, /*remove=*/false));
}

std::filesystem::path CacheFile::getCacheDir() {
  // Defaults to "${HOME}/.cache/fusilli" but having it set via
  // ${FUSILLI_CACHE_DIR} to "/tmp" helps bypass permission issues on
  // the GitHub Actions CI runners as well as for LIT tests that rely
  // on dumping/reading intermediate compilation artifacts to/from disk.
  const char *cacheDirEnv = std::getenv("FUSILLI_CACHE_DIR");
  std::wstring cacheDir;
  if (cacheDirEnv) {
    std::string cacheDirStr(cacheDirEnv);
    cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
  }

#if defined(FUSILLI_PLATFORM_WINDOWS)
  if (cacheDir.empty()) {
    PWSTR pathBuf = nullptr;
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &pathBuf);
    if (SUCCEEDED(hr)) {
      std::string cacheDirStr = std::filesystem::path(pathBuf).string();
      cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
    }
    if (pathBuf)
      CoTaskMemFree(pathBuf);
  }
  return std::filesystem::path(cacheDir) / "fusilli";
#else
  if (cacheDir.empty()) {
    const char *home = std::getenv("HOME");
    if (home) {
      std::string cacheDirStr(home);
      cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
    }
  }
  return std::filesystem::path(cacheDir) / ".cache" / "fusilli";
#endif
}

std::filesystem::path CacheFile::getPath(const std::string &graphName,
                                         const std::string &fileName) {
  // Ensure graphName is safe to use as a directory name, we assume fileName
  // is safe.
  std::string sanitizedGraphName = graphName;
  std::transform(sanitizedGraphName.begin(), sanitizedGraphName.end(),
                 sanitizedGraphName.begin(),
                 [](char c) { return c == ' ' ? '_' : c; });
  std::erase_if(sanitizedGraphName, // NOLINT(misc-include-cleaner)
                [](unsigned char c) {
    return !(std::isalnum(c) || c == '_');
  });

  // Ensure graphName has a value.
  if (sanitizedGraphName.empty())
    sanitizedGraphName = "unnamed_graph";

  std::filesystem::path cacheDir = getCacheDir();
  return cacheDir / sanitizedGraphName / fileName;
}

ErrorObject CacheFile::write(const std::string &content) {
  std::ofstream file(path, std::ios::out | std::ios::binary);
  FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                          "Failed to open file: " + path.string());

  file << content;
  FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                          "Failed to write to file: " + path.string());

  return ok();
}

ErrorOr<std::string> CacheFile::read() {
  // std::ios::ate opens file and moves the cursor to the end, allowing us
  // to get the file size with tellg().
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                          "Failed to open file: " + path.string());

  // Copy the contents of the file into a string.
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(size, '\0');
  file.read(buffer.data(), size);
  FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                          "Failed to read file: " + path.string());

  return ok(buffer);
}

} // namespace fusilli
