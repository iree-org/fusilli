// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains classes for cache file handling of generated artifacts.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_CACHE_H
#define FUSILLI_SUPPORT_CACHE_H

#include "fusilli/support/logging.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace fusilli {

// An RAII type for creating + destroying cache files in
// `${HOME}/.cache/fusilli`.
//
//  void example() {
//    // `remove = true`
//    {
//      // Create ${HOME}/.cache/fusilli/example_graph/input
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/true);
//      assert(isOk(cacheFile));
//
//      assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                  /*filename=*/"input")));
//    }
//    // Try to open the same (now removed) cache file.
//    assert(isError(CacheFile::open(/*graphName=*/"example_graph",
//                                   /*filename=*/"input")));
//
//    // `remove = false`
//    {
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/false);
//      assert(isOk(cacheFile));
//    }
//    // Try to open the same cache file. This time it's found.
//    assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                /*filename=*/"input")));
//  }
class CacheFile {
public:
  // Factory constructor that creates file, overwriting an existing file, and
  // returns an ErrorObject if file could not be created.
  static ErrorOr<CacheFile> create(const std::string &graphName,
                                   const std::string &fileName, bool remove);

  // Factory constructor that opens an existing file and returns ErrorObject if
  // the file does not exist.
  static ErrorOr<CacheFile> open(const std::string &graphName,
                                 const std::string &fileName);

  // Returns the base cache directory path.
  static std::filesystem::path getCacheDir();

  // Utility method to build the path to cache file given `graphName` and
  // `fileName`.
  //
  // Format: ${HOME}/.cache/fusilli/<sanitized version of graphName>/<fileName>
  static std::filesystem::path getPath(const std::string &graphName,
                                       const std::string &fileName);

  // Move constructors.
  CacheFile(CacheFile &&other) noexcept
      : path(std::move(other.path)), remove_(other.remove_) {
    other.path.clear();
    other.remove_ = false;
  }
  CacheFile &operator=(CacheFile &&other) noexcept {
    if (this == &other)
      return *this;
    // If ownership of the cached file is simply changing, we aren't creating a
    // dangling resource that might to be removed.
    bool samePath = path == other.path;
    // Remove current resource if needed.
    if (remove_ && !path.empty() && !samePath)
      std::filesystem::remove(path);
    // Move from other.
    path = std::move(other.path);
    remove_ = other.remove_;
    other.path.clear();
    other.remove_ = false;
    return *this;
  }

  // Delete copy constructors. A copy constructor would likely not be safe, as
  // the destructor for a copy could remove the underlying file while the
  // original is still expecting it to exist.
  CacheFile(const CacheFile &) = delete;
  CacheFile &operator=(const CacheFile &) = delete;

  ~CacheFile() {
    if (remove_ && !path.empty()) {
      std::filesystem::remove(path);
    }
  }

  // Path of file this class wraps.
  std::filesystem::path path;

  // Write to cache file.
  ErrorObject write(const std::string &content);

  // Read contents of cache file.
  ErrorOr<std::string> read();

private:
  // Class should be constructed using one of the factory functions.
  CacheFile(std::filesystem::path path, bool remove)
      : path(std::move(path)), remove_(remove) {}

  // Whether to remove the file on destruction or not.
  bool remove_;
};

// CleanupCacheDirectory removes a sub-directory from the main cache directory
// if it's empty. When used as a base class, C++ destructor ordering (explained
// below) ensures that the directory cleanup in CleanupCacheDirectory destructor
// will happen after any CacheFiles member variables have been destroyed.
//
// Destructor ordering example:
//   struct A   { ~A()  {std::cout << "A";} };
//   struct M1  { ~M1() {std::cout << "M1, ";} };
//   struct M2  { ~M2() {std::cout << "M2, ";} };
//
//   struct B : A {
//       M1 m1;
//       M2 m2;
//       ~B() { std::cout << "B, "; }
//   };
//   // output -> "B, M2, M1, A"
//
// If member destructors (~M1, ~M2) are called inside ~B; the compiler will
// still destroy members afterward, leading to double-destruction (UB).
struct CleanupCacheDirectory {
  std::filesystem::path cacheDir;
  explicit CleanupCacheDirectory(std::filesystem::path dir)
      : cacheDir(std::move(dir)) {}

  ~CleanupCacheDirectory() {
    // This likely indicates the instance in question has been moved from.
    if (cacheDir.empty())
      return;

    if (std::filesystem::exists(cacheDir) &&
        std::filesystem::is_empty(cacheDir))
      std::filesystem::remove(cacheDir);
  }
};

enum class CachedAssetsType : uint8_t {
  Input,
  Output,
  Command,
  Statistics,
};

// Holds cached assets. If `CacheFiles` are set to be removed RAII based removal
// will be tied to the lifetime of this object.
struct CachedAssets : CleanupCacheDirectory {
  CacheFile input;
  CacheFile output;
  CacheFile command;
  CacheFile statistics;

  CachedAssets(CacheFile &&in, CacheFile &&out, CacheFile &&cmd,
               CacheFile &&stats)
      : CleanupCacheDirectory(in.path.parent_path()), input(std::move(in)),
        output(std::move(out)), command(std::move(cmd)),
        statistics(std::move(stats)) {
    // sanity checks:
    assert(input.path.parent_path() == output.path.parent_path() &&
           input.path.parent_path() == command.path.parent_path() &&
           input.path.parent_path() == statistics.path.parent_path() &&
           "Cached assets should be in the same directory.");
    assert(std::filesystem::is_directory(input.path.parent_path()));
  }

  // Default move constructors + destructor.
  CachedAssets(CachedAssets &&) noexcept = default;
  CachedAssets &operator=(CachedAssets &&) noexcept = default;
  ~CachedAssets() = default;

  // Delete copy constructors.
  CachedAssets(const CachedAssets &) = delete;
  CachedAssets &operator=(const CachedAssets &) = delete;
};

} // namespace fusilli

#endif // FUSILLI_SUPPORT_CACHE_H
