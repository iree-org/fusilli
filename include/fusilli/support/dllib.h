// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file provides a cross-platform dynamic library loading utility.
//
// The DynamicLibrary class provides a simple interface for loading shared
// libraries, retrieving symbols, and unloading libraries. The implementation
// is selected at compile time based on the target platform:
// - Linux (glibc) systems: Uses dlmopen/dlsym/dlclose
// - Windows: Uses LoadLibraryEx/GetProcAddress/FreeLibrary
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_DLLIB_H
#define FUSILLI_SUPPORT_DLLIB_H

#include "fusilli/support/logging.h"
#include "fusilli/support/target_platform.h"
#include <string>

#ifdef FUSILLI_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fusilli {

/// DynamicLibrary provides a cross-platform interface for loading and
/// interacting with dynamic/shared libraries.
///
/// Usage:
///   DynamicLibrary lib;
///   ErrorObject loadErr = lib.load("/path/to/library.so");
///   if (isError(loadErr)) {
///     std::cerr << "Error: " << loadErr << std::endl;
///     return;
///   }
///
///   ErrorOr<int(*)(int)> funcPtr = lib.getSymbol<int(*)(int)>("myFunction");
///   if (isError(funcPtr)) {
///     std::cerr << "Symbol not found: " << ErrorObject(funcPtr) << std::endl;
///     return;
///   }
///
///   int result = (*funcPtr)(42);
///   lib.close();
///
class DynamicLibrary {
public:
  DynamicLibrary() = default;

  // Non-copyable.
  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;

  // Movable.
  DynamicLibrary(DynamicLibrary &&other) : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  DynamicLibrary &operator=(DynamicLibrary &&other) {
    if (this != &other) {
      auto err = close();
      assert(isOk(err) && "Error closing library during move assignment");
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  ~DynamicLibrary() {
    auto err = close();
    assert(isOk(err) && "Error closing library during destruction");
  }

  /// Loads a dynamic library from the specified path.
  ///
  /// On POSIX systems, this uses dlmopen with LM_ID_NEWLM to load the library
  /// in a new namespace, providing better isolation.
  ///
  /// On Windows, this uses LoadLibraryEx.
  ///
  /// Returns ErrorObject indicating success or failure.
  ErrorObject load(const std::string &path) {
    if (handle_) {
      auto err = close();
      assert(isOk(err) && "Error closing library during load");
    }

#ifdef FUSILLI_PLATFORM_WINDOWS
    handle_ = LoadLibraryExA(path.c_str(), nullptr, 0);
    if (!handle_) {
      DWORD errorCode = GetLastError();
      char *buffer = nullptr;
      FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          nullptr, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
      std::string errMsg;
      if (buffer) {
        errMsg = buffer;
        LocalFree(buffer);
      } else {
        errMsg = "Unknown error (code: " + std::to_string(errorCode) + ")";
      }
      return error(ErrorCode::FileSystemFailure, errMsg);
    }
#elif FUSILLI_PLATFORM_LINUX
    // Use dlmopen with LM_ID_NEWLM to load in a new namespace for isolation. We
    // use a separate namespace to force reinistialization if another library
    // loaded and shutdown already.
    handle_ = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      const char *err = dlerror();
      return error(ErrorCode::FileSystemFailure, err ? err : "Unknown error");
    }
#else
#error Unsupported platform
#endif

    return ok();
  }

  /// Retrieves a symbol from the loaded library.
  ///
  /// Returns ErrorOr<T> containing the symbol pointer on success, or an error
  /// if the symbol is not found or the library is not loaded.
  template <typename T> ErrorOr<T> getSymbol(const char *name) {
    static_assert(std::is_pointer_v<T>, "T must be a pointer type");
    if (!handle_) {
      return error(ErrorCode::InternalError, "Library not loaded");
    }

#ifdef FUSILLI_PLATFORM_WINDOWS
    void *sym = reinterpret_cast<void *>(GetProcAddress(handle_, name));
    if (!sym) {
      DWORD errorCode = GetLastError();
      char *buffer = nullptr;
      FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          nullptr, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
      std::string errMsg;
      if (buffer) {
        errMsg = buffer;
        LocalFree(buffer);
      } else {
        errMsg = "Symbol not found: " + std::string(name);
      }
      return error(ErrorCode::InternalError, errMsg);
    }
#elif FUSILLI_PLATFORM_LINUX
    // Clear any existing error.
    dlerror();

    void *sym = dlsym(handle_, name);

    if (!sym) {
      const char *err = dlerror();
      std::string errMsg = "Symbol not found: " + std::string(name);
      if (err) {
        errMsg += ": ";
        errMsg += err;
      }
      return error(ErrorCode::InternalError, errMsg);
    }
#else
#error Unsupported platform
#endif

    return ok(reinterpret_cast<T>(sym));
  }

  /// Closes the loaded library.
  ///
  /// Safe to call multiple times or on an unloaded library.
  ErrorObject close() {
    if (handle_) {
#ifdef FUSILLI_PLATFORM_WINDOWS
      FreeLibrary(handle_);
#elif FUSILLI_PLATFORM_LINUX
      dlclose(handle_);
#else
#error Unsupported platform
#endif
      handle_ = nullptr;
    }
    return ok();
  }

  /// Returns true if a library is currently loaded.
  bool isLoaded() const { return handle_ != nullptr; }

private:
#ifdef FUSILLI_PLATFORM_WINDOWS
  HMODULE handle_ = nullptr;
#elif FUSILLI_PLATFORM_LINUX
  void *handle_ = nullptr;
#else
#error Unsupported platform
#endif
};

} // namespace fusilli

#endif // FUSILLI_SUPPORT_DLLIB_H
