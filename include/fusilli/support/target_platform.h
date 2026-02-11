// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILLI_SUPPORT_TARGET_PLATFORM_H
#define FUSILLI_SUPPORT_TARGET_PLATFORM_H

#include <cstdlib>

//==============================================================================
// FUSILLI_PLATFORM_LINUX
//==============================================================================

#if defined(__linux__) || defined(linux) || defined(__linux)
#define FUSILLI_PLATFORM_LINUX 1
#endif // __linux__

//==============================================================================
// FUSILLI_PLATFORM_WINDOWS
//==============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define FUSILLI_PLATFORM_WINDOWS 1
#endif // _WIN32 || _WIN64

#if !defined(FUSILLI_PLATFORM_LINUX) && !defined(FUSILLI_PLATFORM_WINDOWS)
#error Unknown platform.
#endif // all archs

//==============================================================================
// Platform-independent environment variable utilities
//==============================================================================

namespace fusilli {

// Sets an environment variable. Returns 0 on success, non-zero on failure.
inline int setEnv(const char *name, const char *value) {
#if defined(FUSILLI_PLATFORM_WINDOWS)
  return _putenv_s(name, value);
#else
  return setenv(name, value, 1);
#endif
}

// Unsets an environment variable. Returns 0 on success, non-zero on failure.
inline int unsetEnv(const char *name) {
#if defined(FUSILLI_PLATFORM_WINDOWS)
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_TARGET_PLATFORM_H
