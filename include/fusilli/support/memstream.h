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

#ifndef FUSILLI_SUPPORT_MEMSTREAM_H
#define FUSILLI_SUPPORT_MEMSTREAM_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "fusilli/support/target_platform.h"

#ifdef FUSILLI_PLATFORM_LINUX
#include <fcntl.h>
#include <vector>
#elif FUSILLI_PLATFORM_WINDOWS
#include <stdio.h>
#include <windows.h>
#else
#error "MemStream is only implemented for Windows and Linux platforms."
#endif

namespace fusilli {

// A cross-platform memory stream that provides a FILE* interface backed by
// dynamically growing memory. This allows C-style fprintf operations to write
// to an in-memory buffer that can later be retrieved as a std::string.
//
// On Linux/POSIX, this uses the native open_memstream function.
// On Windows, this uses a temporary file.
//
// Example usage:
//   MemStream ms;
//   if (ms.isValid()) {
//     fprintf(ms.stream(), "Hello, %s!", "world");
//     std::string result = ms.str();  // "Hello, world!"
//   }
//
class MemStream {
public:
#ifdef FUSILLI_PLATFORM_WINDOWS
  // Windows implementation using a temporary file.
  MemStream() : stream_(nullptr), tempFilePath_() {
    // Create a temporary file.
    char tempPath[MAX_PATH];
    char tempFileName[MAX_PATH];
    if (GetTempPathA(MAX_PATH, tempPath) == 0) {
      return;
    }
    if (GetTempFileNameA(tempPath, "mem", 0, tempFileName) == 0) {
      return;
    }
    tempFilePath_ = tempFileName;
    stream_ = fopen(tempFilePath_.c_str(), "w+b");
  }

  ~MemStream() {
    if (stream_) {
      fclose(stream_);
      stream_ = nullptr;
    }
    if (!tempFilePath_.empty()) {
      std::remove(tempFilePath_.c_str());
    }
  }

  FILE *stream() { return stream_; }
  bool isValid() const { return stream_ != nullptr; }

  // Retrieve the contents as a string.
  std::string str() {
    if (!stream_)
      return "";

    // Flush pending writes.
    fflush(stream_);

    // Get the size.
    long currentPos = ftell(stream_);
    fseek(stream_, 0, SEEK_END);
    long size = ftell(stream_);
    fseek(stream_, 0, SEEK_SET);

    // Read the contents.
    std::string result(static_cast<size_t>(size), '\0');
    size_t bytesRead =
        fread(result.data(), 1, static_cast<size_t>(size), stream_);
    result.resize(bytesRead);

    // Restore position.
    fseek(stream_, currentPos, SEEK_SET);

    return result;
  }

  // Get current size of the stream.
  size_t size() {
    if (!stream_)
      return 0;
    fflush(stream_);
    long currentPos = ftell(stream_);
    fseek(stream_, 0, SEEK_END);
    long endPos = ftell(stream_);
    fseek(stream_, currentPos, SEEK_SET);
    return static_cast<size_t>(endPos);
  }

private:
  FILE *stream_;
  std::string tempFilePath_;

#elif FUSILLI_PLATFORM_LINUX
  // Linux/POSIX implementation using open_memstream.
  MemStream() : buffer_(nullptr), size_(0), stream_(nullptr) {
    stream_ = open_memstream(&buffer_, &size_);
  }

  ~MemStream() {
    if (stream_) {
      fclose(stream_);
      stream_ = nullptr;
    }
    if (buffer_) {
      free(buffer_);
      buffer_ = nullptr;
    }
  }

  FILE *stream() { return stream_; }
  bool isValid() const { return stream_ != nullptr; }

  // Retrieve the contents as a string.
  std::string str() {
    if (!stream_)
      return "";
    fflush(stream_);
    return std::string(buffer_, size_);
  }

  // Get current size of the stream.
  size_t size() {
    if (!stream_)
      return 0;
    fflush(stream_);
    return size_;
  }

private:
  char *buffer_;
  size_t size_;
  FILE *stream_;
#else
#error "MemStream is only implemented for Windows and Linux platforms."
#endif

public:
  // Implicit conversion to FILE* for convenience.
  operator FILE *() { return stream(); }

  // Delete copy/move operations to prevent resource management issues.
  MemStream(const MemStream &) = delete;
  MemStream &operator=(const MemStream &) = delete;
  MemStream(MemStream &&) = delete;
  MemStream &operator=(MemStream &&) = delete;
};

// An RAII adapter for writing to C++ std::strings using C-style fprints.
// This is useful for interfacing with C APIs that use FILE* for output.
//
// Example usage:
//   std::string output;
//   {
//     FprintAdapter adapter(output);
//     fprintf(adapter, "Value: %d", 42);
//   } // adapter destructor copies content to output
//   // output now contains "Value: 42"
//
class FprintAdapter {
public:
  explicit FprintAdapter(std::string &output) : output_(output), ms_() {}

  ~FprintAdapter() {
    if (ms_.isValid()) {
      output_ = ms_.str();
    }
  }

  FILE *stream() { return ms_.stream(); }
  bool isValid() const { return ms_.isValid(); }

  // Implicit conversion to FILE* for convenience.
  operator FILE *() { return stream(); }

  // Delete copy/move operations.
  FprintAdapter(const FprintAdapter &) = delete;
  FprintAdapter &operator=(const FprintAdapter &) = delete;
  FprintAdapter(FprintAdapter &&) = delete;
  FprintAdapter &operator=(FprintAdapter &&) = delete;

private:
  std::string &output_;
  MemStream ms_;
};

} // namespace fusilli

#endif // FUSILLI_SUPPORT_MEMSTREAM_H
