// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the implementation of CompileCommand and related
// utilities for constructing, serializing, and executing iree-compile
// commands for Fusilli graph compilation.
//
//===----------------------------------------------------------------------===//

#include "fusilli/backend/compile_command.h"

#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/target_platform.h"

#include <cstdlib>
#include <sstream>

namespace fusilli {

std::string escapeArgument(const std::string &arg) {
  std::string escaped;
  escaped.reserve(arg.size() + 2);
  escaped += '"';
  for (char c : arg) {
    if (c == '"' || c == '\\' || c == '$' || c == '`') {
      escaped += '\\';
    }
    escaped += c;
  }
  escaped += '"';
  return escaped;
}

CompileCommand CompileCommand::build(const Handle &handle,
                                     const CacheFile &input,
                                     const CacheFile &output,
                                     const CacheFile &statistics) {
  std::vector<std::string> args = {getIreeCompilePath(), input.path.string()};

  // Get backend-specific flags.
  auto flags = getBackendFlags(handle.getBackend());
  for (const auto &flag : flags) {
    std::string escapedFlag = escapeArgument(flag);
    args.push_back(escapedFlag);
  }

  // TODO(#12): Make this conditional (enabled only for testing/debug).
  args.push_back("--iree-scheduling-dump-statistics-format=json");
  args.push_back("--iree-scheduling-dump-statistics-file=" +
                 statistics.path.string());

  // Add output specification.
  args.push_back("-o");
  args.push_back(output.path.string());

  return CompileCommand(std::move(args));
}

std::string CompileCommand::toString() const {
  std::ostringstream cmdss;
  interleave(
      args_.begin(), args_.end(),
      // each_fn:
      [&](const std::string &arg) { cmdss << arg; },
      // between_fn:
      [&] { cmdss << " "; });
#if defined(FUSILLI_PLATFORM_WINDOWS)
  return cmdss.str() + "\r\n";
#else
  return cmdss.str() + "\n";
#endif
}

ErrorObject CompileCommand::writeTo(CacheFile &cacheFile) const {
  FUSILLI_LOG_LABEL_ENDL("INFO: Writing compile command to cache");
  return cacheFile.write(toString());
}

ErrorObject CompileCommand::execute() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Executing compile command");

  // TODO(#11): in the error case, std::system will dump to stderr, it would
  // be great to capture this for better logging + reproducer production.
  int returnCode = std::system(toString().c_str());
  FUSILLI_RETURN_ERROR_IF(returnCode, ErrorCode::CompileFailure,
                          "iree-compile command failed");

  return ok();
}

} // namespace fusilli
