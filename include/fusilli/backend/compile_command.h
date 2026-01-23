// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains compilation configuration for Fusilli graphs.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_COMPILE_COMMAND_H
#define FUSILLI_BACKEND_COMPILE_COMMAND_H

#include "fusilli/backend/backend.h"
#include "fusilli/backend/handle.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

// Simple argument escaping for command line serialization.
inline std::string escapeArgument(const std::string &arg) {
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

// CompileCommand encapsulates the construction, serialization, and execution
// of iree-compile commands for Fusilli graph compilation.
//
// Usage:
//   ErrorOr<CompileCommand> cmd = CompileCommand::build(
//       handle, inputFile, outputFile, statisticsFile);
//   FUSILLI_CHECK_ERROR(cmd->writeTo(commandFile));
//   FUSILLI_CHECK_ERROR(cmd->execute());
//
// Design principles:
// - Immutable once constructed (move-only semantics)
// - All operations return ErrorOr<T> for error handling
// - Separates concerns: build, serialize, execute
// - Independently testable
class CompileCommand {
public:
  // Factory method to build a compile command for the given configuration.
  // This constructs the full command with:
  // - iree-compile executable path
  // - input file path
  // - backend-specific flags from Handle
  // - statistics output flags
  // - output file specification
  //
  // Returns CompileCommand containing the built command or error.
  static CompileCommand build(const Handle &handle, const CacheFile &input,
                              const CacheFile &output,
                              const CacheFile &statistics) {
    std::vector<std::string> args = {getIreeCompilePath(), input.path};

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
    args.push_back(output.path);

    return CompileCommand(std::move(args));
  }

  // Move constructors (RAII pattern).
  CompileCommand(CompileCommand &&) noexcept = default;
  CompileCommand &operator=(CompileCommand &&) noexcept = default;

  // Delete copy constructors (move-only semantics).
  CompileCommand(const CompileCommand &) = delete;
  CompileCommand &operator=(const CompileCommand &) = delete;

  ~CompileCommand() = default;

  // Serializes the command to a string representation suitable for:
  // - Writing to cache files
  // - Logging
  // - Execution via std::system()
  //
  // Format: space-separated arguments with trailing newline
  // Example: "iree-compile input.mlir --flag1 --flag2 -o output.vmfb\n"
  std::string toString() const {
    std::ostringstream cmdss;
    interleave(
        args_.begin(), args_.end(),
        // each_fn:
        [&](const std::string &arg) { cmdss << arg; },
        // between_fn:
        [&] { cmdss << " "; });
    return cmdss.str() + "\n";
  }

  // Writes the command to the specified cache file.
  // This is a convenience method for: cacheFile.write(cmd.toString())
  //
  // Returns ErrorObject indicating success or failure.
  ErrorObject writeTo(CacheFile &cacheFile) const {
    FUSILLI_LOG_LABEL_ENDL("INFO: Writing compile command to cache");
    return cacheFile.write(toString());
  }

  // Executes the compile command using std::system().
  //
  // Returns ErrorObject:
  // - ok() if compilation succeeds (return code 0)
  // - error(ErrorCode::CompileFailure, msg) if compilation fails
  //
  // Note: stderr output from failed compilation is currently not captured
  // (see TODO #11 in original code).
  ErrorObject execute() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Executing compile command");

    // TODO(#11): in the error case, std::system will dump to stderr, it would
    // be great to capture this for better logging + reproducer production.
    int returnCode = std::system(toString().c_str());
    FUSILLI_RETURN_ERROR_IF(returnCode, ErrorCode::CompileFailure,
                            "iree-compile command failed");

    return ok();
  }

  // Accessor for testing and validation.
  const std::vector<std::string> &getArgs() const { return args_; }

private:
  // Private constructor - use factory method build().
  explicit CompileCommand(std::vector<std::string> args)
      : args_(std::move(args)) {}

  // Command arguments (e.g., ["iree-compile", "input.mlir", "--flag", ...]).
  std::vector<std::string> args_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_COMPILE_COMMAND_H
