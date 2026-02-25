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

#include "fusilli/backend/handle.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/logging.h"

#include <string>
#include <utility>
#include <vector>

namespace fusilli {

// Simple argument escaping for command line serialization.
std::string escapeArgument(const std::string &arg);

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
                              const CacheFile &statistics);

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
  std::string toString() const;

  // Writes the command to the specified cache file.
  // This is a convenience method for: cacheFile.write(cmd.toString())
  //
  // Returns ErrorObject indicating success or failure.
  ErrorObject writeTo(CacheFile &cacheFile) const;

  // Executes the compile command using std::system().
  //
  // Returns ErrorObject:
  // - ok() if compilation succeeds (return code 0)
  // - error(ErrorCode::CompileFailure, msg) if compilation fails
  //
  // Note: stderr output from failed compilation is currently not captured
  // (see TODO #11 in original code).
  ErrorObject execute();

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
