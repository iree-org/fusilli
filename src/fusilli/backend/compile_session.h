// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains CompileContext and CompileSession for IREE C API-based
// compilation.
//
// CompileContext manages the dynamic library loading and global compiler
// state (loaded once per process).
//
// CompileSession represents a single compilation session that can be
// created from a CompileContext and used to compile modules.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_COMPILE_SESSION_H
#define FUSILLI_BACKEND_COMPILE_SESSION_H

#include "fusilli/backend/backend.h"
#include "fusilli/backend/handle.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/dllib.h"
#include "fusilli/support/logging.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fusilli {

// Forward declarations of IREE compiler API types.
struct iree_compiler_error_t;
struct iree_compiler_session_t;
struct iree_compiler_invocation_t;
struct iree_compiler_source_t;
struct iree_compiler_output_t;

// Forward declaration.
class CompileSession;

// CompileContext manages the IREE compiler shared library and global state.
//
// This class:
// - Dynamically loads the IREE compiler shared library
// - Initializes global compiler state
// - Provides function pointers to IREE compiler API
// - Can create multiple CompileSessions
//
// Usage:
//   ErrorOr<CompileContext*> ctx =
//       CompileContext::create();
//   ErrorOr<CompileSession> session = ctx->createSession(handle);
//   FUSILLI_CHECK_ERROR(session->compile(inputFile, outputFile));
//
// Design principles:
// - Singleton-like behavior (library loaded once)
// - Shared ownership via std::shared_ptr
// - Thread-safe session creation
class CompileContext : public std::enable_shared_from_this<CompileContext> {
public:
  // Factory method to create a compiler context.
  // This will:
  // - Load the IREE compiler shared library
  // - Initialize the compiler global state
  // - Load all required function pointers
  //
  // Returns ErrorOr<CompileContext*> containing the context
  // or error.
  static ErrorOr<CompileContext *> create();

  // Delete copy constructors (shared via shared_ptr).
  CompileContext(const CompileContext &) = delete;
  CompileContext &operator=(const CompileContext &) = delete;
  CompileContext(CompileContext &&) = delete;
  CompileContext &operator=(CompileContext &&) = delete;

  // Destructor - cleans up global state and unloads library.
  ~CompileContext();

  // Creates a new compiler session.
  // Sessions can have different flags and configurations.
  //
  // Returns ErrorOr<CompileSession> containing the session or error.
  ErrorOr<CompileSession> createSession(const Handle &handle);

  // Gets the API version of the loaded compiler.
  int getAPIVersion() const;

  // Gets the revision string of the loaded compiler.
  std::string getRevision() const;

  ErrorObject load(const std::string &libPath) noexcept;

private:
  // Private constructor - use factory method create().
  CompileContext();

  // Loads all required function pointers from the shared library.
  ErrorObject loadSymbols() noexcept;

  friend class CompileSession;

  // Handle to the dynamically loaded library.
  DynamicLibrary lib_;

  // Function pointers for IREE compiler API.
  // Global initialization.
  void (*ireeCompilerGlobalInitialize_)() = nullptr;
  void (*ireeCompilerGlobalShutdown_)() = nullptr;
  int (*ireeCompilerGetAPIVersion_)() = nullptr;
  const char *(*ireeCompilerGetRevision_)() = nullptr;

  // Error handling.
  void (*ireeCompilerErrorDestroy_)(iree_compiler_error_t *) = nullptr;
  const char *(*ireeCompilerErrorGetMessage_)(iree_compiler_error_t *) =
      nullptr;

  // Session management.
  iree_compiler_session_t *(*ireeCompilerSessionCreate_)() = nullptr;
  void (*ireeCompilerSessionDestroy_)(iree_compiler_session_t *) = nullptr;
  iree_compiler_error_t *(*ireeCompilerSessionSetFlags_)(
      iree_compiler_session_t *, int, const char *const *) = nullptr;

  // Invocation management.
  iree_compiler_invocation_t *(*ireeCompilerInvocationCreate_)(
      iree_compiler_session_t *) = nullptr;
  void (*ireeCompilerInvocationDestroy_)(iree_compiler_invocation_t *) =
      nullptr;
  void (*ireeCompilerInvocationEnableConsoleDiagnostics_)(
      iree_compiler_invocation_t *) = nullptr;
  bool (*ireeCompilerInvocationParseSource_)(
      iree_compiler_invocation_t *, iree_compiler_source_t *) = nullptr;
  bool (*ireeCompilerInvocationPipeline_)(iree_compiler_invocation_t *,
                                          int /*pipeline*/) = nullptr;

  // Source management.
  iree_compiler_error_t *(*ireeCompilerSourceOpenFile_)(
      iree_compiler_session_t *, const char *,
      iree_compiler_source_t **) = nullptr;
  void (*ireeCompilerSourceDestroy_)(iree_compiler_source_t *) = nullptr;

  // Output management.
  iree_compiler_error_t *(*ireeCompilerOutputOpenFile_)(
      const char *, iree_compiler_output_t **) = nullptr;
  void (*ireeCompilerOutputKeep_)(iree_compiler_output_t *) = nullptr;
  void (*ireeCompilerOutputDestroy_)(iree_compiler_output_t *) = nullptr;
  iree_compiler_error_t *(*ireeCompilerInvocationOutputVMBytecode_)(
      iree_compiler_invocation_t *, iree_compiler_output_t *) = nullptr;
};

// CompileSession represents a single IREE compiler session for compilation.
//
// This class:
// - Manages an IREE compiler session
// - Allows adding compilation flags
// - Compiles input files to output files
//
// Usage:
//   ErrorOr<CompileSession> session = context->createSession(handle);
//   FUSILLI_CHECK_ERROR(session->addFlag("--iree-hal-target-backends=rocm"));
//   FUSILLI_CHECK_ERROR(session->compile(inputFile, outputFile));
//
// Design principles:
// - Move-only semantics
// - RAII-based resource management
// - All operations return ErrorOr<T> for error handling
class CompileSession {
public:
  // Static factory method matching CompileCommand::build().
  static ErrorOr<CompileSession> build(const Handle &handle,
                                       const CacheFile &input,
                                       const CacheFile &output,
                                       const CacheFile &statistics);

  // Move constructors (RAII pattern).
  CompileSession(CompileSession &&other) noexcept;
  CompileSession &operator=(CompileSession &&other) noexcept;

  // Delete copy constructors (move-only semantics).
  CompileSession(const CompileSession &) = delete;
  CompileSession &operator=(const CompileSession &) = delete;

  // Destructor - cleans up session.
  ~CompileSession();

  // Adds a compilation flag to the session.
  // Flags should be in the form: "--flag-name=value" or "--flag-name"
  //
  // Returns ErrorObject indicating success or failure.
  ErrorObject addFlag(const std::string &flag);

  // Adds multiple compilation flags to the session.
  //
  // Returns ErrorObject indicating success or failure.
  ErrorObject addFlags(std::span<const std::string> flags);

  // Compiles an input file to an output file.
  // This will:
  // - Create a new invocation
  // - Parse the source file
  // - Run the standard compilation pipeline
  // - Output VM bytecode to the output file
  //
  // Returns ErrorObject indicating success or failure.
  ErrorObject compile(std::string_view input, std::string_view output);

  // Serialize to string (for caching/logging).
  std::string toString() const;

  // Write command to cache file.
  ErrorObject writeTo(CacheFile &cacheFile) const;

  // Execute compilation (wrapper around compile()).
  ErrorObject execute();

  // Get arguments (for compatibility/testing).
  const std::vector<std::string> &getArgs() const;

private:
  // Private constructor - use CompileContext::createSession().
  CompileSession(CompileContext *context, iree_compiler_session_t *session,
                 Backend backend);

  // Destroys an IREE compiler error object.
  void destroyError(iree_compiler_error_t *error);

  // Gets the error message from an IREE compiler error object.
  std::string getErrorMessage(iree_compiler_error_t *error);

  friend class CompileContext;

  // Shared pointer to the compiler context (keeps library loaded).
  CompileContext *context_;

  // IREE compiler session.
  iree_compiler_session_t *session_ = nullptr;

  // Backend type for this session.
  Backend backend_;

  // Store input/output paths for execute().
  std::string inputPath_;
  std::string outputPath_;

  // Store flags for toString() and getArgs().
  std::vector<std::string> flags_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_COMPILE_SESSION_H
