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
#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

#include <dlfcn.h>
#include <memory>
#include <span>
#include <sstream>
#include <string>
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

private:
  // Private constructor - use factory method create().
  CompileContext(void *libHandle);

  // Loads all required function pointers from the shared library.
  ErrorObject loadSymbols();

  friend class CompileSession;

  // Handle to the dynamically loaded library.
  void *libHandle_ = nullptr;

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

// ============================================================================
// Implementation (header-only)
// ============================================================================

// Pipeline enumeration matching IREE's iree_compiler_pipeline_t. These are
// taken from the IREE C API headers.
enum IREECompilerPipeline : uint8_t {
  IREE_COMPILER_PIPELINE_STD = 0,
  IREE_COMPILER_PIPELINE_HAL_EXECUTABLE = 1,
  IREE_COMPILER_PIPELINE_PRECOMPILE = 2,
  IREE_COMPILER_PIPELINE_VM = 3,
};

// ----------------------------------------------------------------------------
// CompileContext implementation
// ----------------------------------------------------------------------------

inline ErrorOr<CompileContext *> CompileContext::create() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating IREE compiler context");

  static std::mutex instanceMutex;
  static std::unique_ptr<CompileContext> globalInstance;

  // If multiple threads simultaneously request a handle, they will race to get
  // the globalInstance. The first one to acquire the lock will create it,
  // others will see it's already created
  std::lock_guard<std::mutex> lock(instanceMutex);

  if (globalInstance != nullptr)
    return ok(globalInstance.get());

  // Get the path to the IREE compiler shared library.
  std::string libPath = getIreeCompilerLibPath();
  FUSILLI_LOG_LABEL_ENDL(
      "INFO: Loading IREE compiler library from: " << libPath);

  // Load the shared library.
  void *libHandle =
      dlopen(libPath.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);

  if (!libHandle)
    libHandle = dlmopen(LM_ID_NEWLM, libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (!libHandle) {
    std::string error = "Failed to load IREE compiler library: ";
    error += dlerror();
    return fusilli::error(ErrorCode::CompileFailure, error);
  }

  // Create the context object (can't use make_shared due to private ctor).
  globalInstance =
      std::unique_ptr<CompileContext>(new CompileContext(libHandle));

  // Load all required symbols.
  FUSILLI_CHECK_ERROR(globalInstance->loadSymbols());

  // Initialize the compiler.
  FUSILLI_LOG_LABEL_ENDL("INFO: Initializing IREE compiler");
  globalInstance->ireeCompilerGlobalInitialize_();
  return ok(globalInstance.get());
}

inline CompileContext::CompileContext(void *libHandle)
    : libHandle_(libHandle) {}

inline CompileContext::~CompileContext() {
  if (libHandle_) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Shutting down IREE compiler");
    if (ireeCompilerGlobalShutdown_) {
      ireeCompilerGlobalShutdown_();
    }
    dlclose(libHandle_);
  }
}

inline ErrorObject CompileContext::loadSymbols() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Loading IREE compiler symbols");

#define LOAD_SYMBOL(name)                                                      \
  do {                                                                         \
    name##_ = reinterpret_cast<decltype(name##_)>(dlsym(libHandle_, #name));   \
    if (!name##_) {                                                            \
      std::string error = "Failed to load symbol " #name ": ";                 \
      error += dlerror();                                                      \
      return fusilli::error(ErrorCode::CompileFailure, error);                 \
    }                                                                          \
  } while (false)

  // Load all required symbols.
  LOAD_SYMBOL(ireeCompilerGlobalInitialize);
  LOAD_SYMBOL(ireeCompilerGlobalShutdown);
  LOAD_SYMBOL(ireeCompilerGetAPIVersion);
  LOAD_SYMBOL(ireeCompilerGetRevision);
  LOAD_SYMBOL(ireeCompilerErrorDestroy);
  LOAD_SYMBOL(ireeCompilerErrorGetMessage);
  LOAD_SYMBOL(ireeCompilerSessionCreate);
  LOAD_SYMBOL(ireeCompilerSessionDestroy);
  LOAD_SYMBOL(ireeCompilerSessionSetFlags);
  LOAD_SYMBOL(ireeCompilerInvocationCreate);
  LOAD_SYMBOL(ireeCompilerInvocationDestroy);
  LOAD_SYMBOL(ireeCompilerInvocationEnableConsoleDiagnostics);
  LOAD_SYMBOL(ireeCompilerInvocationParseSource);
  LOAD_SYMBOL(ireeCompilerInvocationPipeline);
  LOAD_SYMBOL(ireeCompilerSourceOpenFile);
  LOAD_SYMBOL(ireeCompilerSourceDestroy);
  LOAD_SYMBOL(ireeCompilerOutputOpenFile);
  LOAD_SYMBOL(ireeCompilerOutputDestroy);
  LOAD_SYMBOL(ireeCompilerOutputKeep);
  LOAD_SYMBOL(ireeCompilerInvocationOutputVMBytecode);

#undef LOAD_SYMBOL

  return ok();
}

inline ErrorOr<CompileSession>
CompileContext::createSession(const Handle &handle) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating compiler session");

  // Create a new IREE compiler session.
  iree_compiler_session_t *session = ireeCompilerSessionCreate_();
  if (!session) {
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to create compiler session");
  }

  // Get the backend type from the handle.
  Backend backend = handle.getBackend();

  // Create the CompileSession object.
  CompileSession compileSession(this, session, backend);

  // Get backend-specific flags and apply them.
  auto flags = getBackendFlags(backend);
  FUSILLI_CHECK_ERROR(compileSession.addFlags(flags));

  return ok(std::move(compileSession));
}

inline int CompileContext::getAPIVersion() const {
  return ireeCompilerGetAPIVersion_();
}

inline std::string CompileContext::getRevision() const {
  const char *revision = ireeCompilerGetRevision_();
  return revision ? std::string(revision) : "";
}

// ----------------------------------------------------------------------------
// CompileSession implementation
// ----------------------------------------------------------------------------

inline CompileSession::CompileSession(CompileContext *context,
                                      iree_compiler_session_t *session,
                                      Backend backend)
    : context_(context), session_(session), backend_(backend) {}

inline CompileSession::CompileSession(CompileSession &&other) noexcept
    : context_(std::move(other.context_)), session_(other.session_),
      backend_(other.backend_), inputPath_(std::move(other.inputPath_)),
      outputPath_(std::move(other.outputPath_)),
      flags_(std::move(other.flags_)) {
  other.session_ = nullptr;
}

inline CompileSession &
CompileSession::operator=(CompileSession &&other) noexcept {
  if (this != &other) {
    // Clean up current session.
    if (session_ && context_) {
      context_->ireeCompilerSessionDestroy_(session_);
    }

    // Move from other.
    context_ = std::move(other.context_);
    session_ = other.session_;
    backend_ = other.backend_;
    inputPath_ = std::move(other.inputPath_);
    outputPath_ = std::move(other.outputPath_);
    flags_ = std::move(other.flags_);

    // Clear the other object's state.
    other.session_ = nullptr;
  }
  return *this;
}

inline CompileSession::~CompileSession() {
  if (session_ && context_) {
    context_->ireeCompilerSessionDestroy_(session_);
  }
}

inline void CompileSession::destroyError(iree_compiler_error_t *error) {
  if (error && context_) {
    context_->ireeCompilerErrorDestroy_(error);
  }
}

inline std::string
CompileSession::getErrorMessage(iree_compiler_error_t *error) {
  if (!error || !context_)
    return "";
  const char *msg = context_->ireeCompilerErrorGetMessage_(error);
  return msg ? std::string(msg) : "";
}

inline ErrorObject CompileSession::addFlag(const std::string &flag) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Adding compiler flag: " << flag);

  const char *flagPtr = flag.c_str();
  iree_compiler_error_t *error =
      context_->ireeCompilerSessionSetFlags_(session_, 1, &flagPtr);

  if (error) {
    std::string errMsg = getErrorMessage(error);
    destroyError(error);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to set flag: " + errMsg);
  }

  // Store flag for toString() and getArgs().
  flags_.push_back(flag);

  return ok();
}

inline ErrorObject
CompileSession::addFlags(std::span<const std::string> flags) {
  for (const auto &flag : flags) {
    FUSILLI_CHECK_ERROR(addFlag(flag));
  }
  return ok();
}

inline ErrorObject CompileSession::compile(std::string_view input,
                                           std::string_view output) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Compiling " << input << " to " << output);

  // Create a new invocation.
  iree_compiler_invocation_t *inv =
      context_->ireeCompilerInvocationCreate_(session_);
  if (!inv) {
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to create compiler invocation");
  }

  // Open the source file.
  iree_compiler_source_t *source = nullptr;
  iree_compiler_error_t *error =
      context_->ireeCompilerSourceOpenFile_(session_, input.data(), &source);
  if (error) {
    std::string errMsg = getErrorMessage(error);
    destroyError(error);
    context_->ireeCompilerInvocationDestroy_(inv);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to open source file: " + errMsg);
  }

  // Parse the source.
  bool parseSuccess = context_->ireeCompilerInvocationParseSource_(inv, source);
  context_->ireeCompilerSourceDestroy_(source);

  if (!parseSuccess) {
    context_->ireeCompilerInvocationDestroy_(inv);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to parse source file");
  }

  // Run the standard compilation pipeline.
  bool pipelineSuccess = context_->ireeCompilerInvocationPipeline_(
      inv, IREE_COMPILER_PIPELINE_STD);
  if (!pipelineSuccess) {
    context_->ireeCompilerInvocationDestroy_(inv);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Compilation pipeline failed");
  }

  // Open the output file.
  iree_compiler_output_t *outputHandle = nullptr;
  error = context_->ireeCompilerOutputOpenFile_(output.data(), &outputHandle);
  if (error) {
    std::string errMsg = getErrorMessage(error);
    destroyError(error);
    context_->ireeCompilerInvocationDestroy_(inv);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to open output file: " + errMsg);
  }

  // Output VM bytecode.
  error = context_->ireeCompilerInvocationOutputVMBytecode_(inv, outputHandle);

  // Specify that the written file should be kept after destroying the output
  context_->ireeCompilerOutputKeep_(outputHandle);

  // Close and destroy the handle.
  context_->ireeCompilerOutputDestroy_(outputHandle);

  // Invocation is no longer required.
  context_->ireeCompilerInvocationDestroy_(inv);

  if (error) {
    std::string errMsg = getErrorMessage(error);
    destroyError(error);
    return fusilli::error(ErrorCode::CompileFailure,
                          "Failed to output VM bytecode: " + errMsg);
  }

  if (!std::filesystem::exists(output) ||
      std::filesystem::file_size(output) == 0) {
    return fusilli::error(ErrorCode::CompileFailure,
                          "Output file was not created or is empty");
  }

  FUSILLI_LOG_LABEL_ENDL("INFO: Compilation successful");
  return ok();
}

// ----------------------------------------------------------------------------
// Interface that matches the build behavior of CompileCommand
// ----------------------------------------------------------------------------

inline ErrorOr<CompileSession>
CompileSession::build(const Handle &handle, const CacheFile &input,
                      const CacheFile &output, const CacheFile &statistics) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Building compile session");

  // Create compiler context.
  auto context = FUSILLI_TRY(CompileContext::create());

  // Create session with backend-specific flags.
  auto session = FUSILLI_TRY(context->createSession(handle));

  // Add statistics flags (matching CompileCommand behavior).
  FUSILLI_CHECK_ERROR(
      session.addFlag("--iree-scheduling-dump-statistics-format=json"));
  FUSILLI_CHECK_ERROR(session.addFlag(
      "--iree-scheduling-dump-statistics-file=" + statistics.path.string()));

  // Store paths for later execution.
  session.inputPath_ = input.path.string();
  session.outputPath_ = output.path.string();

  return ok(std::move(session));
}

inline std::string CompileSession::toString() const {
  // Generate command string similar to CompileCommand.
  // Format: "iree-compile <input> <flags> -o <output>\n"
  std::ostringstream cmdss;

  // Add iree-compile path.
  cmdss << getIreeCompilePath();

  // Add input file.
  if (!inputPath_.empty()) {
    cmdss << " " << inputPath_;
  }

  // Add flags.
  for (const auto &flag : flags_) {
    cmdss << " " << flag;
  }

  // Add output specification.
  if (!outputPath_.empty()) {
    cmdss << " -o " << outputPath_;
  }

  return cmdss.str() + "\n";
}

inline ErrorObject CompileSession::writeTo(CacheFile &cacheFile) const {
  FUSILLI_LOG_LABEL_ENDL("INFO: Writing compile command to cache");
  return cacheFile.write(toString());
}

inline ErrorObject CompileSession::execute() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Executing compile session");

  // Call existing compile() method.
  return compile(inputPath_, outputPath_);
}

inline const std::vector<std::string> &CompileSession::getArgs() const {
  return flags_;
}

} // namespace fusilli

#endif // FUSILLI_BACKEND_COMPILE_SESSION_H
