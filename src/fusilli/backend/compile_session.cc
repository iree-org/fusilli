// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Implementation of CompileContext and CompileSession for IREE C API-based
// compilation. See compile_session.h for class documentation.
//
//===----------------------------------------------------------------------===//

#include "fusilli/backend/compile_session.h"

#include "fusilli/backend/backend.h"
#include "fusilli/backend/handle.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/external_tools.h"
#include "fusilli/support/logging.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fusilli {

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

ErrorOr<CompileContext *> CompileContext::create() {
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

  // Create the context object (can't use make_shared due to private ctor).
  globalInstance = std::unique_ptr<CompileContext>(new CompileContext());

  // Load the shared library.
  ErrorObject loadErr = globalInstance->load(libPath);
  if (isError(loadErr)) {
    std::string error = "Failed to load IREE compiler library: ";
    error += loadErr.getMessage();
    globalInstance.reset();
    return fusilli::error(ErrorCode::CompileFailure, error);
  }

  // Load all required symbols.
  FUSILLI_CHECK_ERROR(globalInstance->loadSymbols());

  // Initialize the compiler.
  FUSILLI_LOG_LABEL_ENDL("INFO: Initializing IREE compiler");
  globalInstance->ireeCompilerGlobalInitialize_();
  return ok(globalInstance.get());
}

ErrorObject CompileContext::load(const std::string &libPath) noexcept {
  return lib_.load(libPath);
}

CompileContext::CompileContext() = default;

CompileContext::~CompileContext() {
  // Intentionally skip ireeCompilerGlobalShutdown.
  //
  // The IREE compiler API permanently disables itself after the final
  // shutdown call, making reinitialization impossible [1]. In plugin
  // scenarios, the hosting plugin may be unloaded and reloaded within the
  // same process (destroying and recreating this static singleton). Skipping
  // shutdown allows reinitialization to succeed regardless of whether dlclose
  // actually unmapped the library or it remained resident.
  //
  // [1]:
  // https://github.com/iree-org/iree/blob/76fa637be19b40ec12bb0ac34e852142fb8604f5/compiler/bindings/c/iree/compiler/embedding_api.h#L74-L77
  if (lib_.isLoaded()) {
    auto err = lib_.close();
    assert(isOk(err) &&
           "Error closing IREE compiler library during destruction");
  }
}

ErrorObject CompileContext::loadSymbols() noexcept {
  FUSILLI_LOG_LABEL_ENDL("INFO: Loading IREE compiler symbols");

#define LOAD_SYMBOL(name)                                                      \
  do {                                                                         \
    FUSILLI_ASSIGN_OR_RETURN(name##_,                                          \
                             lib_.getSymbol<decltype(name##_)>(#name));        \
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

ErrorOr<CompileSession> CompileContext::createSession(const Handle &handle) {
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

int CompileContext::getAPIVersion() const {
  return ireeCompilerGetAPIVersion_();
}

std::string CompileContext::getRevision() const {
  const char *revision = ireeCompilerGetRevision_();
  return revision ? std::string(revision) : "";
}

// ----------------------------------------------------------------------------
// CompileSession implementation
// ----------------------------------------------------------------------------

CompileSession::CompileSession(CompileContext *context,
                               iree_compiler_session_t *session,
                               Backend backend)
    : context_(context), session_(session), backend_(backend) {}

CompileSession::CompileSession(CompileSession &&other) noexcept
    : context_(other.context_), session_(other.session_),
      backend_(other.backend_), inputPath_(std::move(other.inputPath_)),
      outputPath_(std::move(other.outputPath_)),
      flags_(std::move(other.flags_)) {
  other.session_ = nullptr;
}

CompileSession &CompileSession::operator=(CompileSession &&other) noexcept {
  if (this != &other) {
    // Clean up current session.
    if (session_ && context_) {
      context_->ireeCompilerSessionDestroy_(session_);
    }

    // Move from other.
    context_ = other.context_;
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

CompileSession::~CompileSession() {
  if (session_ && context_) {
    context_->ireeCompilerSessionDestroy_(session_);
  }
}

void CompileSession::destroyError(iree_compiler_error_t *error) {
  if (error && context_) {
    context_->ireeCompilerErrorDestroy_(error);
  }
}

std::string CompileSession::getErrorMessage(iree_compiler_error_t *error) {
  if (!error || !context_)
    return "";
  const char *msg = context_->ireeCompilerErrorGetMessage_(error);
  return msg ? std::string(msg) : "";
}

ErrorObject CompileSession::addFlag(const std::string &flag) {
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

ErrorObject CompileSession::addFlags(std::span<const std::string> flags) {
  for (const auto &flag : flags) {
    FUSILLI_CHECK_ERROR(addFlag(flag));
  }
  return ok();
}

ErrorObject CompileSession::compile(std::string_view input,
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

ErrorOr<CompileSession> CompileSession::build(const Handle &handle,
                                              const CacheFile &input,
                                              const CacheFile &output,
                                              const CacheFile &statistics) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Building compile session");

  // Create compiler context.
  FUSILLI_ASSIGN_OR_RETURN(auto *context, CompileContext::create());

  // Create session with backend-specific flags.
  FUSILLI_ASSIGN_OR_RETURN(auto session, context->createSession(handle));

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

std::string CompileSession::toString() const {
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

ErrorObject CompileSession::writeTo(CacheFile &cacheFile) const {
  FUSILLI_LOG_LABEL_ENDL("INFO: Writing compile command to cache");
  return cacheFile.write(toString());
}

ErrorObject CompileSession::execute() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Executing compile session");

  // Call existing compile() method.
  return compile(inputPath_, outputPath_);
}

const std::vector<std::string> &CompileSession::getArgs() const {
  return flags_;
}

} // namespace fusilli
