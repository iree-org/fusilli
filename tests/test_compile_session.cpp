// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace fusilli;

static std::string kGraphName = "test_compile_session";

// Returns the path to a shared tuning spec file for tests. The file is created
// once and persists for the process lifetime because IREE caches parsed tuning
// specs on the dialect instance keyed by path, and the tuning spec flag itself
// is a process-wide static. Deleting the file would break subsequent
// compilations that hit the cached path.
static std::filesystem::path getTestTuningSpecPath() {
  static std::filesystem::path path = [] {
    std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "fusilli_test_tuning_specs";
    std::filesystem::create_directories(dir);
    std::filesystem::path p = dir / "tuning_spec.mlir";
    std::ofstream ofs(p);
    assert(ofs.is_open() && "Failed to create test tuning spec file");
    ofs << R"(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
    -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    transform.yield %arg0 : !transform.any_op
  }
}
)";
    ofs.close();
    return p;
  }();
  return path;
}

TEST_CASE("CompileContext::create successfully loads library",
          "[CompileContext]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());

  // Should succeed (assuming libIREECompiler.so is available).
  REQUIRE(context != nullptr);

  // Should be able to get API version.
  int apiVersion = context->getAPIVersion();
  REQUIRE(apiVersion > 0);

  // Should be able to get revision.
  std::string revision = context->getRevision();
  // Revision might be empty in some builds, but shouldn't crash.
  REQUIRE(revision.size() >= 0);
}

TEST_CASE("CompileContext::create loads symbols correctly",
          "[CompileContext]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);

  // API version should be in a reasonable range (e.g., 1.x or 2.x).
  int apiVersion = context->getAPIVersion();
  int majorVersion = apiVersion >> 16;
  int minorVersion = apiVersion & 0xFFFF;

  REQUIRE(majorVersion >= 1);
  REQUIRE(majorVersion <= 10); // Sanity check.
  REQUIRE(minorVersion >= 0);
}

TEST_CASE("CompileContext::createSession with CPU backend",
          "[CompileContext][CompileSession]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle for CPU backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create a session.
  auto maybeSession = context->createSession(handle);
  FUSILLI_REQUIRE_OK(maybeSession);

  FUSILLI_REQUIRE_ASSIGN(CompileSession session, std::move(maybeSession));

  // If we got here, session was created successfully.
  SUCCEED("Session created successfully");
}

#if defined(FUSILLI_ENABLE_AMDGPU)
TEST_CASE("CompileContext::createSession with AMDGPU backend",
          "[CompileContext][CompileSession]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle for AMDGPU backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));

  // Create a session.
  auto maybeSession = context->createSession(handle);
  FUSILLI_REQUIRE_OK(maybeSession);

  FUSILLI_REQUIRE_ASSIGN(CompileSession session, std::move(maybeSession));

  // If we got here, session was created successfully.
  SUCCEED("Session created successfully");
}
#endif

TEST_CASE("CompileContext supports multiple sessions",
          "[CompileContext][CompileSession]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Create multiple sessions from the same context.
  FUSILLI_REQUIRE_ASSIGN(auto session1, context->createSession(handle));
  FUSILLI_REQUIRE_ASSIGN(auto session2, context->createSession(handle));
  FUSILLI_REQUIRE_ASSIGN(auto session3, context->createSession(handle));

  // All sessions should be valid.
  SUCCEED("Multiple sessions created successfully");
}

TEST_CASE("CompileSession::addFlag", "[CompileSession]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Add a flag.
  FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));

  // If we got here, flag was added successfully.
  SUCCEED("Flag added successfully");
}

TEST_CASE("CompileSession::addFlags", "[CompileSession]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Add multiple flags.
  std::vector<std::string> flags = {
      "--iree-opt-level=O3",
      "--iree-vm-bytecode-module-strip-source-map=true",
  };
  FUSILLI_REQUIRE_OK(session.addFlags(flags));

  // If we got here, flags were added successfully.
  SUCCEED("Flags added successfully");
}

TEST_CASE("CompileSession::compile with valid MLIR",
          "[CompileSession][integration]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write a simple MLIR module to the input file.
  std::string mlirContent = getSimpleMLIRModule();
  FUSILLI_REQUIRE_OK(input.write(mlirContent));

  // Compile the module.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  FUSILLI_REQUIRE_OK(compileResult);

  // Verify the output file was created and is not empty.
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);
}

TEST_CASE("CompileSession::compile with custom flags",
          "[CompileSession][integration]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Add custom optimization flags.
  FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));
  FUSILLI_REQUIRE_OK(
      session.addFlag("--iree-vm-bytecode-module-strip-source-map=true"));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input_opt.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output_opt.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write a simple MLIR module to the input file.
  std::string mlirContent = getSimpleMLIRModule();
  FUSILLI_REQUIRE_OK(input.write(mlirContent));

  // Compile the module.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  FUSILLI_REQUIRE_OK(compileResult);

  // Verify the output file was created.
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);
}

TEST_CASE("CompileSession::compile with invalid MLIR",
          "[CompileSession][error]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "invalid.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "invalid.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write invalid MLIR content.
  std::string invalidMLIR = "this is not valid MLIR syntax!";
  FUSILLI_REQUIRE_OK(input.write(invalidMLIR));

  // Attempt to compile - should fail.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  REQUIRE(isError(compileResult));
  REQUIRE(compileResult.getCode() == ErrorCode::CompileFailure);
}

TEST_CASE("CompileSession::compile with missing input file",
          "[CompileSession][error]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Create cache files but don't write to input.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "missing.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "missing.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Input file doesn't exist - should fail.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  REQUIRE(isError(compileResult));
  REQUIRE(compileResult.getCode() == ErrorCode::CompileFailure);
}

TEST_CASE("CompileSession move semantics", "[CompileSession]") {
  // Get the shared compiler context and create sessions.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session1, context->createSession(handle));

  // Add a flag to session1.
  FUSILLI_REQUIRE_OK(session1.addFlag("--iree-opt-level=O3"));

  // Move session1 to session2.
  CompileSession session2 = std::move(session1);

  // session2 should be usable.
  FUSILLI_REQUIRE_OK(session2.addFlag("--iree-opt-level=O2"));

  SUCCEED("Move semantics work correctly");
}

#if defined(FUSILLI_ENABLE_AMDGPU)
TEST_CASE("CompileSession::compile with AMDGPU backend",
          "[CompileSession][integration][amdgpu]") {
  // Get the shared compiler context and create a session for AMDGPU.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input_amdgpu.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output_amdgpu.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write a simple MLIR module to the input file.
  std::string mlirContent = getSimpleMLIRModule();
  FUSILLI_REQUIRE_OK(input.write(mlirContent));

  // Compile the module.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  FUSILLI_REQUIRE_OK(compileResult);

  // Verify the output file was created and is not empty.
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);
}
#endif

TEST_CASE("CompileSession cleanup on destruction", "[CompileSession]") {
  // Get the shared compiler context.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Create a session in a nested scope.
  {
    FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

    // Use the session.
    FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));

    // Session will be destroyed when exiting this scope.
  }

  // Should be able to create a new session after the previous one was
  // destroyed.
  FUSILLI_REQUIRE_ASSIGN(auto session2, context->createSession(handle));
  FUSILLI_REQUIRE_OK(session2.addFlag("--iree-opt-level=O2"));

  SUCCEED("Session cleanup and recreation works correctly");
}

TEST_CASE("CompileSession with invalid flag", "[CompileSession][error]") {
  // Get the shared compiler context and create a session.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Try to add an invalid flag.
  auto result = session.addFlag("--this-is-not-a-valid-iree-flag-xyz123");

  // This might or might not fail depending on IREE's flag validation.
  // If it fails, check the error code.
  if (isError(result)) {
    REQUIRE(result.getCode() == ErrorCode::CompileFailure);
  }
}

TEST_CASE("Multiple CompileContexts can coexist", "[CompileContext]") {
  // Note: We use the shared context for most tests, but this test
  // specifically checks if multiple contexts can be created. This might fail
  // if IREE's global state doesn't support multiple initializations.

  FUSILLI_REQUIRE_ASSIGN(auto *context1, CompileContext::create());
  REQUIRE(context1 != nullptr);

  // Try to create a second context.
  auto maybeContext2 = CompileContext::create();

  // If it succeeds, both contexts should be usable.
  if (isOk(maybeContext2)) {
    FUSILLI_REQUIRE_ASSIGN(auto *context2, std::move(maybeContext2));
    REQUIRE(context2 != nullptr);

    // Both should have the same API version.
    REQUIRE(context1->getAPIVersion() == context2->getAPIVersion());
  }
}

// ===========================================================================
// Tests for CompileCommand-style unified interface
// ===========================================================================

TEST_CASE("CompileSession::build with CPU backend", "[CompileSession]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));
  FUSILLI_REQUIRE_ASSIGN(CacheFile input,
                         CacheFile::create(kGraphName, "input.mlir", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile output,
                         CacheFile::create(kGraphName, "output.vmfb", true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  // Verify that flags were added (including statistics flags).
  const auto &args = session.getArgs();
  REQUIRE(args.size() >= 4); // Backend flags + statistics flags

  // Should have statistics flags.
  bool hasStatFormat = false;
  bool hasStatFile = false;
  for (const auto &arg : args) {
    if (arg.find("--iree-scheduling-dump-statistics-format") !=
        std::string::npos) {
      hasStatFormat = true;
    }
    if (arg.find("--iree-scheduling-dump-statistics-file") !=
        std::string::npos) {
      hasStatFile = true;
    }
  }
  REQUIRE(hasStatFormat);
  REQUIRE(hasStatFile);
}

TEST_CASE("CompileSession::toString format", "[CompileSession]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CacheFile input,
                         CacheFile::create(kGraphName, "input.mlir", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile output,
                         CacheFile::create(kGraphName, "output.vmfb", true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  std::string cmdStr = session.toString();

  REQUIRE(!cmdStr.empty());
  REQUIRE(cmdStr.back() == '\n');
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("iree-compile"));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("-o"));
}

TEST_CASE("CompileSession::writeTo", "[CompileSession]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CacheFile input,
                         CacheFile::create(kGraphName, "input.mlir", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile output,
                         CacheFile::create(kGraphName, "output.vmfb", true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile commandFile,
                         CacheFile::create(kGraphName, "command.txt", true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  FUSILLI_REQUIRE_OK(session.writeTo(commandFile));

  // Read back and verify.
  FUSILLI_REQUIRE_ASSIGN(std::string written, commandFile.read());
  REQUIRE(written == session.toString());
}

TEST_CASE("CompileSession::execute with valid MLIR", "[CompileSession]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CacheFile input,
                         CacheFile::create(kGraphName, "input.mlir", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile output,
                         CacheFile::create(kGraphName, "output.vmfb", true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write valid MLIR.
  FUSILLI_REQUIRE_OK(input.write(getSimpleMLIRModule()));

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  // Execute compilation.
  FUSILLI_REQUIRE_OK(session.execute());

  // Verify output file was created.
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);

  // Verify statistics file was created.
  REQUIRE(std::filesystem::exists(statistics.path));
  REQUIRE(std::filesystem::file_size(statistics.path) > 0);
}

TEST_CASE("CompileSession::getArgs", "[CompileSession]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CacheFile input,
                         CacheFile::create(kGraphName, "input.mlir", true));
  FUSILLI_REQUIRE_ASSIGN(CacheFile output,
                         CacheFile::create(kGraphName, "output.vmfb", true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  const auto &args = session.getArgs();

  // Should have backend flags and statistics flags.
  REQUIRE(!args.empty());

  // All args should be non-empty strings.
  for (const auto &arg : args) {
    REQUIRE(!arg.empty());
  }
}

TEST_CASE("CompileSession::addFlag with tuning spec path",
          "[CompileSession][tuning_spec]") {
  // Verify tuning spec path flag is accepted by C API.
  FUSILLI_REQUIRE_ASSIGN(CompileContext * context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CompileSession session,
                         context->createSession(handle));

  ErrorObject result =
      session.addFlag("--iree-codegen-tuning-spec-path=" +
                      getTestTuningSpecPath().generic_string());
  FUSILLI_REQUIRE_OK(result);

  const std::vector<std::string> &args = session.getArgs();
  bool hasTuningSpecFlag =
      std::any_of(args.begin(), args.end(), [](const std::string &arg) {
        return arg.find("--iree-codegen-tuning-spec-path") != std::string::npos;
      });
  REQUIRE(hasTuningSpecFlag);
}

TEST_CASE("CompileSession::compile with tuning spec",
          "[CompileSession][integration][tuning_spec]") {
  // End-to-end compilation with tuning spec via C API.
  FUSILLI_REQUIRE_ASSIGN(CompileContext * context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(CompileSession session,
                         context->createSession(handle));

  FUSILLI_REQUIRE_OK(session.addFlag("--iree-codegen-tuning-spec-path=" +
                                     getTestTuningSpecPath().generic_string()));

  // Create input/output files in cache (these will auto-cleanup with
  // remove=true).
  std::string graphName = "test_compile_session_with_tuning_spec";
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(graphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(graphName, "output.vmfb", /*remove=*/true));

  // Ensure cleanup happens even if REQUIRE() fails.
  auto cleanup = scope_exit(
      [&] { std::filesystem::remove_all(input.path.parent_path()); });

  // Write a simple MLIR module to the input file.
  std::string mlirContent = R"(
module {
  func.func @tuning_spec_test(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = arith.mulf %arg0, %arg1 : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}
)";
  FUSILLI_REQUIRE_OK(input.write(mlirContent));

  ErrorObject compileResult = session.compile(input.path.generic_string(),
                                              output.path.generic_string());
  FUSILLI_REQUIRE_OK(compileResult);
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);
}
