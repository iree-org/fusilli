// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <string>
#include <vector>

using namespace fusilli;

// Helper to create a simple MLIR module for testing.
static std::string getSimpleMLIRModule() {
  return R"mlir(
module {
  func.func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
}

TEST_CASE("CompileContext::create successfully loads library",
          "[CompileContext]") {
  // Get the shared compiler context.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle for CPU backend.
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

  // Create a session.
  auto maybeSession = context->createSession(handle);
  FUSILLI_REQUIRE_OK(maybeSession);

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(std::move(maybeSession));

  // If we got here, session was created successfully.
  SUCCEED("Session created successfully");
}

#ifdef FUSILLI_ENABLE_AMDGPU
TEST_CASE("CompileContext::createSession with AMDGPU backend",
          "[CompileContext][CompileSession]") {
  // Get the shared compiler context.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle for AMDGPU backend.
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));

  // Create a session.
  auto maybeSession = context->createSession(handle);
  FUSILLI_REQUIRE_OK(maybeSession);

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(std::move(maybeSession));

  // If we got here, session was created successfully.
  SUCCEED("Session created successfully");
}
#endif

TEST_CASE("CompileContext supports multiple sessions",
          "[CompileContext][CompileSession]") {
  // Get the shared compiler context.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);

  // Create a handle.
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

  // Create multiple sessions from the same context.
  auto session1 = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));
  auto session2 = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));
  auto session3 = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // All sessions should be valid.
  SUCCEED("Multiple sessions created successfully");
}

TEST_CASE("CompileSession::addFlag", "[CompileSession]") {
  // Get the shared compiler context and create a session.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Add a flag.
  FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));

  // If we got here, flag was added successfully.
  SUCCEED("Flag added successfully");
}

TEST_CASE("CompileSession::addFlags", "[CompileSession]") {
  // Get the shared compiler context and create a session.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Create temporary cache files.
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "input.mlir", /*remove=*/true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "output.vmfb", /*remove=*/true));

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Add custom optimization flags.
  FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));
  FUSILLI_REQUIRE_OK(
      session.addFlag("--iree-vm-bytecode-module-strip-source-map=true"));

  // Create temporary cache files.
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "input_opt.mlir", /*remove=*/true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "output_opt.vmfb", /*remove=*/true));

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Create temporary cache files.
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "invalid.mlir", /*remove=*/true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "invalid.vmfb", /*remove=*/true));

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Create cache files but don't write to input.
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "missing.mlir", /*remove=*/true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "missing.vmfb", /*remove=*/true));

  // Input file doesn't exist - should fail.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  REQUIRE(isError(compileResult));
  REQUIRE(compileResult.getCode() == ErrorCode::CompileFailure);
}

TEST_CASE("CompileSession move semantics", "[CompileSession]") {
  // Get the shared compiler context and create sessions.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session1 = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Add a flag to session1.
  FUSILLI_REQUIRE_OK(session1.addFlag("--iree-opt-level=O3"));

  // Move session1 to session2.
  CompileSession session2 = std::move(session1);

  // session2 should be usable.
  FUSILLI_REQUIRE_OK(session2.addFlag("--iree-opt-level=O2"));

  SUCCEED("Move semantics work correctly");
}

#ifdef FUSILLI_ENABLE_AMDGPU
TEST_CASE("CompileSession::compile with AMDGPU backend",
          "[CompileSession][integration][amdgpu]") {
  // Get the shared compiler context and create a session for AMDGPU.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Create temporary cache files.
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "input_amdgpu.mlir", /*remove=*/true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      "test_compiler_session", "output_amdgpu.vmfb", /*remove=*/true));

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
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

  // Create a session in a nested scope.
  {
    auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

    // Use the session.
    FUSILLI_REQUIRE_OK(session.addFlag("--iree-opt-level=O3"));

    // Session will be destroyed when exiting this scope.
  }

  // Should be able to create a new session after the previous one was
  // destroyed.
  auto session2 = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));
  FUSILLI_REQUIRE_OK(session2.addFlag("--iree-opt-level=O2"));

  SUCCEED("Session cleanup and recreation works correctly");
}

TEST_CASE("CompileSession with invalid flag", "[CompileSession][error]") {
  // Get the shared compiler context and create a session.
  auto context = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context != nullptr);
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  auto session = FUSILLI_REQUIRE_UNWRAP(context->createSession(handle));

  // Try to add an invalid flag.
  auto result = session.addFlag("--this-is-not-a-valid-iree-flag-xyz123");

  // This might or might not fail depending on IREE's flag validation.
  // If it fails, check the error code.
  if (isError(result)) {
    REQUIRE(result.getCode() == ErrorCode::CompileFailure);
  }
}

TEST_CASE("Multiple CompileContexts can coexist", "[CompileContext]") {
  // Note: We use the shared context for most tests, but this test specifically
  // checks if multiple contexts can be created. This might fail if IREE's
  // global state doesn't support multiple initializations.

  auto context1 = FUSILLI_REQUIRE_UNWRAP(CompileContext::create());
  REQUIRE(context1 != nullptr);

  // Try to create a second context.
  auto maybeContext2 = CompileContext::create();

  // If it succeeds, both contexts should be usable.
  if (isOk(maybeContext2)) {
    auto context2 = FUSILLI_REQUIRE_UNWRAP(std::move(maybeContext2));
    REQUIRE(context2 != nullptr);

    // Both should have the same API version.
    REQUIRE(context1->getAPIVersion() == context2->getAPIVersion());
  }
}

// ===========================================================================
// Tests for CompileCommand-style unified interface
// ===========================================================================

TEST_CASE("CompileSession::build with CPU backend", "[CompileSession]") {
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "input.mlir", true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "output.vmfb", true));
  CacheFile statistics = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "statistics.json", true));

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(
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
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "input.mlir", true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "output.vmfb", true));
  CacheFile statistics = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "statistics.json", true));

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(
      CompileSession::build(handle, input, output, statistics));

  std::string cmdStr = session.toString();

  REQUIRE(!cmdStr.empty());
  REQUIRE(cmdStr.back() == '\n');
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("iree-compile"));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("-o"));
}

TEST_CASE("CompileSession::writeTo", "[CompileSession]") {
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "input.mlir", true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "output.vmfb", true));
  CacheFile statistics = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "statistics.json", true));
  CacheFile commandFile = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "command.txt", true));

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(
      CompileSession::build(handle, input, output, statistics));

  FUSILLI_REQUIRE_OK(session.writeTo(commandFile));

  // Read back and verify.
  std::string written = FUSILLI_REQUIRE_UNWRAP(commandFile.read());
  REQUIRE(written == session.toString());
}

TEST_CASE("CompileSession::execute with valid MLIR", "[CompileSession]") {
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "input.mlir", true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "output.vmfb", true));
  CacheFile statistics = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "statistics.json", true));

  // Write valid MLIR.
  FUSILLI_REQUIRE_OK(input.write(getSimpleMLIRModule()));

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(
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
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  CacheFile input = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "input.mlir", true));
  CacheFile output = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "output.vmfb", true));
  CacheFile statistics = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create("test_compile_session", "statistics.json", true));

  CompileSession session = FUSILLI_REQUIRE_UNWRAP(
      CompileSession::build(handle, input, output, statistics));

  const auto &args = session.getArgs();

  // Should have backend flags and statistics flags.
  REQUIRE(!args.empty());

  // All args should be non-empty strings.
  for (const auto &arg : args) {
    REQUIRE(!arg.empty());
  }
}
