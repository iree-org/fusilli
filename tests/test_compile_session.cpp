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

#include <filesystem>
#include <string>
#include <vector>

using namespace fusilli;

static std::string kGraphName = "test_compile_session";

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
  std::filesystem::remove_all(input.path.parent_path());
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
  std::filesystem::remove_all(input.path.parent_path());
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

  // Write invalid MLIR content.
  std::string invalidMLIR = "this is not valid MLIR syntax!";
  FUSILLI_REQUIRE_OK(input.write(invalidMLIR));

  // Attempt to compile - should fail.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  REQUIRE(isError(compileResult));
  REQUIRE(compileResult.getCode() == ErrorCode::CompileFailure);
  std::filesystem::remove_all(input.path.parent_path());
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

  // Input file doesn't exist - should fail.
  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  REQUIRE(isError(compileResult));
  REQUIRE(compileResult.getCode() == ErrorCode::CompileFailure);
  std::filesystem::remove_all(input.path.parent_path());
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
  std::filesystem::remove_all(input.path.parent_path());
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
  std::filesystem::remove_all(input.path.parent_path());
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

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  std::string cmdStr = session.toString();

  REQUIRE(!cmdStr.empty());
  REQUIRE(cmdStr.back() == '\n');
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("iree-compile"));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("-o"));
  std::filesystem::remove_all(input.path.parent_path());
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

  FUSILLI_REQUIRE_ASSIGN(
      CompileSession session,
      CompileSession::build(handle, input, output, statistics));

  FUSILLI_REQUIRE_OK(session.writeTo(commandFile));

  // Read back and verify.
  FUSILLI_REQUIRE_ASSIGN(std::string written, commandFile.read());
  REQUIRE(written == session.toString());
  std::filesystem::remove_all(input.path.parent_path());
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
  std::filesystem::remove_all(input.path.parent_path());
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
  std::filesystem::remove_all(input.path.parent_path());
}

TEST_CASE("CompileSession::addFlag with tuning spec path",
          "[CompileSession][tuning_spec]") {
  // Verify tuning spec path flag is accepted by C API.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  std::string tuningSpecPath = "/tmp/test_tuning_spec.mlir";

  auto result =
      session.addFlag("--iree-codegen-tuning-spec-path=" + tuningSpecPath);
  FUSILLI_REQUIRE_OK(result);

  // Verify the flag was stored.
  const auto &args = session.getArgs();
  bool hasTuningSpecFlag = false;
  for (const auto &arg : args) {
    if (arg.find("--iree-codegen-tuning-spec-path") != std::string::npos) {
      hasTuningSpecFlag = true;
      break;
    }
  }
  REQUIRE(hasTuningSpecFlag);
}

TEST_CASE("CompileSession::compile with tuning spec",
          "[CompileSession][integration][tuning_spec]") {
  // End-to-end compilation with tuning spec via C API.
  FUSILLI_REQUIRE_ASSIGN(auto *context, CompileContext::create());
  REQUIRE(context != nullptr);
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_ASSIGN(auto session, context->createSession(handle));

  // Create a minimal no-op tuning spec.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile tuningSpec,
      CacheFile::create(kGraphName, "test_tuning_spec.mlir", /*remove=*/true));

  std::string tuningSpecContent = R"(
// Minimal no-op tuning spec for testing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
    -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    transform.yield %arg0 : !transform.any_op
  }
}
)";
  FUSILLI_REQUIRE_OK(tuningSpec.write(tuningSpecContent));

  FUSILLI_REQUIRE_OK(session.addFlag("--iree-codegen-tuning-spec-path=" +
                                     tuningSpec.path.string()));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input_with_spec.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output_with_spec.vmfb", /*remove=*/true));

  std::string mlirContent = getSimpleMLIRModule();
  FUSILLI_REQUIRE_OK(input.write(mlirContent));

  auto compileResult =
      session.compile(input.path.string(), output.path.string());
  FUSILLI_REQUIRE_OK(compileResult);
  REQUIRE(std::filesystem::exists(output.path));
  REQUIRE(std::filesystem::file_size(output.path) > 0);

  std::filesystem::remove_all(input.path.parent_path());
}
