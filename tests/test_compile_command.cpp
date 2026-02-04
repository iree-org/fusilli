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

using namespace fusilli;

static std::string kGraphName = "test_compile_command";

TEST_CASE("CompileCommand::build with CPU backend", "[CompileCommand]") {
  // Create test handle for CPU backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));

  // Write simple MLIR module to input file.
  REQUIRE(input.write(getSimpleMLIRModule()).isOk());

  // Build the compile command.
  CompileCommand cmd = CompileCommand::build(handle, input, output, statistics);

  // Verify the command arguments.
  const auto &args = cmd.getArgs();

  // Should have at least: iree-compile, input, backend flags, stats flags, -o,
  // output.
  REQUIRE(args.size() >= 7);

  // Check first argument is iree-compile (or path to it).
  REQUIRE_THAT(args[0], Catch::Matchers::ContainsSubstring("iree-compile"));

  // Check input file is included.
  REQUIRE(args[1] == input.path.string());

  // Check CPU backend flags are present.
  bool hasCPUBackend = false;
  bool hasCPUTarget = false;
  for (const auto &arg : args) {
    if (arg.find("--iree-hal-target-backends=llvm-cpu") != std::string::npos) {
      hasCPUBackend = true;
    }
    if (arg.find("--iree-llvmcpu-target-cpu=host") != std::string::npos) {
      hasCPUTarget = true;
    }
  }
  REQUIRE(hasCPUBackend);
  REQUIRE(hasCPUTarget);

  // Check statistics flags are present.
  bool hasStatisticsFormat = false;
  bool hasStatisticsFile = false;
  for (const auto &arg : args) {
    if (arg.find("--iree-scheduling-dump-statistics-format=json") !=
        std::string::npos) {
      hasStatisticsFormat = true;
    }
    if (arg.find("--iree-scheduling-dump-statistics-file=") !=
        std::string::npos) {
      hasStatisticsFile = true;
    }
  }
  REQUIRE(hasStatisticsFormat);
  REQUIRE(hasStatisticsFile);

  // Check output specification is at the end.
  REQUIRE(args[args.size() - 2] == "-o");
  REQUIRE(args[args.size() - 1] == output.path.string());

  auto result = cmd.execute();
  REQUIRE(result.isOk());

  auto outputSize = std::filesystem::file_size(output.path);
  REQUIRE(outputSize > 0);
  std::filesystem::remove_all(input.path.parent_path());
}

#ifdef FUSILLI_ENABLE_AMDGPU
TEST_CASE("CompileCommand::build with AMDGPU backend", "[CompileCommand]") {
  // Create test handle for AMDGPU backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));

  // Write simple MLIR module to input file.
  REQUIRE(input.write(getSimpleMLIRModule()).isOk());

  // Build the compile command.
  CompileCommand cmd = CompileCommand::build(handle, input, output, statistics);

  // Verify the command arguments.
  const auto &args = cmd.getArgs();

  // Should have more arguments than CPU due to additional AMDGPU flags.
  REQUIRE(args.size() >= 12);

  // Check AMDGPU backend flags are present.
  bool hasROCMBackend = false;
  bool hasHIPTarget = false;
  bool hasOptLevel = false;
  for (const auto &arg : args) {
    if (arg.find("--iree-hal-target-backends=rocm") != std::string::npos) {
      hasROCMBackend = true;
    }
    if (arg.find("--iree-hip-target=") != std::string::npos) {
      hasHIPTarget = true;
    }
    if (arg.find("--iree-opt-level=O3") != std::string::npos) {
      hasOptLevel = true;
    }
  }
  REQUIRE(hasROCMBackend);
  REQUIRE(hasHIPTarget);
  REQUIRE(hasOptLevel);

  auto result = cmd.execute();
  REQUIRE(result.isOk());

  auto outputSize = std::filesystem::file_size(output.path);
  REQUIRE(outputSize > 0);
  std::filesystem::remove_all(input.path.parent_path());
}
#endif

TEST_CASE("CompileCommand::toString format", "[CompileCommand]") {
  // Create test handle.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));

  // Build the compile command.
  CompileCommand cmd = CompileCommand::build(handle, input, output, statistics);

  // Get string representation.
  std::string cmdStr = cmd.toString();

  // Verify format: should be space-separated with trailing newline.
  REQUIRE(!cmdStr.empty());
  REQUIRE(cmdStr.back() == '\n');

  // Verify it contains key components.
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("iree-compile"));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring(input.path.string()));
  REQUIRE_THAT(cmdStr,
               Catch::Matchers::ContainsSubstring(output.path.string()));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring(
                           "--iree-hal-target-backends=llvm-cpu"));
  REQUIRE_THAT(cmdStr, Catch::Matchers::ContainsSubstring("-o"));
  std::filesystem::remove_all(input.path.parent_path());
}

TEST_CASE("CompileCommand::writeTo", "[CompileCommand]") {
  // Create test handle.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile commandFile,
      CacheFile::create(kGraphName, "command.txt", /*remove=*/true));

  // Build the compile command.
  CompileCommand cmd = CompileCommand::build(handle, input, output, statistics);

  // Write to cache file.
  FUSILLI_REQUIRE_OK(cmd.writeTo(commandFile));

  // Read back and verify.
  FUSILLI_REQUIRE_ASSIGN(std::string writtenContent, commandFile.read());
  std::string expectedContent = cmd.toString();

  REQUIRE(writtenContent == expectedContent);
  std::filesystem::remove_all(input.path.parent_path());
}

TEST_CASE("CompileCommand::getArgs", "[CompileCommand]") {
  // Create test handle.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));

  // Build the compile command.
  CompileCommand cmd = CompileCommand::build(handle, input, output, statistics);

  // Get args and verify structure.
  const auto &args = cmd.getArgs();

  REQUIRE(!args.empty());

  // First arg should be the compiler executable.
  REQUIRE_THAT(args[0], Catch::Matchers::ContainsSubstring("iree-compile"));

  // Second arg should be the input file.
  REQUIRE(args[1] == input.path.string());

  // Last two args should be "-o" and output path.
  REQUIRE(args.size() >= 2);
  REQUIRE(args[args.size() - 2] == "-o");
  REQUIRE(args[args.size() - 1] == output.path.string());
  std::filesystem::remove_all(input.path.parent_path());
}

TEST_CASE("CompileCommand round-trip serialization", "[CompileCommand]") {
  // Create test handle.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  // Create temporary cache files.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile input,
      CacheFile::create(kGraphName, "input.mlir", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile output,
      CacheFile::create(kGraphName, "output.vmfb", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile statistics,
      CacheFile::create(kGraphName, "statistics.json", /*remove=*/true));
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile commandFile,
      CacheFile::create(kGraphName, "command.txt", /*remove=*/true));

  // Build and write command.
  CompileCommand cmd1 =
      CompileCommand::build(handle, input, output, statistics);
  FUSILLI_REQUIRE_OK(cmd1.writeTo(commandFile));

  // Build another command with the same parameters.
  CompileCommand cmd2 =
      CompileCommand::build(handle, input, output, statistics);

  // Read the serialized command.
  FUSILLI_REQUIRE_ASSIGN(std::string serializedCmd, commandFile.read());

  // Verify both produce the same string.
  REQUIRE(cmd1.toString() == cmd2.toString());
  REQUIRE(serializedCmd == cmd1.toString());
  std::filesystem::remove_all(input.path.parent_path());
}
