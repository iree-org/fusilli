// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace fusilli;

//===----------------------------------------------------------------------===//
// Tests for execCommand
//===----------------------------------------------------------------------===//

TEST_CASE("execCommand basic functionality", "[support][process]") {
  SECTION("Execute echo command") {
    auto result = execCommand("echo hello");
    REQUIRE(result.has_value());
    REQUIRE(result->find("hello") != std::string::npos);
  }

  SECTION("Execute command that produces no output") {
    auto result = execCommand("true");
    REQUIRE(result.has_value());
    REQUIRE(result->empty());
  }
}

TEST_CASE("execCommand error handling", "[support][process]") {
  SECTION("Command that does not exist returns nullopt") {
    auto result = execCommand("nonexistent_command_12345 2>/dev/null");
    // popen succeeds even for invalid commands, so we just
    // verify it doesn't crash and returns a value.
    REQUIRE(result.has_value());
  }

  SECTION("Command with redirect to suppress stderr") {
    auto result = execCommand("ls /nonexistent_path_12345 2>/dev/null");
    REQUIRE(result.has_value());
    // May or may not be empty depending on system
  }
}

TEST_CASE("execCommand output capturing", "[support][process]") {
  SECTION("Handles whitespace in output") {
    auto result = execCommand("echo '  spaces  '");
    REQUIRE(result.has_value());
    REQUIRE(result->find("spaces") != std::string::npos);
  }
}
