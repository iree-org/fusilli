// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace fusilli;

// =============================================================================
// needsShellQuoting Tests
// =============================================================================

TEST_CASE("needsShellQuoting returns false for simple flags",
          "[needsShellQuoting]") {
  REQUIRE_FALSE(needsShellQuoting("--iree-hal-target-backends=rocm"));
  REQUIRE_FALSE(needsShellQuoting("--iree-opt-level=O3"));
  REQUIRE_FALSE(needsShellQuoting("-o"));
  REQUIRE_FALSE(needsShellQuoting("iree-compile"));
  REQUIRE_FALSE(needsShellQuoting("/tmp/cache/output.vmfb"));
}

TEST_CASE("needsShellQuoting returns true for parentheses",
          "[needsShellQuoting]") {
  REQUIRE(needsShellQuoting(
      "--iree-preprocessing-pass-pipeline="
      "builtin.module(util.func(iree-preprocessing-convert-conv-filter-"
      "to-channels-last))"));
}

TEST_CASE("needsShellQuoting returns true for spaces", "[needsShellQuoting]") {
  REQUIRE(needsShellQuoting("/path/with spaces/file.mlir"));
}

TEST_CASE("needsShellQuoting returns true for shell metacharacters",
          "[needsShellQuoting]") {
  REQUIRE(needsShellQuoting("foo;bar"));
  REQUIRE(needsShellQuoting("foo|bar"));
  REQUIRE(needsShellQuoting("foo&bar"));
  REQUIRE(needsShellQuoting("$HOME/file"));
  REQUIRE(needsShellQuoting("`cmd`"));
  REQUIRE(needsShellQuoting("file*.mlir"));
  REQUIRE(needsShellQuoting("file?.mlir"));
  REQUIRE(needsShellQuoting("foo>bar"));
  REQUIRE(needsShellQuoting("foo<bar"));
  REQUIRE(needsShellQuoting("~user"));
  REQUIRE(needsShellQuoting("foo#comment"));
  REQUIRE(needsShellQuoting("!history"));
  REQUIRE(needsShellQuoting("{a,b}"));
  REQUIRE(needsShellQuoting("[abc]"));
  REQUIRE(needsShellQuoting("foo\\bar"));
  REQUIRE(needsShellQuoting("foo\"bar"));
  REQUIRE(needsShellQuoting("foo'bar"));
  REQUIRE(needsShellQuoting("foo\tbar"));
}

TEST_CASE("needsShellQuoting returns false for empty string",
          "[needsShellQuoting]") {
  // Empty string has no special characters, but note that it may still need
  // quoting for semantic reasons (not handled by this function).
  REQUIRE_FALSE(needsShellQuoting(""));
}

// =============================================================================
// escapeArgument Tests
// =============================================================================

TEST_CASE("escapeArgument returns simple args unchanged", "[escapeArgument]") {
  REQUIRE(escapeArgument("--iree-hal-target-backends=rocm") ==
          "--iree-hal-target-backends=rocm");
  REQUIRE(escapeArgument("-o") == "-o");
  REQUIRE(escapeArgument("/tmp/cache/output.vmfb") == "/tmp/cache/output.vmfb");
}

TEST_CASE("escapeArgument wraps parentheses in single quotes",
          "[escapeArgument]") {
  std::string flag = "--iree-preprocessing-pass-pipeline="
                     "builtin.module(util.func(convert-filter))";
  std::string expected = "'--iree-preprocessing-pass-pipeline="
                         "builtin.module(util.func(convert-filter))'";
  REQUIRE(escapeArgument(flag) == expected);
}

TEST_CASE("escapeArgument wraps spaces in single quotes", "[escapeArgument]") {
  REQUIRE(escapeArgument("/path/with spaces/file") ==
          "'/path/with spaces/file'");
}

TEST_CASE("escapeArgument escapes embedded single quotes", "[escapeArgument]") {
  // Input: it's
  // Expected: 'it'\''s' (end quote, escaped quote, restart quote)
  REQUIRE(escapeArgument("it's") == "'it'\\''s'");
}

TEST_CASE("escapeArgument handles multiple single quotes", "[escapeArgument]") {
  // Input: a'b'c (contains single quotes and no other metachar... but ' itself
  // is a metachar)
  REQUIRE(escapeArgument("a'b'c") == "'a'\\''b'\\''c'");
}

TEST_CASE("escapeArgument handles dollar signs", "[escapeArgument]") {
  // Single-quoted strings prevent variable expansion.
  REQUIRE(escapeArgument("$HOME/file") == "'$HOME/file'");
}

TEST_CASE("escapeArgument handles backticks", "[escapeArgument]") {
  // Single-quoted strings prevent command substitution.
  REQUIRE(escapeArgument("`whoami`") == "'`whoami`'");
}

TEST_CASE("escapeArgument handles the actual problematic flag",
          "[escapeArgument]") {
  // This is the exact flag that caused the original bash syntax error.
  std::string flag =
      "--iree-preprocessing-pass-pipeline="
      "builtin.module(util.func("
      "iree-preprocessing-convert-conv-filter-to-channels-last))";
  std::string result = escapeArgument(flag);

  // Should be wrapped in single quotes.
  REQUIRE(result.front() == '\'');
  REQUIRE(result.back() == '\'');

  // The content between quotes should be the original flag unchanged
  // (no single quotes in the original, so no escaping needed inside).
  REQUIRE(result == "'" + flag + "'");
}
