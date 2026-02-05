// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains tests for utilities in tests/utils.h, including
// Catch2 StringMaker specializations for Fusilli types.
//
//===----------------------------------------------------------------------===//

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace fusilli;

TEST_CASE("Catch2 StringMaker for half type", "[utils][half]") {
  SECTION("converts half to readable string") {
    half val1 = half(129.0f);
    std::string str1 = Catch::StringMaker<half>::convert(val1);
    // Should convert to float representation
    REQUIRE_FALSE(str1.empty());
    REQUIRE_FALSE(str1 == "{?}");

    // Verify it contains the numeric representation
    REQUIRE(str1.find("129") != std::string::npos);
  }

  SECTION("handles fractional values") {
    half val2 = half(128.5f);
    std::string str2 = Catch::StringMaker<half>::convert(val2);
    REQUIRE_FALSE(str2.empty());
    REQUIRE_FALSE(str2 == "{?}");

    // Should show fractional part
    REQUIRE(str2.find("128") != std::string::npos);
  }

  SECTION("handles zero") {
    half val3 = half(0.0f);
    std::string str3 = Catch::StringMaker<half>::convert(val3);
    REQUIRE_FALSE(str3.empty());
    REQUIRE_FALSE(str3 == "{?}");

    REQUIRE(str3.find('0') != std::string::npos);
  }

  SECTION("handles negative values") {
    half val4 = half(-42.5f);
    std::string str4 = Catch::StringMaker<half>::convert(val4);
    REQUIRE_FALSE(str4.empty());
    REQUIRE_FALSE(str4 == "{?}");

    // Should show negative sign.
    REQUIRE(str4.find('-') != std::string::npos);
    REQUIRE(str4.find("42") != std::string::npos);
  }

  SECTION("different values produce different strings") {
    half valA = half(100.0f);
    half valB = half(200.0f);
    std::string strA = Catch::StringMaker<half>::convert(valA);
    std::string strB = Catch::StringMaker<half>::convert(valB);

    // Different values should produce different strings.
    REQUIRE(strA != strB);
  }
}

TEST_CASE("Catch2 StringMaker for bf16 type", "[utils][bf16]") {
  SECTION("converts bf16 to readable string") {
    bf16 val1 = bf16(256.0f);
    std::string str1 = Catch::StringMaker<bf16>::convert(val1);
    // Should convert to float representation
    REQUIRE_FALSE(str1.empty());
    REQUIRE_FALSE(str1 == "{?}");

    // Verify it contains the numeric representation
    REQUIRE(str1.find("256") != std::string::npos);
  }

  SECTION("handles zero") {
    bf16 val2 = bf16(0.0f);
    std::string str2 = Catch::StringMaker<bf16>::convert(val2);
    REQUIRE_FALSE(str2.empty());
    REQUIRE_FALSE(str2 == "{?}");

    REQUIRE(str2.find('0') != std::string::npos);
  }

  SECTION("handles negative values") {
    bf16 val3 = bf16(-128.0f);
    std::string str3 = Catch::StringMaker<bf16>::convert(val3);
    REQUIRE_FALSE(str3.empty());
    REQUIRE_FALSE(str3 == "{?}");

    // Should show negative sign.
    REQUIRE(str3.find('-') != std::string::npos);
    REQUIRE(str3.find("128") != std::string::npos);
  }

  SECTION("different values produce different strings") {
    bf16 valA = bf16(50.0f);
    bf16 valB = bf16(150.0f);
    std::string strA = Catch::StringMaker<bf16>::convert(valA);
    std::string strB = Catch::StringMaker<bf16>::convert(valB);

    // Different values should produce different strings.
    REQUIRE(strA != strB);
  }
}

TEST_CASE("Catch2 comparison works with half type",
          "[utils][half][comparison]") {
  half val1 = half(100.0f);
  half val2 = half(100.0f);
  REQUIRE(val1 == val2);
}

TEST_CASE("Catch2 comparison works with bf16 type",
          "[utils][bf16][comparison]") {
  bf16 val1 = bf16(200.0f);
  bf16 val2 = bf16(200.0f);
  REQUIRE(val1 == val2);
}

TEST_CASE("castToSizeT utility function", "[utils]") {
  SECTION("converts int64_t vector to size_t vector") {
    std::vector<int64_t> input = {1, 2, 3, 4, 5};
    std::vector<size_t> result = castToSizeT(input);

    REQUIRE(result.size() == input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      REQUIRE(result[i] == static_cast<size_t>(input[i]));
    }
  }

  SECTION("handles empty vector") {
    std::vector<int64_t> input = {};
    std::vector<size_t> result = castToSizeT(input);

    REQUIRE(result.empty());
  }

  SECTION("handles large dimensions") {
    std::vector<int64_t> input = {1024, 2048, 4096, 8192};
    std::vector<size_t> result = castToSizeT(input);

    REQUIRE(result.size() == 4);
    REQUIRE(result[0] == 1024);
    REQUIRE(result[1] == 2048);
    REQUIRE(result[2] == 4096);
    REQUIRE(result[3] == 8192);
  }
}

TEST_CASE("allocateWorkspace helper function", "[utils][workspace]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  SECTION("returns nullptr for nullopt") {
    FUSILLI_REQUIRE_ASSIGN(auto workspace, allocateWorkspace(handle, std::nullopt));
    REQUIRE(workspace == nullptr);
  }

  SECTION("returns nullptr for size 0") {
    FUSILLI_REQUIRE_ASSIGN(auto workspace, allocateWorkspace(handle, 0));
    REQUIRE(workspace == nullptr);
  }

  SECTION("allocates buffer for non-zero size") {
    constexpr size_t bufferSize = 1024;
    FUSILLI_REQUIRE_ASSIGN(auto workspace, allocateWorkspace(handle, bufferSize));
    REQUIRE(workspace != nullptr);
    REQUIRE(*workspace != nullptr);

    // Verify the underlying buffer view has the expected shape (1D i8).
    iree_hal_buffer_view_t *bufferView = *workspace;
    REQUIRE(iree_hal_buffer_view_shape_rank(bufferView) == 1);
    REQUIRE(iree_hal_buffer_view_shape_dim(bufferView, 0) == bufferSize);
  }

  SECTION("allocates different sizes correctly") {
    // Small allocation
    FUSILLI_REQUIRE_ASSIGN(auto small, allocateWorkspace(handle, 64));
    REQUIRE(small != nullptr);
    REQUIRE(iree_hal_buffer_view_shape_dim(*small, 0) == 64);

    // Larger allocation (1 MB)
    constexpr size_t oneMB = size_t{1024} * size_t{1024};
    FUSILLI_REQUIRE_ASSIGN(auto large, allocateWorkspace(handle, oneMB));
    REQUIRE(large != nullptr);
    REQUIRE(iree_hal_buffer_view_shape_dim(*large, 0) == oneMB);
  }
}
