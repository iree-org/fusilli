// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace fusilli;

// =============================================================================
// Int4 Struct Tests
// =============================================================================

TEST_CASE("Int4 default constructor initializes to zero", "[Int4]") {
  Int4 x;
  REQUIRE(x.toInt() == 0);
  REQUIRE(x.toBits() == 0);
}

TEST_CASE("Int4 construction from in-range values", "[Int4]") {
  SECTION("positive values") {
    for (int8_t v = 0; v <= 7; ++v) {
      Int4 x(v);
      REQUIRE(x.toInt() == v);
    }
  }

  SECTION("negative values") {
    for (int8_t v = -8; v < 0; ++v) {
      Int4 x(v);
      REQUIRE(x.toInt() == v);
    }
  }

  SECTION("boundary values") {
    Int4 minVal(int8_t(-8));
    REQUIRE(minVal.toInt() == -8);

    Int4 maxVal(int8_t(7));
    REQUIRE(maxVal.toInt() == 7);

    Int4 zero(int8_t(0));
    REQUIRE(zero.toInt() == 0);
  }
}

TEST_CASE("Int4 clamps out-of-range values", "[Int4]") {
  SECTION("positive overflow") {
    Int4 x(int8_t(10));
    REQUIRE(x.toInt() == 7);

    Int4 y(int8_t(127));
    REQUIRE(y.toInt() == 7);
  }

  SECTION("negative overflow") {
    Int4 x(int8_t(-10));
    REQUIRE(x.toInt() == -8);

    Int4 y(int8_t(-128));
    REQUIRE(y.toInt() == -8);
  }
}

TEST_CASE("Int4 implicit conversion to int8_t", "[Int4]") {
  Int4 x(int8_t(5));
  int8_t val = x;
  REQUIRE(val == 5);

  Int4 y(int8_t(-3));
  int8_t neg = y;
  REQUIRE(neg == -3);
}

TEST_CASE("Int4 fromBits and toBits roundtrip", "[Int4]") {
  for (uint8_t bits = 0; bits < 16; ++bits) {
    Int4 x = Int4::fromBits(bits);
    REQUIRE(x.toBits() == bits);
  }
}

TEST_CASE("Int4 fromBits sign extension", "[Int4]") {
  // Bit pattern 0b1000 = 8 unsigned, but -8 as signed 4-bit.
  Int4 x = Int4::fromBits(0x08);
  REQUIRE(x.toInt() == -8);

  // Bit pattern 0b1111 = 15 unsigned, but -1 as signed 4-bit.
  Int4 y = Int4::fromBits(0x0F);
  REQUIRE(y.toInt() == -1);

  // Bit pattern 0b0111 = 7 as both unsigned and signed 4-bit.
  Int4 z = Int4::fromBits(0x07);
  REQUIRE(z.toInt() == 7);
}

// =============================================================================
// Packing / Unpacking Tests
// =============================================================================

TEST_CASE("Int4::pack and Int4::unpack roundtrip with even count", "[Int4]") {
  std::vector<Int4> elements = {Int4(0),  Int4(1), Int4(-1), Int4(7),
                                Int4(-8), Int4(3), Int4(-5), Int4(6)};

  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.size() == 4); // 8 elements -> 4 bytes

  std::vector<Int4> unpacked = Int4::unpack(packed.data(), elements.size());
  REQUIRE(unpacked.size() == elements.size());

  for (size_t i = 0; i < elements.size(); ++i)
    REQUIRE(unpacked[i].toInt() == elements[i].toInt());
}

TEST_CASE("Int4::pack and Int4::unpack roundtrip with odd count", "[Int4]") {
  std::vector<Int4> elements = {Int4(1), Int4(2), Int4(3)};

  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.size() == 2); // 3 elements -> 2 bytes (ceil(3/2))

  std::vector<Int4> unpacked = Int4::unpack(packed.data(), elements.size());
  REQUIRE(unpacked.size() == elements.size());

  for (size_t i = 0; i < elements.size(); ++i)
    REQUIRE(unpacked[i].toInt() == elements[i].toInt());
}

TEST_CASE("Int4::pack single element", "[Int4]") {
  std::vector<Int4> elements = {Int4(5)};

  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.size() == 1);
  REQUIRE((packed[0] & 0x0F) == 5); // low nibble

  std::vector<Int4> unpacked = Int4::unpack(packed.data(), 1);
  REQUIRE(unpacked[0].toInt() == 5);
}

TEST_CASE("Int4::pack empty vector", "[Int4]") {
  std::vector<Int4> elements;
  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.empty());
}

TEST_CASE("Int4::pack nibble placement", "[Int4]") {
  // Verify convention: even index in low nibble, odd index in high nibble.
  std::vector<Int4> elements = {Int4(3), Int4(5)};
  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.size() == 1);
  REQUIRE((packed[0] & 0x0F) == 3);        // low nibble = element[0]
  REQUIRE(((packed[0] >> 4) & 0x0F) == 5); // high nibble = element[1]
}

TEST_CASE("Int4::pack all values roundtrip", "[Int4]") {
  // Test all 16 possible 4-bit signed values.
  std::vector<Int4> elements;
  for (int8_t v = -8; v <= 7; ++v)
    elements.push_back(Int4(v));

  std::vector<uint8_t> packed = Int4::pack(elements);
  REQUIRE(packed.size() == 8); // 16 elements -> 8 bytes

  std::vector<Int4> unpacked = Int4::unpack(packed.data(), elements.size());
  REQUIRE(unpacked.size() == 16);

  for (size_t i = 0; i < elements.size(); ++i)
    REQUIRE(unpacked[i].toInt() == elements[i].toInt());
}
