// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

using namespace fusilli;

// =============================================================================
// Float16 Tests
// =============================================================================

TEST_CASE("Float16 default constructor initializes to zero", "[Float16]") {
  Float16 x;
  REQUIRE(x.toFloat() == 0.0f);
  REQUIRE(x.toBits() == 0);
}

TEST_CASE("Float16 conversion from float", "[Float16]") {
  SECTION("positive values") {
    Float16 x(1.0f);
    REQUIRE(x.toFloat() == 1.0f);

    Float16 y(2.5f);
    REQUIRE(y.toFloat() == 2.5f);

    Float16 z(0.5f);
    REQUIRE(z.toFloat() == 0.5f);
  }

  SECTION("negative values") {
    Float16 x(-1.0f);
    REQUIRE(x.toFloat() == -1.0f);

    Float16 y(-3.5f);
    REQUIRE(y.toFloat() == -3.5f);
  }

  SECTION("zero") {
    Float16 posZero(0.0f);
    REQUIRE(posZero.toFloat() == 0.0f);

    Float16 negZero(-0.0f);
    REQUIRE(negZero.toFloat() == -0.0f);
  }
}

TEST_CASE("Float16 handles special values", "[Float16]") {
  SECTION("infinity") {
    Float16 posInf(std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(posInf.toFloat()));
    REQUIRE(posInf.toFloat() > 0);

    Float16 negInf(-std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(negInf.toFloat()));
    REQUIRE(negInf.toFloat() < 0);
  }

  SECTION("NaN") {
    Float16 nanVal(std::numeric_limits<float>::quiet_NaN());
    REQUIRE(std::isnan(nanVal.toFloat()));
  }

  SECTION("overflow to infinity") {
    // Float16 max is ~65504, values larger overflow to inf
    Float16 large(100000.0f);
    REQUIRE(std::isinf(large.toFloat()));
  }

  SECTION("underflow to zero") {
    // Very small values underflow to zero
    Float16 tiny(1e-10f);
    REQUIRE(tiny.toFloat() == 0.0f);
  }
}

TEST_CASE("Float16 arithmetic operations", "[Float16]") {
  Float16 a(2.0f);
  Float16 b(3.0f);

  SECTION("addition") {
    Float16 result = a + b;
    REQUIRE(result.toFloat() == 5.0f);
  }

  SECTION("subtraction") {
    Float16 result = b - a;
    REQUIRE(result.toFloat() == 1.0f);
  }

  SECTION("multiplication") {
    Float16 result = a * b;
    REQUIRE(result.toFloat() == 6.0f);
  }

  SECTION("division") {
    Float16 result = b / a;
    REQUIRE(result.toFloat() == 1.5f);
  }

  SECTION("negation") {
    Float16 result = -a;
    REQUIRE(result.toFloat() == -2.0f);
  }
}

TEST_CASE("Float16 fromBits and toBits", "[Float16]") {
  // 0x3C00 is 1.0 in Float16
  Float16 one = Float16::fromBits(0x3C00);
  REQUIRE(one.toFloat() == 1.0f);
  REQUIRE(one.toBits() == 0x3C00);

  // 0x4000 is 2.0 in Float16
  Float16 two = Float16::fromBits(0x4000);
  REQUIRE(two.toFloat() == 2.0f);
  REQUIRE(two.toBits() == 0x4000);
}

TEST_CASE("Float16 explicit float conversion operator", "[Float16]") {
  Float16 x(3.5f);
  float f = static_cast<float>(x);
  REQUIRE(f == 3.5f);
}

TEST_CASE("Float16 precision limits", "[Float16]") {
  // Float16 has ~3 decimal digits of precision
  // Values that differ by less than the precision should round to the same
  Float16 a(1.0f);
  Float16 b(1.0001f); // Very close values
  // They might round to same value due to Float16 precision
  float diff = std::abs(a.toFloat() - b.toFloat());
  REQUIRE(diff < 0.002f); // Within Float16 precision
}

TEST_CASE("Float16 denormalized numbers", "[Float16]") {
  // Smallest positive denormalized Float16: 2^-24 ≈ 5.96e-8
  Float16 minDenorm = Float16::fromBits(0x0001);
  REQUIRE(minDenorm.toFloat() > 0.0f);
  REQUIRE(minDenorm.toFloat() < 1e-6f);

  // Largest denormalized Float16: (2^-14) * (1 - 2^-10)
  Float16 maxDenorm = Float16::fromBits(0x03FF);
  REQUIRE(maxDenorm.toFloat() > 0.0f);
  REQUIRE(std::isnormal(maxDenorm.toFloat()));
}

// =============================================================================
// BFloat16 Tests
// =============================================================================

TEST_CASE("BFloat16 default constructor initializes to zero", "[BFloat16]") {
  BFloat16 x;
  REQUIRE(x.toFloat() == 0.0f);
  REQUIRE(x.toBits() == 0);
}

TEST_CASE("BFloat16 conversion from float", "[BFloat16]") {
  SECTION("positive values") {
    BFloat16 x(1.0f);
    REQUIRE(x.toFloat() == 1.0f);

    BFloat16 y(2.5f);
    REQUIRE(y.toFloat() == 2.5f);

    BFloat16 z(0.5f);
    REQUIRE(z.toFloat() == 0.5f);
  }

  SECTION("negative values") {
    BFloat16 x(-1.0f);
    REQUIRE(x.toFloat() == -1.0f);

    BFloat16 y(-3.5f);
    REQUIRE(y.toFloat() == -3.5f);
  }

  SECTION("zero") {
    BFloat16 posZero(0.0f);
    REQUIRE(posZero.toFloat() == 0.0f);

    BFloat16 negZero(-0.0f);
    REQUIRE(negZero.toFloat() == -0.0f);
  }
}

TEST_CASE("BFloat16 handles special values", "[BFloat16]") {
  SECTION("infinity") {
    BFloat16 posInf(std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(posInf.toFloat()));
    REQUIRE(posInf.toFloat() > 0);

    BFloat16 negInf(-std::numeric_limits<float>::infinity());
    REQUIRE(std::isinf(negInf.toFloat()));
    REQUIRE(negInf.toFloat() < 0);
  }

  SECTION("NaN") {
    BFloat16 nanVal(std::numeric_limits<float>::quiet_NaN());
    REQUIRE(std::isnan(nanVal.toFloat()));
  }

  SECTION("large values preserved") {
    // BFloat16 has same exponent range as float32
    BFloat16 large(1e30f);
    float result = large.toFloat();
    // Should be close (within BFloat16 precision)
    REQUIRE(std::abs(result - 1e30f) / 1e30f < 0.01f);
  }

  SECTION("small values preserved") {
    BFloat16 small(1e-30f);
    float result = small.toFloat();
    // Should be close (within BFloat16 precision)
    REQUIRE(std::abs(result - 1e-30f) / 1e-30f < 0.01f);
  }
}

TEST_CASE("BFloat16 fromBits and toBits", "[BFloat16]") {
  // 0x3F80 is 1.0 in BFloat16 (upper 16 bits of float32 1.0)
  BFloat16 one = BFloat16::fromBits(0x3F80);
  REQUIRE(one.toFloat() == 1.0f);
  REQUIRE(one.toBits() == 0x3F80);

  // 0x4000 is 2.0 in BFloat16
  BFloat16 two = BFloat16::fromBits(0x4000);
  REQUIRE(two.toFloat() == 2.0f);
  REQUIRE(two.toBits() == 0x4000);
}

TEST_CASE("BFloat16 explicit float conversion operator", "[BFloat16]") {
  BFloat16 x(3.5f);
  float f = static_cast<float>(x);
  REQUIRE(f == 3.5f);
}

TEST_CASE("BFloat16 precision limits", "[BFloat16]") {
  // BFloat16 has ~2 decimal digits of precision (7 mantissa bits)
  BFloat16 a(1.0f);
  BFloat16 b(1.01f); // Close values
  // They might round to same value due to BFloat16 precision
  float diff = std::abs(a.toFloat() - b.toFloat());
  REQUIRE(diff < 0.02f); // Within BFloat16 precision
}

// =============================================================================
// Cross-type Tests
// =============================================================================

TEST_CASE("Float16 and BFloat16 have different bit representations",
          "[Float16][BFloat16]") {
  // 1.0 has different bit patterns in Float16 vs BFloat16
  Float16 float16One(1.0f);
  BFloat16 bfloat16One(1.0f);

  // Float16: 0x3C00, BFloat16: 0x3F80
  REQUIRE(float16One.toBits() != bfloat16One.toBits());
  REQUIRE(float16One.toBits() == 0x3C00);
  REQUIRE(bfloat16One.toBits() == 0x3F80);

  // But they both convert back to 1.0
  REQUIRE(float16One.toFloat() == bfloat16One.toFloat());
}

TEST_CASE("Float16 has more precision but smaller range than BFloat16",
          "[Float16][BFloat16]") {
  // Float16: 5 exp bits, 10 mantissa bits - max ~65504
  // BFloat16: 8 exp bits, 7 mantissa bits - max ~3.4e38

  // Large value that BFloat16 can represent but Float16 cannot
  float large = 100000.0f;
  Float16 float16Large(large);
  BFloat16 bfloat16Large(large);

  REQUIRE(std::isinf(float16Large.toFloat()));   // Overflows in Float16
  REQUIRE(!std::isinf(bfloat16Large.toFloat())); // Fits in BFloat16

  // Small precision test - Float16 is more precise
  // 1.001 rounded differently
  float precise = 1.001f;
  Float16 float16Precise(precise);
  BFloat16 bfloat16Precise(precise);

  // Float16 should be closer to original value (more mantissa bits)
  float float16Err = std::abs(float16Precise.toFloat() - precise);
  float bfloat16Err = std::abs(bfloat16Precise.toFloat() - precise);
  // Float16 has better precision for small values
  REQUIRE(float16Err <= bfloat16Err);
}
