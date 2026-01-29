// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains portable Float16 and BFloat16 implementations represented
// as structs with int16_t storage. These types support conversion to/from float
// and perform arithmetic operations in 32-bit float precision.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_FLOAT_TYPES_H
#define FUSILLI_SUPPORT_FLOAT_TYPES_H

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace fusilli {

// IEEE 754 half-precision floating point (Float16)
// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
//
// This type provides implicit conversions to/from float, allowing seamless
// interoperability with float arithmetic. All operations are performed in
// float precision through these conversions.
struct Float16 {
  int16_t data;

  constexpr Float16() : data(0) {}

  // Construct from float (handles double via implicit conversion)
  constexpr Float16(float f) : data(floatToFp16Bits(f)) {}

  // Convert to float
  constexpr float toFloat() const { return fp16BitsToFloat(data); }

  // Implicit conversion to float for seamless interoperability
  // Arithmetic and comparisons work through this conversion
  constexpr operator float() const { return toFloat(); }

  // Unary negation
  constexpr Float16 operator-() const { return Float16(-toFloat()); }

  // Compound assignment operators
  constexpr Float16 &operator+=(Float16 other) {
    *this = Float16(toFloat() + other.toFloat());
    return *this;
  }

  constexpr Float16 &operator-=(Float16 other) {
    *this = Float16(toFloat() - other.toFloat());
    return *this;
  }

  constexpr Float16 &operator*=(Float16 other) {
    *this = Float16(toFloat() * other.toFloat());
    return *this;
  }

  constexpr Float16 &operator/=(Float16 other) {
    *this = Float16(toFloat() / other.toFloat());
    return *this;
  }

  // Create from raw bits
  static constexpr Float16 fromBits(int16_t bits) {
    Float16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  constexpr int16_t toBits() const { return data; }

private:
  static constexpr int16_t floatToFp16Bits(float f) {
    uint32_t bits = std::bit_cast<uint32_t>(f);

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    // Handle special cases
    if (exp == 128) {
      // Inf or NaN
      if (mantissa == 0) {
        // Infinity
        return static_cast<int16_t>((sign << 15) | 0x7C00);
      } else {
        // NaN - preserve some mantissa bits
        return static_cast<int16_t>((sign << 15) | 0x7C00 |
                                    (mantissa >> 13 ? mantissa >> 13 : 1));
      }
    }

    if (exp < -24) {
      // Too small, round to zero
      return static_cast<int16_t>(sign << 15);
    }

    if (exp < -14) {
      // Denormalized number
      mantissa |= 0x800000; // Add implicit leading 1
      int shift = -exp - 14 + 13;
      uint32_t fp16Mantissa = mantissa >> shift;
      // Round to nearest even
      uint32_t remainder = mantissa & ((1 << shift) - 1);
      uint32_t halfway = 1 << (shift - 1);
      if (remainder > halfway || (remainder == halfway && (fp16Mantissa & 1))) {
        fp16Mantissa++;
      }
      return static_cast<int16_t>((sign << 15) | fp16Mantissa);
    }

    if (exp > 15) {
      // Overflow to infinity
      return static_cast<int16_t>((sign << 15) | 0x7C00);
    }

    // Normalized number
    uint32_t fp16Exp = static_cast<uint32_t>(exp + 15);
    uint32_t fp16Mantissa = mantissa >> 13;
    // Round to nearest even
    uint32_t remainder = mantissa & 0x1FFF;
    if (remainder > 0x1000 || (remainder == 0x1000 && (fp16Mantissa & 1))) {
      fp16Mantissa++;
      if (fp16Mantissa > 0x3FF) {
        fp16Mantissa = 0;
        fp16Exp++;
        if (fp16Exp > 30) {
          // Overflow to infinity
          return static_cast<int16_t>((sign << 15) | 0x7C00);
        }
      }
    }

    return static_cast<int16_t>((sign << 15) | (fp16Exp << 10) | fp16Mantissa);
  }

  static constexpr float fp16BitsToFloat(int16_t bits) {
    uint16_t ubits = static_cast<uint16_t>(bits);
    uint32_t sign = (ubits >> 15) & 0x1;
    uint32_t exp = (ubits >> 10) & 0x1F;
    uint32_t mantissa = ubits & 0x3FF;

    uint32_t result;
    if (exp == 0) {
      if (mantissa == 0) {
        // Zero
        result = sign << 31;
      } else {
        // Denormalized number
        exp = 1;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3FF;
        result = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
      }
    } else if (exp == 31) {
      // Inf or NaN
      result = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
      // Normalized number
      result = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    return std::bit_cast<float>(result);
  }
};

// Brain floating point (BFloat16)
// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Same exponent range as float32, just truncated mantissa
//
// This type provides implicit conversions to/from float, allowing seamless
// interoperability with float arithmetic. All operations are performed in
// float precision through these conversions.
struct BFloat16 {
  int16_t data;

  constexpr BFloat16() : data(0) {}

  // Construct from float (handles double via implicit conversion)
  constexpr BFloat16(float f) : data(floatToBf16Bits(f)) {}

  // Convert to float
  constexpr float toFloat() const { return bf16BitsToFloat(data); }

  // Implicit conversion to float for seamless interoperability
  // Arithmetic and comparisons work through this conversion
  constexpr operator float() const { return toFloat(); }

  // Unary negation
  constexpr BFloat16 operator-() const { return BFloat16(-toFloat()); }

  // Compound assignment operators
  constexpr BFloat16 &operator+=(BFloat16 other) {
    *this = BFloat16(toFloat() + other.toFloat());
    return *this;
  }

  constexpr BFloat16 &operator-=(BFloat16 other) {
    *this = BFloat16(toFloat() - other.toFloat());
    return *this;
  }

  constexpr BFloat16 &operator*=(BFloat16 other) {
    *this = BFloat16(toFloat() * other.toFloat());
    return *this;
  }

  constexpr BFloat16 &operator/=(BFloat16 other) {
    *this = BFloat16(toFloat() / other.toFloat());
    return *this;
  }

  // Create from raw bits
  static constexpr BFloat16 fromBits(int16_t bits) {
    BFloat16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  constexpr int16_t toBits() const { return data; }

private:
  static constexpr int16_t floatToBf16Bits(float f) {
    uint32_t bits = std::bit_cast<uint32_t>(f);

    // Round to nearest even
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;

    // Take the upper 16 bits
    return static_cast<int16_t>(bits >> 16);
  }

  static constexpr float bf16BitsToFloat(int16_t bits) {
    // bf16 is just the upper 16 bits of float32
    uint32_t result = static_cast<uint32_t>(static_cast<uint16_t>(bits)) << 16;
    return std::bit_cast<float>(result);
  }
};

// Half precision floating point types.
// On Windows, use portable struct implementations with int16_t storage.
// On other platforms, use compiler extensions for native support.
#ifdef _WIN32
using half = Float16;
using bf16 = BFloat16;
#else
// https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point
// These should be supported by GCC as well.
// TODO(#14): When on C++23, switch to using `std::float16_t`
// and `std::bfloat16_t` from <stdfloat> (C++23).
// https://en.cppreference.com/w/cpp/types/floating-point.html
using half = _Float16;
using bf16 = __bf16;
#endif

} // namespace fusilli

#endif // FUSILLI_SUPPORT_FLOAT_TYPES_H
