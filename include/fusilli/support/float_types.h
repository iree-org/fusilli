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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fusilli {

// IEEE 754 half-precision floating point (Float16)
// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
struct Float16 {
  int16_t data;

  Float16() : data(0) {}
  explicit Float16(int16_t raw) : data(raw) {}

  // Construct from float
  explicit Float16(float f) { data = floatToFp16Bits(f); }

  // Convert to float
  float toFloat() const { return fp16BitsToFloat(data); }

  // Implicit conversion to float for arithmetic
  explicit operator float() const { return toFloat(); }

  // Arithmetic operators (perform math in float32)
  Float16 operator+(const Float16 &other) const {
    return Float16(toFloat() + other.toFloat());
  }

  Float16 operator-(const Float16 &other) const {
    return Float16(toFloat() - other.toFloat());
  }

  Float16 operator*(const Float16 &other) const {
    return Float16(toFloat() * other.toFloat());
  }

  Float16 operator/(const Float16 &other) const {
    return Float16(toFloat() / other.toFloat());
  }

  Float16 operator-() const { return Float16(-toFloat()); }

  // Compound assignment operators
  Float16 &operator+=(const Float16 &other) {
    *this = *this + other;
    return *this;
  }

  Float16 &operator-=(const Float16 &other) {
    *this = *this - other;
    return *this;
  }

  Float16 &operator*=(const Float16 &other) {
    *this = *this * other;
    return *this;
  }

  Float16 &operator/=(const Float16 &other) {
    *this = *this / other;
    return *this;
  }

  // Comparison operators
  bool operator==(const Float16 &other) const {
    return toFloat() == other.toFloat();
  }

  bool operator!=(const Float16 &other) const {
    return toFloat() != other.toFloat();
  }

  bool operator<(const Float16 &other) const {
    return toFloat() < other.toFloat();
  }

  bool operator<=(const Float16 &other) const {
    return toFloat() <= other.toFloat();
  }

  bool operator>(const Float16 &other) const {
    return toFloat() > other.toFloat();
  }

  bool operator>=(const Float16 &other) const {
    return toFloat() >= other.toFloat();
  }

  // Create from raw bits
  static Float16 fromBits(int16_t bits) {
    Float16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  int16_t toBits() const { return data; }

private:
  static int16_t floatToFp16Bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

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

  static float fp16BitsToFloat(int16_t bits) {
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

    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
  }
};

// Brain floating point (BFloat16)
// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Same exponent range as float32, just truncated mantissa
struct BFloat16 {
  int16_t data;

  BFloat16() : data(0) {}
  explicit BFloat16(int16_t raw) : data(raw) {}

  // Construct from float
  explicit BFloat16(float f) { data = floatToBf16Bits(f); }

  // Convert to float
  float toFloat() const { return bf16BitsToFloat(data); }

  // Implicit conversion to float for arithmetic
  explicit operator float() const { return toFloat(); }

  // Arithmetic operators (perform math in float32)
  BFloat16 operator+(const BFloat16 &other) const {
    return BFloat16(toFloat() + other.toFloat());
  }

  BFloat16 operator-(const BFloat16 &other) const {
    return BFloat16(toFloat() - other.toFloat());
  }

  BFloat16 operator*(const BFloat16 &other) const {
    return BFloat16(toFloat() * other.toFloat());
  }

  BFloat16 operator/(const BFloat16 &other) const {
    return BFloat16(toFloat() / other.toFloat());
  }

  BFloat16 operator-() const { return BFloat16(-toFloat()); }

  // Compound assignment operators
  BFloat16 &operator+=(const BFloat16 &other) {
    *this = *this + other;
    return *this;
  }

  BFloat16 &operator-=(const BFloat16 &other) {
    *this = *this - other;
    return *this;
  }

  BFloat16 &operator*=(const BFloat16 &other) {
    *this = *this * other;
    return *this;
  }

  BFloat16 &operator/=(const BFloat16 &other) {
    *this = *this / other;
    return *this;
  }

  // Comparison operators
  bool operator==(const BFloat16 &other) const {
    return toFloat() == other.toFloat();
  }

  bool operator!=(const BFloat16 &other) const {
    return toFloat() != other.toFloat();
  }

  bool operator<(const BFloat16 &other) const {
    return toFloat() < other.toFloat();
  }

  bool operator<=(const BFloat16 &other) const {
    return toFloat() <= other.toFloat();
  }

  bool operator>(const BFloat16 &other) const {
    return toFloat() > other.toFloat();
  }

  bool operator>=(const BFloat16 &other) const {
    return toFloat() >= other.toFloat();
  }

  // Create from raw bits
  static BFloat16 fromBits(int16_t bits) {
    BFloat16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  int16_t toBits() const { return data; }

private:
  static int16_t floatToBf16Bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    // Round to nearest even
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;

    // Take the upper 16 bits
    return static_cast<int16_t>(bits >> 16);
  }

  static float bf16BitsToFloat(int16_t bits) {
    // bf16 is just the upper 16 bits of float32
    uint32_t result = static_cast<uint32_t>(static_cast<uint16_t>(bits)) << 16;
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
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
