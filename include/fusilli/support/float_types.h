// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains portable Float16 and BFloat16 implementations represented
// as structs with uint16_t storage. These types support conversion to/from
// float and perform arithmetic operations in 32-bit float precision.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_FLOAT_TYPES_H
#define FUSILLI_SUPPORT_FLOAT_TYPES_H

#include "fusilli/support/target_platform.h"

#include <iree/base/internal/math.h>

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
  uint16_t data;

  constexpr Float16() : data(0) {}

  // Construct from float (handles double via implicit conversion)
  constexpr Float16(float f) : data(iree_math_f32_to_f16(f)) {}

  // Convert to float
  [[nodiscard]] constexpr float toFloat() const {
    return iree_math_f16_to_f32(data);
  }

  // Implicit conversion to float for seamless interoperability
  // Arithmetic and comparisons work through this conversion
  [[nodiscard]] constexpr operator float() const { return toFloat(); }

  [[nodiscard]] static constexpr Float16 fromBits(uint16_t bits) {
    Float16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  [[nodiscard]] constexpr uint16_t toBits() const { return data; }
};

// Brain floating point (BFloat16)
// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
// Same exponent range as float32, just truncated mantissa
//
// This type provides implicit conversions to/from float, allowing seamless
// interoperability with float arithmetic. All operations are performed in
// float precision through these conversions.
struct BFloat16 {
  uint16_t data;

  constexpr BFloat16() : data(0) {}

  // Construct from float (handles double via implicit conversion)
  constexpr BFloat16(float f) : data(iree_math_f32_to_bf16(f)) {}

  // Convert to float
  [[nodiscard]] constexpr float toFloat() const {
    return iree_math_bf16_to_f32(data);
  }

  // Implicit conversion to float for seamless interoperability
  // Arithmetic and comparisons work through this conversion
  [[nodiscard]] constexpr operator float() const { return toFloat(); }

  [[nodiscard]] static constexpr BFloat16 fromBits(uint16_t bits) {
    BFloat16 result;
    result.data = bits;
    return result;
  }

  // Get raw bits
  [[nodiscard]] constexpr uint16_t toBits() const { return data; }
};

// Half precision floating point types.
// On Windows, use portable struct implementations with uint16_t storage.
// On other platforms, use compiler extensions for native support.
#ifdef FUSILLI_PLATFORM_WINDOWS
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
