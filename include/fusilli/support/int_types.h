// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Signed 4-bit integer type (Int4) with pack/unpack for IREE's dense
// sub-byte encoding (2 elements per byte, LSB-first).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_INT_TYPES_H
#define FUSILLI_SUPPORT_INT_TYPES_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fusilli {

// Signed 4-bit integer. Range: [-8, 7].
// Stored in the low nibble of a uint8_t.
struct Int4 {
  static constexpr unsigned kBitWidth = 4;

  constexpr Int4() = default;

  // Construct from int8_t with clamping to [-8, 7].
  Int4(int8_t val) {
    data_ =
        static_cast<uint8_t>(std::clamp(static_cast<int>(val), -8, 7) & 0x0F);
  }

  // Convert to int8_t with sign extension from bit 3.
  [[nodiscard]] int8_t toInt() const {
    uint8_t bits = toBits();
    return (bits & 0x08) ? static_cast<int8_t>(bits | 0xF0)
                         : static_cast<int8_t>(bits);
  }

  // Implicit conversion to int8_t.
  [[nodiscard]] operator int8_t() const { return toInt(); }

  // Get raw 4-bit pattern.
  [[nodiscard]] constexpr uint8_t toBits() const { return data_ & 0x0F; }

  // Construct from raw 4-bit pattern without clamping.
  [[nodiscard]] static constexpr Int4 fromBits(uint8_t bits) {
    Int4 result;
    result.data_ = bits & 0x0F;
    return result;
  }

  // Pack N elements into ceil(N/2) bytes.
  // Convention: element[2k] in low nibble, element[2k+1] in high nibble.
  static std::vector<uint8_t> pack(const std::vector<Int4> &elements) {
    size_t packedSize = (elements.size() + 1) / 2;
    std::vector<uint8_t> packed(packedSize, 0);
    for (size_t i = 0; i < elements.size(); i++) {
      unsigned slot = i % 2;
      packed[i / 2] |= static_cast<uint8_t>(elements[i].toBits() << (slot * 4));
    }
    return packed;
  }

  // Unpack bytes into N elements.
  static std::vector<Int4> unpack(const uint8_t *data, size_t count) {
    std::vector<Int4> elements(count);
    for (size_t i = 0; i < count; i++) {
      unsigned slot = i % 2;
      elements[i] =
          Int4::fromBits(static_cast<uint8_t>(data[i / 2] >> (slot * 4)));
    }
    return elements;
  }

private:
  uint8_t data_ = 0;
};

// True if T requires sub-byte packing for IREE HAL buffers.
template <typename T> inline constexpr bool kIsSubByteElement = false;
template <> inline constexpr bool kIsSubByteElement<Int4> = true;

// Type alias matching the `half` and `bf16` pattern.
using int4 = Int4;

} // namespace fusilli

#endif // FUSILLI_SUPPORT_INT_TYPES_H
