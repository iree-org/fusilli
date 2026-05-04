// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the element types used throughout Fusilli datastructures.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_TYPES_H
#define FUSILLI_ATTRIBUTES_TYPES_H

#include "fusilli/external/torch_types.h"
#include "fusilli/support/float_types.h"
#include "fusilli/support/int_types.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace fusilli {

// Define a macro to iterate over all fusilli datatypes and the corresponding
// torch datatypes and mlir asm.
#define FUSILLI_FORALL_DATA_TYPES(_)                                           \
  _(Half, Half, "f16")                                                         \
  _(BFloat16, BFloat16, "bf16")                                                \
  _(Float, Float, "f32")                                                       \
  _(Double, Double, "f64")                                                     \
  _(Uint8, Byte, "ui8")                                                        \
  _(Int4, Undefined, "si4")                                                    \
  _(Int8, Char, "si8")                                                         \
  _(Int16, Short, "si16")                                                      \
  _(Int32, Int, "si32")                                                        \
  _(Int64, Long, "si64")                                                       \
  _(Boolean, Bool, "i1")                                                       \
  _(FP8E5M2, Float8_e5m2, "f8E5M2")

enum class DataType : uint8_t {
  NotSet,
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE) FUSILLI_TYPE,
  FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to MLIR types.
static const std::unordered_map<DataType, std::string> kDataTypeToMlirTypeAsm =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE)                       \
  {DataType::FUSILLI_TYPE, MLIR_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to Torch types.
static const std::unordered_map<DataType, torch_upstream::ScalarType>
    kDataTypeToTorchType = {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE)                       \
  {DataType::FUSILLI_TYPE, torch_upstream::ScalarType::TORCH_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from MLIR type ASM strings to Fusilli types.
static const std::unordered_map<std::string, DataType> kMlirTypeAsmToDataType =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE)                       \
  {MLIR_TYPE, DataType::FUSILLI_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Returns whether a dtype has a valid Torch scalar type enum that can be passed
// to ops such as `torch.aten.to.dtype`.
inline bool hasTorchScalarType(DataType dtype) {
  if (dtype == DataType::NotSet)
    return false;
  return kDataTypeToTorchType.at(dtype) !=
         torch_upstream::ScalarType::Undefined;
}

// Returns the storage bitwidth for the given Fusilli dtype.
inline uint8_t getDataTypeBitWidth(DataType dtype) {
  switch (dtype) {
  case DataType::NotSet:
    return 0;
  case DataType::Boolean:
    return 1;
  case DataType::Int4:
    return 4;
  case DataType::Uint8:
  case DataType::Int8:
  case DataType::FP8E5M2:
    return 8;
  case DataType::Half:
  case DataType::BFloat16:
  case DataType::Int16:
    return 16;
  case DataType::Float:
  case DataType::Int32:
    return 32;
  case DataType::Double:
  case DataType::Int64:
    return 64;
  }
  return 0;
}

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_TYPES_H
