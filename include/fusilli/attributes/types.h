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

// Category of a DataType. Boolean is treated as integer since it is stored as
// i1 and participates in integer arithmetic / index_cast.
enum class DataTypeCategory : uint8_t {
  Float,
  Integer,
};

// Define a macro to iterate over all fusilli datatypes and the corresponding
// torch datatypes, mlir asm, and category.
#define FUSILLI_FORALL_DATA_TYPES(_)                                           \
  _(Half, Half, "f16", Float)                                                  \
  _(BFloat16, BFloat16, "bf16", Float)                                         \
  _(Float, Float, "f32", Float)                                                \
  _(Double, Double, "f64", Float)                                              \
  _(Uint8, Byte, "ui8", Integer)                                               \
  _(Int4, Undefined, "si4", Integer)                                           \
  _(Int8, Char, "si8", Integer)                                                \
  _(Int16, Short, "si16", Integer)                                             \
  _(Int32, Int, "si32", Integer)                                               \
  _(Int64, Long, "si64", Integer)                                              \
  _(Boolean, Bool, "i1", Integer)                                              \
  _(FP8E5M2, Float8_e5m2, "f8E5M2", Float)

enum class DataType : uint8_t {
  NotSet,
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE, CATEGORY) FUSILLI_TYPE,
  FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to MLIR types.
static const std::unordered_map<DataType, std::string> kDataTypeToMlirTypeAsm =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE, CATEGORY)             \
  {DataType::FUSILLI_TYPE, MLIR_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to Torch types.
static const std::unordered_map<DataType, torch_upstream::ScalarType>
    kDataTypeToTorchType = {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE, CATEGORY)             \
  {DataType::FUSILLI_TYPE, torch_upstream::ScalarType::TORCH_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from MLIR type ASM strings to Fusilli types.
static const std::unordered_map<std::string, DataType> kMlirTypeAsmToDataType =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE, CATEGORY)             \
  {MLIR_TYPE, DataType::FUSILLI_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to their category (float vs integer).
static const std::unordered_map<DataType, DataTypeCategory> kDataTypeCategory =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE, CATEGORY)             \
  {DataType::FUSILLI_TYPE, DataTypeCategory::CATEGORY},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Returns true iff `dtype` is a floating-point type (e.g. f16, bf16, f32).
inline bool isFloatDataType(DataType dtype) {
  auto it = kDataTypeCategory.find(dtype);
  return it != kDataTypeCategory.end() && it->second == DataTypeCategory::Float;
}

// Returns true iff `dtype` is an integer type (including Boolean, which is
// stored as i1).
inline bool isIntegerDataType(DataType dtype) {
  auto it = kDataTypeCategory.find(dtype);
  return it != kDataTypeCategory.end() &&
         it->second == DataTypeCategory::Integer;
}

// Returns the signless MLIR element type for a DataType, stripping the
// leading "s"/"u" that kDataTypeToMlirTypeAsm carries on integer types
// ("si32" -> "i32", "ui8" -> "i8"). Builtin MLIR tensors and arith ops use
// signless integers, unlike torch vtensors which preserve signedness.
inline std::string getSignlessElementTypeAsm(DataType dtype) {
  std::string elemType = kDataTypeToMlirTypeAsm.at(dtype);
  if (elemType.size() >= 2 &&
      (elemType.substr(0, 2) == "si" || elemType.substr(0, 2) == "ui"))
    elemType = "i" + elemType.substr(2);
  return elemType;
}

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_TYPES_H
