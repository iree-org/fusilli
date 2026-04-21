// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

using namespace fusilli;

TEST_CASE("getSignlessElementTypeAsm strips integer signedness prefix",
          "[asm_emitter_helpers]") {
  REQUIRE(getSignlessElementTypeAsm(DataType::Int8) == "i8");
  REQUIRE(getSignlessElementTypeAsm(DataType::Int16) == "i16");
  REQUIRE(getSignlessElementTypeAsm(DataType::Int32) == "i32");
  REQUIRE(getSignlessElementTypeAsm(DataType::Int64) == "i64");
  REQUIRE(getSignlessElementTypeAsm(DataType::Uint8) == "i8");
}

TEST_CASE("getSignlessElementTypeAsm leaves floats and i1 untouched",
          "[asm_emitter_helpers]") {
  REQUIRE(getSignlessElementTypeAsm(DataType::Half) == "f16");
  REQUIRE(getSignlessElementTypeAsm(DataType::BFloat16) == "bf16");
  REQUIRE(getSignlessElementTypeAsm(DataType::Float) == "f32");
  REQUIRE(getSignlessElementTypeAsm(DataType::Double) == "f64");
  REQUIRE(getSignlessElementTypeAsm(DataType::FP8E5M2) == "f8E5M2");
  REQUIRE(getSignlessElementTypeAsm(DataType::Boolean) == "i1");
}

TEST_CASE("isFloatDataType classifies all float flavours",
          "[asm_emitter_helpers]") {
  REQUIRE(isFloatDataType(DataType::Half));
  REQUIRE(isFloatDataType(DataType::BFloat16));
  REQUIRE(isFloatDataType(DataType::Float));
  REQUIRE(isFloatDataType(DataType::Double));
  REQUIRE(isFloatDataType(DataType::FP8E5M2));
  REQUIRE_FALSE(isFloatDataType(DataType::Int32));
  REQUIRE_FALSE(isFloatDataType(DataType::Boolean));
  REQUIRE_FALSE(isFloatDataType(DataType::NotSet));
}

TEST_CASE("isIntegerDataType classifies integer types (including Boolean)",
          "[asm_emitter_helpers]") {
  REQUIRE(isIntegerDataType(DataType::Uint8));
  REQUIRE(isIntegerDataType(DataType::Int4));
  REQUIRE(isIntegerDataType(DataType::Int8));
  REQUIRE(isIntegerDataType(DataType::Int16));
  REQUIRE(isIntegerDataType(DataType::Int32));
  REQUIRE(isIntegerDataType(DataType::Int64));
  REQUIRE(isIntegerDataType(DataType::Boolean));
  REQUIRE_FALSE(isIntegerDataType(DataType::Float));
  REQUIRE_FALSE(isIntegerDataType(DataType::Half));
  REQUIRE_FALSE(isIntegerDataType(DataType::NotSet));
}

TEST_CASE("buildBuiltinTensorTypeStr uses signless element type",
          "[asm_emitter_helpers]") {
  const std::vector<int64_t> dims = {16, 256};
  REQUIRE(buildBuiltinTensorTypeStr(dims, DataType::Float) ==
          "tensor<16x256xf32>");
  REQUIRE(buildBuiltinTensorTypeStr(dims, DataType::Int32) ==
          "tensor<16x256xi32>");
  REQUIRE(buildBuiltinTensorTypeStr(dims, DataType::Uint8) ==
          "tensor<16x256xi8>");
  REQUIRE(buildBuiltinTensorTypeStr({4}, DataType::Boolean) == "tensor<4xi1>");
}

TEST_CASE("getIdentityAffineMapAsm produces (dN) -> (dN)",
          "[asm_emitter_helpers]") {
  REQUIRE(getIdentityAffineMapAsm(1) == "affine_map<(d0) -> (d0)>");
  REQUIRE(getIdentityAffineMapAsm(2) == "affine_map<(d0, d1) -> (d0, d1)>");
  REQUIRE(getIdentityAffineMapAsm(4) ==
          "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>");
}

TEST_CASE("getIteratorTypesAsm repeats the kind `rank` times",
          "[asm_emitter_helpers]") {
  REQUIRE(getIteratorTypesAsm(1, "parallel") == "\"parallel\"");
  REQUIRE(getIteratorTypesAsm(3, "parallel") ==
          "\"parallel\", \"parallel\", \"parallel\"");
  REQUIRE(getIteratorTypesAsm(2, "reduction") ==
          "\"reduction\", \"reduction\"");
}
