// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("RmsnormAttr default constructor", "[rmsnorm_attr]") {
  RmsnormAttr attr;
  REQUIRE(attr.getForwardPhase() == NormFwdPhase::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("RmsnormAttr setters and getters", "[rmsnorm_attr]") {
  RmsnormAttr attr;

  NormFwdPhase phase = NormFwdPhase::INFERENCE;

  attr.setForwardPhase(phase);

  REQUIRE(attr.getForwardPhase() == phase);

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto s = std::make_shared<TensorAttr>(2.0f);
  auto e = std::make_shared<TensorAttr>(3.0f);
  auto y = std::make_shared<TensorAttr>(4.0f);
  auto r = std::make_shared<TensorAttr>(5.0f);

  attr.setX(x).setSCALE(s).setY(y).setINV_RMS(r);
  attr.setEpsilon(e);

  REQUIRE(attr.inputs.size() == 3);
  REQUIRE(attr.outputs.size() == 2);

#define CHECK_TENSOR_PROPERTIES(NAME, TENSOR)                                  \
  REQUIRE(attr.get##NAME() == TENSOR);                                         \
  REQUIRE(attr.get##NAME()->getDataType() == DataType::Float);                 \
  REQUIRE(attr.get##NAME()->getDim() == std::vector<int64_t>{1});              \
  REQUIRE(attr.get##NAME()->getStride() == std::vector<int64_t>{1});           \
  REQUIRE(attr.get##NAME()->isScalar() == true);                               \
  REQUIRE(attr.get##NAME()->isVirtual() == false)

  CHECK_TENSOR_PROPERTIES(X, x);
  CHECK_TENSOR_PROPERTIES(SCALE, s);
  CHECK_TENSOR_PROPERTIES(Y, y);
  CHECK_TENSOR_PROPERTIES(INV_RMS, r);

  CHECK_TENSOR_PROPERTIES(Epsilon, e);

#undef CHECK_TENSOR_PROPERTIES
}
