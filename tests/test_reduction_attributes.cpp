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

TEST_CASE("ReductionAttr default constructor", "[reduction_attr]") {
  ReductionAttr attr;
  REQUIRE(attr.getMode() == ReductionAttr::Mode::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("ReductionAttr setters and getters", "[reduction_attr]") {
  ReductionAttr attr;
  ReductionAttr::Mode mode = ReductionAttr::Mode::SUM;

  attr.setMode(mode);

  REQUIRE(attr.getMode() == mode);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto y = std::make_shared<TensorAttr>(2.0f);

  attr.setX(x).setY(y);

  REQUIRE(attr.inputs.size() == 1);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getY() == y);

  REQUIRE(attr.getX()->getDataType() == DataType::Float);
  REQUIRE(attr.getY()->getDataType() == DataType::Float);

  REQUIRE(attr.getX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->isScalar() == true);
  REQUIRE(attr.getY()->isScalar() == true);

  REQUIRE(attr.getX()->isVirtual() == false);
  REQUIRE(attr.getY()->isVirtual() == false);
}
