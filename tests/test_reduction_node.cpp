// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("ReductionNode getName correctly propagates the attribute name",
          "[reduction_node]") {
  Context ctx;
  ReductionAttr attr;
  attr.setName("reduction");

  ReductionNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "reduction");
}

TEST_CASE("ReductionNode getType returns correct type", "[reduction_node]") {
  Context ctx;
  ReductionAttr attr;
  attr.setMode(ReductionAttr::Mode::SUM);

  ReductionNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::Reduction);
}

TEST_CASE("ReductionNode preValidateNode detects missing mode",
          "[reduction_node]") {
  Context ctx;

  SECTION("Mode not set") {
    ReductionAttr attr;
    ReductionNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Reduction mode not set");
  }

  SECTION("Mode set to SUM without input X") {
    ReductionAttr attr;
    attr.setMode(ReductionAttr::Mode::SUM);
    ReductionNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Reduction operation requires X input");
  }

  SECTION("Mode set to SUM without output Y") {
    ReductionAttr attr;
    attr.setMode(ReductionAttr::Mode::SUM);
    auto x = std::make_shared<TensorAttr>(1.0f);
    attr.setX(x);
    ReductionNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Reduction operation requires Y output");
  }
}

TEST_CASE("ReductionNode postValidateNode detects rank mismatch",
          "[reduction_node]") {
  Context ctx;

  SECTION("Input and output ranks don't match") {
    ReductionAttr attr;
    attr.setMode(ReductionAttr::Mode::SUM);
    auto x = std::make_shared<TensorAttr>();
    x->setDim({2, 3, 4, 5}); // 4D tensor
    auto y = std::make_shared<TensorAttr>();
    y->setDim({2, 3}); // 2D tensor - mismatched rank
    attr.setX(x).setY(y);
    ReductionNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() ==
            "Reduction input and output must have the same rank");
  }
}

TEST_CASE("ReductionNode with tensor attributes", "[reduction_node]") {
  Context ctx;
  ReductionAttr attr;

  attr.setMode(ReductionAttr::Mode::SUM);

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto y = std::make_shared<TensorAttr>(2.0f);

  attr.setX(x).setY(y);

  ReductionNode node(std::move(attr), ctx);

  // Verify the node has access to the attributes
  REQUIRE(node.reductionAttr.getX() == x);
  REQUIRE(node.reductionAttr.getY() == y);
  REQUIRE(node.reductionAttr.getMode() == ReductionAttr::Mode::SUM);

  // Verify tensor properties
  REQUIRE(node.reductionAttr.getX()->getDataType() == DataType::Float);
  REQUIRE(node.reductionAttr.getY()->getDataType() == DataType::Float);

  REQUIRE(node.reductionAttr.getX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(node.reductionAttr.getY()->getDim() == std::vector<int64_t>{1});
}

TEST_CASE("ReductionNode with SUM mode", "[reduction_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  ReductionAttr attr;
  attr.setMode(ReductionAttr::Mode::SUM);

  int64_t n = 16, c = 256, h = 64, w = 32;

  auto x = std::make_shared<TensorAttr>();
  x->setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1});

  auto y = std::make_shared<TensorAttr>();
  y->setDim({n, c, 1, 1}).setStride({c, 1, 1, 1});

  attr.setX(x).setY(y);

  ReductionNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
}

TEST_CASE("ReductionNode with MAX mode full tensor reduction",
          "[reduction_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  ReductionAttr attr;
  attr.setMode(ReductionAttr::Mode::MAX);

  int64_t n = 16, c = 256, h = 64, w = 32;

  auto x = std::make_shared<TensorAttr>();
  x->setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1});

  auto y = std::make_shared<TensorAttr>();
  // Full reduction - all dimensions become 1
  y->setDim({1, 1, 1, 1}).setStride({1, 1, 1, 1});

  attr.setX(x).setY(y);

  ReductionNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
}
