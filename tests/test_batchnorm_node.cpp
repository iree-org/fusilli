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

TEST_CASE("BatchNormNode getName correctly propagates the attribute name",
          "[batchnorm_node]") {
  Context ctx;
  BatchnormAttr attr;
  attr.setName("foo_batchnorm");

  BatchNormNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_batchnorm");
}

TEST_CASE("BatchNormNode getType returns correct type", "[batchnorm_node]") {
  Context ctx;
  BatchnormAttr attr;

  BatchNormNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::BatchNorm);
}

TEST_CASE("BatchNormNode preValidateNode detects missing attributes",
          "[batchnorm_node]") {
  Context ctx;

  SECTION("Forward phase not set") {
    BatchnormAttr attr;
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm forward phase not set");
  }

  SECTION("Input X missing") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm input tensor X not set");
  }

  SECTION("Output Y missing") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm output tensor Y not set");
  }

  SECTION("Epsilon missing") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setMEAN(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setVAR(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm epsilon not set");
  }

  SECTION("Momentum missing") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setMEAN(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setVAR(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm momentum not set");
  }

  SECTION("Inference mode missing running MEAN") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm inference requires running MEAN");
  }

  SECTION("Inference mode missing running VAR") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    attr.setMEAN(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BatchNorm inference requires running VAR");
  }

  SECTION("Training mode missing SAVED_MEAN output") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::TRAINING);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4, 8, 8}).setStride({4 * 8 * 8, 8 * 8, 8, 1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() ==
            "BatchNorm training requires SAVED_MEAN output");
  }

  SECTION("X rank too low") {
    BatchnormAttr attr;
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setY(
        std::make_shared<TensorAttr>(TensorAttr().setDim({4}).setStride({1})));
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setMomentum(std::make_shared<TensorAttr>(0.1f));
    BatchNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "BatchNorm input tensor X must have a rank of at least 2");
  }
}

TEST_CASE("BatchNormNode inferPropertiesNode infers Y shape from X",
          "[batchnorm_node]") {
  int64_t n = 2, c = 4, h = 8, w = 8;
  Context ctx;
  ctx.setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = std::make_shared<TensorAttr>(
      TensorAttr()
          .setName("x")
          .setDim({n, c, h, w})
          .setDataType(DataType::Float)
          .setStride({c * h * w, h * w, w, 1})); // NCHW
  auto meanT = std::make_shared<TensorAttr>(TensorAttr()
                                                .setName("mean")
                                                .setDim({c})
                                                .setDataType(DataType::Float)
                                                .setStride({1}));
  auto varT = std::make_shared<TensorAttr>(TensorAttr()
                                               .setName("var")
                                               .setDim({c})
                                               .setDataType(DataType::Float)
                                               .setStride({1}));
  auto epsT = std::make_shared<TensorAttr>(TensorAttr(1e-5f).setName("eps"));
  auto momT = std::make_shared<TensorAttr>(TensorAttr(0.1f).setName("mom"));
  auto yT = std::make_shared<TensorAttr>(
      TensorAttr().setName("y").setIsVirtual(true));

  BatchnormAttr attr;
  attr.setName("bn")
      .setForwardPhase(NormFwdPhase::INFERENCE)
      .setX(xT)
      .setMEAN(meanT)
      .setVAR(varT)
      .setEpsilon(epsT)
      .setMomentum(momT)
      .setY(yT);

  BatchNormNode node(std::move(attr), ctx);

  REQUIRE(!isError(node.preValidateNode()));
  REQUIRE(!isError(node.inferPropertiesNode()));

  REQUIRE(yT->getDim() == std::vector<int64_t>({n, c, h, w}));
  REQUIRE(yT->getStride() == xT->getStride());
}
