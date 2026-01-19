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

TEST_CASE("LayerNormNode getName correctly propagates the attribute name",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setName("foo_layernorm");

  LayerNormNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_layernorm");
}

TEST_CASE("LayerNormNode getType returns correct type", "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  LayerNormNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::LayerNorm);
}

TEST_CASE("LayerNormNode preValidateNode detects missing attributes",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  SECTION("Forward phase not set") {
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm forward phase not set");
  }

  SECTION("Input X missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm input tensor X not set");
  }

  SECTION("Output Y missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm output tensor Y not set");
  }

  SECTION("Epsilon missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm epsilon not set");
  }

  SECTION("All required attributes present for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for INFERENCE forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Extra output MEAN for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor MEAN should not be set");
  }

  SECTION("Extra output INV_VARIANCE for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor INV_VARIANCE should not be set");
  }

  SECTION("Output MEAN missing for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm output tensor MEAN not set");
  }

  SECTION("Output INV_VARIANCE missing for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor INV_VARIANCE not set");
  }

  SECTION("All required attributes present for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for TRAINING forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE(
    "LayerNormNode inferPropertiesNode when output tensors are fully specified",
    "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setMEAN(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1}).setStride({1, 1})));
  attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1}).setStride({1, 1})));

  LayerNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.layernormAttr.getY();
  auto mT = node.layernormAttr.getMEAN();
  auto vT = node.layernormAttr.getINV_VARIANCE();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(mT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(mT->getStride() == std::vector<int64_t>{1, 1});
  REQUIRE(vT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(vT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE(
    "LayerNormNode inferPropertiesNode when output tensors are under-specified",
    "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>());
  attr.setMEAN(std::make_shared<TensorAttr>());
  attr.setINV_VARIANCE(std::make_shared<TensorAttr>());

  LayerNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.layernormAttr.getY();
  auto mT = node.layernormAttr.getMEAN();
  auto vT = node.layernormAttr.getINV_VARIANCE();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(mT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(mT->getStride() == std::vector<int64_t>{1, 1});
  REQUIRE(vT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(vT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE("LayerNormNode shape checks on SCALE and BIAS tensors",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  int64_t n = 2, c = 3, d = 4;

  SECTION("Incorrect SCALE shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm input tensor SCALE must have shape as "
            "tensor X with single batch");
  }

  SECTION("Incorrect BIAS shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm input tensor BIAS must have shape as "
            "tensor X with single batch");
  }
}

TEST_CASE("LayerNormNode postValidateNode detects incorrect shapes and strides",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  int64_t n = 2, c = 3, d = 4;

  SECTION("Output Y has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n + 1, c, d}).setStride({c * d, d, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "LayerNorm output Y tensor must have the same shape as input X tensor");
  }

  SECTION("Output Y has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(TensorAttr()
                                               .setDim({n, c, d})
                                               .setStride({d, c * d, 1})
                                               .setName("Y_invalid_layout")));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
    REQUIRE(status.getMessage() ==
            "Tensor 'Y_invalid_layout' is neither contiguous nor channels-last "
            "as defined by its stride");
  }

  SECTION("Output MEAN has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Layernorm output MEAN tensor must have shape [B, 1, ..., 1] with "
            "rank equal to input X tensor's rank, and batch dimension equal "
            "to input X tensor's batch dimension");
  }

  SECTION("Output MEAN has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({n, 1, 1}).setName(
            "MEAN_invalid_layout")));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output MEAN tensor must have unit strides");
  }

  SECTION("Output INV_VARIANCE has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output INV_VARIANCE tensor must have "
            "shape [B, 1, ..., 1] with  rank equal to "
            "input X tensor's rank, and batch dimension equal "
            "to input X tensor's batch dimension");
  }

  SECTION("Output INV_VARIANCE has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, n}).setName(
            "INV_VARIANCE_invalid_layout")));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output INV_VARIANCE tensor must have unit strides");
  }
}
