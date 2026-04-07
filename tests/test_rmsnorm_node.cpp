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

TEST_CASE("RmsNormNode getName correctly propagates the attribute name",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setName("foo_rmsnorm");

  RmsNormNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_rmsnorm");
}

TEST_CASE("RmsNormNode getType returns correct type", "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  RmsNormNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::RmsNorm);
}

TEST_CASE("RmsNormNode preValidateNode detects missing attributes",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  SECTION("Forward phase not set") {
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RmsNorm forward phase not set");
  }

  SECTION("Input X missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RmsNorm input tensor X not set");
  }

  SECTION("Output Y missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RmsNorm output tensor Y not set");
  }

  SECTION("Epsilon missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RmsNorm epsilon not set");
  }

  SECTION("All required attributes present for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for INFERENCE forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    // For 2D input [N, C], scale is [1, C] (channel-only).
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Extra output INV_RMS for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm output tensor INV_RMS should not be set");
  }

  SECTION("Output INV_RMS missing for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RmsNorm output tensor INV_RMS not set");
  }

  SECTION("All required attributes present for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for TRAINING forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    // For 2D input [N, C], scale is [1, C] (channel-only).
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE(
    "RmsNormNode inferPropertiesNode when output tensors are fully specified",
    "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5, h = 3, w = 4;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
  attr.setY(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
  // INV_RMS has all non-batch dims reduced: [N, 1, 1, 1].
  attr.setINV_RMS(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1, 1, 1}).setStride({1, 1, 1, 1})));

  RmsNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.rmsnormAttr.getY();
  auto rT = node.rmsnormAttr.getINV_RMS();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c, h, w});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c * h * w, h * w, w, 1});
  REQUIRE(rT->getDim() == std::vector<int64_t>{n, 1, 1, 1});
  REQUIRE(rT->getStride() == std::vector<int64_t>{1, 1, 1, 1});
}

TEST_CASE(
    "RmsNormNode inferPropertiesNode when output tensors are under-specified",
    "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5, h = 3, w = 4;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
  attr.setY(std::make_shared<TensorAttr>());
  attr.setINV_RMS(std::make_shared<TensorAttr>());

  RmsNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.rmsnormAttr.getY();
  auto rT = node.rmsnormAttr.getINV_RMS();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c, h, w});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c * h * w, h * w, w, 1});
  REQUIRE(rT->getDim() == std::vector<int64_t>{n, 1, 1, 1});
  REQUIRE(rT->getStride() == std::vector<int64_t>{1, 1, 1, 1});
}

TEST_CASE("RmsNormNode inferPropertiesNode when SCALE tensor is unspecified",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::INFERENCE)
      .setEpsilon(std::make_shared<TensorAttr>(1e-5f));

  int64_t n = 2, c = 5, h = 3, w = 4;

  SECTION("SCALE shape and strides are unspecified (NCHW)") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(TensorAttr()));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    // RMSNorm scale is per-channel: [1, C, 1, 1].
    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, 1, 1});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c, 1, 1, 1});
  }

  SECTION("SCALE shape and strides are partially specified (NHWC)") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, 1, c * w, c})));
    attr.setSCALE(
        std::make_shared<TensorAttr>(TensorAttr().setDim({1, c, 1, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, 1, 1});
    // Stride preserves channels-last format from X; dims 0, 2, 3 are
    // size 1 so their stride values don't affect memory layout.
    REQUIRE(sT->getStride() == std::vector<int64_t>{c, 1, c, c});
  }

  SECTION("SCALE shape and strides are set") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, 1, c * w, c})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, 1, 1}).setStride({c, 1, 1, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, 1, 1});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c, 1, 1, 1});
  }
}

TEST_CASE("RmsNormNode shape checks on SCALE tensor", "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  int64_t n = 2, c = 3, h = 4, w = 5;

  SECTION("Incorrect SCALE shape - matches X (non-unit batch)") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm input tensor SCALE must have per-channel shape "
            "[1, C, 1, ..., 1]");
  }

  SECTION("Incorrect SCALE shape - non-unit spatial dims") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    // [1, C, H, W] is wrong for RMSNorm; should be [1, C, 1, 1].
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm input tensor SCALE must have per-channel shape "
            "[1, C, 1, ..., 1]");
  }
}

TEST_CASE("RmsNormNode postValidateNode detects incorrect shapes and strides",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  int64_t n = 2, c = 3, h = 4, w = 5;

  SECTION("Output Y has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n + 1, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "RmsNorm output Y tensor must have the same shape as input X tensor");
  }

  SECTION("Output Y has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({w, c * h * w, h * w, 1})
                                         .setName("Y_invalid_layout")));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
    REQUIRE(status.getMessage() ==
            "Tensor 'Y_invalid_layout' is neither contiguous nor channels-last "
            "as defined by its stride");
  }

  SECTION("Output INV_RMS has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    // Wrong: batch dim should be N, not 1.
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1, 1}).setStride({1, 1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm output INV_RMS tensor must have shape [N, 1, ..., 1] "
            "with rank equal to input X tensor's rank, and all non-batch "
            "dimensions set to 1");
  }

  SECTION("Output INV_RMS has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setINV_RMS(
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, 1, 1, 1})
                                         .setStride({1, 1, n, 1})
                                         .setName("INV_RMS_invalid_layout")));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm output INV_RMS tensor must have unit strides");
  }

  SECTION("TRAINING forward phase is not yet supported") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1, 1}).setStride({1, 1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  }
}
