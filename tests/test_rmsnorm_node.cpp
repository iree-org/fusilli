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

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setINV_RMS(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1}).setStride({1, 1})));

  RmsNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.rmsnormAttr.getY();
  auto rT = node.rmsnormAttr.getINV_RMS();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(rT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(rT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE(
    "RmsNormNode inferPropertiesNode when output tensors are under-specified",
    "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>());
  attr.setINV_RMS(std::make_shared<TensorAttr>());

  RmsNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.rmsnormAttr.getY();
  auto rT = node.rmsnormAttr.getINV_RMS();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(rT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(rT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE("RmsNormNode inferPropertiesNode when SCALE tensor is unspecified",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::INFERENCE)
      .setEpsilon(std::make_shared<TensorAttr>(1e-5f));

  int64_t n = 2, c = 5, d = 10;

  SECTION("SCALE shape and strides are unspecified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(TensorAttr()));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, d, 1});
  }

  SECTION("SCALE shape and strides are partially specified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setSCALE(std::make_shared<TensorAttr>(TensorAttr().setDim({1, c, d})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }

  SECTION("SCALE shape and strides are set") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, d}).setStride({c * d, 1, c})));
    attr.setY(std::make_shared<TensorAttr>());
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.rmsnormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }
}

TEST_CASE("RmsNormNode shape checks on SCALE tensor", "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  // 4D NCHW-packed x for trailing-suffix coverage.
  int64_t n = 2, c = 4, h = 8, w = 8;
  std::vector<int64_t> xDim = {n, c, h, w};
  std::vector<int64_t> xStride = {c * h * w, h * w, w, 1};

  attr.setForwardPhase(NormFwdPhase::INFERENCE)
      .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim(xDim).setStride(xStride)));
  attr.setY(std::make_shared<TensorAttr>());

  // Each SECTION re-runs from the top of the TEST_CASE body, so `attr` is
  // freshly populated; we override SCALE per SECTION to test the trailing-
  // suffix rule:
  //   reduction = maximal trailing suffix where scale[i] == x[i],
  //   leading region (excluding batch) must be all-1,
  //   batch dim (scale[0]) must be 1.

  SECTION("Canonical full suffix accepted") {
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Trailing suffix [H,W] accepted") {
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, h, w}).setStride({h * w, h * w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Trailing suffix [W] only accepted") {
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1, w}).setStride({w, w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Leading region non-1 rejected (sandwich)") {
    // scale=[1, c, 1, w]: trailing match is [w] (matchCount=1), but the
    // leading region (positions 1..2) contains scale[1]=c != 1.
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, 1, w}).setStride({c * w, w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "RmsNorm SCALE leading region (before "
                                   "normalized shape) must be 1");
  }

  SECTION("No trailing match rejected (per-channel)") {
    // scale=[1, c, 1, 1]: trailing dim 1 vs x's W=w → no match.
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, 1, 1}).setStride({c, 1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm SCALE has no trailing dims matching X — at least "
            "one normalized dim is required");
  }

  SECTION("All-1 scale on non-degenerate x rejected") {
    // x=[N,C,H,W] with non-1 spatials, scale=[1,1,1,1] → no trailing
    // match (scale[3]=1 vs x[3]=W).
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1, 1}).setStride({1, 1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm SCALE has no trailing dims matching X — at least "
            "one normalized dim is required");
  }

  SECTION("Different rank rejected") {
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({c, h, w}).setStride({h * w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm SCALE tensor must have the same rank as X");
  }

  SECTION("Non-1 batch dim rejected") {
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1})));
    RmsNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm SCALE tensor must have batch dim equal to 1 "
            "(broadcast across batch)");
  }
}

TEST_CASE("RmsNormNode degenerate all-1 x accepted with all-1 scale",
          "[rmsnorm_node]") {
  // Special case from the cuDNN/hipDNN rule: x=[N,1,1,...] with
  // scale=[1,1,1,...] passes because trailing match goes all the way to
  // the batch boundary.
  Context ctx;
  RmsnormAttr attr;

  attr.setForwardPhase(NormFwdPhase::INFERENCE)
      .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({2, 1, 1, 1}).setStride({1, 1, 1, 1})));
  attr.setSCALE(std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 1, 1, 1}).setStride({1, 1, 1, 1})));
  attr.setY(std::make_shared<TensorAttr>());
  RmsNormNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
}

TEST_CASE("RmsNormNode postValidateNode detects incorrect shapes and strides",
          "[rmsnorm_node]") {
  Context ctx;
  RmsnormAttr attr;

  int64_t n = 2, c = 3, d = 4;

  SECTION("Output Y has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n + 1, c, d}).setStride({c * d, d, 1})));
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
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(TensorAttr()
                                               .setDim({n, c, d})
                                               .setStride({d, c * d, 1})
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
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RmsNorm output INV_RMS tensor must have shape [B, 1, ..., 1] with "
            "rank equal to input X tensor's rank, and batch dimension equal "
            "to input X tensor's batch dimension");
  }

  SECTION("Output INV_RMS has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, n}).setName(
            "INV_RMS_invalid_layout")));
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
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setINV_RMS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    RmsNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  }
}
