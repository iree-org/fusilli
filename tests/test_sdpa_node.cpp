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
#include <string>
#include <utility>
#include <vector>

using namespace fusilli;

// Helper to create a contiguous 4D tensor.
static std::shared_ptr<TensorAttr> makeTensor4D(const std::string &name,
                                                int64_t b, int64_t h, int64_t s,
                                                int64_t d) {
  std::vector<int64_t> dim = {b, h, s, d};
  auto stride =
      generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));
  return std::make_shared<TensorAttr>(
      TensorAttr().setName(name).setDim(dim).setStride(stride));
}

TEST_CASE("SdpaNode getName correctly propagates the attribute name",
          "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;
  attr.setName("foo_sdpa");

  SdpaNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_sdpa");
}

TEST_CASE("SdpaNode getType returns correct type", "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;
  attr.setName("test_sdpa");

  SdpaNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::Sdpa);
}

TEST_CASE("SdpaNode preValidateNode detects missing attributes",
          "[sdpa_node]") {
  Context ctx;

  SECTION("Input Q missing") {
    SdpaAttr attr;
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "SDPA input tensor Q not set");
  }

  SECTION("Input K missing") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "SDPA input tensor K not set");
  }

  SECTION("Input V missing") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "SDPA input tensor V not set");
  }

  SECTION("Output O missing") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "SDPA output tensor O not set");
  }
}

TEST_CASE("SdpaNode preValidateNode rank checks", "[sdpa_node]") {
  Context ctx;

  SECTION("Q must be rank 4") {
    SdpaAttr attr;
    auto q = std::make_shared<TensorAttr>(
        TensorAttr().setDim({64, 64}).setStride({64, 1}));
    attr.setQ(q);
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensor Q must be rank 4 [batch, heads, seq_len, "
            "head_dim]");
  }

  SECTION("K must be rank 4") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    auto k = std::make_shared<TensorAttr>(
        TensorAttr().setDim({8, 64, 64}).setStride({64L * 64, 64, 1}));
    attr.setK(k);
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensor K must be rank 4 [batch, heads, seq_len, "
            "head_dim]");
  }
}

TEST_CASE("SdpaNode preValidateNode dimension checks", "[sdpa_node]") {
  Context ctx;

  SECTION("Batch dimension mismatch") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 2, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensors Q, K, V must have matching batch dimension");
  }

  SECTION("Head dimension mismatch between Q and K") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 32));
    attr.setV(makeTensor4D("V", 1, 8, 64, 32));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensors Q and K must have matching head_dim");
  }

  SECTION("Heads mismatch without GQA") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 4, 64, 64));
    attr.setV(makeTensor4D("V", 1, 4, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA without GQA requires Q heads (8) to equal KV heads (4)");
  }

  SECTION("K and V heads mismatch") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 4, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensors K and V must have matching heads dimension");
  }

  SECTION("K and V sequence length mismatch") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 128, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA input tensors K and V must have matching sequence length");
  }
}

TEST_CASE("SdpaNode GQA validation", "[sdpa_node]") {
  Context ctx;

  SECTION("Valid GQA: Q heads is multiple of KV heads") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 2, 64, 64));
    attr.setV(makeTensor4D("V", 1, 2, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    attr.setEnableGqa(true);
    SdpaNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Invalid GQA: Q heads not a multiple of KV heads") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 7, 64, 64));
    attr.setK(makeTensor4D("K", 1, 2, 64, 64));
    attr.setV(makeTensor4D("V", 1, 2, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    attr.setEnableGqa(true);
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA with GQA requires Q heads (7) to be a multiple of KV heads "
            "(2)");
  }
}

TEST_CASE("SdpaNode mask and is_causal mutual exclusion", "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;

  attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
  attr.setK(makeTensor4D("K", 1, 8, 64, 64));
  attr.setV(makeTensor4D("V", 1, 8, 64, 64));
  attr.setMASK(makeTensor4D("MASK", 1, 1, 64, 64));
  attr.setO(std::make_shared<TensorAttr>());
  attr.setIsCausal(true);

  SdpaNode node(std::move(attr), ctx);

  auto status = node.preValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() ==
          "SDPA attention mask and is_causal are mutually exclusive");
}

TEST_CASE("SdpaNode dropout range validation", "[sdpa_node]") {
  Context ctx;

  SECTION("Negative dropout") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    attr.setDropout(-0.1f);
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA dropout probability must be in [0, 1)");
  }

  SECTION("Dropout >= 1") {
    SdpaAttr attr;
    attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
    attr.setK(makeTensor4D("K", 1, 8, 64, 64));
    attr.setV(makeTensor4D("V", 1, 8, 64, 64));
    attr.setO(std::make_shared<TensorAttr>());
    attr.setDropout(1.0f);
    SdpaNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "SDPA dropout probability must be in [0, 1)");
  }
}

TEST_CASE("SdpaNode valid basic MHA configuration", "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;

  attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
  attr.setK(makeTensor4D("K", 1, 8, 64, 64));
  attr.setV(makeTensor4D("V", 1, 8, 64, 64));
  attr.setO(std::make_shared<TensorAttr>());

  SdpaNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto oT = node.sdpaAttr.getO();
  REQUIRE(oT->getDim() == std::vector<int64_t>{1, 8, 64, 64});
  REQUIRE(oT->getStride() ==
          std::vector<int64_t>{8L * 64 * 64, 64L * 64, 64, 1});

  FUSILLI_REQUIRE_OK(node.postValidateNode());
}

TEST_CASE("SdpaNode output shape inference with cross-attention dimensions",
          "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;

  // Q: [2, 8, 32, 64], K/V: [2, 8, 128, 64] (different seq lengths)
  attr.setQ(makeTensor4D("Q", 2, 8, 32, 64));
  attr.setK(makeTensor4D("K", 2, 8, 128, 64));
  attr.setV(makeTensor4D("V", 2, 8, 128, 64));
  attr.setO(std::make_shared<TensorAttr>());

  SdpaNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto oT = node.sdpaAttr.getO();
  // Output: [batch=2, headsQ=8, seqQ=32, headDim=64]
  REQUIRE(oT->getDim() == std::vector<int64_t>{2, 8, 32, 64});

  FUSILLI_REQUIRE_OK(node.postValidateNode());
}

TEST_CASE("SdpaNode postValidateNode dimension validation", "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;

  attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
  attr.setK(makeTensor4D("K", 1, 8, 64, 64));
  attr.setV(makeTensor4D("V", 1, 8, 64, 64));
  // Wrong O dimensions.
  attr.setO(makeTensor4D("O", 1, 8, 32, 64));

  SdpaNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto status = node.postValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() ==
          "SDPA output tensor O dimensions do not match expected shape "
          "[batch, headsQ, seqQ, headDim]");
}

TEST_CASE("SdpaNode causal flag passes validation", "[sdpa_node]") {
  Context ctx;
  SdpaAttr attr;

  attr.setQ(makeTensor4D("Q", 1, 8, 64, 64));
  attr.setK(makeTensor4D("K", 1, 8, 64, 64));
  attr.setV(makeTensor4D("V", 1, 8, 64, 64));
  attr.setO(std::make_shared<TensorAttr>());
  attr.setIsCausal(true);

  SdpaNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());
}
