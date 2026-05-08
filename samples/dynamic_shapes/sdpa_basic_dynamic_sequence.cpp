// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("Dynamic sequence basic SDPA f16", "[dynamic][sdpa][graph]") {
  const int64_t batch = 1;
  const int64_t heads = 8;
  const int64_t representativeSeqLen = 64;
  const int64_t headDim = 64;
  const std::vector<int64_t> runtimeSeqLens = {16, 32, representativeSeqLen};

  auto graph = std::make_shared<Graph>();
  graph->setName("dynamic_sdpa_basic_mha")
      .setIODataType(DataType::Half)
      .setIntermediateDataType(DataType::Half);

  std::vector<int64_t> dims = {batch, heads, representativeSeqLen, headDim};
  std::vector<int64_t> stride =
      generateStrideFromDim(dims, getContiguousStrideOrder(dims.size()));

  auto qT = graph->tensor(
      TensorAttr().setName("q").setDim(dims).setDynamicDims({2}).setStride(
          stride));
  auto kT = graph->tensor(
      TensorAttr().setName("k").setDim(dims).setDynamicDims({2}).setStride(
          stride));
  auto vT = graph->tensor(
      TensorAttr().setName("v").setDim(dims).setDynamicDims({2}).setStride(
          stride));

  SdpaAttr sdpaAttr;
  sdpaAttr.setName("sdpa")
      .setDropout(0.0f)
      .setIsCausal(false)
      .setScale(std::nullopt)
      .setEnableGqa(false);

  auto oT = graph->sdpa(qT, kT, vT, /*mask=*/nullptr, sdpaAttr);
  oT->setDynamicDims({2}).setOutput(true);

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  for (int64_t runtimeSeqLen : runtimeSeqLens) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Running runtimeSeqLen=" << runtimeSeqLen);

    std::vector<int64_t> runtimeDims = {batch, heads, runtimeSeqLen, headDim};
    size_t runtimeSize =
        static_cast<size_t>(batch * heads * runtimeSeqLen * headDim);

    FUSILLI_REQUIRE_ASSIGN(
        auto qRawBuf,
        Buffer::allocate(handle, castToSizeT(runtimeDims),
                         std::vector<half>(runtimeSize, half(0.01f))));
    auto qBuf = std::make_shared<Buffer>(std::move(qRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto kRawBuf,
        Buffer::allocate(handle, castToSizeT(runtimeDims),
                         std::vector<half>(runtimeSize, half(0.01f))));
    auto kBuf = std::make_shared<Buffer>(std::move(kRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto vRawBuf,
        Buffer::allocate(handle, castToSizeT(runtimeDims),
                         std::vector<half>(runtimeSize, half(0.01f))));
    auto vBuf = std::make_shared<Buffer>(std::move(vRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto oRawBuf,
        Buffer::allocate(handle, castToSizeT(runtimeDims),
                         std::vector<half>(runtimeSize, half(0.0f))));
    auto oBuf = std::make_shared<Buffer>(std::move(oRawBuf));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {{qT, qBuf}, {kT, kBuf}, {vT, vBuf}, {oT, oBuf}};

    FUSILLI_REQUIRE_ASSIGN(auto workspaceSize, graph->getWorkspaceSize());
    FUSILLI_REQUIRE_ASSIGN(auto workspace,
                           allocateWorkspace(handle, workspaceSize));

    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    std::vector<half> result;
    FUSILLI_REQUIRE_OK(oBuf->read(handle, result));
    REQUIRE(result.size() == runtimeSize);
    for (auto val : result)
      REQUIRE(std::abs(static_cast<float>(val) - 0.01f) < 1e-2f);
  }
}
