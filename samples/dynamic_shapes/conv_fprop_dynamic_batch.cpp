// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("Dynamic batch convolution fprop with zero workspace",
          "[dynamic][conv][graph]") {
  const int64_t n = 4, c = 4, h = 4, w = 4, k = 4, r = 1, s = 1;
  const std::vector<int64_t> runtimeNs = {1, 2, n};

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("dynamic_conv_fprop_nchw_kcrs_1x1_nopad");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("image")
                                .setDim({n, c, h, w})
                                .setDynamicDims({0})
                                .setStride({c * h * w, h * w, w, 1}));

    auto wT = graph->tensor(TensorAttr()
                                .setName("filter")
                                .setDim({k, c, r, s})
                                .setStride({c * r * s, r * s, s, 1}));

    auto convAttr = ConvFPropAttr()
                        .setPadding({0, 0})
                        .setStride({1, 1})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto yT = graph->convFProp(xT, wT, convAttr);
    yT->setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    REQUIRE(yT->getDynamicDims() == std::vector<size_t>{0});
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, wT, yT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  auto [graph, xT, wT, yT] = buildNewGraph(handle);

  for (auto runtimeN : runtimeNs) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Running runtimeN=" << runtimeN);

    FUSILLI_REQUIRE_ASSIGN(
        auto xRawBuf,
        Buffer::allocate(handle, castToSizeT({runtimeN, c, h, w}),
                         std::vector<half>(runtimeN * c * h * w, half(1.0f))));
    auto xBuf = std::make_shared<Buffer>(std::move(xRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto wBuf, allocateBufferOfType(handle, wT, DataType::Half, 1.0f));

    FUSILLI_REQUIRE_ASSIGN(
        auto yRawBuf,
        Buffer::allocate(handle, castToSizeT({runtimeN, k, h, w}),
                         std::vector<half>(runtimeN * k * h * w, half(0.0f))));
    auto yBuf = std::make_shared<Buffer>(std::move(yRawBuf));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {{xT, xBuf}, {wT, wBuf}, {yT, yBuf}};

    FUSILLI_REQUIRE_ASSIGN(auto workspaceSize, graph->getWorkspaceSize());
    REQUIRE(workspaceSize == 0);
    FUSILLI_REQUIRE_ASSIGN(auto workspace,
                           allocateWorkspace(handle, workspaceSize));

    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    std::vector<half> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    REQUIRE(result.size() == static_cast<size_t>(runtimeN * k * h * w));
    for (auto val : result)
      REQUIRE(val == half(4.0f));
  }
}
