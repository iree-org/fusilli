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
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(DataType type,
                                const std::vector<int64_t> &dim) {
  std::string name =
      std::format("pointwise_relu_bwd_dt{}", kDataTypeToMlirTypeAsm.at(type));
  for (const auto &d : dim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Pointwise RELU_BWD", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};
  constexpr DataType kDt = DataType::Half;

  auto execute = [&](Handle &handle, half dy, half x) {
    auto buildNewGraph = [&](Handle &handleArg) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(kDt, dim));
      graph->setIODataType(kDt).setComputeDataType(kDt);

      // Initialize input tensors: IN_0 = grad_output (dy), IN_1 = self (x)
      auto dyT =
          graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
              generateStrideFromDim(dim,
                                    getContiguousStrideOrder(dim.size()))));
      auto xT = graph->tensor(TensorAttr().setName("in1").setDim(dim).setStride(
          generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

      // Create Pointwise RELU_BWD op
      auto pointwiseAttr =
          PointwiseAttr().setMode(PointwiseAttr::Mode::RELU_BWD);
      auto pointwiseResult = graph->pointwise(dyT, xT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

      return std::make_tuple(graph, dyT, xT, pointwiseResult);
    };
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, dyT, xT, dxT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(auto dyBuf,
                           allocateBufferOfType(handle, dyT, kDt, dy));
    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, kDt, x));

    // Allocate output buffer.
    FUSILLI_REQUIRE_ASSIGN(auto dxBuf,
                           allocateBufferOfType(handle, dxT, kDt, 0.0f));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {dyT, dyBuf},
            {xT, xBuf},
            {dxT, dxBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate reference value: dx = dy if x > 0 else 0.
    half expected = static_cast<float>(x) > 0.0f ? dy : half(0.0f);

    // Read output buffer.
    std::vector<half> result;
    FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(static_cast<float>(val) == static_cast<float>(expected));

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; ++i)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(static_cast<float>(val) == static_cast<float>(expected));
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Positive x case: dx = dy.
  execute(handle, half(2.5f), half(1.25f));
  // Negative x case: dx = 0.
  execute(handle, half(2.5f), half(-1.25f));
}
