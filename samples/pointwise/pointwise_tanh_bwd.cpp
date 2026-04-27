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
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Pointwise TANH_BWD op", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};

  auto buildNewGraph = [&](Handle &handle) {
    // Create graph
    auto graph = std::make_shared<Graph>();
    graph->setName("pointwise_tanh_bwd");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Half);

    // Initialize input tensors: dy (grad_output) and x (forward input).
    auto dyT = graph->tensor(TensorAttr().setName("dy").setDim(dim).setStride(
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));
    auto xT = graph->tensor(TensorAttr().setName("x").setDim(dim).setStride(
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

    // Create Pointwise TANH_BWD op
    auto pointwiseAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::TANH_BWD);
    auto dxT = graph->pointwise(dyT, xT, pointwiseAttr);

    dxT->setName("result").setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, dxT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, dyT, xT, dxT] = buildNewGraph(handle);

  // Input values.
  const half dy = half(2.0f);
  const half x = half(0.5f);

  // Allocate input buffers.
  FUSILLI_REQUIRE_ASSIGN(auto dyBuf,
                         allocateBufferOfType(handle, dyT, DataType::Half, dy));
  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, DataType::Half, x));

  // Allocate output buffer.
  FUSILLI_REQUIRE_ASSIGN(
      auto dxBuf, allocateBufferOfType(handle, dxT, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dxT, dxBuf},
      };

  // Allocate workspace buffer if needed.
  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  // Calculate reference value: y = tanh(x); dx = dy * (1 - y*y)
  const double xD = static_cast<double>(x);
  const double dyD = static_cast<double>(dy);
  const double yD = std::tanh(xD);
  const half expected = half(dyD * (1.0 - yD * yD));

  // Read output buffer.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));

  auto isClose = [](half lhs, half rhs) -> bool {
    const double lhsD = static_cast<double>(lhs);
    const double rhsD = static_cast<double>(rhs);
    const double absTol = 1e-3;
    const double relTol = 1e-3;
    return std::abs(lhsD - rhsD) <= absTol + relTol * std::abs(rhsD);
  };

  for (auto val : result) {
    REQUIRE(isClose(val, expected));
  }

  // Execute graph a few times and re-check.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; ++i)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  result.clear();
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(isClose(val, expected));
}
