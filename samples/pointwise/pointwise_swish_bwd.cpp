// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Pointwise SWISH_BWD op", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};
  constexpr float kSwishBeta = 1.0f;
  const DataType dt = DataType::Half;

  auto buildNewGraph = [&](Handle &handleArg) {
    // Create graph
    auto graph = std::make_shared<Graph>();
    graph->setName("pointwise_swish_bwd");
    graph->setIODataType(dt).setComputeDataType(dt);

    // Initialize input tensors: IN_0 = grad_output (dy), IN_1 = self (x).
    auto dyT =
        graph->tensor(TensorAttr()
                          .setName("grad_out")
                          .setDim(dim)
                          .setStride(generateStrideFromDim(
                              dim, getContiguousStrideOrder(dim.size()))));
    auto xT = graph->tensor(TensorAttr().setName("self").setDim(dim).setStride(
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

    // Create Pointwise SWISH_BWD op.
    auto pointwiseAttr = PointwiseAttr()
                             .setMode(PointwiseAttr::Mode::SWISH_BWD)
                             .setSwishBeta(kSwishBeta);
    auto pointwiseResult = graph->pointwise(dyT, xT, pointwiseAttr);

    pointwiseResult->setName("result").setOutput(true);

    // Validate, infer missing properties.
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile.
    FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, pointwiseResult);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, dyT, xT, dxT] = buildNewGraph(handle);

  // Input values.
  constexpr double kDyVal = 0.5;
  constexpr double kXVal = 1.25;

  // Allocate input buffers.
  FUSILLI_REQUIRE_ASSIGN(auto dyBuf,
                         allocateBufferOfType(handle, dyT, dt, kDyVal));
  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, dt, kXVal));

  // Allocate output buffer.
  FUSILLI_REQUIRE_ASSIGN(auto dxBuf,
                         allocateBufferOfType(handle, dxT, dt, 0.0f));

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

  // Execute graph.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  // Reference computation: for y = x * sigmoid(beta * x),
  // dx = dy * (sig + beta * x * sig * (1 - sig))
  // where sig = sigmoid(beta * x) = 1 / (1 + exp(-beta * x)).
  // Match the device's fp16 rounding of inputs by round-tripping through half.
  const double dyD = static_cast<double>(half(kDyVal));
  const double xD = static_cast<double>(half(kXVal));
  const double betaD = static_cast<double>(kSwishBeta);
  const double sig = 1.0 / (1.0 + std::exp(-betaD * xD));
  const double dinner = sig + betaD * xD * sig * (1.0 - sig);
  const half expected(dyD * dinner);

  auto isClose = [](half lhs, half rhs) -> bool {
    const double lhsD = static_cast<double>(lhs);
    const double rhsD = static_cast<double>(rhs);
    const double absTol = 1e-3;
    const double relTol = 1e-3;
    return std::abs(lhsD - rhsD) <= absTol + relTol * std::abs(rhsD);
  };

  // Read output buffer and verify against reference.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(isClose(val, expected));

  // Execute graph again to exercise repeat dispatch.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));
  result.clear();
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(isClose(val, expected));
}
