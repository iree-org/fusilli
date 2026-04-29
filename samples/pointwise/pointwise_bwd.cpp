// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph.
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &dim) {
  std::string name =
      std::format("pointwise_{}_dt{}", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (const auto &d : dim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Pointwise binary backward ops", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};
  constexpr float kSwishBeta = 2.0f;

  // clang-format off
  const auto mode = GENERATE(
      PointwiseAttr::Mode::SWISH_BWD,
      PointwiseAttr::Mode::TANH_BWD);
  // clang-format on

  auto execute = [&]<typename T>(Handle &handle, DataType dt, T dy, T x) {
    auto buildNewGraph = [&](Handle &handleArg) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dim));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Initialize input tensors: IN_0 = grad_output (dy), IN_1 = self (x).
      auto dyT =
          graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
              generateStrideFromDim(dim,
                                    getContiguousStrideOrder(dim.size()))));
      auto xT = graph->tensor(TensorAttr().setName("in1").setDim(dim).setStride(
          generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

      // Create Pointwise op.
      auto pointwiseAttr =
          PointwiseAttr().setMode(mode).setSwishBeta(kSwishBeta);
      auto pointwiseResult = graph->pointwise(dyT, xT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties.
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile.
      FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

      return std::make_tuple(graph, dyT, xT, pointwiseResult);
    };
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, dyT, xT, dxT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(auto dyBuf,
                           allocateBufferOfType(handle, dyT, dt, dy));
    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, dt, x));

    // Allocate output buffer.
    FUSILLI_REQUIRE_ASSIGN(auto dxBuf,
                           allocateBufferOfType(handle, dxT, dt, 0.0f));

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

    // Calculate reference value. Match the device's rounding by round-tripping
    // inputs through T before computing in double.
    const double dyD = static_cast<double>(T(dy));
    const double xD = static_cast<double>(T(x));
    T expected = T(0.0f);
    switch (mode) {
    case PointwiseAttr::Mode::SWISH_BWD: {
      // For y = x * sigmoid(beta * x):
      //   dx = dy * (sig + beta * x * sig * (1 - sig)),
      // where sig = sigmoid(beta * x).
      const double betaD = static_cast<double>(kSwishBeta);
      const double sig = 1.0 / (1.0 + std::exp(-betaD * xD));
      const double comp = sig + betaD * xD * sig * (1.0 - sig);
      expected = T(dyD * comp);
      break;
    }
    case PointwiseAttr::Mode::TANH_BWD: {
      // For y = tanh(x): dx = dy * (1 - y * y).
      const double yD = std::tanh(xD);
      expected = T(dyD * (1.0 - yD * yD));
      break;
    }
    default:
      FAIL(
          "Unsupported pointwise mode: " << PointwiseAttr::kModeToStr.at(mode));
    }

    auto isClose = [](T lhs, T rhs) -> bool {
      const double lhsD = static_cast<double>(lhs);
      const double rhsD = static_cast<double>(rhs);
      const double absTol = 1e-3;
      const double relTol = 1e-3;
      return std::abs(lhsD - rhsD) <= absTol + relTol * std::abs(rhsD);
    };

    // Read output buffer and verify against reference.
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(isClose(val, expected));

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; ++i)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(dxBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(isClose(val, expected));
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // fp16
  execute(handle, DataType::Half, half(0.5f), half(1.25f));
}
