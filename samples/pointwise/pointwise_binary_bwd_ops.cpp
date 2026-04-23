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

// Based on parameters, generates a unique name for the graph
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<std::vector<int64_t>> &dims) {
  std::string name =
      std::format("pointwise_{}_dt{}", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (size_t i = 0; i < dims.size(); ++i) {
    name += std::format("_in{}", i);
    for (const auto &d : dims[i]) {
      name += std::format("_{}", d);
    }
  }
  return name;
};

TEST_CASE("Pointwise binary backward ops", "[pointwise][graph]") {
  const auto dims = std::vector<std::vector<int64_t>>{
      std::vector<int64_t>{2, 16, 64, 64}, std::vector<int64_t>{2, 16, 64, 64}};

  // clang-format off
  const auto mode = GENERATE(
      PointwiseAttr::Mode::GELU_BWD,
      PointwiseAttr::Mode::GELU_APPROX_TANH_BWD);
  // clang-format on

  auto execute = [&]<typename T>(Handle &handle, DataType dt, T gradOut, T x) {
    auto buildNewGraph = [&](Handle &handleArg) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dims));
      graph->setIODataType(dt).setComputeDataType(dt);

      // IN_0 is grad_output, IN_1 is the original input (self).
      auto gradOutT = graph->tensor(
          TensorAttr()
              .setName("grad_out")
              .setDim(dims[0])
              .setStride(generateStrideFromDim(
                  dims[0], getContiguousStrideOrder(dims[0].size()))));
      auto xT =
          graph->tensor(TensorAttr().setName("x").setDim(dims[1]).setStride(
              generateStrideFromDim(dims[1],
                                    getContiguousStrideOrder(dims[1].size()))));

      // Create Pointwise op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult = graph->pointwise(gradOutT, xT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

      return std::make_tuple(graph, gradOutT, xT, pointwiseResult);
    };
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, gradOutT, xT, yT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(auto gradOutBuf,
                           allocateBufferOfType(handle, gradOutT, dt, gradOut));
    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, dt, x));

    // Allocate output buffer.
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, dt, 0.0f));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {gradOutT, gradOutBuf},
            {xT, xBuf},
            {yT, yBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate reference value.
    T y = 0;
    const double gradOutD = static_cast<double>(gradOut);
    const double xD = static_cast<double>(x);
    switch (mode) {
    case PointwiseAttr::Mode::GELU_BWD: {
      // d/dx[0.5 * x * (1 + erf(x/sqrt(2)))]
      //   = 0.5 * (1 + erf(x/sqrt(2))) + x/sqrt(2*pi) * exp(-x^2/2)
      constexpr double kSqrt2 = 1.4142135623730951;
      constexpr double kInvSqrt2Pi = 0.3989422804014327;
      const double cdf = 0.5 * (1.0 + std::erf(xD / kSqrt2));
      const double pdf = kInvSqrt2Pi * std::exp(-0.5 * xD * xD);
      y = gradOutD * (cdf + xD * pdf);
      break;
    }
    case PointwiseAttr::Mode::GELU_APPROX_TANH_BWD: {
      // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
      // dinner/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
      // d/dx[0.5 * x * (1 + tanh(inner))]
      //   = 0.5 * (1 + tanh(inner))
      //     + 0.5 * x * (1 - tanh(inner)^2) * dinner/dx
      constexpr double kSqrt2OverPi = 0.7978845608028654;
      const double inner = kSqrt2OverPi * (xD + 0.044715 * xD * xD * xD);
      const double tanhInner = std::tanh(inner);
      const double dInner = kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * xD * xD);
      y = gradOutD * (0.5 * (1.0 + tanhInner) +
                      0.5 * xD * (1.0 - tanhInner * tanhInner) * dInner);
      break;
    }
    default:
      FAIL(
          "Unsupported pointwise mode: " << PointwiseAttr::kModeToStr.at(mode));
    }

    // Read output buffers.
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));

    auto isClose = [](T lhs, T rhs) -> bool {
      const double lhsD = static_cast<double>(lhs);
      const double rhsD = static_cast<double>(rhs);
      const double absTol = 1e-3;
      const double relTol = 1e-3;
      return std::abs(lhsD - rhsD) <= absTol + relTol * std::abs(rhsD);
    };

    for (auto val : result)
      REQUIRE(isClose(val, y));

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; ++i)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(isClose(val, y));
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // fp16: positive and negative inputs.
  execute(handle, DataType::Half, half(1.5f), half(0.75f));
  execute(handle, DataType::Half, half(0.5f), half(-1.25f));
}
