// Copyright 2025 Advanced Micro Devices, Inc.
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
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &dim) {
  std::string name =
      std::format("pointwise_{}_dt{}_in0", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (const auto &d : dim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Pointwise unary ops", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};

  auto supportsInteger = [](PointwiseAttr::Mode m) {
    switch (m) {
    case PointwiseAttr::Mode::ABS:
    case PointwiseAttr::Mode::NEG:
    case PointwiseAttr::Mode::RELU_FWD:
      return true;
    default:
      return false;
    }
  };

  auto supportsFloat = [](PointwiseAttr::Mode m) {
    switch (m) {
    case PointwiseAttr::Mode::ABS:
    case PointwiseAttr::Mode::CEIL:
    case PointwiseAttr::Mode::ERF:
    case PointwiseAttr::Mode::EXP:
    case PointwiseAttr::Mode::FLOOR:
    case PointwiseAttr::Mode::NEG:
    case PointwiseAttr::Mode::RECIPROCAL:
    case PointwiseAttr::Mode::RELU_FWD:
    case PointwiseAttr::Mode::SIGMOID_FWD:
    case PointwiseAttr::Mode::TANH_FWD:
      return true;
    default:
      return false;
    }
  };

  // clang-format off
  const auto mode = GENERATE(
      PointwiseAttr::Mode::ABS,
      PointwiseAttr::Mode::CEIL,
      PointwiseAttr::Mode::ELU_FWD,
      PointwiseAttr::Mode::ERF,
      PointwiseAttr::Mode::EXP,
      PointwiseAttr::Mode::FLOOR,
      PointwiseAttr::Mode::NEG,
      PointwiseAttr::Mode::RECIPROCAL,
      PointwiseAttr::Mode::RELU_FWD,
      PointwiseAttr::Mode::SIGMOID_FWD,
      PointwiseAttr::Mode::TANH_FWD);
  // clang-format on

  auto execute = [&]<typename T>(Handle &handle, DataType dt, T x) {
    auto buildNewGraph = [&](Handle &handleArg) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dim));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Initialize input tensors
      auto xT = graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
          generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

      // Create Pointwise unary op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult = graph->pointwise(xT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

      return std::make_tuple(graph, xT, pointwiseResult);
    };
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, xT, yT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, dt, x));

    // Allocate output buffer.
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, dt, 0.0f));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {xT, xBuf},
            {yT, yBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate reference value
    T y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::ABS: {
      double xD = static_cast<double>(x);
      y = std::abs(xD);
      break;
    }
    case PointwiseAttr::Mode::ELU_FWD: {
      double xD = static_cast<double>(x);
      // ELU(x) = x if x > 0 else alpha * (exp(x) - 1).
      // The graph uses the default alpha (1.0), so std::expm1 suffices.
      y = xD > 0 ? xD : std::expm1(xD);
      break;
    }
    case PointwiseAttr::Mode::RELU_FWD: {
      y = std::max(x, T(0));
      break;
    }
    case PointwiseAttr::Mode::SIGMOID_FWD: {
      double xD = static_cast<double>(x);
      y = T(1) / (T(1) + std::exp(-xD));
      break;
    }
    case PointwiseAttr::Mode::TANH_FWD: {
      double xD = static_cast<double>(x);
      y = std::tanh(xD);
      break;
    }
    case PointwiseAttr::Mode::CEIL: {
      double xD = static_cast<double>(x);
      y = std::ceil(xD);
      break;
    }
    case PointwiseAttr::Mode::ERF: {
      double xD = static_cast<double>(x);
      y = std::erf(xD);
      break;
    }
    case PointwiseAttr::Mode::EXP: {
      double xD = static_cast<double>(x);
      y = std::exp(xD);
      break;
    }
    case PointwiseAttr::Mode::FLOOR: {
      double xD = static_cast<double>(x);
      y = std::floor(xD);
      break;
    }
    case PointwiseAttr::Mode::NEG: {
      y = -x;
      break;
    }
    case PointwiseAttr::Mode::RECIPROCAL: {
      double xD = static_cast<double>(x);
      y = 1.0 / xD;
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
      if (std::is_floating_point<T>::value || std::is_same<T, half>::value) {
        return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs)) <
               1e-3;
      }
      return lhs == rhs;
    };

    for (auto val : result) {
      REQUIRE(isClose(val, y));
    }

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

  // int32
  if (supportsInteger(mode))
    execute(handle, DataType::Int32, int(-128));
  // fp16
  if (supportsFloat(mode))
    execute(handle, DataType::Half, half(3.14));
}
