// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
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

TEST_CASE("Pointwise unary logical ops", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};

  // clang-format off
  const auto mode = GENERATE(
      PointwiseAttr::Mode::LOGICAL_NOT);
  // clang-format on

  auto execute = [&]<typename T>(Handle &handle, DataType dt, T x) {
    // Create graph.
    auto graph = std::make_shared<Graph>();
    graph->setName(generateName(mode, dt, dim));
    graph->setIODataType(dt).setComputeDataType(dt);

    // Initialize input tensor.
    auto xT = graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

    // Create Pointwise op.
    auto pointwiseAttr = PointwiseAttr().setMode(mode);
    auto yT = graph->pointwise(xT, pointwiseAttr);

    yT->setName("result").setOutput(true);

    // Validate and compile.
    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    // Allocate input buffer.
    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, dt, x));

    // Allocate output buffer.
    DataType yDt = DataType::Boolean;
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, yDt, false));

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
    bool y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::LOGICAL_NOT: {
      y = !(x != T(0));
      break;
    }
    default:
      FAIL(
          "Unsupported pointwise mode: " << PointwiseAttr::kModeToStr.at(mode));
    }

    // Read output buffers.
    std::vector<uint8_t> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result) {
      REQUIRE(val == y);
    }

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; ++i)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // int32: zero (false) and non-zero (true) cases.
  execute(handle, DataType::Int32, int(0));
  execute(handle, DataType::Int32, int(-50));
  // fp16: zero (false) and non-zero (true) cases.
  execute(handle, DataType::Half, half(0.0));
  execute(handle, DataType::Half, half(1.0));
}
