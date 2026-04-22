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

TEST_CASE("Pointwise ternary ops", "[pointwise][graph]") {
  const auto dims = std::vector<std::vector<int64_t>>{
      std::vector<int64_t>{2, 16, 64, 64}, std::vector<int64_t>{2, 16, 64, 64},
      GENERATE(std::vector<int64_t>{2, 16, 64, 64},
               std::vector<int64_t>{1, 16, 1, 1})};

  // clang-format off
  const auto mode = GENERATE(PointwiseAttr::Mode::BINARY_SELECT);
  // clang-format on

  auto execute = [&]<typename T>(Handle &handle, DataType dt, bool cond,
                                 T xTrue, T xFalse) {
    auto buildNewGraph = [&](Handle &handleArg) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dims));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Condition tensor is boolean regardless of the IO data type.
      auto condT = graph->tensor(
          TensorAttr()
              .setName("cond")
              .setDataType(DataType::Boolean)
              .setDim(dims[0])
              .setStride(generateStrideFromDim(
                  dims[0], getContiguousStrideOrder(dims[0].size()))));
      auto xTrueT = graph->tensor(
          TensorAttr().setName("x_true").setDim(dims[1]).setStride(
              generateStrideFromDim(dims[1],
                                    getContiguousStrideOrder(dims[1].size()))));
      auto xFalseT = graph->tensor(
          TensorAttr().setName("x_false").setDim(dims[2]).setStride(
              generateStrideFromDim(dims[2],
                                    getContiguousStrideOrder(dims[2].size()))));

      // Create Pointwise op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult =
          graph->pointwise(condT, xTrueT, xFalseT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handleArg, /*remove=*/true));

      return std::make_tuple(graph, condT, xTrueT, xFalseT, pointwiseResult);
    };
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, condT, xTrueT, xFalseT, yT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(
        auto condBuf,
        allocateBufferOfType(handle, condT, DataType::Boolean, cond));
    FUSILLI_REQUIRE_ASSIGN(auto xTrueBuf,
                           allocateBufferOfType(handle, xTrueT, dt, xTrue));
    FUSILLI_REQUIRE_ASSIGN(auto xFalseBuf,
                           allocateBufferOfType(handle, xFalseT, dt, xFalse));

    // Allocate output buffer.
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, dt, 0.0f));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {condT, condBuf},
            {xTrueT, xTrueBuf},
            {xFalseT, xFalseBuf},
            {yT, yBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate reference value.
    T y = cond ? xTrue : xFalse;

    // Read output buffers.
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);

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

  // int32: true/false condition cases.
  execute(handle, DataType::Int32, true, int(7), int(-3));
  execute(handle, DataType::Int32, false, int(7), int(-3));
  // fp16: true/false condition cases.
  execute(handle, DataType::Half, true, half(1.5f), half(-2.5f));
  execute(handle, DataType::Half, false, half(1.5f), half(-2.5f));
}
