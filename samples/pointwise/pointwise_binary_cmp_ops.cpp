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

TEST_CASE("Pointwise binary compare ops", "[pointwise][graph]") {
  const auto dims = std::vector<std::vector<int64_t>>{
      std::vector<int64_t>{2, 16, 64, 64},
      GENERATE(std::vector<int64_t>{2, 16, 64, 64},
               std::vector<int64_t>{1, 16, 1, 1})};

  const auto mode =
      GENERATE(PointwiseAttr::Mode::CMP_EQ, PointwiseAttr::Mode::CMP_LT,
               PointwiseAttr::Mode::CMP_LE, PointwiseAttr::Mode::CMP_GT,
               PointwiseAttr::Mode::CMP_GE, PointwiseAttr::Mode::CMP_NEQ);

  auto execute = [&]<typename T>(const std::shared_ptr<Handle> &handlePtr,
                                 DataType dt, T x0, T x1) {
    auto buildNewGraph = [&](const Handle &handle) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dims));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Initialize input tensors
      auto x0T =
          graph->tensor(TensorAttr().setName("in0").setDim(dims[0]).setStride(
              generateStrideFromDim(dims[0],
                                    getContiguousStrideOrder(dims[0].size()))));
      auto x1T =
          graph->tensor(TensorAttr().setName("in1").setDim(dims[1]).setStride(
              generateStrideFromDim(dims[1],
                                    getContiguousStrideOrder(dims[1].size()))));

      // Create Pointwise op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult = graph->pointwise(x0T, x1T, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

      return std::make_tuple(graph, x0T, x1T, pointwiseResult);
    };

    Handle &handle = *handlePtr;
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, x0T, x1T, yT] = buildNewGraph(handle);

    // Allocate input buffers.
    FUSILLI_REQUIRE_ASSIGN(auto x0Buf,
                           allocateBufferOfType(handle, x0T, dt, x0));
    FUSILLI_REQUIRE_ASSIGN(auto x1Buf,
                           allocateBufferOfType(handle, x1T, dt, x1));

    // Allocate output buffer.
    DataType yDt = DataType::Boolean;
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, yDt, false));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {x0T, x0Buf},
            {x1T, x1Buf},
            {yT, yBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate reference value
    bool y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::CMP_EQ: {
      y = (x0 == x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_LT: {
      y = (x0 < x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_LE: {
      y = (x0 <= x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_GT: {
      y = (x0 > x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_GE: {
      y = (x0 >= x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_NEQ: {
      y = (x0 != x1);
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
    for (size_t i = 0; i < numIters; i++)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#endif

  // int32
  execute(handlePtr, DataType::Int32, int(-50), int(-50));
  execute(handlePtr, DataType::Int32, int(-50), int(-51));
  execute(handlePtr, DataType::Int32, int(-51), int(-50));
  execute(handlePtr, DataType::Int32, int(-51), int(-51));
  // fp16
  execute(handlePtr, DataType::Half, half(1.0), half(1.0));
  execute(handlePtr, DataType::Half, half(1.0), half(1.1));
  execute(handlePtr, DataType::Half, half(1.1), half(1.1));
  execute(handlePtr, DataType::Half, half(1.1), half(1.0));
}
