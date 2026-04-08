// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Builds the input tensor data for a given reduction mode. The reduction
// in this sample is checked at output index (d0=0, d1=0), which corresponds
// to xData indices [0, 64). Each branch picks values that produce a
// non-trivial, deterministic expected value at that output index while
// staying in range for fp16/int32.
template <typename T>
static std::vector<T> generateReductionInputData(ReductionAttr::Mode mode,
                                                 int64_t xSize) {
  std::vector<T> xData(xSize);
  switch (mode) {
  case ReductionAttr::Mode::MUL: {
    // Mostly 1s with a 2 and a 3 inside the (d0=0, d1=0) block so the
    // product over that block is 6 — non-trivial enough to verify the
    // multiplication actually happens (rather than degenerating to 1).
    std::fill(xData.begin(), xData.end(), T(1));
    xData[10] = T(2);
    xData[50] = T(3);
    break;
  }
  case ReductionAttr::Mode::MUL_NO_ZEROS: {
    // Same shape as MUL but with zeros sprinkled in. The expected
    // product over the (d0=0, d1=0) block is still 6 because zeros
    // are excluded from the reduction.
    std::fill(xData.begin(), xData.end(), T(1));
    xData[10] = T(2);
    xData[50] = T(3);
    xData[5] = T(0);
    xData[20] = T(0);
    break;
  }
  default: {
    // Values from -50 to 49.
    for (int64_t i = 0; i < xSize; ++i)
      xData[i] = static_cast<T>(i % 100 - 50);
    break;
  }
  }
  return xData;
}

// Based on parameters, generates a unique name for the graph
static std::string generateName(ReductionAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &xDim,
                                const std::vector<int64_t> &yDim) {
  std::string name =
      std::format("reduction_{}_dt{}", ReductionAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  name += "_x";
  for (const auto &d : xDim)
    name += std::format("_{}", d);

  name += "_y";
  for (const auto &d : yDim)
    name += std::format("_{}", d);
  return name;
};

TEST_CASE("Reduction ops", "[reduction][graph]") {
  const auto xDims = std::vector<int64_t>{2, 16, 8, 8};
  const auto yDims = std::vector<int64_t>{2, 16, 1, 1};

  const auto mode =
      GENERATE(ReductionAttr::Mode::SUM, ReductionAttr::Mode::MIN,
               ReductionAttr::Mode::MAX, ReductionAttr::Mode::NORM1,
               ReductionAttr::Mode::AMAX, ReductionAttr::Mode::AVG,
               ReductionAttr::Mode::NORM2, ReductionAttr::Mode::MUL,
               ReductionAttr::Mode::MUL_NO_ZEROS);

  auto execute = [&]<typename T>(Handle &handle, DataType dt, T initValue) {
    // Create graph.
    auto graph = std::make_shared<Graph>();
    graph->setName(generateName(mode, dt, xDims, yDims));
    graph->setIODataType(dt).setComputeDataType(dt);

    // Initialize input tensor.
    auto xT = graph->tensor(TensorAttr().setName("x").setDim(xDims).setStride(
        generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))));

    // Create Reduction op.
    auto reductionAttr = ReductionAttr().setMode(mode);
    auto yT = graph->reduction(xT, reductionAttr);

    // Set output dimensions for dimension reduction.
    yT->setDim(yDims).setStride(
        generateStrideFromDim(yDims, getContiguousStrideOrder(yDims.size())));

    yT->setName("result").setOutput(true);

    // Validate and compile.
    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    // Calculate total input size
    int64_t xSize = 1;
    for (auto d : xDims)
      xSize *= d;

    std::vector<T> xData = generateReductionInputData<T>(mode, xSize);

    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, xData));
    FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                           allocateBufferOfType(handle, yT, dt, initValue));
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {xT, xBuf},
            {yT, yBuf},
        };

    // Allocate workspace buffer if needed.
    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    // Execute graph once
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Calculate expected output
    int64_t ySize = 1;
    for (auto d : yDims)
      ySize *= d;

    std::vector<T> expectedY(ySize);

    // Compute reduction manually for a single output value (d0=0, d1=0)
    // Input is [2, 16, 8, 8], output is [2, 16, 1, 1]
    // We reduce over the last two dimensions
    int64_t d0 = 0, d1 = 0;

    // Initialize to same value as output buffer
    T expectedValue = initValue;
    double sumSq = 0.0;

    // Perform reduction
    for (int64_t d2 = 0; d2 < 8; ++d2) {
      for (int64_t d3 = 0; d3 < 8; ++d3) {
        int64_t inIdx = ((d0 * 16 + d1) * 8 + d2) * 8 + d3;
        switch (mode) {
        case ReductionAttr::Mode::SUM:
          expectedValue = expectedValue + xData[inIdx];
          break;
        case ReductionAttr::Mode::MIN:
          expectedValue = std::min(expectedValue, xData[inIdx]);
          break;
        case ReductionAttr::Mode::MAX:
          expectedValue = std::max(expectedValue, xData[inIdx]);
          break;
        case ReductionAttr::Mode::NORM1:
          expectedValue =
              static_cast<T>(static_cast<double>(expectedValue) +
                             std::abs(static_cast<double>(xData[inIdx])));
          break;
        case ReductionAttr::Mode::AMAX:
          expectedValue = std::max(
              expectedValue,
              static_cast<T>(std::abs(static_cast<double>(xData[inIdx]))));
          break;
        case ReductionAttr::Mode::AVG:
          expectedValue = expectedValue + xData[inIdx];
          break;
        case ReductionAttr::Mode::NORM2: {
          double v = static_cast<double>(xData[inIdx]);
          sumSq += v * v;
          break;
        }
        case ReductionAttr::Mode::MUL:
          expectedValue = expectedValue * xData[inIdx];
          break;
        case ReductionAttr::Mode::MUL_NO_ZEROS:
          if (xData[inIdx] != T(0))
            expectedValue = expectedValue * xData[inIdx];
          break;
        default:
          break;
        }
      }
    }
    if (mode == ReductionAttr::Mode::NORM2)
      expectedValue = static_cast<T>(std::sqrt(sumSq));

    // Finalize AVG by dividing the accumulated sum by the number of
    // reduced elements (8 * 8 = 64). Integer types use integer division.
    if (mode == ReductionAttr::Mode::AVG)
      expectedValue = expectedValue / T(64);

    // Read output buffer
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));

    // Validate output size and check the first value (d0=0, d1=0)
    REQUIRE(result.size() == ySize);
    int64_t checkIdx = d0 * 16 + d1;
    if constexpr (std::is_floating_point_v<T>) {
      REQUIRE(std::abs(result[checkIdx] - expectedValue) < T(0.01));
    } else {
      if (mode == ReductionAttr::Mode::NORM2) {
        // sqrt may round; allow off-by-one for integer types.
        T diff = result[checkIdx] > expectedValue
                     ? result[checkIdx] - expectedValue
                     : expectedValue - result[checkIdx];
        REQUIRE(diff <= T(1));
      } else {
        REQUIRE(result[checkIdx] == expectedValue);
      }
    }
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Determine initial value based on reduction mode
  auto getInitValue = [&]<typename T>() -> T {
    switch (mode) {
    case ReductionAttr::Mode::SUM:
      return T(0);
    case ReductionAttr::Mode::MIN:
      return std::numeric_limits<T>::max();
    case ReductionAttr::Mode::MAX:
      return std::numeric_limits<T>::lowest();
    case ReductionAttr::Mode::MUL:
    case ReductionAttr::Mode::MUL_NO_ZEROS:
      return T(1);
    case ReductionAttr::Mode::NORM1:
    case ReductionAttr::Mode::AMAX:
    case ReductionAttr::Mode::AVG:
    case ReductionAttr::Mode::NORM2:
    default:
      return T(0);
    }
  };

  // torch.aten.mean.dim is not defined on integer tensors, so AVG only
  // exercises the floating-point paths.
  if (mode != ReductionAttr::Mode::AVG)
    execute(handle, DataType::Int32, getInitValue.template operator()<int>());
  // fp16
  execute(handle, DataType::Half, getInitValue.template operator()<half>());
  // fp32
  execute(handle, DataType::Float, getInitValue.template operator()<float>());
}
