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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

namespace {

template <typename T>
void executeDynamicReduction(Handle &handle, ReductionAttr::Mode mode,
                             DataType dt, T initValue) {
  const std::vector<int64_t> xDims = {2, 16, 8, 8};
  const std::vector<int64_t> yDims = {2, 16, 1, 1};

  auto graph = std::make_shared<Graph>();
  graph->setName("dynamic_reduction_" + ReductionAttr::kModeToStr.at(mode));
  graph->setIODataType(dt).setComputeDataType(dt);

  auto xT = graph->tensor(
      TensorAttr().setName("x").setDim(xDims).setDynamicDims({0}).setStride(
          generateStrideFromDim(xDims,
                                getContiguousStrideOrder(xDims.size()))));

  auto reductionAttr = ReductionAttr().setMode(mode);
  auto yT = graph->reduction(xT, reductionAttr);
  yT->setName("result")
      .setDim(yDims)
      .setDynamicDims({0})
      .setStride(
          generateStrideFromDim(yDims, getContiguousStrideOrder(yDims.size())))
      .setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  int64_t xSize = 1;
  for (auto dim : xDims)
    xSize *= dim;

  std::vector<T> xData(xSize);
  for (int64_t i = 0; i < xSize; ++i)
    xData[i] = static_cast<T>(i % 100 - 50);

  FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, xData));
  FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                         allocateBufferOfType(handle, yT, dt, initValue));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{xT, xBuf}, {yT, yBuf}};

  // XFAIL: MIN/MAX reductions currently reflect iree.abi.transients.size
  // instead of iree.abi.transients.size.constant for dynamic batch. Fusilli
  // rejects that dynamic workspace query until runtime size functions are
  // supported.
  FUSILLI_REQUIRE_ASSIGN(auto workspaceSize, graph->getWorkspaceSize());
  REQUIRE(workspaceSize.value_or(0) > 0);
  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, workspaceSize));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<T> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  REQUIRE(result.size() == static_cast<size_t>(yDims[0] * yDims[1]));

  for (int64_t n = 0; n < xDims[0]; ++n) {
    for (int64_t c = 0; c < xDims[1]; ++c) {
      T expectedValue = initValue;
      for (int64_t d2 = 0; d2 < xDims[2]; ++d2) {
        for (int64_t d3 = 0; d3 < xDims[3]; ++d3) {
          int64_t inIdx = ((n * xDims[1] + c) * xDims[2] + d2) * xDims[3] + d3;
          if (mode == ReductionAttr::Mode::MIN)
            expectedValue = std::min(expectedValue, xData[inIdx]);
          else
            expectedValue = std::max(expectedValue, xData[inIdx]);
        }
      }

      int64_t outIdx = n * xDims[1] + c;
      REQUIRE(result[outIdx] == expectedValue);
    }
  }
}

} // namespace

// TODO(#400): Remove [!shouldfail] once Fusilli supports querying IREE dynamic
// transient workspace size functions (iree.abi.transients.size) at runtime.
TEST_CASE("Dynamic batch MIN/MAX reductions with dynamic workspace",
          "[dynamic][reduction][graph][!shouldfail]") {
  const auto mode =
      GENERATE(ReductionAttr::Mode::MIN, ReductionAttr::Mode::MAX);

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  executeDynamicReduction(handle, mode, DataType::Int32,
                          mode == ReductionAttr::Mode::MIN
                              ? std::numeric_limits<int>::max()
                              : std::numeric_limits<int>::lowest());
}
