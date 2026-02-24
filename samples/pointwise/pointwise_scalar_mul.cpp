// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Pointwise MUL with a scalar operand: result = tensor * scalar.
TEST_CASE("Pointwise MUL with scalar operand", "[pointwise][scalar][graph]") {
  const float inputVal = 3.0f;
  const float scalarVal = 2.0f;
  const float expectedVal = inputVal * scalarVal; // 6.0f

  const std::vector<int64_t> dims = {2, 16, 64, 64};

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Build graph.
  auto graph = std::make_shared<Graph>();
  graph->setName("pointwise_scalar_mul_sample");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr().setName("in0").setDim(dims).setStride(
      generateStrideFromDim(dims, getContiguousStrideOrder(dims.size()))));

  auto alphaT = graph->tensor(TensorAttr(scalarVal));
  alphaT->setName("alpha");

  auto pointwiseAttr =
      PointwiseAttr().setMode(PointwiseAttr::Mode::MUL).setName("pw_mul");

  auto yT = graph->pointwise(xT, alphaT, pointwiseAttr);
  yT->setName("result").setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  // Allocate input buffer (all elements = inputVal).
  FUSILLI_REQUIRE_ASSIGN(
      auto xBuf, allocateBufferOfType(handle, xT, DataType::Float, inputVal));

  // Allocate output buffer (zeroed).
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, 0.0f));

  // Scalar is not in the variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {yT, yBuf},
      };

  // Allocate workspace buffer if needed.
  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  // Verify output.
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == expectedVal);
}
