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

TEST_CASE("Pointwise gen_index", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 3, 4, 5};
  const int64_t axis = GENERATE(int64_t(0), int64_t(1), int64_t(2), int64_t(3));

  auto buildNewGraph = [&](Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName(std::format("pointwise_gen_index_axis{}", axis));
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

    auto pointwiseAttr = PointwiseAttr()
                             .setMode(PointwiseAttr::Mode::GEN_INDEX)
                             .setGenIdxAxis(axis);
    auto pointwiseResult = graph->pointwise(xT, pointwiseAttr);

    pointwiseResult->setName("result").setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, pointwiseResult);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  auto [graph, xT, yT] = buildNewGraph(handle);

  FUSILLI_REQUIRE_ASSIGN(
      auto xBuf, allocateBufferOfType(handle, xT, DataType::Float, float(0)));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, float(0)));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{xT, xBuf}, {yT, yBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));

  // Expected value at flat index `i` is the coordinate along `axis`.
  const size_t rank = dim.size();
  std::vector<int64_t> strides(rank, 1);
  for (int64_t k = static_cast<int64_t>(rank) - 2; k >= 0; --k)
    strides[k] = strides[k + 1] * dim[k + 1];

  for (size_t i = 0; i < result.size(); ++i) {
    int64_t coord = (static_cast<int64_t>(i) / strides[axis]) % dim[axis];
    REQUIRE(result[i] == static_cast<float>(coord));
  }
}
