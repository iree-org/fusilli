// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Int4 x fp16 matmul", "[matmul][int4]") {
  // Mixed precision matmul requires rank-3 tensors (batch + M + K) due to
  // torch-mlir constraints (torch.bmm is used for mixed element types).
  int64_t batch = 1, m = 4, k = 8, n = 4;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("matmul_int4_fp16");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto aT = graph->tensor(TensorAttr()
                                .setName("matrix_a")
                                .setDim({batch, m, k})
                                .setStride({m * k, k, 1})
                                .setDataType(DataType::Int4));

    auto bT = graph->tensor(TensorAttr()
                                .setName("matrix_b")
                                .setDim({batch, k, n})
                                .setStride({k * n, n, 1}));

    auto cT = graph->matmul(aT, bT, MatmulAttr().setName("matmul"));
    cT->setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, cT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  auto [graph, aT, bT, cT] = buildNewGraph(handle);

  // A: Int4 filled with 1.
  FUSILLI_REQUIRE_ASSIGN(auto aBuf,
                         allocateBufferOfType(handle, aT, DataType::Int4, 1.0));

  // B: fp16 filled with 1.
  FUSILLI_REQUIRE_ASSIGN(auto bBuf,
                         allocateBufferOfType(handle, bT, DataType::Half, 1.0));

  // C: fp16 zero-initialized output.
  FUSILLI_REQUIRE_ASSIGN(auto cBuf,
                         allocateBufferOfType(handle, cT, DataType::Half, 0.0));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{aT, aBuf}, {bT, bBuf}, {cT, cBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<half> result;
  FUSILLI_REQUIRE_OK(cBuf->read(handle, result));

  // A is all int4(1), B is all half(1), K=8: each dot product = 8.0.
  float expected = static_cast<float>(k);
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(static_cast<float>(result[i]) == expected);
  }
}
