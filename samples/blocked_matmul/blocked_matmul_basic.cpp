// Copyright 2025 Advanced Micro Devices, Inc.
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

// Blocked matmul (mmt4d):
//   LHS [M0, K0, M1, K1] x RHS [K0, N0, K1, N1] -> OUT [M0, N0, M1, N1]
//   RHS is transposed: physical layout [N0, K0, N1, K1]
//
// When all inputs are ones, each output element equals K0 * K1 (the
// contraction dimension product).
TEST_CASE("Blocked matmul; LHS [M0,K0,M1,K1], RHS transposed; basic mmt4d",
          "[blocked_matmul][graph]") {
  int64_t m0 = 4, k0 = 8, m1 = 4, k1 = 2;
  int64_t n0 = 6, n1 = 4;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("blocked_matmul_basic_sample");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    // LHS: logical [m0, k0, m1, k1], contiguous
    auto lhsT = graph->tensor(TensorAttr()
                                  .setName("lhs")
                                  .setDim({m0, k0, m1, k1})
                                  .setStride({k0 * m1 * k1, m1 * k1, k1, 1}));

    // RHS: logical [k0, n0, k1, n1], physical [n0, k0, n1, k1] (transposed)
    auto rhsT = graph->tensor(TensorAttr()
                                  .setName("rhs")
                                  .setDim({k0, n0, k1, n1})
                                  .setStride({n1 * k1, k0 * n1 * k1, 1, k1}));

    auto bmAttr = BlockedMatmulAttr().setName("blocked_matmul");
    auto outT = graph->blockedMatmul(lhsT, rhsT, bmAttr);
    outT->setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, lhsT, rhsT, outT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, lhsT, rhsT, outT] = buildNewGraph(handle);

  // Allocate input buffers (all ones).
  FUSILLI_REQUIRE_ASSIGN(
      auto lhsBuf, allocateBufferOfType(handle, lhsT, DataType::Float, 1.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto rhsBuf, allocateBufferOfType(handle, rhsT, DataType::Float, 1.0f));

  // Allocate output buffer (zeros).
  FUSILLI_REQUIRE_ASSIGN(
      auto outBuf, allocateBufferOfType(handle, outT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {lhsT, lhsBuf},
          {rhsT, rhsBuf},
          {outT, outBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));

  // When LHS and RHS are all ones, each output element = k0 * k1.
  float expected = static_cast<float>(k0 * k1);
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(result[i] == expected);
  }
}
