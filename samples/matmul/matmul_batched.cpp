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

TEST_CASE(
    "Batched matrix multiplication; A (B, M, K), B (B, K, N); batched matmul",
    "[matmul][graph]") {
  int64_t batch = 16, m = 32, k = 64, n = 128;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("matmul_batched_sample");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto aT = graph->tensor(TensorAttr()
                                .setName("matrix_a_batched")
                                .setDim({batch, m, k})
                                .setStride({m * k, k, 1}));

    auto bT = graph->tensor(TensorAttr()
                                .setName("matrix_b_batched")
                                .setDim({batch, k, n})
                                .setStride({k * n, n, 1}));

    auto matmulAttr = MatmulAttr().setName("batched_matmul");

    auto cT = graph->matmul(aT, bT, matmulAttr);
    cT->setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, cT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, aT, bT, cT] = buildNewGraph(handle);

  // Allocate input buffer for A.
  float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, aT, DataType::Float, inputScalar));

  // Allocate input buffer for B.
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, bT, DataType::Float, inputScalar));

  // Allocate output buffer for C.
  FUSILLI_REQUIRE_ASSIGN(
      auto cBuf, allocateBufferOfType(handle, cT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {cT, cBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers.
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(cBuf->read(handle, result));

  // Verify output.
  // When A and B are all ones, C = A @ B should have all elements equal to k.
  float expected = static_cast<float>(k);
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(result[i] == expected);
  }
}

TEST_CASE("Batched matrix multiplication with broadcast; A (B, M, K), B (1, K, "
          "N); broadcasted batch",
          "[matmul][graph][broadcast]") {
  int64_t batch = 16, m = 32, k = 64, n = 128;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("matmul_batched_broadcast_sample");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto aT = graph->tensor(TensorAttr()
                                .setName("matrix_a_batched")
                                .setDim({batch, batch, m, k})
                                .setStride({batch * m * k, m * k, k, 1}));

    auto bT = graph->tensor(TensorAttr()
                                .setName("matrix_b_broadcast")
                                .setDim({1, 1, k, n})
                                .setStride({1, 1, n, 1}));

    auto matmulAttr = MatmulAttr().setName("batched_broadcast_matmul");

    auto cT = graph->matmul(aT, bT, matmulAttr);
    cT->setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, cT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, aT, bT, cT] = buildNewGraph(handle);

  // Allocate input buffer for A.
  float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, aT, DataType::Float, inputScalar));

  // Allocate input buffer for B (single batch, will be broadcasted).
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, bT, DataType::Float, inputScalar));

  // Allocate output buffer for C.
  FUSILLI_REQUIRE_ASSIGN(
      auto cBuf, allocateBufferOfType(handle, cT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {cT, cBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers.
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(cBuf->read(handle, result));

  // Verify output.
  // When A and B are all ones, C = A @ B should have all elements equal to k.
  // This holds true even with broadcasting, as each batch uses the same B.
  float expected = static_cast<float>(k);
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(result[i] == expected);
  }
}
