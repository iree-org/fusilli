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

TEST_CASE("Matrix multiplication with bias; A (M, K), B (K, N), bias (1, N); "
          "basic matmul with bias",
          "[matmul][graph][bias]") {
  int64_t m = 64, k = 128, n = 256;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("matmul_basic_with_bias_sample");
    graph->setIODataType(DataType::Half)
        .setComputeDataType(DataType::Float)
        .setIntermediateDataType(DataType::Half);

    auto aT = graph->tensor(
        TensorAttr().setName("matrix_a").setDim({m, k}).setStride({k, 1}));

    auto bT = graph->tensor(
        TensorAttr().setName("matrix_b").setDim({k, n}).setStride({n, 1}));

    auto matmulAttr = MatmulAttr().setName("matmul");

    auto cT = graph->matmul(aT, bT, matmulAttr);

    // Add bias vector with shape (1, N) that broadcasts to (M, N)
    auto biasT = graph->tensor(
        TensorAttr().setName("bias").setDim({1, n}).setStride({n, 1}));

    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    auto resultT = graph->pointwise(cT, biasT, biasAttr);
    resultT->setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, biasT, resultT);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif
  Handle &handle = *handlePtr;

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, aT, bT, biasT, resultT] = buildNewGraph(handle);

  // Allocate input buffer for A.
  auto aBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, aT, DataType::Half, 1.0f));

  // Allocate input buffer for B.
  auto bBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, bT, DataType::Half, 1.0f));

  // Allocate bias buffer.
  float biasValue = 2.0f;
  auto biasBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, biasT, DataType::Half, biasValue));

  // Allocate output buffer for result.
  auto resultBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, resultT, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {biasT, biasBuf},
          {resultT, resultBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(resultBuf->read(handle, result));

  // Verify output.
  // When A and B are all ones, C = A @ B should have all elements equal to k.
  // After adding bias of 2.0, result should be k + 2.0.
  float expected = static_cast<float>(k) + biasValue;
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(static_cast<float>(result[i]) == expected);
  }
}

TEST_CASE("Matrix multiplication with bias; A (M, K), B^T (N, K), bias (1, N); "
          "matmul with transpose B and bias",
          "[matmul][graph][transpose][bias]") {
  int64_t m = 64, k = 128, n = 256;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("matmul_transpose_b_with_bias_sample");
    graph->setIODataType(DataType::Half)
        .setComputeDataType(DataType::Float)
        .setIntermediateDataType(DataType::Half);

    auto aT = graph->tensor(
        TensorAttr().setName("matrix_a").setDim({m, k}).setStride({k, 1}));

    auto bT = graph->tensor(TensorAttr()
                                .setName("matrix_b_transposed")
                                .setDim({k, n})
                                .setStride({1, k}));

    auto matmulAttr = MatmulAttr().setName("matmul_transpose_b");

    auto cT = graph->matmul(aT, bT, matmulAttr);

    // Add bias vector with shape (1, N) that broadcasts to (M, N)
    auto biasT = graph->tensor(
        TensorAttr().setName("bias").setDim({1, n}).setStride({n, 1}));

    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    auto resultT = graph->pointwise(cT, biasT, biasAttr);
    resultT->setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, biasT, resultT);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif
  Handle &handle = *handlePtr;

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, aT, bT, biasT, resultT] = buildNewGraph(handle);

  // Allocate input buffer for A.
  auto aBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, aT, DataType::Half, 1.0f));

  // Allocate input buffer for B (transposed layout).
  auto bBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, bT, DataType::Half, 1.0f));

  // Allocate bias buffer.
  float biasValue = 2.0f;
  auto biasBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, biasT, DataType::Half, biasValue));

  // Allocate output buffer for result.
  auto resultBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, resultT, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {biasT, biasBuf},
          {resultT, resultBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(resultBuf->read(handle, result));

  // Verify output.
  // When A and B are all ones, C = A @ B^T should have all elements equal to k.
  // After adding bias of 2.0, result should be k + 2.0.
  float expected = static_cast<float>(k) + biasValue;
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(static_cast<float>(result[i]) == expected);
  }
}
