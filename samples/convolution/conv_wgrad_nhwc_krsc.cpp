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

TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 1x1; no padding",
          "[conv][graph]") {
  int64_t n = 4, c = 8, h = 8, w = 8, k = 16, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_1x1_nopad");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto wgradAttr = ConvWGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_wgrad");

    auto dwT = graph->convWGrad(dyT, xT, wgradAttr);
    dwT->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, c, r, s});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, dwT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, xT, dwT] = buildNewGraph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DW for 1x1 conv equals N*H*W.
  const float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto dyBuf,
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto xBuf,
      allocateBufferOfType(handle, xT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto dwBuf, allocateBufferOfType(handle, dwT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dwT, dwBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffer and validate values for 1x1, stride=1, no padding.
  std::vector<float> dwVals;
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));

  const float expected =
      static_cast<float>(n * h * w) * inputScalar * inputScalar;
  for (auto val : dwVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  dwVals.clear();
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));
  for (auto val : dwVals)
    REQUIRE(val == expected);
}

TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 1x1; no padding; bias",
          "[conv][graph]") {
  int64_t n = 4, c = 8, h = 8, w = 8, k = 16, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_1x1_nopad_bias");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, 1, c * w, c})); // NHWC

    // Bias gradient: reduce DY over batch and spatial dims.
    auto reductionAttr = ReductionAttr()
                             .setMode(ReductionAttr::Mode::SUM)
                             .setName("bias_reduction");
    auto dbT = graph->reduction(dyT, reductionAttr);
    dbT->setName("db")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({1, k, 1, 1})
        .setStride({1, 1, 1, 1});

    auto wgradAttr = ConvWGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_wgrad");

    auto dwT = graph->convWGrad(dyT, xT, wgradAttr);
    dwT->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, c, r, s});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, dbT, dwT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, xT, dbT, dwT] = buildNewGraph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DW for 1x1 conv equals N*H*W.
  const float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto dyBuf,
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto xBuf,
      allocateBufferOfType(handle, xT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto dbBuf, allocateBufferOfType(handle, dbT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto dwBuf, allocateBufferOfType(handle, dwT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dbT, dbBuf},
          {dwT, dwBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers and validate values for 1x1, stride=1, no padding.
  // DW: each element equals N*H*W * inputScalar^2 = 256.
  std::vector<float> dwVals;
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));

  const float expectedDw =
      static_cast<float>(n * h * w) * inputScalar * inputScalar;
  for (auto val : dwVals)
    REQUIRE(val == expectedDw);

  // DB: each element equals N*H*W * inputScalar = 256.
  std::vector<float> dbVals;
  FUSILLI_REQUIRE_OK(dbBuf->read(handle, dbVals));

  const float expectedDb = static_cast<float>(n * h * w) * inputScalar;
  for (auto val : dbVals)
    REQUIRE(val == expectedDb);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  dwVals.clear();
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));
  for (auto val : dwVals)
    REQUIRE(val == expectedDw);

  dbVals.clear();
  FUSILLI_REQUIRE_OK(dbBuf->read(handle, dbVals));
  for (auto val : dbVals)
    REQUIRE(val == expectedDb);
}
