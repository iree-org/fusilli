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

TEST_CASE("Convolution dgrad; DY/W (NHWC/KRSC), DX (NHWC); 1x1; no padding",
          "[conv][graph]") {
  int64_t n = 4, c = 8, h = 8, w = 8, k = 16, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_dgrad_sample_nhwc_krsc_1x1_nopad");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto wT = graph->tensor(TensorAttr()
                                .setName("w")
                                .setDim({k, c, r, s})
                                .setStride({c * r * s, r * s, s, 1})); // KRSC

    auto dgradAttr = ConvDGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_dgrad");

    auto dxT = graph->convDGrad(dyT, wT, dgradAttr);
    dxT->setName("dx")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({n, c, h, w});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, wT, dxT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, wT, dxT] = buildNewGraph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DX for 1x1 conv equals k.
  const float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto dyBuf,
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto wBuf,
      allocateBufferOfType(handle, wT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto dxBuf, allocateBufferOfType(handle, dxT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dxT, dxBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  std::vector<float> dxVals;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));

  const float expected = static_cast<float>(k) * inputScalar * inputScalar;
  for (auto val : dxVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  dxVals.clear();
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));
  for (auto val : dxVals)
    REQUIRE(val == expected);
}

TEST_CASE("Convolution dgrad; DY/W (NHWC/KRSC), DX (NHWC); 1x1; no padding; "
          "bias",
          "[conv][graph]") {
  int64_t n = 4, c = 8, h = 8, w = 8, k = 16, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_dgrad_sample_nhwc_krsc_1x1_nopad_bias");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto wT = graph->tensor(TensorAttr()
                                .setName("w")
                                .setDim({k, c, r, s})
                                .setStride({c * r * s, r * s, s, 1})); // KRSC

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

    auto dgradAttr = ConvDGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_dgrad");

    auto dxT = graph->convDGrad(dyT, wT, dgradAttr);
    dxT->setName("dx")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({n, c, h, w});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, wT, dbT, dxT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, wT, dbT, dxT] = buildNewGraph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DX for 1x1 conv equals k.
  const float inputScalar = 1.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto dyBuf,
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto wBuf,
      allocateBufferOfType(handle, wT, DataType::Float, inputScalar));
  FUSILLI_REQUIRE_ASSIGN(
      auto dbBuf, allocateBufferOfType(handle, dbT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto dxBuf, allocateBufferOfType(handle, dxT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dbT, dbBuf},
          {dxT, dxBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers and validate.
  // DX: each element equals k * inputScalar^2 = 16.
  std::vector<float> dxVals;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));

  const float expectedDx = static_cast<float>(k) * inputScalar * inputScalar;
  for (auto val : dxVals)
    REQUIRE(val == expectedDx);

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
  dxVals.clear();
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));
  for (auto val : dxVals)
    REQUIRE(val == expectedDx);

  dbVals.clear();
  FUSILLI_REQUIRE_OK(dbBuf->read(handle, dbVals));
  for (auto val : dbVals)
    REQUIRE(val == expectedDb);
}
