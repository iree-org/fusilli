// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Layer normalization; inference mode; NCHW layout; no bias/scale",
          "[layernorm][graph]") {
  constexpr int64_t n = 2, c = 3, h = 32, w = 32;
  constexpr float eps = 1e-5f;

  auto generateInputAndExpectedTensors = [=]() {
    // For each batch b, we fill:
    //   - first half of elements with x0 = 2*b
    //   - second half of elements with x1 = 2*b + 2
    //
    // Layer norm formula: y = (x - mean) / sqrt(variance + eps)
    // With two distinct values x0, x1:
    //   y0 = -1 / sqrt(1 + eps)
    //   y1 = 1 / sqrt(1 + eps)
    const float div = std::sqrt(1.0f + eps);
    const float y0 = -1.0f / div;
    const float y1 = 1.0f / div;

    constexpr size_t size = n * c * h * w;
    std::vector<float> inputVals(size, 0.0f);
    std::vector<float> expectedVals(size, 0.0f);
    for (int64_t b = 0; b < n; ++b) {
      const float x0 = 2.0f * static_cast<float>(b);
      const float x1 = x0 + 2.0f;

      int64_t start = b * c * h * w;
      int64_t interm = start + c * h * w / 2;
      int64_t end = interm + c * h * w / 2;

      std::fill(inputVals.begin() + start, inputVals.begin() + interm, x0);
      std::fill(inputVals.begin() + interm, inputVals.begin() + end, x1);

      std::fill(expectedVals.begin() + start, expectedVals.begin() + interm,
                y0);
      std::fill(expectedVals.begin() + interm, expectedVals.begin() + end, y1);
    }
    return std::make_pair(inputVals, expectedVals);
  };

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("layernorm_infer_sample_nchw");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1})); // NCHW

    auto epsilonT = graph->tensor(TensorAttr(eps));

    auto layernormAttr = LayernormAttr()
                             .setForwardPhase(NormFwdPhase::INFERENCE)
                             .setEpsilon(epsilonT)
                             .setName("layernorm");

    // Layernorm
    auto [yT, mT, vT] = graph->layernorm(xT, nullptr, nullptr, layernormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, yT);
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

  auto [graph, xT, yT] = buildNewGraph(handle);

  auto [inputVals, expectedVals] = generateInputAndExpectedTensors();

  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT(xT->getPhysicalDim()), inputVals)));
  auto yBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, yT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {yT, yBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  std::vector<float> yVals;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));

  REQUIRE(yVals.size() == expectedVals.size());
  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < yVals.size(); i++) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }

  // Execute graph a few times to verify consistent results.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  yVals.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));

  REQUIRE(yVals.size() == expectedVals.size());
  for (size_t i = 0; i < yVals.size(); i++) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
}
