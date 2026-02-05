// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "layernorm_utils.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Layer normalization; training mode; NCHW layout; scale, bias",
          "[layernorm][graph]") {
  constexpr int64_t n = 2, c = 3, h = 32, w = 32;
  constexpr float scale = 0.5f, bias = 1.0f;
  constexpr float eps = 1e-5f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("layernorm_train_sample_nchw_scale_bias");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1})); // NCHW

    // Shape and strides will be inferred later in inferPropertiesNode()
    auto sT = graph->tensor(TensorAttr().setName("scale"));
    auto bT = graph->tensor(TensorAttr().setName("bias"));

    auto epsilonT = graph->tensor(TensorAttr(eps));

    auto layernormAttr = LayernormAttr()
                             .setForwardPhase(NormFwdPhase::TRAINING)
                             .setEpsilon(epsilonT)
                             .setName("layernorm");

    // Layernorm
    auto [yT, mT, vT] = graph->layernorm(xT, sT, bT, layernormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);
    mT->setName("mean").setDataType(DataType::Float).setOutput(true);
    vT->setName("inv_variance").setDataType(DataType::Float).setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, sT, bT, yT, mT, vT);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#endif
  Handle &handle = *handlePtr;

  auto [graph, xT, sT, bT, yT, mT, vT] = buildNewGraph(handle);

  auto [inputVals, expectedVals, expectedMeans, expectedVariances] =
      layernorm_utils::generateIOTensorsForTrainForward(n, c, h, w, scale, bias,
                                                        eps);

  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, inputVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto sBuf, allocateBufferOfType(handle, sT, DataType::Float, scale));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf, allocateBufferOfType(handle, bT, DataType::Float, bias));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto mBuf, allocateBufferOfType(handle, mT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto vBuf, allocateBufferOfType(handle, vT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf}, {sT, sBuf}, {bT, bBuf},
          {yT, yBuf}, {mT, mBuf}, {vT, vBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  std::vector<float> yVals, mVals, vVals;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));
  FUSILLI_REQUIRE_OK(mBuf->read(handle, mVals));
  FUSILLI_REQUIRE_OK(vBuf->read(handle, vVals));

  REQUIRE(yVals.size() == expectedVals.size());
  REQUIRE(mVals.size() == expectedMeans.size());
  REQUIRE(vVals.size() == expectedVariances.size());
  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < yVals.size(); i++) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
  for (size_t i = 0; i < mVals.size(); i++) {
    REQUIRE(std::abs(mVals[i] - expectedMeans[i]) < tolerance);
    REQUIRE(std::abs(vVals[i] - expectedVariances[i]) < tolerance);
  }

  // Execute graph a few times to verify consistent results.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  yVals.clear();
  mVals.clear();
  vVals.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));
  FUSILLI_REQUIRE_OK(mBuf->read(handle, mVals));
  FUSILLI_REQUIRE_OK(vBuf->read(handle, vVals));

  REQUIRE(yVals.size() == expectedVals.size());
  REQUIRE(mVals.size() == expectedMeans.size());
  REQUIRE(vVals.size() == expectedVariances.size());
  for (size_t i = 0; i < yVals.size(); i++) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
  for (size_t i = 0; i < mVals.size(); i++) {
    REQUIRE(std::abs(mVals[i] - expectedMeans[i]) < tolerance);
    REQUIRE(std::abs(vVals[i] - expectedVariances[i]) < tolerance);
  }
}
