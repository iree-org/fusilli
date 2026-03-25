// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "batchnorm_utils.h"
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

TEST_CASE("Batch normalization; training mode; NCHW layout; scale, bias",
          "[batchnorm][graph]") {
  constexpr int64_t n = 2, c = 4, h = 8, w = 8;
  constexpr float scale = 0.5f, bias = 1.0f;
  constexpr float eps = 1e-5f;
  constexpr float momentum = 0.1f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("batchnorm_train_sample_nchw_scale_bias");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1})); // NCHW

    // Shape and strides inferred by inferPropertiesNode().
    auto sT = graph->tensor(TensorAttr().setName("scale"));
    auto bT = graph->tensor(TensorAttr().setName("bias"));

    auto epsilonT = graph->tensor(TensorAttr(eps).setName("epsilon"));
    auto momentumT = graph->tensor(TensorAttr(momentum).setName("momentum"));

    auto batchnormAttr = BatchnormAttr()
                             .setForwardPhase(NormFwdPhase::TRAINING)
                             .setEpsilon(epsilonT)
                             .setMomentum(momentumT)
                             .setName("batchnorm");

    // Training without running statistics; scale and bias are provided.
    auto [yT, smT, sivT] =
        graph->batchnorm(xT, sT, bT, nullptr, nullptr, batchnormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);
    smT->setName("saved_mean").setDataType(DataType::Float).setOutput(true);
    sivT->setName("saved_inv_var")
        .setDataType(DataType::Float)
        .setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, sT, bT, yT, smT, sivT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, xT, sT, bT, yT, smT, sivT] = buildNewGraph(handle);

  auto [inputVals, expectedVals, expectedSavedMean, expectedSavedInvVar] =
      batchnorm_utils::generateNchwForTrainForward(n, c, h, w, scale, bias,
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
      auto smBuf, allocateBufferOfType(handle, smT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto sivBuf, allocateBufferOfType(handle, sivT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf}, {sT, sBuf},    {bT, bBuf},
          {yT, yBuf}, {smT, smBuf}, {sivT, sivBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> yVals, smVals, sivVals;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));
  FUSILLI_REQUIRE_OK(smBuf->read(handle, smVals));
  FUSILLI_REQUIRE_OK(sivBuf->read(handle, sivVals));

  REQUIRE(yVals.size() == expectedVals.size());
  REQUIRE(smVals.size() == expectedSavedMean.size());
  REQUIRE(sivVals.size() == expectedSavedInvVar.size());

  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < yVals.size(); ++i) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
  for (size_t i = 0; i < smVals.size(); ++i) {
    REQUIRE(std::abs(smVals[i] - expectedSavedMean[i]) < tolerance);
    REQUIRE(std::abs(sivVals[i] - expectedSavedInvVar[i]) < tolerance);
  }

  // Verify consistent results across multiple executions.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; ++i)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  yVals.clear();
  smVals.clear();
  sivVals.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));
  FUSILLI_REQUIRE_OK(smBuf->read(handle, smVals));
  FUSILLI_REQUIRE_OK(sivBuf->read(handle, sivVals));

  REQUIRE(yVals.size() == expectedVals.size());
  REQUIRE(smVals.size() == expectedSavedMean.size());
  REQUIRE(sivVals.size() == expectedSavedInvVar.size());
  for (size_t i = 0; i < yVals.size(); ++i) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
  for (size_t i = 0; i < smVals.size(); ++i) {
    REQUIRE(std::abs(smVals[i] - expectedSavedMean[i]) < tolerance);
    REQUIRE(std::abs(sivVals[i] - expectedSavedInvVar[i]) < tolerance);
  }
}
