// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates that momentum is optional for batch normalization. When omitted,
// the emitter uses PyTorch's default value of 0.1. Momentum does not affect
// the output Y in inference mode (training=false), so the numerical result is
// identical to the case where momentum is explicitly set.

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

TEST_CASE("Batch normalization; inference mode; NCHW layout; no momentum",
          "[batchnorm][graph]") {
  constexpr int64_t n = 2, c = 4, h = 8, w = 8;
  constexpr float eps = 1e-5f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("batchnorm_infer_nchw_no_momentum");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1}));

    auto meanT = graph->tensor(TensorAttr().setName("running_mean"));
    auto varT = graph->tensor(TensorAttr().setName("running_var"));

    auto epsilonT = graph->tensor(TensorAttr(eps).setName("epsilon"));

    // No momentum — omitted entirely; emitter defaults to 0.1.
    auto batchnormAttr = BatchnormAttr()
                             .setForwardPhase(NormFwdPhase::INFERENCE)
                             .setEpsilon(epsilonT)
                             .setName("batchnorm");

    auto [yT, smT, sivT] =
        graph->batchnorm(xT, nullptr, nullptr, meanT, varT, batchnormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, meanT, varT, yT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, xT, meanT, varT, yT] = buildNewGraph(handle);

  auto [inputVals, runningMeanVals, runningVarVals, expectedVals] =
      batchnorm_utils::generateNchwForInferForward(n, c, h, w, /*scale=*/1.0f,
                                                   /*bias=*/0.0f, eps);

  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, inputVals));
  FUSILLI_REQUIRE_ASSIGN(auto meanBuf,
                         allocateBufferOfType(handle, meanT, runningMeanVals));
  FUSILLI_REQUIRE_ASSIGN(auto varBuf,
                         allocateBufferOfType(handle, varT, runningVarVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {meanT, meanBuf},
          {varT, varBuf},
          {yT, yBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> yVals;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));

  REQUIRE(yVals.size() == expectedVals.size());
  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < yVals.size(); ++i)
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
}
