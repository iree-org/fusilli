// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates that a 5D (NCDHW) input works with rank-matched [1,C,1,1,1]
// channel tensors (scale, bias, running mean, running var). The numerical
// result must be identical to the 1D case.

#include <fusilli.h>

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

// Generates input and expected output tensors for BatchNorm inference in NCDHW
// physical memory order.
//
// Fill pattern: x[n, c, d, h, w] = float(c + 1). Running stats are mean=0,
// var=1, so the output reduces to: y = scale * (c + 1) / sqrt(1 + eps) + bias.
static std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
                  std::vector<float>>
generateNcdhwForInferForward(int64_t n, int64_t c, int64_t d, int64_t h,
                             int64_t w, float scale, float bias, float eps) {
  const size_t total = static_cast<size_t>(n * c * d * h * w);
  std::vector<float> inputVals(total), meanVals(c, 0.0f), varVals(c, 1.0f),
      expectedVals(total);
  const float invStd = 1.0f / std::sqrt(1.0f + eps);
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ci = 0; ci < c; ++ci) {
      const float xVal = static_cast<float>(ci + 1);
      const float yVal = scale * xVal * invStd + bias;
      for (int64_t di = 0; di < d; ++di) {
        for (int64_t hi = 0; hi < h; ++hi) {
          for (int64_t wi = 0; wi < w; ++wi) {
            int64_t idx =
                ni * c * d * h * w + ci * d * h * w + di * h * w + hi * w + wi;
            inputVals[idx] = xVal;
            expectedVals[idx] = yVal;
          }
        }
      }
    }
  }
  return {inputVals, meanVals, varVals, expectedVals};
}

TEST_CASE("Batch normalization; inference mode; NCDHW layout; rank-matched "
          "[1,C,1,1,1] scale, bias, mean, var",
          "[batchnorm][graph]") {
  constexpr int64_t n = 2, c = 4, d = 4, h = 8, w = 8;
  constexpr float scale = 0.5f, bias = 1.0f;
  constexpr float eps = 1e-5f;
  constexpr float momentum = 0.1f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("batchnorm_infer_ncdhw_rank_matched");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    // 5D NCDHW input with contiguous stride.
    auto xT =
        graph->tensor(TensorAttr()
                          .setName("x")
                          .setDim({n, c, d, h, w})
                          .setStride({c * d * h * w, d * h * w, h * w, w, 1}));

    // Rank-matched [1, C, 1, 1, 1] tensors with contiguous stride {C,1,1,1,1}.
    auto sT = graph->tensor(TensorAttr()
                                .setName("scale")
                                .setDim({1, c, 1, 1, 1})
                                .setStride({c, 1, 1, 1, 1}));
    auto bT = graph->tensor(TensorAttr()
                                .setName("bias")
                                .setDim({1, c, 1, 1, 1})
                                .setStride({c, 1, 1, 1, 1}));
    auto meanT = graph->tensor(TensorAttr()
                                   .setName("running_mean")
                                   .setDim({1, c, 1, 1, 1})
                                   .setStride({c, 1, 1, 1, 1}));
    auto varT = graph->tensor(TensorAttr()
                                  .setName("running_var")
                                  .setDim({1, c, 1, 1, 1})
                                  .setStride({c, 1, 1, 1, 1}));

    auto epsilonT = graph->tensor(TensorAttr(eps).setName("epsilon"));
    auto momentumT = graph->tensor(TensorAttr(momentum).setName("momentum"));

    auto batchnormAttr = BatchnormAttr()
                             .setForwardPhase(NormFwdPhase::INFERENCE)
                             .setEpsilon(epsilonT)
                             .setMomentum(momentumT)
                             .setName("batchnorm");

    auto [yT, smT, sivT] =
        graph->batchnorm(xT, sT, bT, meanT, varT, batchnormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, sT, bT, meanT, varT, yT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, xT, sT, bT, meanT, varT, yT] = buildNewGraph(handle);

  auto [inputVals, runningMeanVals, runningVarVals, expectedVals] =
      generateNcdhwForInferForward(n, c, d, h, w, scale, bias, eps);

  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, inputVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto sBuf, allocateBufferOfType(handle, sT, DataType::Float, scale));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf, allocateBufferOfType(handle, bT, DataType::Float, bias));
  FUSILLI_REQUIRE_ASSIGN(auto meanBuf,
                         allocateBufferOfType(handle, meanT, runningMeanVals));
  FUSILLI_REQUIRE_ASSIGN(auto varBuf,
                         allocateBufferOfType(handle, varT, runningVarVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},       {sT, sBuf},     {bT, bBuf},
          {meanT, meanBuf}, {varT, varBuf}, {yT, yBuf},
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
