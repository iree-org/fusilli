// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "rmsnorm_utils.h"
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

TEST_CASE("RMS normalization; inference mode; NHWC layout; with scale",
          "[rmsnorm][graph]") {
  constexpr int64_t n = 2, c = 3, h = 32, w = 32;
  constexpr float eps = 1e-5f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("rmsnorm_infer_sample_nhwc_scale");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto scaleT = graph->tensor(TensorAttr().setName("scale"));

    auto epsilonT = graph->tensor(TensorAttr(eps));

    auto rmsnormAttr = RmsnormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setName("rmsnorm");

    // RmsNorm
    auto [yT, rT] = graph->rmsnorm(xT, scaleT, rmsnormAttr);

    yT->setName("y").setDataType(DataType::Float).setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, scaleT, yT);
  };

  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, xT, scaleT, yT] = buildNewGraph(handle);

  auto [inputVals, expectedVals] =
      rmsnorm_utils::generateIOTensorsForInferForward(n, c, h, w, 1.f, eps);

  // Scale tensor is all ones (identity).
  size_t scaleSize = 1 * c * h * w;
  std::vector<float> scaleVals(scaleSize, 1.0f);

  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, xT, inputVals));
  FUSILLI_REQUIRE_ASSIGN(auto scaleBuf,
                         allocateBufferOfType(handle, scaleT, scaleVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, yT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {scaleT, scaleBuf},
          {yT, yBuf},
      };

  // Allocate workspace buffer if needed.
  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> yVals;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, yVals));

  REQUIRE(yVals.size() == expectedVals.size());
  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < yVals.size(); ++i) {
    REQUIRE(std::abs(yVals[i] - expectedVals[i]) < tolerance);
  }
}
