// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// MLIR template for a custom negate function with static shape [4,8].
//
// Uses logical dimensions — the permute ops in the emitter handle
// physical-to-logical conversion before the call.
static std::string getCustomNegateStaticMlir() {
  return R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[4,8],{IN0_DTYPE}>)
                                    -> !torch.vtensor<[4,8],{OUT0_DTYPE}> {
    %0 = torch.aten.neg %arg0 : !torch.vtensor<[4,8],{IN0_DTYPE}>
        -> !torch.vtensor<[4,8],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[4,8],{OUT0_DTYPE}>
  }
)";
}

// Compose a built-in pointwise add with a static custom negate where one
// input is transposed. This exercises the physical-to-logical permute path:
//   a: contiguous dim={4,8} stride={8,1} — physical matches logical
//   b: transposed  dim={4,8} stride={1,4} — physical layout is [8,4]
// The permute converts b from physical [8,4] to logical [4,8] before the
// static func.call which expects [4,8].
TEST_CASE(
    "Custom op static transposed: add with transposed input + static negate",
    "[custom_op][graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto graph = std::make_shared<Graph>();
  graph->setName("custom_neg_static_transposed");
  graph->setIODataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  // Contiguous: dim={4,8}, stride={8,1}.
  auto aT =
      graph->tensor(TensorAttr().setName("a").setDim({4, 8}).setStride({8, 1}));

  // Transposed: dim={4,8}, stride={1,4} — physical layout is [8,4].
  auto bT =
      graph->tensor(TensorAttr().setName("b").setDim({4, 8}).setStride({1, 4}));

  // Step 1: Built-in pointwise add.
  auto pwAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD).setName("pw");
  auto pwOut = graph->pointwise(aT, bT, pwAttr);

  // Step 2: Custom negate with static MLIR using logical shape [4,8].
  CustomOpAttr negAttr;
  negAttr.setName("my_neg")
      .setMlir(getCustomNegateStaticMlir())
      .setNumOutputs(1)
      .setIsStatic(true);

  auto outs = graph->customOp({pwOut}, negAttr);
  outs[0]
      ->setDim({4, 8})
      .setStride({8, 1})
      .setDataType(DataType::Float)
      .setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  float fillVal = 3.0f;
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, aT, DataType::Float, double(fillVal)));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, bT, DataType::Float, double(fillVal)));
  FUSILLI_REQUIRE_ASSIGN(
      auto outBuf, allocateBufferOfType(handle, outs[0], DataType::Float, 0.0));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{aT, aBuf}, {bT, bBuf}, {outs[0], outBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  // Verify: -(a + b) where all elements are 3.0f → -6.0f.
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));

  float expected = -2.0f * fillVal;
  for (const auto &val : result)
    REQUIRE(val == expected);
}
