// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// MLIR template for a custom negate function.
//
// Placeholders resolved at emission time:
//   {FUNC_NAME}  — CustomOpAttr name (e.g., "my_neg")
//   {IN0_DTYPE}  — input 0's MLIR element type (e.g., "f32", "f16")
//   {OUT0_DTYPE} — output 0's MLIR element type
//
// The function signature uses `!torch.vtensor<[?],dtype>` with dynamic
// dimensions — fusilli inserts static-to-dynamic casts before the call and
// dynamic-to-static casts after.
static std::string getCustomNegateMlir() {
  return R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>)
                                    -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %0 = torch.aten.neg %arg0 : !torch.vtensor<[?],{IN0_DTYPE}>
        -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";
}

// Compose a built-in pointwise add with a custom negate to compute -(a + b).
// Parameterized over dtype and input shape.
TEST_CASE("Custom op: compose built-in pointwise add with custom negate",
          "[custom_op][graph]") {
  auto execute = [&]<typename T>(Handle &handle, DataType dt,
                                 const std::vector<int64_t> &dim, T fillVal) {
    auto graph = std::make_shared<Graph>();
    graph->setName(std::format("custom_neg_dt{}_n{}",
                               kDataTypeToMlirTypeAsm.at(dt), dim[0]));
    graph->setIODataType(dt).setIntermediateDataType(dt);

    auto stride =
        generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));
    auto aT =
        graph->tensor(TensorAttr().setName("a").setDim(dim).setStride(stride));
    auto bT =
        graph->tensor(TensorAttr().setName("b").setDim(dim).setStride(stride));

    // Step 1: Built-in pointwise add.
    auto pwAttr =
        PointwiseAttr().setMode(PointwiseAttr::Mode::ADD).setName("pw");
    auto pwOut = graph->pointwise(aT, bT, pwAttr);

    // Step 2: Custom negate — takes the pointwise output as input.
    CustomOpAttr negAttr;
    negAttr.setName("my_neg").setMlir(getCustomNegateMlir()).setNumOutputs(1);

    auto outs = graph->customOp(negAttr, pwOut);

    // IMPORTANT: Unlike built-in ops, custom op outputs cannot be inferred.
    // You must manually set dim, stride, and dataType on each output.
    outs[0]->setDim(dim).setStride(stride).setDataType(dt).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    FUSILLI_REQUIRE_ASSIGN(
        auto aBuf, allocateBufferOfType(handle, aT, dt, double(fillVal)));
    FUSILLI_REQUIRE_ASSIGN(
        auto bBuf, allocateBufferOfType(handle, bT, dt, double(fillVal)));
    FUSILLI_REQUIRE_ASSIGN(auto outBuf,
                           allocateBufferOfType(handle, outs[0], dt, 0.0));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {{aT, aBuf}, {bT, bBuf}, {outs[0], outBuf}};

    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    // Verify: -(a + b) where all elements are fillVal → -(2 * fillVal).
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));

    T expected = T(-2.0 * double(fillVal));
    for (const auto &val : result)
      REQUIRE(val == expected);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Float32, n=1
  execute(handle, DataType::Float, /*dim=*/{1}, /*fillVal=*/3.0f);
  // Float32, n=6
  execute(handle, DataType::Float, /*dim=*/{6}, /*fillVal=*/3.0f);
  // Float16, n=1
  execute(handle, DataType::Half, /*dim=*/{1}, /*fillVal=*/half(1.5));
  // Float16, n=6
  execute(handle, DataType::Half, /*dim=*/{6}, /*fillVal=*/half(1.5));
}
