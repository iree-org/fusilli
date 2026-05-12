// Copyright 2026 Advanced Micro Devices, Inc.
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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace fusilli;

static std::string getCustomAddMlir() {
  return R"(
  func.func private @{FUNC_NAME}(%arg0: {IN0_TYPE},
                                   %arg1: {IN1_TYPE})
                                   -> {OUT0_TYPE} {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : {IN0_TYPE}, {IN1_TYPE}, !torch.int
        -> {OUT0_TYPE}
    return %0 : {OUT0_TYPE}
  }
)";
}

TEST_CASE("Dynamic batch custom add with zero workspace",
          "[dynamic][custom_op][graph]") {
  const int64_t n = 4, c = 8;
  const std::vector<int64_t> runtimeNs = {1, 2, n};

  auto graph = std::make_shared<Graph>();
  graph->setName("dynamic_custom_add")
      .setIODataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto aT = graph->tensor(TensorAttr()
                              .setName("a")
                              .setDim({n, c})
                              .setDynamicDims({0})
                              .setStride({c, 1})
                              .setDataType(DataType::Float));
  auto bT = graph->tensor(TensorAttr()
                              .setName("b")
                              .setDim({n, c})
                              .setDynamicDims({0})
                              .setStride({c, 1})
                              .setDataType(DataType::Float));

  CustomOpAttr addAttr;
  addAttr.setName("my_dynamic_add")
      .setMlir(getCustomAddMlir())
      .setNumOutputs(1);

  auto outs = graph->customOp({aT, bT}, addAttr);
  outs[0]
      ->setDim({n, c})
      .setStride({c, 1})
      .setDataType(DataType::Float)
      .setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  REQUIRE(outs[0]->getDynamicDims() == std::vector<size_t>{0});

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  for (auto runtimeN : runtimeNs) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Running runtimeN=" << runtimeN);

    std::vector<float> aData(runtimeN * c);
    std::vector<float> bData(runtimeN * c);
    for (int64_t i = 0; i < runtimeN * c; ++i) {
      aData[i] = static_cast<float>(i);
      bData[i] = static_cast<float>(100 + i);
    }

    FUSILLI_REQUIRE_ASSIGN(
        auto aRawBuf,
        Buffer::allocate(handle, castToSizeT({runtimeN, c}), aData));
    auto aBuf = std::make_shared<Buffer>(std::move(aRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto bRawBuf,
        Buffer::allocate(handle, castToSizeT({runtimeN, c}), bData));
    auto bBuf = std::make_shared<Buffer>(std::move(bRawBuf));

    FUSILLI_REQUIRE_ASSIGN(
        auto outRawBuf,
        Buffer::allocate(handle, castToSizeT({runtimeN, c}),
                         std::vector<float>(runtimeN * c, 0.0f)));
    auto outBuf = std::make_shared<Buffer>(std::move(outRawBuf));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {{aT, aBuf}, {bT, bBuf}, {outs[0], outBuf}};

    FUSILLI_REQUIRE_ASSIGN(auto workspaceSize, graph->getWorkspaceSize());
    REQUIRE(workspaceSize == 0);
    FUSILLI_REQUIRE_ASSIGN(auto workspace,
                           allocateWorkspace(handle, workspaceSize));

    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    std::vector<float> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
    REQUIRE(result.size() == static_cast<size_t>(runtimeN * c));
    for (size_t i = 0; i < result.size(); ++i)
      REQUIRE(result[i] == aData[i] + bData[i]);
  }
}
