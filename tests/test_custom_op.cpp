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

static std::string getCustomAddMlir() {
  return R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>,
                                   %arg1: !torch.vtensor<[?],{IN1_DTYPE}>)
                                   -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[?],{IN0_DTYPE}>, !torch.vtensor<[?],{IN1_DTYPE}>, !torch.int
        -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";
}

TEST_CASE("CustomOpAttr stores name, MLIR, and numOutputs", "[custom_op]") {
  CustomOpAttr attr;
  attr.setName("my_op").setMlir("some mlir").setNumOutputs(1);

  REQUIRE(attr.getName() == "my_op");
  REQUIRE(attr.getMlir() == "some mlir");
  REQUIRE(attr.getNumOutputs() == 1);
}

TEST_CASE("CustomOpNode getName and getType", "[custom_op]") {
  CustomOpAttr attr;
  attr.setName("test_node").setMlir("mlir");
  Context ctx;
  CustomOpNode node(std::move(attr), ctx);

  REQUIRE(node.getName() == "test_node");
  REQUIRE(node.getType() == INode::Type::Custom);
}

TEST_CASE("Graph customOp() adds CustomOpNode and output tensors",
          "[custom_op]") {
  Graph g;
  g.setName("custom_op_graph").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b =
      g.tensor(TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({a, b}, addAttr);
  outs[0]->setDim({4}).setStride({1}).setDataType(DataType::Float);

  REQUIRE(outs.size() == 1);
  REQUIRE(outs[0]->getName() == "my_add_OUT_0");
  REQUIRE(outs[0]->isVirtual() == true);

  outs[0]->setOutput(true);
  REQUIRE(outs[0]->isVirtual() == false);

  FUSILLI_REQUIRE_OK(g.validate());
}

TEST_CASE("Graph customOp() auto-generates names", "[custom_op]") {
  Graph g;
  g.setName("auto_name_graph").setIODataType(DataType::Float);

  auto a = g.tensor(
      TensorAttr().setDim({4}).setStride({1}).setDataType(DataType::Float));

  CustomOpAttr attr;
  attr.setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({a}, attr);
  outs[0]->setDim({4}).setStride({1}).setDataType(DataType::Float);

  // Name should be auto-populated.
  REQUIRE(a->getName() == "custom_op_0_IN_0");
  REQUIRE(outs[0]->getName() == "custom_op_0_OUT_0");
}

TEST_CASE("CustomOp compile + execute round-trip", "[custom_op]") {
  Graph g;
  g.setName("custom_op_execute").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b =
      g.tensor(TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({a, b}, addAttr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  FUSILLI_REQUIRE_OK(g.validate());

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(g.compile(handle, /*remove=*/true));

  // Allocate buffers.
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, a, std::vector<float>{1, 2, 3, 4}));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, b, std::vector<float>{5, 6, 7, 8}));
  FUSILLI_REQUIRE_ASSIGN(
      auto outBuf,
      allocateBufferOfType(handle, outs[0], DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{a, aBuf}, {b, bBuf}, {outs[0], outBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, g.getWorkspaceSize()));
  FUSILLI_REQUIRE_OK(g.execute(handle, variantPack, workspace));

  std::vector<float> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
  REQUIRE(result.size() == 4);
  REQUIRE(result[0] == 6.0f);
  REQUIRE(result[1] == 8.0f);
  REQUIRE(result[2] == 10.0f);
  REQUIRE(result[3] == 12.0f);
}

TEST_CASE("CustomOp composition: built-in op -> custom op", "[custom_op]") {
  Graph g;
  g.setName("composition_graph")
      .setIODataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b =
      g.tensor(TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  // Built-in pointwise add first.
  auto pwAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD).setName("pw");
  auto pwOut = g.pointwise(a, b, pwAttr);

  // Custom op takes the pointwise output.
  std::string negateMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>)
                                    -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %0 = torch.aten.neg %arg0 : !torch.vtensor<[?],{IN0_DTYPE}>
        -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";

  CustomOpAttr negAttr;
  negAttr.setName("my_neg").setMlir(negateMlir).setNumOutputs(1);

  auto outs = g.customOp({pwOut}, negAttr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  FUSILLI_REQUIRE_OK(g.validate());

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(g.compile(handle, /*remove=*/true));

  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, a, std::vector<float>{1, 2, 3, 4}));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, b, std::vector<float>{5, 6, 7, 8}));
  FUSILLI_REQUIRE_ASSIGN(
      auto outBuf,
      allocateBufferOfType(handle, outs[0], DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{a, aBuf}, {b, bBuf}, {outs[0], outBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, g.getWorkspaceSize()));
  FUSILLI_REQUIRE_OK(g.execute(handle, variantPack, workspace));

  std::vector<float> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
  REQUIRE(result.size() == 4);
  // a+b = {6,8,10,12}, negated = {-6,-8,-10,-12}
  REQUIRE(result[0] == -6.0f);
  REQUIRE(result[1] == -8.0f);
  REQUIRE(result[2] == -10.0f);
  REQUIRE(result[3] == -12.0f);
}

TEST_CASE("CustomOp error: missing MLIR", "[custom_op]") {
  Graph g;
  g.setName("error_no_mlir").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomOpAttr attr;
  attr.setName("bad_op").setNumOutputs(1);

  auto outs = g.customOp({a}, attr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
  REQUIRE(status.getMessage() == "CustomOp MLIR not set");
}

TEST_CASE("CustomOp error: missing outputs", "[custom_op]") {
  Graph g;
  g.setName("error_no_outputs").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomOpAttr attr;
  attr.setName("bad_op").setMlir("some mlir");
  // No setNumOutputs call — defaults to 0, so the node will have empty outputs.
  auto outs = g.customOp({a}, attr);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
}

// Scalar tensor input must return validation error, not assertion crash.
TEST_CASE("CustomOp error: scalar input", "[custom_op]") {
  Graph g;
  g.setName("scalar_input_graph").setIODataType(DataType::Float);

  auto scalar = g.tensor(TensorAttr(1.0f).setName("s"));

  CustomOpAttr attr;
  attr.setName("bad_op").setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({scalar}, attr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "CustomOp input 0 is scalar (not supported)");
}

// Scalar tensor output must return validation error.
TEST_CASE("CustomOp error: scalar output", "[custom_op]") {
  Graph g;
  g.setName("scalar_output_graph").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomOpAttr attr;
  attr.setName("bad_op").setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({a}, attr);
  // Force the output tensor to be scalar. setOutput(true) clears the virtual
  // flag first so that the CustomOpNode scalar check fires before TensorAttr's
  // own "virtual + scalar" validation.
  outs[0]->setOutput(true).setIsScalar(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "CustomOp output 0 is scalar (not supported)");
}

// Null shared_ptr input must return validation error, not segfault.
TEST_CASE("CustomOp error: null input shared_ptr", "[custom_op]") {
  Graph g;
  g.setName("null_input_graph").setIODataType(DataType::Float);

  std::shared_ptr<TensorAttr> nullInput;

  CustomOpAttr attr;
  attr.setName("null_op").setMlir(getCustomAddMlir()).setNumOutputs(1);

  auto outs = g.customOp({nullInput}, attr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
  REQUIRE(status.getMessage() == "CustomOp input 0 is null");
}
