// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WORKSPACE_SAMPLES_POINTWISE_UTILS_H
#define WORKSPACE_SAMPLES_POINTWISE_UTILS_H

// Include test utilities (FUSILLI_REQUIRE_OK, allocateBufferOfType, etc.)
// Uses angle brackets to search include paths rather than relative directory
#include <utils.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fusilli {

// Helper to create TensorAttr with contiguous strides
inline TensorAttr getTensorAttr(DataType dt, const std::vector<int64_t> &dims) {
  return TensorAttr()
      .setDim(dims)
      .setStride(
          generateStrideFromDim(dims, getContiguousStrideOrder(dims.size())))
      .setDataType(dt);
}

// Builder for binary pointwise operation graphs.
// Encapsulates graph creation, tensor setup, compilation, and execution.
class PointwiseBinaryGraphBuilder {
public:
  PointwiseBinaryGraphBuilder(const std::string &name, DataType dt,
                              PointwiseAttr::Mode mode, const TensorAttr &lhsTy,
                              const TensorAttr &rhsTy)
      : mode_(mode) {
    graph_ = std::make_shared<Graph>();
    graph_->setName(name);
    graph_->setIODataType(dt).setComputeDataType(dt);

    lhsT_ = graph_->tensor(lhsTy);
    rhsT_ = graph_->tensor(rhsTy);

    auto pointwiseAttr = PointwiseAttr().setMode(mode);
    outputT_ = graph_->pointwise(lhsT_, rhsT_, pointwiseAttr);
    outputT_->setName("result").setOutput(true);

    FUSILLI_REQUIRE_OK(graph_->validate());
  }

  // Compile the graph for the given handle
  void compile(const Handle &handle) {
    FUSILLI_REQUIRE_OK(graph_->compile(handle, /*remove=*/true));
  }

  // Execute the graph with input values, returning the output buffer.
  // InputT is the type for input values, OutputT for initializing output
  // buffer.
  template <typename InputT, typename OutputT>
  std::shared_ptr<Buffer> execute(Handle &handle, DataType inputDt, InputT x0,
                                  InputT x1, DataType outputDt,
                                  OutputT outputInit) {
    auto x0Buf = FUSILLI_REQUIRE_UNWRAP(
        allocateBufferOfType(handle, lhsT_, inputDt, x0));
    auto x1Buf = FUSILLI_REQUIRE_UNWRAP(
        allocateBufferOfType(handle, rhsT_, inputDt, x1));
    auto outBuf = FUSILLI_REQUIRE_UNWRAP(
        allocateBufferOfType(handle, outputT_, outputDt, outputInit));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {lhsT_, x0Buf},
            {rhsT_, x1Buf},
            {outputT_, outBuf},
        };

    FUSILLI_REQUIRE_OK(graph_->execute(handle, variantPack));
    return outBuf;
  }

  // Re-execute the graph with an existing variant pack
  void execute(Handle &handle,
               const std::unordered_map<std::shared_ptr<TensorAttr>,
                                        std::shared_ptr<Buffer>> &variantPack) {
    FUSILLI_REQUIRE_OK(graph_->execute(handle, variantPack));
  }

  // Accessors
  std::shared_ptr<Graph> getGraph() const { return graph_; }
  std::shared_ptr<TensorAttr> getLhsTensor() const { return lhsT_; }
  std::shared_ptr<TensorAttr> getRhsTensor() const { return rhsT_; }
  std::shared_ptr<TensorAttr> getOutputTensor() const { return outputT_; }
  PointwiseAttr::Mode getMode() const { return mode_; }

private:
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<TensorAttr> lhsT_;
  std::shared_ptr<TensorAttr> rhsT_;
  std::shared_ptr<TensorAttr> outputT_;
  PointwiseAttr::Mode mode_;
};

// Builder for unary pointwise operation graphs.
// Encapsulates graph creation, tensor setup, compilation, and execution.
class PointwiseUnaryGraphBuilder {
public:
  PointwiseUnaryGraphBuilder(const std::string &name, DataType dt,
                             PointwiseAttr::Mode mode,
                             const TensorAttr &inputTy)
      : mode_(mode) {
    graph_ = std::make_shared<Graph>();
    graph_->setName(name);
    graph_->setIODataType(dt).setComputeDataType(dt);

    inputT_ = graph_->tensor(inputTy);

    auto pointwiseAttr = PointwiseAttr().setMode(mode);
    outputT_ = graph_->pointwise(inputT_, pointwiseAttr);
    outputT_->setName("result").setOutput(true);

    FUSILLI_REQUIRE_OK(graph_->validate());
  }

  // Compile the graph for the given handle
  void compile(const Handle &handle) {
    FUSILLI_REQUIRE_OK(graph_->compile(handle, /*remove=*/true));
  }

  // Execute the graph with input value, returning the output buffer.
  // InputT is the type for input value, OutputT for initializing output buffer.
  template <typename InputT, typename OutputT>
  std::shared_ptr<Buffer> execute(Handle &handle, DataType inputDt, InputT x,
                                  DataType outputDt, OutputT outputInit) {
    auto xBuf = FUSILLI_REQUIRE_UNWRAP(
        allocateBufferOfType(handle, inputT_, inputDt, x));
    auto outBuf = FUSILLI_REQUIRE_UNWRAP(
        allocateBufferOfType(handle, outputT_, outputDt, outputInit));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {inputT_, xBuf},
            {outputT_, outBuf},
        };

    FUSILLI_REQUIRE_OK(graph_->execute(handle, variantPack));
    return outBuf;
  }

  // Re-execute the graph with an existing variant pack
  void execute(Handle &handle,
               const std::unordered_map<std::shared_ptr<TensorAttr>,
                                        std::shared_ptr<Buffer>> &variantPack) {
    FUSILLI_REQUIRE_OK(graph_->execute(handle, variantPack));
  }

  // Accessors
  std::shared_ptr<Graph> getGraph() const { return graph_; }
  std::shared_ptr<TensorAttr> getInputTensor() const { return inputT_; }
  std::shared_ptr<TensorAttr> getOutputTensor() const { return outputT_; }
  PointwiseAttr::Mode getMode() const { return mode_; }

private:
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<TensorAttr> inputT_;
  std::shared_ptr<TensorAttr> outputT_;
  PointwiseAttr::Mode mode_;
};

} // namespace fusilli

#endif // WORKSPACE_SAMPLES_POINTWISE_UTILS_H
