// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for fusilli tests.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_TESTS_UTILS_H
#define FUSILLI_TESTS_UTILS_H

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <utility> // IWYU pragma: export
#include <vector>

// Test side dual to FUSILLI_CHECK_ERROR. REQUIRE expression that evaluates to
// (or in the case of ErrorOr<T> is convertible to) an ErrorObject to be in ok
// state; exactly equivalent to `REQUIRE(isOk(expr))` but prints a nicer error
// message when a test fails.
//
// Usage:
//   ErrorObject bar();
//
//   TEST_CASE("thing", "[example]") {
//     REQUIRE(isOk(bar()));      // No helpful error message.
//     FUSILLI_REQUIRE_OK(bar()); // Nice error message.
//   }
#define FUSILLI_REQUIRE_OK(expr)                                               \
  do {                                                                         \
    const fusilli::ErrorObject &error = (expr);                                \
    if (isError(error)) {                                                      \
      FUSILLI_LOG_LABEL_RED("ERROR: " << error << " ");                        \
      FUSILLI_LOG_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__);        \
    }                                                                          \
    REQUIRE(isOk(error));                                                      \
  } while (false)

// Test side dual to FUSILLI_ASSIGN_OR_RETURN. Unwrap the value from an
// expression that evaluates to an ErrorOr<T>, and fail the test using Catch2's
// REQUIRE if the result is an error. The unwrapped value is assigned to the
// provided variable declaration.
//
// This is very similar to FUSILLI_ASSIGN_OR_RETURN, but
// FUSILLI_ASSIGN_OR_RETURN propagates an error to callers on the error path,
// while this macro fails the test on the error path. The two macros are
// analogous to Rust's `?` (try) operator and `.unwrap()` call respectively.
//
// Usage:
//   ErrorOr<int> getValue();
//
//   TEST_CASE("thing", "[example]") {
//     FUSILLI_REQUIRE_ASSIGN(auto value, getValue());
//     REQUIRE(value == 42);
//   }
#define FUSILLI_REQUIRE_ASSIGN_IMPL(errorOr, var, expr)                        \
  auto errorOr = (expr);                                                       \
  FUSILLI_REQUIRE_OK(errorOr);                                                 \
  var = std::move(*errorOr);

#define FUSILLI_REQUIRE_ASSIGN(varDecl, expr)                                  \
  FUSILLI_REQUIRE_ASSIGN_IMPL(FUSILLI_ERROR_VAR(_errorOr), varDecl, expr)

// Utility to convert vector of dims from int64_t to size_t (unsigned long)
// which is compatible with `iree_hal_dim_t` and fixes narrowing conversion
// warnings.
inline std::vector<size_t> castToSizeT(const std::vector<int64_t> &input) {
  return std::vector<size_t>(input.begin(), input.end());
}

namespace fusilli {

// Helper to create a simple MLIR module for testing.
inline std::string getSimpleMLIRModule() {
  return R"mlir(
module {
  func.func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
}

inline ErrorOr<std::shared_ptr<Buffer>>
allocateBufferOfType(Handle &handle, const std::shared_ptr<TensorAttr> &tensor,
                     DataType type, float initVal) {
  FUSILLI_RETURN_ERROR_IF(!tensor, ErrorCode::AttributeNotSet,
                          "Tensor is not set");

  switch (type) {
  case DataType::Float: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<float>(tensor->getVolume(), float(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::Int32: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(handle,
                         /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
                         /*bufferData=*/
                         std::vector<int>(tensor->getVolume(), int(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::Half: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<half>(tensor->getVolume(), half(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::BFloat16: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<bf16>(tensor->getVolume(), bf16(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::Int16: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<int16_t>(tensor->getVolume(), int16_t(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::Int8: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<int8_t>(tensor->getVolume(), int8_t(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  case DataType::Boolean: {
    FUSILLI_ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::allocate(
            handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
            /*bufferData=*/
            std::vector<int8_t>(tensor->getVolume(), int8_t(initVal))));
    return std::make_shared<Buffer>(std::move(buffer));
  }
  default:
    return error(ErrorCode::InvalidAttribute, "Unsupported DataType");
  }
}

inline TensorAttr createTestTensorAttr(const std::string &name,
                                       std::span<const int64_t> dim) {

  std::vector<int64_t> stride(dim.size());
  int64_t strideAcc = 1;
  for (int64_t i = static_cast<int64_t>(dim.size()) - 1; i >= 0; --i) {
    stride[i] = strideAcc;
    strideAcc *= dim[i];
  }

  TensorAttr attr;
  attr.setName(name).setDim(dim).setStride(stride);
  return attr;
}

inline std::shared_ptr<TensorAttr>
createTestTensor(const std::string &name, std::span<const int64_t> dim,
                 Graph *graph) {
  return graph->tensor(createTestTensorAttr(name, dim));
}

inline ErrorObject testUnaryPointwiseAsmEmitter(const std::string &graphName,
                                                const std::string &opName,
                                                const std::string &mode,
                                                PointwiseAttr::Mode pwMode,
                                                std::vector<int64_t> inDims) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = createTestTensor("arg0", inDims, graph.get());

  auto pointwiseAttr = PointwiseAttr().setMode(pwMode).setName(opName);

  auto yT = graph->pointwise(xT, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::AMDGPU));
#else
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

inline ErrorObject testBinaryPointwiseAsmEmitter(const std::string &graphName,
                                                 const std::string &opName,
                                                 const std::string &mode,
                                                 PointwiseAttr::Mode pwMode,
                                                 std::vector<int64_t> lhsDims,
                                                 std::vector<int64_t> rhsDims) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = createTestTensor("arg0", lhsDims, graph.get());
  auto bT = createTestTensor("arg1", rhsDims, graph.get());

  auto pointwiseAttr = PointwiseAttr().setMode(pwMode).setName(opName);

  auto yT = graph->pointwise(xT, bT, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::AMDGPU));
#else
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

} // namespace fusilli

#endif // FUSILLI_TESTS_UTILS_H
