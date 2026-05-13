// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Pointwise-specific test helpers for lit tests that verify ASM emitter output.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_TESTS_POINTWISE_UTILS_H
#define FUSILLI_TESTS_POINTWISE_UTILS_H

#include "utils.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fusilli {

inline ErrorObject testUnaryPointwiseAsmEmitter(const std::string &graphName,
                                                const std::string &mode,
                                                PointwiseAttr &pointwiseAttr,
                                                std::vector<int64_t> inDims) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = createTestTensor("arg0", inDims, graph.get());

  auto yT = graph->pointwise(xT, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(kDefaultBackend));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

inline ErrorObject testBinaryPointwiseAsmEmitter(const std::string &graphName,
                                                 const std::string &mode,
                                                 PointwiseAttr &pointwiseAttr,
                                                 std::vector<int64_t> lhsDims,
                                                 std::vector<int64_t> rhsDims) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = createTestTensor("arg0", lhsDims, graph.get());
  auto bT = createTestTensor("arg1", rhsDims, graph.get());

  auto yT = graph->pointwise(xT, bT, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(kDefaultBackend));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

inline ErrorObject testTernaryPointwiseAsmEmitter(
    const std::string &graphName, const std::string &opName,
    const std::string &mode, PointwiseAttr::Mode pwMode,
    std::vector<int64_t> in0Dims, std::vector<int64_t> in1Dims,
    std::vector<int64_t> in2Dims) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // IN_0 is a boolean predicate; IN_1 and IN_2 carry the selected values.
  auto in0StrideOrder = getContiguousStrideOrder(in0Dims.size());
  auto in0T = graph->tensor(
      TensorAttr()
          .setName("arg0")
          .setDataType(DataType::Boolean)
          .setDim(in0Dims)
          .setStride(generateStrideFromDim(in0Dims, in0StrideOrder)));
  auto in1T = createTestTensor("arg1", in1Dims, graph.get());
  auto in2T = createTestTensor("arg2", in2Dims, graph.get());

  auto pointwiseAttr = PointwiseAttr().setMode(pwMode).setName(opName);

  auto yT = graph->pointwise(in0T, in1T, in2T, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(kDefaultBackend));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

} // namespace fusilli

#endif // FUSILLI_TESTS_POINTWISE_UTILS_H
