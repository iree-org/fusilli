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

inline ErrorObject testSwishPointwiseAsmEmitter(const std::string &graphName,
                                                const std::string &opName,
                                                const std::string &mode,
                                                std::vector<int64_t> inDims,
                                                float beta) {

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = createTestTensor("arg0", inDims, graph.get());

  auto pointwiseAttr = PointwiseAttr()
                           .setMode(PointwiseAttr::Mode::SWISH_FWD)
                           .setName(opName)
                           .setSwishBeta(beta);

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

} // namespace fusilli

#endif // FUSILLI_TESTS_POINTWISE_UTILS_H
