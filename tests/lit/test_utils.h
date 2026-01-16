// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WORKSPACE_TESTS_LIT_TEST_UTILS_H
#define WORKSPACE_TESTS_LIT_TEST_UTILS_H

#include <fusilli.h>

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fusilli {
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
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
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
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
  }

  return ok();
}

} // namespace fusilli

#endif // WORKSPACE_TESTS_LIT_TEST_UTILS_H
