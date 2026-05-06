// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

namespace {

struct PointwiseGraph {
  std::shared_ptr<Graph> graph;
  std::shared_ptr<TensorAttr> x0;
  std::shared_ptr<TensorAttr> x1;
  std::shared_ptr<TensorAttr> y;
};

PointwiseGraph buildPointwiseAddGraph(const std::string &graphName) {
  const std::vector<int64_t> dims = {4};

  auto graph = std::make_shared<Graph>();
  graph->setName(graphName);
  graph->setIODataType(DataType::Int32).setComputeDataType(DataType::Int32);

  auto x0T =
      graph->tensor(TensorAttr().setName("lhs").setDim(dims).setStride({1}));
  auto x1T =
      graph->tensor(TensorAttr().setName("rhs").setDim(dims).setStride({1}));

  auto pointwiseAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
  auto yT = graph->pointwise(x0T, x1T, pointwiseAttr);
  yT->setName("result").setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  return {graph, x0T, x1T, yT};
}

std::vector<uint8_t> compileArtifact(Backend backend) {
  auto compileGraph = buildPointwiseAddGraph("aot_multi_backend_compile_graph");
  FUSILLI_REQUIRE_ASSIGN(
      auto compiledArtifactBytes,
      compileGraph.graph->compileToArtifact(backend, /*remove=*/true));

  REQUIRE(!compiledArtifactBytes.empty());
  REQUIRE(!compileGraph.graph->getWorkspaceSize().has_value());
  return compiledArtifactBytes;
}

void loadAndExecuteArtifact(Backend backend,
                            const std::vector<uint8_t> &vmfbBytes,
                            const std::string &graphName) {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(backend));

  auto runtimeGraph = buildPointwiseAddGraph(graphName);
  FUSILLI_REQUIRE_OK(runtimeGraph.graph->loadFromArtifact(handle, vmfbBytes));

  REQUIRE(runtimeGraph.graph->getWorkspaceSize().has_value());

  FUSILLI_REQUIRE_ASSIGN(
      auto x0Buf,
      allocateBufferOfType(handle, runtimeGraph.x0, DataType::Int32, 2));
  FUSILLI_REQUIRE_ASSIGN(
      auto x1Buf,
      allocateBufferOfType(handle, runtimeGraph.x1, DataType::Int32, 3));
  FUSILLI_REQUIRE_ASSIGN(auto yBuf, allocateBufferOfType(handle, runtimeGraph.y,
                                                         DataType::Int32, 0));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {runtimeGraph.x0, x0Buf},
          {runtimeGraph.x1, x1Buf},
          {runtimeGraph.y, yBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(
      auto workspace,
      allocateWorkspace(handle, runtimeGraph.graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(
      runtimeGraph.graph->execute(handle, variantPack, workspace));

  std::vector<int> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == 5);
}

} // namespace

TEST_CASE("AOT multi-backend artifacts can be selected per execution backend",
          "[aot][graph]") {
  // Compile Phase:
  // Compile by backend only. Handles and devices are created later, matching an
  // AOT flow where compilation and execution happen on different machines.
  auto cpuArtifact = compileArtifact(Backend::CPU);
#if defined(FUSILLI_ENABLE_AMDGPU)
  auto gpuArtifact = compileArtifact(Backend::AMDGPU);
#endif

  // Caller code may serialize and de-serialize the compiled artifacts here.
  // <serialize / de-serialize>

  // Execute Phase:
  // Load needs handles / devices and the compiled artifact (bytes)
  loadAndExecuteArtifact(Backend::CPU, cpuArtifact,
                         "aot_multi_backend_runtime_cpu_graph");
#if defined(FUSILLI_ENABLE_AMDGPU)
  loadAndExecuteArtifact(Backend::AMDGPU, gpuArtifact,
                         "aot_multi_backend_runtime_amdgpu_graph");
#endif
}
