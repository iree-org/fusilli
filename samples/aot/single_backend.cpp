// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
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

void executeAndCheck(Handle &handle, PointwiseGraph &ctx) {
  FUSILLI_REQUIRE_ASSIGN(
      auto x0Buf, allocateBufferOfType(handle, ctx.x0, DataType::Int32, 2));
  FUSILLI_REQUIRE_ASSIGN(
      auto x1Buf, allocateBufferOfType(handle, ctx.x1, DataType::Int32, 3));
  FUSILLI_REQUIRE_ASSIGN(
      auto yBuf, allocateBufferOfType(handle, ctx.y, DataType::Int32, 0));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {ctx.x0, x0Buf},
          {ctx.x1, x1Buf},
          {ctx.y, yBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(
      auto workspace, allocateWorkspace(handle, ctx.graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(ctx.graph->execute(handle, variantPack, workspace));

  std::vector<int> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == 5);
}

void compileAndCopyArtifact(Graph &graph, Backend backend,
                            const std::filesystem::path &callerOwnedArtifact) {
  FUSILLI_REQUIRE_ASSIGN(auto compiledArtifact,
                         graph.compileToArtifact(backend, /*remove=*/true));

  REQUIRE(std::filesystem::exists(compiledArtifact));
  REQUIRE(!graph.getWorkspaceSize().has_value());

  std::error_code err;
  std::filesystem::copy_file(compiledArtifact, callerOwnedArtifact,
                             std::filesystem::copy_options::overwrite_existing,
                             err);
  REQUIRE(!err);
}

} // namespace

TEST_CASE("AOT single-backend artifact compile/load/execute round trip",
          "[aot][graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  const std::filesystem::path callerOwnedArtifact =
      std::filesystem::temp_directory_path() /
      "fusilli_aot_single_backend_sample.vmfb";
  auto cleanup =
      ScopeExit([&] { std::filesystem::remove(callerOwnedArtifact); });

  {
    // With `remove=true`, Fusilli removes the compile-side cache files when
    // the compile graph is destroyed. Copy the VMFB into caller-owned storage
    // before leaving this scope.
    auto compileGraph =
        buildPointwiseAddGraph("aot_single_backend_compile_graph");
    compileAndCopyArtifact(*compileGraph.graph, handle.getBackend(),
                           callerOwnedArtifact);
  }

  REQUIRE(std::filesystem::exists(callerOwnedArtifact));

  auto runtimeGraph =
      buildPointwiseAddGraph("aot_single_backend_runtime_graph");
  FUSILLI_REQUIRE_OK(
      runtimeGraph.graph->loadFromArtifact(handle, callerOwnedArtifact));

  REQUIRE(runtimeGraph.graph->getWorkspaceSize().has_value());
  executeAndCheck(handle, runtimeGraph);
}
