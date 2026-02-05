// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Graph getName correctly propagates the context name", "[graph]") {
  Graph g;
  g.setName("foo_graph");
  REQUIRE(g.getName() == "foo_graph");
}

TEST_CASE("Graph tensor() adds input tensor", "[graph]") {
  Graph g;
  g.setName("adds_input_tensor");
  auto t =
      g.tensor(TensorAttr().setName("input").setDim({2, 2}).setStride({2, 1}));
  REQUIRE(t->getName() == "input");
  REQUIRE(t->getDim() == std::vector<int64_t>({2, 2}));
  REQUIRE(t->getStride() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("Graph conv_fprop() adds ConvFPropNode and output tensor",
          "[graph]") {
  Graph g;
  g.setName("adds_convfpropnode_and_output_tensor");
  auto x =
      g.tensor(TensorAttr().setDim({1, 8, 8, 3}).setStride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr().setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
  auto y = g.convFProp(x, w, attr);

  // Names for inputs are auto-populated when not set.
  REQUIRE(x->getName() == "conv_fprop_0_X");
  REQUIRE(w->getName() == "conv_fprop_0_W");
  REQUIRE(y->getName() == "conv_fprop_0_Y");

  // Y is virtual (intermediate tensor) unless specified as output.
  REQUIRE(y->isVirtual() == true);
  y->setOutput(true);
  REQUIRE(y->isVirtual() == false);
}

TEST_CASE("Graph validate() fails if name is not set", "[graph]") {
  Graph g;
  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
  REQUIRE(status.getMessage() == "Graph name not set");

  g.setName("name_is_set_now");
  FUSILLI_REQUIRE_OK(g.validate());
}

TEST_CASE("Graph validate() fails on missing attributes", "[graph]") {
  Graph g;
  g.setName("validate_returns_ok_for_valid_graph");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 3, 8, 8})
                        .setStride({192, 64, 8, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);

  // shape and strides of output tensor are not inferred yet
  REQUIRE(y->getDim().empty());
  REQUIRE(y->getStride().empty());

  // This runs shape/stride inference
  FUSILLI_REQUIRE_OK(g.validate());

  REQUIRE(y->getDim() == std::vector<int64_t>{1, 4, 6, 6});
  REQUIRE(y->getStride() == std::vector<int64_t>{144, 36, 6, 1});
}

// Helper function to create graph for testing.
static Graph testGraph(bool validate) {
  Graph g;
  g.setName("unvalidated_graph");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;
  auto xT = g.tensor(TensorAttr()
                         .setName("image")
                         .setDim({n, c, h, w})
                         .setStride({c * h * w, h * w, w, 1}));
  auto wT = g.tensor(TensorAttr()
                         .setName("filter")
                         .setDim({k, c, r, s})
                         .setStride({c * r * s, r * s, s, 1}));
  auto conv = ConvFPropAttr()
                  .setPadding({0, 0})
                  .setStride({1, 1})
                  .setDilation({1, 1})
                  .setName("conv_fprop");
  auto yT = g.convFProp(xT, wT, conv);
  yT->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
  yT->setOutput(true);
  if (validate) {
    g.setName("validated_graph");
    FUSILLI_REQUIRE_OK(g.validate());
  }
  return g;
};

TEST_CASE("Graph asm_emitter requires validation to be run first", "[graph]") {
  Graph g = testGraph(/*validate=*/false);

  // ASM emitter without validation should throw an error.
  auto status = g.emitAsm();
  REQUIRE(isError(status));
  REQUIRE(ErrorObject(status).getCode() == ErrorCode::NotValidated);
  REQUIRE(ErrorObject(status).getMessage() ==
          "Graph must be validated before emitting MLIR assembly");

  // Validate the graph first.
  FUSILLI_REQUIRE_OK(g.validate());

  // ASM emitter should now work.
  FUSILLI_REQUIRE_OK(g.emitAsm());
}

TEST_CASE("Graph `getCompiledArtifact` cache generation and invalidation",
          "[graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle cpuHandle, Handle::create(Backend::CPU));
#ifdef FUSILLI_ENABLE_AMDGPU
  FUSILLI_REQUIRE_ASSIGN(Handle gpuHandle, Handle::create(Backend::AMDGPU));
#endif

  Graph g = testGraph(/*validate=*/true);

  FUSILLI_REQUIRE_ASSIGN(std::string generatedAsm, g.emitAsm());

  // Cache should be empty, compilation artifacts should be generated.
  std::optional<bool> reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(cpuHandle, generatedAsm,
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit, no compilation should be required.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(cpuHandle, generatedAsm,
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());

#ifdef FUSILLI_ENABLE_AMDGPU
  // Cache should miss based on different handle / device / compile command.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(gpuHandle, generatedAsm,
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit with a re-run on the different handle.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(gpuHandle, generatedAsm,
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());
#endif

  // Cache should miss because of different generated asm.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit with the same generated asm.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());

  // Cache should miss because graph name change.
  g.setName("new_graph_name");
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());
}

TEST_CASE("Graph `getCompiledArtifact` should not read cached items from "
          "other/previous Graph instances",
          "[graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  std::string generatedAsm;
  {
    Graph g = testGraph(/*validate=*/true);

    FUSILLI_REQUIRE_ASSIGN(generatedAsm, g.emitAsm());

    // Cache should be empty.
    std::optional<bool> reCompiled = std::nullopt;
    FUSILLI_REQUIRE_OK(g.getCompiledArtifact(handle, generatedAsm,
                                             /*remove=*/false, &reCompiled));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit with the same generated asm.
    reCompiled = std::nullopt;
    FUSILLI_REQUIRE_OK(g.getCompiledArtifact(handle, generatedAsm,
                                             /*remove=*/false, &reCompiled));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());
  }

  Graph g = testGraph(/*validate=*/true);

  // Check that the generated asm matches the cache.
  FUSILLI_REQUIRE_ASSIGN(
      CacheFile asmCache,
      CacheFile::open(g.getName(), IREE_COMPILE_INPUT_FILENAME));
  FUSILLI_REQUIRE_ASSIGN(std::string asmContent, asmCache.read());
  REQUIRE(asmContent == generatedAsm);

  // Nonetheless a new instance should regenerate cache.
  std::optional<bool> reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(g.getCompiledArtifact(handle, generatedAsm,
                                           /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());
}

TEST_CASE("Graph `getCompiledArtifact` invalid input IR", "[graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));
  std::string graphName;
  {
    Graph g;
    g.setName("invalid_input_ir");
    ErrorObject err =
        g.getCompiledArtifact(handle, "invalid mlir", /*remove=*/true);
    REQUIRE(isError(err));
    REQUIRE(err.getCode() == ErrorCode::CompileFailure);
    // Error message varies between subprocess and C API backends.
    // Subprocess: "iree-compile command failed"
    // C API: "Failed to parse source file" (more detailed)
    REQUIRE(!err.getMessage().empty());
  }
  // Cache created with "remove", ensure it is removed after the test.
  REQUIRE(!std::filesystem::exists(
      CacheFile::getPath(graphName, "test").parent_path()));
}

TEST_CASE("Graph `compile` method fails without validation", "[graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));

  Graph g = testGraph(/*validate=*/false);

  auto status = g.compile(handle, /*remove=*/true);
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotValidated);
  REQUIRE(status.getMessage() ==
          "Graph must be validated before being compiled");
}

TEST_CASE("Graph `compile` recompilations with changed handle", "[graph]") {
  // This test constructs a single graph but compiles it with different
  // handles and backends, ensuring that the graph did not use cached
  // artifacts from a previous compilation and correctly re-compiled
  // for the new handle/backend.
  Graph g = testGraph(/*validate=*/true);

  // Path to compile command cache file.
  auto cacheDir = CacheFile::getCacheDir();
  std::filesystem::path cmdPath =
      cacheDir / g.getName() / "iree-compile-command.txt";

  FUSILLI_REQUIRE_ASSIGN(Handle cpuHandle, Handle::create(Backend::CPU));
  FUSILLI_REQUIRE_OK(g.compile(cpuHandle, /*remove=*/true));

  std::string cpuCmd;
  REQUIRE(std::filesystem::exists(cmdPath));
  std::ifstream cpuCmdFile(cmdPath);
  REQUIRE(cpuCmdFile.is_open());
  std::getline(cpuCmdFile, cpuCmd);
  REQUIRE(!cpuCmd.empty());

#ifdef FUSILLI_ENABLE_AMDGPU
  FUSILLI_REQUIRE_ASSIGN(Handle gpuHandle, Handle::create(Backend::AMDGPU));
  FUSILLI_REQUIRE_OK(g.compile(gpuHandle, /*remove=*/true));

  std::string gpuCmd;
  REQUIRE(std::filesystem::exists(cmdPath));
  std::ifstream gpuCmdFile(cmdPath);
  REQUIRE(gpuCmdFile.is_open());
  std::getline(gpuCmdFile, gpuCmd);
  REQUIRE(!gpuCmd.empty());

  // The compile commands should be different for CPU and GPU handles.
  REQUIRE(cpuCmd != gpuCmd);
#endif
}

TEST_CASE("Graph `execute`", "[graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("fprop_sample");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("image")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1}));

    auto wT = graph->tensor(TensorAttr()
                                .setName("filter")
                                .setDim({k, c, r, s})
                                .setStride({c * r * s, r * s, s, 1}));

    auto convAttr = ConvFPropAttr()
                        .setPadding({0, 0})
                        .setStride({1, 1})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto yT = graph->convFProp(xT, wT, convAttr);

    // Specify Y's dimensions and strides.
    yT->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
    yT->setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());

    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, wT, yT);
  };

  // Parameterize by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::CPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(Backend::AMDGPU));
    handlePtr = std::make_shared<Handle>(std::move(handle));
  }
#endif
  Handle &handle = *handlePtr;

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, X, W, Y] = buildNewGraph(handle);

  // Allocate input buffer.
  // xBuf is a shared_ptr<Buffer> and *xBuf is the de-referenced Buffer obj.
  // Hence checking `*xBuf != nullptr` might seem weird at first, but due to
  // the implicit automatic cast from `Buffer` -> `iree_hal_buffer_view_t *`,
  // `*xBuf != nullptr` simply checks that the underlying raw
  // `iree_hal_buffer_view_t *` is not NULL which is what we expect.
  FUSILLI_REQUIRE_ASSIGN(auto xBuf,
                         allocateBufferOfType(handle, X, DataType::Half, 1.0f));
  REQUIRE(*xBuf != nullptr);

  // Allocate weight buffer.
  FUSILLI_REQUIRE_ASSIGN(auto wBuf,
                         allocateBufferOfType(handle, W, DataType::Half, 1.0f));
  REQUIRE(*wBuf != nullptr);

  // Allocate output buffer.
  FUSILLI_REQUIRE_ASSIGN(auto yBuf,
                         allocateBufferOfType(handle, Y, DataType::Half, 0.0f));
  REQUIRE(*yBuf != nullptr);

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));
  REQUIRE(*yBuf != nullptr);

  // Make sure input/weight buffers are held until `xBuf` and `yBuf` are alive.
  // If `Graph::execute` were to release them (via iree_hal_buffer_view_release)
  // right after the call to iree_runtime_call_inputs_push_back_buffer_view,
  // this would seg-fault with a use-after-free so this test guards against
  // that.
  std::vector<half> input;
  FUSILLI_REQUIRE_OK(xBuf->read(handle, input));
  for (auto val : input)
    REQUIRE(val == half(1.0f));
  std::vector<half> weight;
  FUSILLI_REQUIRE_OK(wBuf->read(handle, weight));
  for (auto val : weight)
    REQUIRE(val == half(1.0f));
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(128.0f));
}
