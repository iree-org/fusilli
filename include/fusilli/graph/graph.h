// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `Graph` class which derives from the
// `INode` class (like other nodes).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_GRAPH_H
#define FUSILLI_GRAPH_GRAPH_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/attributes/custom_op_attributes.h"
#include "fusilli/attributes/layernorm_attributes.h"
#include "fusilli/attributes/matmul_attributes.h"
#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/reduction_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/compile_command.h"
#include "fusilli/backend/compile_session.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/custom_op_node.h"
#include "fusilli/node/layernorm_node.h"
#include "fusilli/node/matmul_node.h"
#include "fusilli/node/node.h"
#include "fusilli/node/pointwise_node.h"
#include "fusilli/node/reduction_node.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define IREE_COMPILE_INPUT_FILENAME "iree-compile-input.mlir"
#define IREE_COMPILE_OUTPUT_FILENAME "iree-compile-output.vmfb"
#define IREE_COMPILE_COMMAND_FILENAME "iree-compile-command.txt"
#define IREE_COMPILE_STATISTICS_FILENAME "iree-compile-statistics.json"

namespace fusilli {

inline bool checkCompileBackendEnv() {
  const char *backend = std::getenv("FUSILLI_COMPILE_BACKEND_USE_CLI");
  return backend && strcmp(backend, "0") != 0;
}

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  // Validates the graph for correctness and infers missing properties.
  ErrorObject validate() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating Graph");
    FUSILLI_RETURN_ERROR_IF(getName().empty(), ErrorCode::AttributeNotSet,
                            "Graph name not set");
    // Validate nodes:
    // This infers missing tensor properties such as dims,
    // stride, dtype based on context.
    FUSILLI_CHECK_ERROR(validateSubtree());
    // Validate inputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on inputs first.
    for (const auto &input : fullGraphInputs_) {
      FUSILLI_CHECK_ERROR(input->validate());
    }
    // Validate outputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on outputs first.
    for (const auto &output : fullGraphOutputs_) {
      FUSILLI_CHECK_ERROR(output->validate());
    }
    FUSILLI_LOG_LABEL_ENDL("INFO: Graph validation completed successfully");
    isValidated_ = true;
    return ok();
  }

  // Compiles the graph using IREE compiler and sets up the IREE VM
  // context for future g->execute calls.
  //
  // Set `remove = true` to remove compilation artifacts (cache files) when
  // this `Graph` instance goes out of scope.
  ErrorObject compile(const Handle &handle, bool remove = false) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Compiling Graph");
    FUSILLI_RETURN_ERROR_IF(!isValidated_, ErrorCode::NotValidated,
                            "Graph must be validated before being compiled");

    // Generate MLIR assembly for this graph.
    FUSILLI_ASSIGN_OR_RETURN(std::string generatedAsm, emitAsm());

    // Compile using IREE compiler or reuse cached artifact.
    FUSILLI_ASSIGN_OR_RETURN(auto vmfbPath,
                             getCompiledArtifact(handle, generatedAsm, remove));

    FUSILLI_LOG_LABEL_ENDL("INFO: Compiled Graph cached at \"" +
                           vmfbPath.string() + "\"");

    // Create per-graph IREE VM context and load the compiled artifact.
    FUSILLI_CHECK_ERROR(createVmContext(handle, vmfbPath.string()));

    return ok();
  }

  // Executes the graph using IREE runtime. Requires a `variantPack` which is a
  // map from `TensorAttr` to `Buffer` wrapping the `iree_hal_buffer_view_t *`.
  // Definition in `fusilli/backend/runtime.h`.
  //
  // Backend Specific Execution Behavior
  //   For some backends execution will be async. The specifics of how one
  //   should launch a kernel and synchronize work items vary per backend.
  //
  // CPU
  //   Synchronous execution, no synchronization necessary.
  //
  // AMDGPU
  //   Asynchronous execution. Synchronization will depend on if you're using an
  //   external hip stream.

  //   `Handle::create(Backend::AMDGPU)`
  //      When not using an external hip stream, kernel launches are async and
  //      launched on the default (null) stream. Reads, writes, and allocations
  //      done through the fusilli APIs (`fusilli::Buffer::allocate`,
  //      `fusilli::Buffer::read`, etc.) are stream ordered (aka happen in the
  //      order executed). Read implicitly synchronizes (waits for anything in
  //      the stream behind it to finish) ensuring correct data is read. Any GPU
  //      read or write done outside of the fusilli APIs would lead to undefined
  //      behavior.
  //
  //      Note: The default stream has implicit synchronization with all other
  //      streams and will therefore limit concurrency with other streams.
  //
  //   `Handle::create(Backend::AMDGPU, /*deviceID=*/0, /*stream=*/stream)`
  //      With an external hip stream all kernel launches will be async and
  //      stream ordered on the stream provided. Assuming the default stream
  //      isn't used, there will be no synchronization with other streams. Any
  //      stream interaction will maintain normal stream ordering:
  //      `hipMallocAsync` (`hipMalloc` is synchronous so by default safe),
  //      `hipMemcpyAsync`, etc. are all fine.  Fusilli APIs are (still) stream
  //      ordered, and `fusilli::Buffer::read` maintains the synchronization
  //      behavior from the previous case.
  //
  //   Example Usage:
  //
  //     // Default stream  ===
  //     auto handle = Handle::create(Backend::AMDGPU);
  //     Graph graph;
  //
  //     // Allocate outputs
  //     auto outputBuf = std::make_shared<Buffer>(Buffer::allocate(...));
  //
  //     // Execute (async, stream ordered)
  //     graph.execute(handle, {{outputAttr, outputBuf}, ...});
  //
  //     // Read output (implicitly synchronizes stream)
  //     std::vector<half> result;
  //     outputBuf->read(handle, result);
  //
  //     // External stream  ===
  //     hipStream_t stream;
  //     hipStreamCreate(&stream);
  //     auto handle = Handle::create(Backend::AMDGPU, /*deviceID=*/0,
  //                                    /*stream=*/stream);
  //
  //     void *devicePtr;
  //     hipMallocAsync(&devicePtr, bufferSize, stream));
  //     ....
  //     auto outputBuf = std::make_shared<Buffer>(
  //                         Buffer::import(ireeBufferViewFromDevicePtr));
  //
  //     // All operations are stream ordered on the provided stream
  //     auto outputBuf = std::make_shared<Buffer>(Buffer::allocate(...));
  //     graph.execute(handle, {{outputAttr, outputBuf}, ...});
  //
  //     // Can mix with HIP operations on same stream
  //     hipMemcpyAsync(hostData, devicePtr, size,
  //                     hipMemcpyDeviceToDevice, stream);
  //     hipStreamSynchronize(stream);
  //     doSomethingWith(hostData);
  //
  // Workspace Buffer Usage:
  //   After calling compile(), query getWorkspaceSize() to determine if a
  //   workspace buffer is needed. If size > 0, allocate using
  //   Buffer::allocateRaw() and pass it to execute(). The same workspace
  //   buffer can be reused across multiple execute() calls.
  //
  //   Example:
  //     graph.compile(handle);
  //     auto wsSize = graph.getWorkspaceSize();
  //     std::shared_ptr<Buffer> workspace = nullptr;
  //     if (wsSize.value_or(0) > 0) {
  //       FUSILLI_ASSIGN_OR_RETURN(auto wsBuf,
  //                                Buffer::allocateRaw(handle, *wsSize));
  //       workspace = std::make_shared<Buffer>(std::move(wsBuf));
  //     }
  //     graph.execute(handle, variantPack, workspace);
  ErrorObject
  execute(const Handle &handle,
          const std::unordered_map<std::shared_ptr<TensorAttr>,
                                   std::shared_ptr<Buffer>> &variantPack,
          const std::shared_ptr<Buffer> &workspace) const;

  // Delete copy constructors, keep default move constructor and destructor.
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;
  ~Graph() = default;

  // Getters and setters for graph context.
  const std::string &getName() const override final {
    return context.getName();
  }
  Type getType() const override final { return Type::Composite; }

  Graph &setName(const std::string &name) {
    context.setName(name);
    return *this;
  }

  Graph &setIODataType(DataType type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &setComputeDataType(DataType type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &setIntermediateDataType(DataType type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  // Declarations for tensor and op builder methods go here.
  // Definitions are towards the end of this file below.
  std::shared_ptr<TensorAttr> tensor(const TensorAttr &tensor);

  std::shared_ptr<TensorAttr> convFProp(const std::shared_ptr<TensorAttr> &x,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvFPropAttr &attributes);
  std::shared_ptr<TensorAttr> convWGrad(const std::shared_ptr<TensorAttr> &dy,
                                        const std::shared_ptr<TensorAttr> &x,
                                        ConvWGradAttr &attributes);
  std::shared_ptr<TensorAttr> convDGrad(const std::shared_ptr<TensorAttr> &dy,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvDGradAttr &attributes);
  std::array<std::shared_ptr<TensorAttr>, 3>
  layernorm(const std::shared_ptr<TensorAttr> &x,
            const std::shared_ptr<TensorAttr> &scale,
            const std::shared_ptr<TensorAttr> &bias, LayernormAttr &attributes);
  std::shared_ptr<TensorAttr> matmul(const std::shared_ptr<TensorAttr> &a,
                                     const std::shared_ptr<TensorAttr> &b,
                                     MatmulAttr &attributes);
  std::shared_ptr<TensorAttr> pointwise(const std::shared_ptr<TensorAttr> &in,
                                        PointwiseAttr &attributes);

  std::shared_ptr<TensorAttr> pointwise(const std::shared_ptr<TensorAttr> &in0,
                                        const std::shared_ptr<TensorAttr> &in1,
                                        PointwiseAttr &attributes);

  std::shared_ptr<TensorAttr> reduction(const std::shared_ptr<TensorAttr> &x,
                                        ReductionAttr &attributes);

  template <typename... Tensors>
  std::vector<std::shared_ptr<TensorAttr>> customOp(CustomOpAttr &customOpAttr,
                                                    Tensors &&...inputArgs) {
    std::vector<std::shared_ptr<TensorAttr>> inputs{
        std::forward<Tensors>(inputArgs)...};

    // Populate name when not set.
    if (customOpAttr.getName().empty())
      customOpAttr.setName("custom_op_" + std::to_string(subNodes_.size()));

    // Auto-name unnamed inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] && inputs[i]->getName().empty())
        inputs[i]->setName(customOpAttr.getName() + "_IN_" + std::to_string(i));
    }

    FUSILLI_LOG_LABEL_ENDL("INFO: Adding CustomOpNode '"
                           << customOpAttr.getName() << "' to Graph");

    // Create output tensors. The caller sets dim/stride/datatype on the
    // returned tensors (same pattern as convDGrad/convWGrad).
    std::vector<std::shared_ptr<TensorAttr>> outputTensors;
    outputTensors.reserve(customOpAttr.getNumOutputs());
    for (int i = 0; i < customOpAttr.getNumOutputs(); ++i)
      outputTensors.push_back(
          outputTensor(customOpAttr.getName() + "_OUT_" + std::to_string(i)));

    // Create node and add to Graph's subNodes_.
    auto node =
        std::make_unique<CustomOpNode>(std::move(customOpAttr), context);
    node->inputs = inputs;
    node->outputs = outputTensors;
    subNodes_.emplace_back(std::move(node));

    return outputTensors;
  }

  // Query required workspace buffer size.
  // Returns std::nullopt if not compiled, 0 if no workspace needed,
  // or the required size in bytes.
  std::optional<size_t> getWorkspaceSize() const { return workspaceSize_; }

  // ASM emitter driver method.
  //
  // TODO(#13): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::string> emitAsm() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Emitting MLIR assembly for Graph");
    FUSILLI_RETURN_ERROR_IF(
        !isValidated_, ErrorCode::NotValidated,
        "Graph must be validated before emitting MLIR assembly");
    std::ostringstream oss;
    emitAsmSubtree(oss);
    FUSILLI_LOG_ENDL(oss.str());
    return ok(oss.str());
  }

  // Return compiled artifact. The first invocation will always generate
  // compiled artifact, subsequent invocations may return cached versions
  // assuming cache invalidation checks pass. Set `remove = true` to remove
  // cache files when this `Graph` instance goes out of scope.
  //
  // `reCompiled` will be set to true if a value is passed and the cache was
  // (re)generated; this parameter is useful for testing.
  //
  // TODO(#13): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::filesystem::path>
  getCompiledArtifact(const Handle &handle, const std::string &generatedAsm,
                      bool remove, std::optional<bool> *reCompiled = nullptr) {
    // Check for cache hit.
    FUSILLI_ASSIGN_OR_RETURN(bool cacheValid,
                             validateCache(handle, generatedAsm));
    if (cacheValid) {
      if (reCompiled)
        *reCompiled = false;
      return ok(cache_->output.path);
    }
    // (Re)generate cache.
    FUSILLI_ASSIGN_OR_RETURN(
        auto generatedCache,
        generateCompiledArtifact(handle, generatedAsm, remove));
    cache_ = std::move(generatedCache);
    if (reCompiled)
      *reCompiled = true;
    return ok(cache_->output.path);
  }

  ErrorOr<std::string> readCompilationCacheFile(CachedAssetsType type) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Getting cached assets path");
    FUSILLI_RETURN_ERROR_IF(!cache_.has_value(), ErrorCode::FileSystemFailure,
                            "Cache not populated yet");

    // `CacheFile::read` already returns an `ErrorOr<std::string>`
    // so don't wrap it in another `ok()` here.
    switch (type) {
    case CachedAssetsType::Input:
      return cache_->input.read();
    case CachedAssetsType::Command:
      return cache_->command.read();
    case CachedAssetsType::Output:
      return cache_->output.read();
    case CachedAssetsType::Statistics:
      return cache_->statistics.read();
    default:
      return error(ErrorCode::InvalidAttribute, "Unknown CachedAssetsType");
    }
  }

private:
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject createVmContext(const Handle &handle,
                              const std::string &vmfbPath);

  // Queries the required transient/workspace buffer size from the compiled
  // module. Returns the size in bytes, or 0 if no transients are needed.
  // Returns an error if the module requires dynamic transient sizes.
  // Definition in `fusilli/backend/runtime.h`.
  ErrorOr<size_t> queryTransientSize();

  // Create compiled artifacts from graph writing results to the cache. Set
  // `remove = true` to remove cache files when returned `CachedAssets` lifetime
  // ends.
  ErrorOr<CachedAssets>
  generateCompiledArtifact(const Handle &handle,
                           const std::string &generatedAsm, bool remove) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Generating compiled artifacts");

    // Create cache files.
    FUSILLI_ASSIGN_OR_RETURN(auto inputCache,
                             CacheFile::create(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_INPUT_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto outputCache,
                             CacheFile::create(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto commandCache,
                             CacheFile::create(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_COMMAND_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto statisticsCache,
                             CacheFile::create(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME,
                                 /*remove=*/remove));
    CachedAssets cache = CachedAssets(
        /*in=*/std::move(inputCache),
        /*out=*/std::move(outputCache),
        /*cmd=*/std::move(commandCache),
        /*stats=*/std::move(statisticsCache));

    // Write input asm to cache.
    FUSILLI_CHECK_ERROR(cache.input.write(generatedAsm));

    // determine which implementation to use.
    if (checkCompileBackendEnv()) {
      // Use CompileCommand (CLI).
      CompileCommand cmd = CompileCommand::build(
          handle, cache.input, cache.output, cache.statistics);
      FUSILLI_CHECK_ERROR(cmd.writeTo(cache.command));
      FUSILLI_LOG_LABEL_ENDL("INFO: iree-compile command (CLI)");
      FUSILLI_LOG_ENDL(cmd.toString());
      FUSILLI_CHECK_ERROR(cmd.execute());
    } else {
      // Use CompileSession (C API) - DEFAULT.
      FUSILLI_ASSIGN_OR_RETURN(CompileSession session,
                               CompileSession::build(handle, cache.input,
                                                     cache.output,
                                                     cache.statistics));
      FUSILLI_CHECK_ERROR(session.writeTo(cache.command));
      FUSILLI_LOG_LABEL_ENDL("INFO: iree-compile command (C API)");
      FUSILLI_LOG_ENDL(session.toString());
      FUSILLI_CHECK_ERROR(session.execute());
    }

    return ok(std::move(cache));
  }

  // Check for cache validity. Cache should be invalidated if:
  //  - Cache has not been generated for this instance yet
  //  - Graph name (and therefore cache path) has changed
  //  - Generated assembly differs
  //  - Compile commands have changed
  //  - Handle/backend (and therefore compile command) has changed
  ErrorOr<bool> validateCache(const Handle &handle,
                              const std::string &generatedAsm) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating cache");

    // Check for cache miss if cache hasn't been generated.
    if (!cache_.has_value()) {
      FUSILLI_LOG_ENDL("Cache not previously populated.");
      return ok(false);
    }

    // Check for cache miss if paths don't match (e.g., if graph name changed).
    if (cache_->input.path != CacheFile::getPath(
                                  /*graphName=*/getName(),
                                  /*fileName=*/IREE_COMPILE_INPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache input paths differ.");
      return ok(false);
    }
    if (cache_->output.path != CacheFile::getPath(
                                   /*graphName=*/getName(),
                                   /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache output paths differ.");
      return ok(false);
    }
    if (cache_->command.path !=
        CacheFile::getPath(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_COMMAND_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile command paths differ.");
      return ok(false);
    }
    if (cache_->statistics.path !=
        CacheFile::getPath(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile statistics paths differ.");
      return ok(false);
    }

    // Open expected files.
    FUSILLI_ASSIGN_OR_RETURN(CacheFile input,
                             CacheFile::open(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_INPUT_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(CacheFile output,
                             CacheFile::open(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(CacheFile command,
                             CacheFile::open(
                                 /*graphName=*/getName(),
                                 /*fileName=*/IREE_COMPILE_COMMAND_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(
        CacheFile statistics,
        CacheFile::open(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME));

    // Check for a cache miss on generated assembly.
    FUSILLI_ASSIGN_OR_RETURN(std::string inputContents, input.read());
    if (inputContents != generatedAsm) {
      FUSILLI_LOG_ENDL("Generated assembly does not match");
      return ok(false);
    }

    // Check for a cache miss on compile command.
    std::string cmdString;

    if (checkCompileBackendEnv()) {
      // Use CompileCommand (CLI).
      CompileCommand cmd =
          CompileCommand::build(handle, input, output, statistics);
      cmdString = cmd.toString();
    } else {
      // Use CompileSession (C API) - DEFAULT.
      FUSILLI_ASSIGN_OR_RETURN(
          auto session,
          CompileSession::build(handle, input, output, statistics));
      cmdString = session.toString();
    }

    FUSILLI_ASSIGN_OR_RETURN(std::string commandContents, command.read());
    if (commandContents != cmdString) {
      FUSILLI_LOG_ENDL("Compile command does not match");
      return ok(false);
    }

    return ok(true);
  }

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Adding output tensor '"
                           << name << "' to Graph outputs");
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating Graph");
    // Validate input/output names are unique (requirement for SSA).
    std::unordered_set<std::string> usedSymbols;
    for (const auto &t : fullGraphInputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    for (const auto &t : fullGraphOutputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    // Recursively validate node names are unique (requirement for SSA).
    FUSILLI_CHECK_ERROR(checkNodeNamesAreUnique(usedSymbols));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for Graph");
    // Populate sorted inputs / outputs after graph is fully constructed
    // and pre-validated (to ensure no symbol conflict).
    fullGraphInputsSorted_.insert(fullGraphInputs_.begin(),
                                  fullGraphInputs_.end());
    fullGraphOutputsSorted_.insert(fullGraphOutputs_.begin(),
                                   fullGraphOutputs_.end());
    return ok();
  }

  ErrorObject postValidateNode() const override final { return ok(); }

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string emitNodePostAsm() const override final;
  std::string getOperandNamesAndTypesAsm() const;
  std::string getResultNamesAndTypesAsm() const;

  // This is set after `validate()` is run at least once successfully.
  bool isValidated_ = false;

  // Required workspace buffer size in bytes. Set during createVmContext()
  // by querying the iree.abi.transients.size.constant attribute.
  // std::nullopt indicates the graph has not been compiled yet.
  std::optional<size_t> workspaceSize_;

  // IREE VM context lifetime managed by the `Graph` object
  // (deleted when the `Graph` object goes out of scope).
  IreeVmContextUniquePtrType vmContext_;

  // Memoized function handle resolved during createVmContext().
  // Avoids repeated function lookup on every execute() call.
  std::optional<iree_vm_function_t> vmFunction_;

  // Pre-computed VM input list capacity for iree_vm_list_create().
  // Set during createVmContext() to avoid recomputing on every execute().
  iree_host_size_t vmInputListCapacity_ = 0;

  // Cache set by `getCompiledArtifact()`.
  //
  // Note: new instances should always re-generate cache even if the results
  // could be read from the file system. Old results may have been generated
  // with a different version of IREE, it would not be safe to use them.
  std::optional<CachedAssets> cache_;

  // This is safe for post-insertion updates of TensorAttr (e.g. setting name
  // or other properties) since it uses the pointer value itself for hashing.
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphInputs_;
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphOutputs_;

  // These are sorted by the TensorAttr name, so post-insertion modification is
  // UB (undefined behavior). These are to be populated after the graph is fully
  // constructed and validated, and no further updates are expected.
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphInputsSorted_;
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphOutputsSorted_;
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(const TensorAttr &tensor) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Adding input tensor '" << tensor.getName()
                                                       << "' to Graph inputs");
  auto tensorPtr = std::make_shared<TensorAttr>(tensor);
  fullGraphInputs_.insert(tensorPtr);
  return tensorPtr;
}

// Create a ConvFPropNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convFProp(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvFPropAttr &convAttr) {
  // Populate names when not set.
  if (convAttr.getName().empty())
    convAttr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(convAttr.getName() + "_X");
  if (w && w->getName().empty())
    w->setName(convAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvFPropNode '" << convAttr.getName()
                                                        << "' to Graph");

  // Set inputs.
  convAttr.setX(x).setW(w);

  // Set outputs.
  auto y = outputTensor(convAttr.getName() + "_Y");
  convAttr.setY(y);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(convAttr), context));

  return y;
}

// Create a ConvWGradNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convWGrad(const std::shared_ptr<TensorAttr> &dy,
                 const std::shared_ptr<TensorAttr> &x,
                 ConvWGradAttr &convWGradAttr) {
  // Populate names when not set.
  if (convWGradAttr.getName().empty())
    convWGradAttr.setName("conv_wgrad_" + std::to_string(subNodes_.size()));
  if (dy && dy->getName().empty())
    dy->setName(convWGradAttr.getName() + "_DY");
  if (x && x->getName().empty())
    x->setName(convWGradAttr.getName() + "_X");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvWGradNode '"
                         << convWGradAttr.getName() << "' to Graph");

  // Set inputs.
  convWGradAttr.setDY(dy).setX(x);

  // Set outputs.
  auto dw = outputTensor(convWGradAttr.getName() + "_DW");
  convWGradAttr.setDW(dw);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvWGradNode>(std::move(convWGradAttr), context));

  return dw;
}

// Create a ConvDGradNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convDGrad(const std::shared_ptr<TensorAttr> &dy,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvDGradAttr &convDGradAttr) {
  // Populate names when not set.
  if (convDGradAttr.getName().empty())
    convDGradAttr.setName("conv_dgrad_" + std::to_string(subNodes_.size()));
  if (dy && dy->getName().empty())
    dy->setName(convDGradAttr.getName() + "_DY");
  if (w && w->getName().empty())
    w->setName(convDGradAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvDGradNode '"
                         << convDGradAttr.getName() << "' to Graph");

  // Set inputs.
  convDGradAttr.setDY(dy).setW(w);

  // Set outputs.
  auto dx = outputTensor(convDGradAttr.getName() + "_DX");
  convDGradAttr.setDX(dx);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvDGradNode>(std::move(convDGradAttr), context));

  return dx;
}

// Create a LayerNormNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes
inline std::array<std::shared_ptr<TensorAttr>, 3>
Graph::layernorm(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &scale,
                 const std::shared_ptr<TensorAttr> &bias,
                 LayernormAttr &layernormAttr) {
  // Populate names when not set.
  if (layernormAttr.getName().empty())
    layernormAttr.setName("layernorm_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(layernormAttr.getName() + "_X");
  if (scale && scale->getName().empty())
    scale->setName(layernormAttr.getName() + "_SCALE");
  if (bias && bias->getName().empty())
    bias->setName(layernormAttr.getName() + "_BIAS");
  auto eps = layernormAttr.getEpsilon();
  if (eps && eps->getName().empty())
    eps->setName(layernormAttr.getName() + "_EPSILON");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding LayerNorm '" << layernormAttr.getName()
                                                    << "' to Graph");

  // Set inputs.
  layernormAttr.setX(x);
  layernormAttr.setSCALE(scale);
  layernormAttr.setBIAS(bias);

  // Set outputs.
  std::shared_ptr<TensorAttr> y = outputTensor(layernormAttr.getName() + "_Y");
  std::shared_ptr<TensorAttr> m = nullptr;
  std::shared_ptr<TensorAttr> v = nullptr;
  if (layernormAttr.getForwardPhase() == NormFwdPhase::TRAINING) {
    m = outputTensor(layernormAttr.getName() + "_MEAN");
    v = outputTensor(layernormAttr.getName() + "_INV_VARIANCE");
  }
  layernormAttr.setY(y);
  layernormAttr.setMEAN(m);
  layernormAttr.setINV_VARIANCE(v);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<LayerNormNode>(std::move(layernormAttr), context));

  // `std::move` is useful for this case because we're returning an
  // array initialized from lvalues and `std::move` avoids unnecessary
  // copy and ref count operations on the shared pointers. This isn't
  // necessary in methods where a single local variable is returned
  // as NRVO would handle it.
  return {std::move(y), std::move(m), std::move(v)};
}

// Create a MatmulNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::matmul(const std::shared_ptr<TensorAttr> &a,
              const std::shared_ptr<TensorAttr> &b, MatmulAttr &matmulAttr) {
  // Populate names when not set.
  if (matmulAttr.getName().empty())
    matmulAttr.setName("matmul_" + std::to_string(subNodes_.size()));
  if (a && a->getName().empty())
    a->setName(matmulAttr.getName() + "_A");
  if (b && b->getName().empty())
    b->setName(matmulAttr.getName() + "_B");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding MatmulNode '" << matmulAttr.getName()
                                                     << "' to Graph");

  // Set inputs.
  matmulAttr.setA(a).setB(b);

  // Set outputs.
  auto c = outputTensor(matmulAttr.getName() + "_C");
  matmulAttr.setC(c);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<MatmulNode>(std::move(matmulAttr), context));

  return c;
}

// Create a PointwiseNode for single operand cases (e.g. RELU), populate it with
// the specified attributes, create output tensors and add the node to the
// graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::pointwise(const std::shared_ptr<TensorAttr> &in,
                 PointwiseAttr &pointwiseAttr) {
  // Populate names when not set.
  if (pointwiseAttr.getName().empty())
    pointwiseAttr.setName("pointwise_" + std::to_string(subNodes_.size()));
  if (in && in->getName().empty())
    in->setName(pointwiseAttr.getName() + "_IN_0");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding PointwiseNode '"
                         << pointwiseAttr.getName() << "' to Graph");

  // Set inputs.
  pointwiseAttr.setIN_0(in);

  // Set outputs.
  auto out = outputTensor(pointwiseAttr.getName() + "_OUT_0");
  pointwiseAttr.setOUT_0(out);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<PointwiseNode>(std::move(pointwiseAttr), context));

  return out;
}

// Create a PointwiseNode for cases with two operands (e.g. ADD), populate it
// with the specified attributes, create output tensors and add the node to the
// graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::pointwise(const std::shared_ptr<TensorAttr> &in0,
                 const std::shared_ptr<TensorAttr> &in1,
                 PointwiseAttr &pointwiseAttr) {
  // Populate names when not set.
  if (pointwiseAttr.getName().empty())
    pointwiseAttr.setName("pointwise_" + std::to_string(subNodes_.size()));
  if (in0 && in0->getName().empty())
    in0->setName(pointwiseAttr.getName() + "_IN_0");
  if (in1 && in1->getName().empty())
    in1->setName(pointwiseAttr.getName() + "_IN_1");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding PointwiseNode '"
                         << pointwiseAttr.getName() << "' to Graph");

  // Set inputs.
  pointwiseAttr.setIN_0(in0).setIN_1(in1);

  // Set outputs.
  auto out = outputTensor(pointwiseAttr.getName() + "_OUT_0");
  pointwiseAttr.setOUT_0(out);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<PointwiseNode>(std::move(pointwiseAttr), context));

  return out;
}

// Create a ReductionNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::reduction(const std::shared_ptr<TensorAttr> &x,
                 ReductionAttr &reductionAttr) {
  // Populate names when not set.
  if (reductionAttr.getName().empty())
    reductionAttr.setName("reduction_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(reductionAttr.getName() + "_X");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ReductionNode '"
                         << reductionAttr.getName() << "' to Graph");

  // Set inputs.
  reductionAttr.setX(x);

  // Set outputs.
  auto y = outputTensor(reductionAttr.getName() + "_Y");
  reductionAttr.setY(y);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ReductionNode>(std::move(reductionAttr), context));

  return y;
}

} // namespace fusilli

#endif // FUSILLI_GRAPH_GRAPH_H
