// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the declaration for the `Graph` class which derives from
// the `INode` class (like other nodes).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_GRAPH_H
#define FUSILLI_GRAPH_GRAPH_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/attributes/layernorm_attributes.h"
#include "fusilli/attributes/matmul_attributes.h"
#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/reduction_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/layernorm_node.h"
#include "fusilli/node/matmul_node.h"
#include "fusilli/node/node.h"
#include "fusilli/node/pointwise_node.h"
#include "fusilli/node/reduction_node.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/logging.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define IREE_COMPILE_INPUT_FILENAME "iree-compile-input.mlir"
#define IREE_COMPILE_OUTPUT_FILENAME "iree-compile-output.vmfb"
#define IREE_COMPILE_COMMAND_FILENAME "iree-compile-command.txt"
#define IREE_COMPILE_STATISTICS_FILENAME "iree-compile-statistics.json"

namespace fusilli {

bool checkCompileBackendEnv();

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  // Validates the graph for correctness and infers missing properties.
  ErrorObject validate();

  // Compiles the graph using IREE compiler and sets up the IREE VM
  // context for future g->execute calls.
  //
  // Set `remove = true` to remove compilation artifacts (cache files) when
  // this `Graph` instance goes out of scope.
  ErrorObject compile(const Handle &handle, bool remove = false);

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

  // Declarations for tensor and op builder methods.
  // Definitions are in graph.cc.
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

  // Query required workspace buffer size.
  // Returns std::nullopt if not compiled, 0 if no workspace needed,
  // or the required size in bytes.
  std::optional<size_t> getWorkspaceSize() const { return workspaceSize_; }

  // ASM emitter driver method.
  //
  // TODO(#13): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::string> emitAsm();

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
                      bool remove, std::optional<bool> *reCompiled = nullptr);

  ErrorOr<std::string> readCompilationCacheFile(CachedAssetsType type);

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
                           const std::string &generatedAsm, bool remove);

  // Check for cache validity. Cache should be invalidated if:
  //  - Cache has not been generated for this instance yet
  //  - Graph name (and therefore cache path) has changed
  //  - Generated assembly differs
  //  - Compile commands have changed
  //  - Handle/backend (and therefore compile command) has changed
  ErrorOr<bool> validateCache(const Handle &handle,
                              const std::string &generatedAsm);

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name);

  ErrorObject preValidateNode() const override final;

  ErrorObject inferPropertiesNode() override final;

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

} // namespace fusilli

#endif // FUSILLI_GRAPH_GRAPH_H
