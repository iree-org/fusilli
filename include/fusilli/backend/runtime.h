// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the wrapper code around
// IREE runtime C-APIs to create and manage VM instances, HAL devices, VM
// contexts and function invocations.
//
// Here's a rough mapping of Fusilli constructs to IREE runtime constructs
// (based on scope and lifetime):
//
//  - Group of `Handle`s manage the IREE VM instance lifetime.
//    An instance is shared across handles/threads/contexts and released
//    when the last handle goes out of scope.
//  - `Handle` manages IREE HAL device lifetime. Handles may be shared
//    by multiple graphs (as long as they intend to run on the same device).
//    Separate physical devices should have their own handles (hence logical
//    HAL device) created. Graphs running on the same physical devices should
//    reuse the same handle (hence logical HAL device). The device is released
//    when the handle holding it goes out of scope.
//  - `Graph` manages IREE VM context lifetime. A context holds state on
//    the HAL device and the loaded VM modules.
//  - `Buffer` manages IREE HAL buffer view lifetime. The buffer view is
//    released when the `Buffer` object holding it goes out of scope.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_RUNTIME_H
#define FUSILLI_BACKEND_RUNTIME_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/graph.h"
#include "fusilli/support/logging.h"

#include <iree/hal/api.h>
#include <iree/hal/drivers/hip/api.h>
#include <iree/hal/drivers/init.h>
#include <iree/io/file_contents.h>
#include <iree/modules/hal/module.h>
#include <iree/vm/api.h>
#include <iree/vm/bytecode/module.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
//
// Handle Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Register HAL drivers once in the global driver registry. The global
// registry persists for the entire process lifetime, so drivers must only
// be registered once even if all VM instances are destroyed and recreated.
inline ErrorObject registerHalDriversOnce() {
  static std::once_flag driversRegistered;
  static iree_status_t registrationStatus = iree_ok_status();
  std::call_once(driversRegistered, []() {
    registrationStatus = iree_hal_register_all_available_drivers(
        iree_hal_driver_registry_default());
  });
  FUSILLI_CHECK_ERROR(registrationStatus);
  return ok();
}

// Create static singleton IREE VM instance shared across handles/threads.
inline ErrorOr<IreeVmInstanceSharedPtrType> Handle::createSharedInstance() {
  // Mutex for thread-safe initialization of weakInstance.
  static std::mutex instanceMutex;

  // Static weak_ptr to the IREE VM instance ensures that the
  // instance is only created once and shared across all handles
  // without prolonging its lifetime till program termination. This
  // allows the instance to be released when the last handle owning
  // it goes out of scope, as opposed to hogging on to it until the
  // static variable goes out of scope upon program termination.
  static std::weak_ptr<iree_vm_instance_t> weakInstance;

  // Register HAL drivers in the global registry (idempotent via call_once).
  FUSILLI_CHECK_ERROR(registerHalDriversOnce());

  // If multiple threads simultaneously request a handle, they will
  // race into `createSharedInstance()` but only one will succeed in
  // creating the instance, and others will use it.
  std::lock_guard<std::mutex> lock(instanceMutex);

  // Try to get the shared_ptr from the weak_ptr (if it exists).
  IreeVmInstanceSharedPtrType sharedInstance = weakInstance.lock();

  // If weak_ptr expired, it means no handles are alive and holding the
  // instance, so create a new instance.
  if (sharedInstance == nullptr) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating shared IREE VM instance");
    iree_vm_instance_t *rawInstance = nullptr;
    FUSILLI_CHECK_ERROR(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &rawInstance));

    // Register HAL types with the VM instance.
    FUSILLI_CHECK_ERROR(iree_hal_module_register_all_types(rawInstance));

    // Wrap the raw instance ptr with a shared_ptr and custom deleter
    // for lifetime management.
    sharedInstance =
        IreeVmInstanceSharedPtrType(rawInstance, IreeVmInstanceDeleter());

    weakInstance = sharedInstance;
  }

  return ok(sharedInstance);
}

inline ErrorObject Handle::createCPUDevice() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-handle IREE HAL device");

  // Create a driver from the global driver registry.
  iree_hal_driver_t *driver = nullptr;
  FUSILLI_CHECK_ERROR(iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_cstring_view(kHalDriver.at(backend_)), iree_allocator_system(),
      &driver));

  // Create the default device from the driver.
  iree_hal_device_t *rawDevice = nullptr;
  iree_status_t status = iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &rawDevice);
  iree_hal_driver_release(driver);
  FUSILLI_CHECK_ERROR(status);

  // Wrap the raw device ptr with a unique_ptr and custom deleter
  // for lifetime management.
  device_ = IreeHalDeviceUniquePtrType(rawDevice);

  return ok();
}

// Copied from the IREE runtime code.
#define HIP_DEVICE_ID_TO_IREE_DEVICE_ID(device)                                \
  (iree_hal_device_id_t)((device) + 1)

inline ErrorObject Handle::createAMDGPUDevice(int deviceId, uintptr_t stream) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-handle IREE HAL device on device: "
                         << deviceId
                         << " stream: " << reinterpret_cast<void *>(stream));
// Hide some of the IREE HAL HIP driver symbols to avoid linking errors
// when building Fusilli without AMDGPU support (which disables IREE HAL
// HIP driver from being built).
#if defined(FUSILLI_ENABLE_AMDGPU)
  // Device parms.
  iree_hal_hip_device_params_t params;
  setDefaultIreeHalHipDeviceParams(&params);
  params.external_stream = stream; // set stream to provided stream

  // Create driver.
  iree_hal_hip_driver_options_t driverOptions;
  iree_hal_hip_driver_options_initialize(&driverOptions);
  iree_hal_driver_t *driver;
  FUSILLI_CHECK_ERROR(iree_hal_hip_driver_create(
      iree_make_cstring_view(kHalDriver.at(backend_)), &driverOptions, &params,
      iree_allocator_system(), &driver));

  // Create device.
  iree_hal_device_t *rawDevice = nullptr;
  FUSILLI_CHECK_ERROR(iree_hal_driver_create_device_by_id(
      driver, HIP_DEVICE_ID_TO_IREE_DEVICE_ID(deviceId), /*param_count=*/0,
      /*params=*/nullptr, iree_allocator_system(), &rawDevice));

  // Wrap the raw device ptr with a unique_ptr and custom deleter
  // for lifetime management.
  device_ = IreeHalDeviceUniquePtrType(rawDevice);
  return ok();
#else
  return ErrorObject(ErrorCode::InternalError,
                     "AMDGPU backend not supported on this platform.");
#endif
}

#undef HIP_DEVICE_ID_TO_IREE_DEVICE_ID

//===----------------------------------------------------------------------===//
//
// Graph Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Create IREE VM context for this graph and load the compiled artifact.
inline ErrorObject Graph::createVmContext(const Handle &handle,
                                          const std::string &vmfbPath) {
  // Create a context even if one was created earlier, since the handle
  // (hence device) might have changed and we might be re-compiling the graph
  // for the new device.
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-graph IREE VM context");
  iree_allocator_t allocator = iree_allocator_system();

  iree_vm_context_t *rawContext = nullptr;
  FUSILLI_CHECK_ERROR(iree_vm_context_create(
      handle.getInstance(), IREE_VM_CONTEXT_FLAG_NONE, allocator, &rawContext));

  // Create HAL module and register it with the context.
  {
    iree_hal_device_t *device = handle.getDevice();
    iree_vm_module_t *halModule = nullptr;
    FUSILLI_CHECK_ERROR(iree_hal_module_create(
        handle.getInstance(), iree_hal_module_device_policy_default(),
        /*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_NONE,
        iree_hal_module_debug_sink_null(), allocator, &halModule));
    iree_status_t status = iree_vm_context_register_modules(
        rawContext, /*module_count=*/1, &halModule);
    iree_vm_module_release(halModule);
    FUSILLI_CHECK_ERROR(status);
  }

  // Read the VMFB file and create a bytecode module from it.
  FUSILLI_LOG_LABEL_ENDL("INFO: Loading bytecode module into IREE VM context");
  {
    iree_io_file_contents_t *fileContents = nullptr;
    FUSILLI_CHECK_ERROR(iree_io_file_contents_read(
        iree_make_cstring_view(vmfbPath.c_str()), allocator, &fileContents));

    iree_vm_module_t *bytecodeModule = nullptr;
    iree_status_t status = iree_vm_bytecode_module_create(
        handle.getInstance(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        fileContents->const_buffer,
        iree_io_file_contents_deallocator(fileContents), allocator,
        &bytecodeModule);
    if (!iree_status_is_ok(status)) {
      iree_io_file_contents_free(fileContents);
      FUSILLI_CHECK_ERROR(status);
    }
    // File contents ownership transferred to bytecode module on success
    // so there's no `iree_io_file_contents_free` on the success path.

    status = iree_vm_context_register_modules(rawContext, /*module_count=*/1,
                                              &bytecodeModule);
    iree_vm_module_release(bytecodeModule);
    FUSILLI_CHECK_ERROR(status);
  }

  // Wrap the raw context ptr with a unique_ptr and custom deleter
  // for lifetime management.
  vmContext_ = IreeVmContextUniquePtrType(rawContext);

  // Resolve and cache the function handle for `module.main` or
  // `module.main$async`.
  bool executeAsync = kBackendExecuteAsync.at(handle.getBackend());
  iree_vm_function_t function;
  FUSILLI_CHECK_ERROR(iree_vm_context_resolve_function(
      vmContext_.get(),
      iree_make_cstring_view(executeAsync ? "module.main$async"
                                          : "module.main"),
      &function));
  vmFunction_ = function;

  // Pre-compute the VM input list capacity for execute().
  vmInputListCapacity_ = 0;
  // Count the number of output buffers.
  for (const auto &output : fullGraphOutputsSorted_)
    if (!output->isVirtual())
      vmInputListCapacity_++;
  // Count the number of input buffers.
  for (const auto &input : fullGraphInputsSorted_)
    if (!input->isScalar())
      vmInputListCapacity_++;
  // Count the workspace buffer (or null ref when size = 0).
  vmInputListCapacity_++;
  // Count the wait fence and signal fence for asynchronous execution.
  if (executeAsync)
    vmInputListCapacity_ += 2;

  // Query the required workspace size from the compiled module.
  FUSILLI_LOG_LABEL_ENDL("INFO: Querying workspace size from compiled module");
  FUSILLI_ASSIGN_OR_RETURN(workspaceSize_, queryTransientSize());

  return ok();
}

// Queries the required transient/workspace buffer size from the compiled
// module. The --iree-torch-externalize-transients compiler flag adds an
// attribute "iree.abi.transients.size.constant" with the required buffer size
// for the constant workspace size case, or an "iree.abi.transients.size"
// function for the data-dependent workspace size case. Only the former is
// supported by Fusilli at the moment.
inline ErrorOr<size_t> Graph::queryTransientSize() {
  // Always resolve the async function for attribute queries. The
  // iree.abi.transients.size.constant attribute is stored in the
  // iree.reflection dict on the @main$async entry point. The sync wrapper
  // @main is auto-generated and does not carry reflection attributes.
  iree_vm_function_t mainFunc;
  FUSILLI_CHECK_ERROR(iree_vm_context_resolve_function(
      vmContext_.get(), iree_make_cstring_view("module.main$async"),
      &mainFunc));

  // First check for constant transient size attribute.
  iree_string_view_t sizeAttr = iree_vm_function_lookup_attr_by_name(
      &mainFunc, IREE_SV("iree.abi.transients.size.constant"));
  if (!iree_string_view_is_empty(sizeAttr)) {
    uint64_t size = 0;
    if (iree_string_view_atoi_uint64(sizeAttr, &size)) {
      FUSILLI_LOG_LABEL_ENDL("INFO: Workspace size required: " << size
                                                               << " bytes");
      return ok(static_cast<size_t>(size));
    }
  }

  // Check if dynamic transient size function is present. Fusilli doesn't
  // support dynamic transient sizes yet.
  iree_string_view_t dynamicSizeAttr = iree_vm_function_lookup_attr_by_name(
      &mainFunc, IREE_SV("iree.abi.transients.size"));
  FUSILLI_RETURN_ERROR_IF(
      !iree_string_view_is_empty(dynamicSizeAttr), ErrorCode::NotImplemented,
      "Dynamic workspace sizes are not supported. The compiled module "
      "requires a data-dependent transient size that must be queried at "
      "runtime.");

  // No transient size attributes found, no workspace needed.
  // This is a catch-all for cases where the module was compiled without
  // the --iree-torch-externalize-transients flag (could be the case for
  // certain backends).
  FUSILLI_LOG_LABEL_ENDL("INFO: No workspace allocation required");
  return ok(static_cast<size_t>(0));
}

// Executes the graph using IREE runtime. Requires a `variantPack` which is a
// map from `TensorAttr` to `Buffer` wrapping the `iree_hal_buffer_view_t *`.
// The `workspace` parameter provides transient storage for intermediate values
// when required by the compiled module.
inline ErrorObject
Graph::execute(const Handle &handle,
               const std::unordered_map<std::shared_ptr<TensorAttr>,
                                        std::shared_ptr<Buffer>> &variantPack,
               const std::shared_ptr<Buffer> &workspace) const {
  FUSILLI_LOG_LABEL_ENDL("INFO: Executing Graph");
  FUSILLI_RETURN_ERROR_IF(vmContext_ == nullptr, ErrorCode::NotCompiled,
                          "Graph::execute requires a successful compile() first"
                          " (VM context not created)");
  FUSILLI_RETURN_ERROR_IF(!vmFunction_.has_value(), ErrorCode::NotCompiled,
                          "Graph::execute requires a successful compile() first"
                          " (VM function not resolved)");

  if (!kBackendExecuteAsync.contains(handle.getBackend())) // C++ 20
    return ErrorObject(ErrorCode::InternalError,
                       "Graph::execute got an unknown backend");
  bool executeAsync = kBackendExecuteAsync.at(handle.getBackend());

  iree_allocator_t allocator = iree_allocator_system();

  // Create input list. No output list needed since compiled functions write
  // results in-place to the buffer views passed as inputs (void return).
  iree_vm_list_t *rawInputList = nullptr;
  FUSILLI_CHECK_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          vmInputListCapacity_, allocator,
                                          &rawInputList));
  // The unique_ptr ensures the list is released on all exit paths
  // (success or error).
  IreeVmListUniquePtrType inputList(rawInputList);

  // Populate output buffers.
  for (const auto &output : fullGraphOutputsSorted_) {
    // Virtual tensors are internal to the function (intermediate outputs) and
    // aren't exposed in the call's signature (not part of the variantPack).
    if (output->isVirtual()) {
      FUSILLI_RETURN_ERROR_IF(variantPack.contains(output),
                              ErrorCode::VariantPackError,
                              "Virtual output tensor found in variantPack");
      continue;
    }
    FUSILLI_RETURN_ERROR_IF(!variantPack.contains(output), // C++20
                            ErrorCode::VariantPackError,
                            "Output tensor missing from variantPack");
    iree_vm_ref_t ref =
        iree_hal_buffer_view_retain_ref(*(variantPack.at(output)));
    FUSILLI_CHECK_ERROR(iree_vm_list_push_ref_move(inputList.get(), &ref));
  }

  // Populate input buffers.
  for (const auto &input : fullGraphInputsSorted_) {
    // Scalar constants are inlined in the function and aren't exposed
    // in the call's signature (not part of the variantPack).
    if (input->isScalar()) {
      FUSILLI_RETURN_ERROR_IF(variantPack.contains(input),
                              ErrorCode::VariantPackError,
                              "Scalar constant tensor found in variantPack");
      continue;
    }

    FUSILLI_RETURN_ERROR_IF(!variantPack.contains(input), // C++20
                            ErrorCode::VariantPackError,
                            "Input tensor missing from variantPack");
    iree_vm_ref_t ref =
        iree_hal_buffer_view_retain_ref(*(variantPack.at(input)));
    FUSILLI_CHECK_ERROR(iree_vm_list_push_ref_move(inputList.get(), &ref));
  }

  // Push workspace buffer. The --iree-torch-externalize-transients flag always
  // adds a !hal.buffer argument to the generated function signature, even when
  // no transient storage is needed (size = 0). We must always push a buffer
  // (or null ref when size = 0) to satisfy the function signature.
  if (workspaceSize_.value_or(0) > 0) {
    FUSILLI_RETURN_ERROR_IF(
        workspace == nullptr, ErrorCode::InvalidArgument,
        "Workspace buffer required but not provided (size=" +
            std::to_string(*workspaceSize_) + " bytes)");
    iree_hal_buffer_t *halBuffer = iree_hal_buffer_view_buffer(*workspace);
    FUSILLI_RETURN_ERROR_IF(
        iree_hal_buffer_byte_length(halBuffer) < *workspaceSize_,
        ErrorCode::InvalidArgument,
        "Workspace buffer too small: provided " +
            std::to_string(iree_hal_buffer_byte_length(halBuffer)) +
            " bytes, required " + std::to_string(*workspaceSize_) + " bytes");
    iree_vm_ref_t bufferRef = iree_hal_buffer_retain_ref(halBuffer);
    FUSILLI_CHECK_ERROR(
        iree_vm_list_push_ref_move(inputList.get(), &bufferRef));
  } else {
    // Size is 0 - no workspace needed. Accept (and ignore) a non-null
    // workspace buffer for caller convenience.
    if (workspace != nullptr)
      FUSILLI_LOG_LABEL_ENDL("WARNING: Workspace buffer provided but not "
                             "needed (size=0), ignoring");
    // Push a null ref to satisfy IREE function signature
    iree_vm_ref_t nullRef = iree_vm_ref_null();
    FUSILLI_CHECK_ERROR(iree_vm_list_push_ref_move(inputList.get(), &nullRef));
  }

  // In the asynchronous case, the IREE generated `@main$async` function
  // expects two additional `hal.fence` arguments. Since we rely on
  // stream-ordered synchronization, the fences may be dummy just to
  // align with the function signature without doing anything useful.
  if (executeAsync) {
    constexpr iree_host_size_t kDummyFenceCapacity = 0;
    // Create dummy wait fence (tells generated function that inputs are ready)
    // that's already completed.
    {
      iree_hal_fence_t *waitFence;
      FUSILLI_CHECK_ERROR(
          iree_hal_fence_create(kDummyFenceCapacity, allocator, &waitFence));

      iree_vm_ref_t waitFenceRef = iree_hal_fence_move_ref(waitFence);
      FUSILLI_CHECK_ERROR(
          iree_vm_list_push_ref_move(inputList.get(), &waitFenceRef));
    }
    // Create dummy signal fence (tells downstream consumers that kernel has
    // ran) that's already completed.
    {
      iree_hal_fence_t *signalFence;
      FUSILLI_CHECK_ERROR(
          iree_hal_fence_create(kDummyFenceCapacity, allocator, &signalFence));

      iree_vm_ref_t signalFenceRef = iree_hal_fence_move_ref(signalFence);
      FUSILLI_CHECK_ERROR(
          iree_vm_list_push_ref_move(inputList.get(), &signalFenceRef));
    }
  }

  // Invoke the function.
  FUSILLI_CHECK_ERROR(iree_vm_invoke(
      vmContext_.get(), *vmFunction_, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/nullptr, inputList.get(), /*outputs=*/nullptr, allocator));

  return ok();
}

// Factory: Allocates a new buffer view and takes ownership.
template <typename T>
inline ErrorOr<Buffer>
Buffer::allocate(const Handle &handle,
                 const std::vector<iree_hal_dim_t> &bufferShape,
                 const std::vector<T> &bufferData) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Allocating new device buffer");

  // Validate that bufferData size matches the product of bufferShape dimensions
  size_t expectedSize = 1;
  for (auto dim : bufferShape) {
    expectedSize *= dim;
  }
  FUSILLI_RETURN_ERROR_IF(
      expectedSize == 0 || bufferShape.empty(), ErrorCode::RuntimeFailure,
      "Buffer::allocate failed: cannot allocate a buffer with zero size");
  FUSILLI_RETURN_ERROR_IF(
      bufferData.size() != expectedSize, ErrorCode::RuntimeFailure,
      "Buffer::allocate failed: bufferData size (" +
          std::to_string(bufferData.size()) +
          ") does not match product of bufferShape dimensions (" +
          std::to_string(expectedSize) + ")");

  iree_hal_buffer_view_t *rawBufferView = nullptr;
  iree_hal_buffer_params_t bufferParams = {
      // Intended usage of this buffer (transfers, dispatches, etc):
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      // Access to allow to this memory:
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      // Where to allocate (host or device):
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  FUSILLI_CHECK_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      // IREE HAL device and allocator:
      handle.getDevice(), iree_hal_device_allocator(handle.getDevice()),
      // Shape rank and dimensions:
      bufferShape.size(), bufferShape.data(),
      // Element type:
      getIreeHalElementTypeForT<T>(),
      // Encoding type:
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, bufferParams,
      // The actual heap buffer to wrap or clone and its allocator:
      iree_make_const_byte_span(bufferData.data(),
                                bufferData.size() * sizeof(T)),
      // Buffer view + storage are returned and owned by the caller
      // (this Buffer object in this case):
      &rawBufferView));

  return ok(Buffer(IreeHalBufferViewUniquePtrType(rawBufferView)));
}

//===----------------------------------------------------------------------===//
//
// Buffer Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Factory: Imports an existing buffer view and retains ownership.
inline ErrorOr<Buffer>
Buffer::import(iree_hal_buffer_view_t *externalBufferView) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Importing pre-allocated device buffer");
  FUSILLI_RETURN_ERROR_IF(
      externalBufferView == nullptr, ErrorCode::RuntimeFailure,
      "Buffer::import failed as externalBufferView* is NULL");
  iree_hal_buffer_view_retain(externalBufferView);
  return ok(Buffer(IreeHalBufferViewUniquePtrType(externalBufferView)));
}

// Factory: Allocates a raw buffer for workspace/transient usage.
inline ErrorOr<Buffer> Buffer::allocateRaw(const Handle &handle,
                                           size_t sizeInBytes) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Allocating raw device buffer of size "
                         << sizeInBytes << " bytes");
  FUSILLI_RETURN_ERROR_IF(sizeInBytes == 0, ErrorCode::RuntimeFailure,
                          "Buffer::allocateRaw failed: cannot allocate "
                          "zero-size buffer");

  // Allocate raw buffer using IREE HAL allocator.
  iree_hal_buffer_t *rawBuffer = nullptr;
  iree_hal_buffer_params_t bufferParams = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  FUSILLI_CHECK_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(handle.getDevice()), bufferParams, sizeInBytes,
      &rawBuffer));

  // Wrap in buffer view for API compatibility (1D i8 shape).
  iree_hal_buffer_view_t *bufferView = nullptr;
  iree_hal_dim_t shape[] = {static_cast<iree_hal_dim_t>(sizeInBytes)};
  FUSILLI_CHECK_ERROR(iree_hal_buffer_view_create(
      rawBuffer, IREE_ARRAYSIZE(shape), shape, IREE_HAL_ELEMENT_TYPE_INT_8,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(),
      &bufferView));

  iree_hal_buffer_release(rawBuffer); // buffer view now owns it
  return ok(Buffer(IreeHalBufferViewUniquePtrType(bufferView)));
}

// Reads device buffer by initiating a device-to-host transfer and
// populating `outData`.
template <typename T>
inline ErrorObject Buffer::read(const Handle &handle, std::vector<T> &outData) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Reading device buffer through D2H transfer");
  FUSILLI_RETURN_ERROR_IF(outData.size() != 0, ErrorCode::RuntimeFailure,
                          "Buffer::read failed as outData is NOT empty");

  // Get the underlying buffer from the buffer view.
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(getBufferView());

  // Resize output vector `outData` based on buffer size.
  iree_device_size_t byteLength =
      iree_hal_buffer_view_byte_length(getBufferView());
  outData.resize(byteLength / sizeof(T));

  // Copy results back from device.
  FUSILLI_CHECK_ERROR(iree_hal_device_transfer_d2h(
      handle.getDevice(), buffer, 0, outData.data(), byteLength,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  return ok();
}

} // namespace fusilli

#endif // FUSILLI_BACKEND_RUNTIME_H
