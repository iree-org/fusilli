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

#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/support/logging.h"

#include <iree/hal/api.h>

#include <string>
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
ErrorObject registerHalDriversOnce();

//===----------------------------------------------------------------------===//
//
// Buffer Runtime API Methods
//
//===----------------------------------------------------------------------===//

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
