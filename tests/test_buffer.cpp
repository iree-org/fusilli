// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <iree/hal/api.h>

#include <cstddef>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("Buffer allocation, move semantics and lifetime", "[buffer]") {
  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float).
  std::vector<float> data(6, 1.0f);
  FUSILLI_REQUIRE_ASSIGN(Buffer buf,
                         Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents.
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
  for (auto val : result)
    REQUIRE(val == 1.0f);

  // Test move semantics.
  Buffer movedBuf = std::move(buf);

  // Moved-to buffer is not NULL.
  // Moved-from buffer is NULL.
  REQUIRE(movedBuf != nullptr);
  REQUIRE(buf == nullptr);

  // Read moved buffer and check contents.
  result.clear();
  FUSILLI_REQUIRE_OK(movedBuf.read(handle, result));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer import and lifetimes", "[buffer]") {
  // Create handle for the target backend.
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Allocate a buffer of shape [2, 3] with all elements set to half(1.0f).
  std::vector<half> data(6, half(1.0f));
  FUSILLI_REQUIRE_ASSIGN(Buffer buf,
                         Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(1.0f));

  // Test import in local scope.
  {
    FUSILLI_REQUIRE_ASSIGN(Buffer importedBuf, Buffer::import(buf));
    // Both buffers co-exist and retain ownership (reference count tracked).
    REQUIRE(importedBuf != nullptr);
    REQUIRE(buf != nullptr);

    // Read imported buffer and check contents.
    result.clear();
    FUSILLI_REQUIRE_OK(importedBuf.read(handle, result));
    for (auto val : result)
      REQUIRE(val == half(1.0f));
  }

  // Initial buffer still exists in outer scope.
  REQUIRE(buf != nullptr);

  // Read original buffer and check contents.
  result.clear();
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer errors", "[buffer]") {
  SECTION("Import NULL buffer") {
    // Importing a NULL buffer view should fail.
    iree_hal_buffer_view_t *nullBuf = nullptr;
    ErrorObject status = Buffer::import(nullBuf);
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status.getMessage() ==
            "Buffer::import failed as externalBufferView* is NULL");
  }

  SECTION("Reading into a non-empty vector") {
    // Reading into a non-empty vector should fail.
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

    // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float).
    std::vector<float> data(6, 0.0f);
    FUSILLI_REQUIRE_ASSIGN(Buffer buf,
                           Buffer::allocate(handle, castToSizeT({2, 3}), data));

    // Read buffer into a non-empty vector.
    std::vector<float> result(6, 1.0f);
    REQUIRE(!result.empty());
    ErrorObject status = buf.read(handle, result);
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status.getMessage() ==
            "Buffer::read failed as outData is NOT empty");

    // Read buffer into an empty vector should work.
    result.clear();
    FUSILLI_REQUIRE_OK(buf.read(handle, result));
    for (auto val : result)
      REQUIRE(val == 0.0f);
  }

  SECTION("Buffer allocation with mismatched data size") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

    // Test case 1: bufferData has more elements than bufferShape expects.
    std::vector<float> tooMuchData(10, 1.0f);
    ErrorObject status1 =
        Buffer::allocate(handle, castToSizeT({2, 3}), tooMuchData);
    REQUIRE(isError(status1));
    REQUIRE(status1.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status1.getMessage() ==
            "Buffer::allocate failed: bufferData size (10) does not match "
            "product of bufferShape dimensions (6)");

    // Test case 2: bufferData has fewer elements than bufferShape expects.
    std::vector<float> tooLittleData(4, 1.0f);
    ErrorObject status2 =
        Buffer::allocate(handle, castToSizeT({2, 3}), tooLittleData);
    REQUIRE(isError(status2));
    REQUIRE(status2.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status2.getMessage() ==
            "Buffer::allocate failed: bufferData size (4) does not match "
            "product of bufferShape dimensions (6)");
  }

  SECTION("Buffer allocation with zero dimension") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

    // Test case 1: bufferShape with a zero dimension.
    std::vector<float> someData(5, 1.0f);
    ErrorObject status1 =
        Buffer::allocate(handle, castToSizeT({2, 0, 3}), someData);
    REQUIRE(isError(status1));
    REQUIRE(status1.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status1.getMessage() ==
            "Buffer::allocate failed: cannot allocate a buffer with zero size");

    // Test case 2: Empty bufferData and empty bufferShape.
    std::vector<float> noData;
    ErrorObject status2 = Buffer::allocate(handle, castToSizeT({}), noData);
    REQUIRE(isError(status2));
    REQUIRE(status2.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status2.getMessage() ==
            "Buffer::allocate failed: cannot allocate a buffer with zero size");
  }
}

TEST_CASE("Buffer::allocateRaw for workspace buffers", "[buffer]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Allocate a raw buffer of 1024 bytes.
  constexpr size_t bufferSize = 1024;
  FUSILLI_REQUIRE_ASSIGN(Buffer buf, Buffer::allocateRaw(handle, bufferSize));
  REQUIRE(buf != nullptr);

  // Verify the buffer view is valid and has the expected size.
  // allocateRaw creates a 1D i8 buffer view with shape [sizeInBytes].
  iree_hal_buffer_view_t *bufferView = buf;
  REQUIRE(bufferView != nullptr);
  REQUIRE(iree_hal_buffer_view_shape_rank(bufferView) == 1);
  REQUIRE(iree_hal_buffer_view_shape_dim(bufferView, 0) == bufferSize);
  REQUIRE(iree_hal_buffer_view_element_type(bufferView) ==
          IREE_HAL_ELEMENT_TYPE_INT_8);

  // Verify the underlying HAL buffer byte length matches the requested size.
  // Graph::execute relies on this property to validate workspace buffer sizes.
  iree_hal_buffer_t *halBuffer = iree_hal_buffer_view_buffer(bufferView);
  REQUIRE(halBuffer != nullptr);
  REQUIRE(iree_hal_buffer_byte_length(halBuffer) >= bufferSize);

  // Test move semantics.
  Buffer movedBuf = std::move(buf);
  REQUIRE(movedBuf != nullptr);
  REQUIRE(buf == nullptr);
}

TEST_CASE("Buffer::allocateRaw with various sizes", "[buffer]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Test small allocation (1 byte).
  FUSILLI_REQUIRE_ASSIGN(Buffer smallBuf, Buffer::allocateRaw(handle, 1));
  REQUIRE(smallBuf != nullptr);
  REQUIRE(iree_hal_buffer_view_shape_dim(smallBuf, 0) == 1);

  // Test larger allocation (1 MB).
  constexpr size_t oneMB = 1024lu * 1024lu;
  FUSILLI_REQUIRE_ASSIGN(Buffer largeBuf, Buffer::allocateRaw(handle, oneMB));
  REQUIRE(largeBuf != nullptr);
  REQUIRE(iree_hal_buffer_view_shape_dim(largeBuf, 0) == oneMB);
}

TEST_CASE("Buffer::allocateRaw errors", "[buffer]") {
  SECTION("Zero-size allocation") {
    FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

    // Allocating a zero-size buffer should fail.
    ErrorObject status = Buffer::allocateRaw(handle, 0);
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status.getMessage() ==
            "Buffer::allocateRaw failed: cannot allocate zero-size buffer");
  }
}
