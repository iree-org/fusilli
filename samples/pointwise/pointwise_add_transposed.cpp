// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "pointwise_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("Pointwise add with transposed operand", "[pointwise][graph]") {
  const int64_t n = 3, m = 2;

  // clang-format off
  const std::vector<float> inputData = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
  };

  // Result of inputData + transpose(inputData)
  const std::vector<float> expectedResult = {
    2.0f, 6.0f,
    5.0f, 9.0f,
    8.0f, 12.0f
  };
  // clang-format on

  // Parameterize sample by backend and create device-specific handles
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif

  Handle &handle = *handlePtr;

  // Tensor A: contiguous nxm tensor (row-major)
  TensorAttr aTy =
      TensorAttr().setName("input_a").setDim({n, m}).setStride({m, 1});

  // Tensor B: transposed nxm tensor
  // Logical dim={n, m}, but stored with transposed strides
  TensorAttr bTy = TensorAttr()
                       .setName("input_b_transposed")
                       .setDim({n, m})
                       .setStride({1, n});

  PointwiseBinaryGraphBuilder builder("pointwise_add_transposed",
                                      DataType::Float, PointwiseAttr::Mode::ADD,
                                      aTy, bTy);
  builder.compile(handle);

  // Allocate input buffers and initialize with input data
  auto aBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle, castToSizeT(builder.getLhsTensor()->getPhysicalDim()),
      inputData)));
  auto bBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle, castToSizeT(builder.getRhsTensor()->getPhysicalDim()),
      inputData)));

  // Allocate output buffer
  auto resultBuf = FUSILLI_REQUIRE_UNWRAP(allocateBufferOfType(
      handle, builder.getOutputTensor(), DataType::Float, 0.0f));

  // Create variant pack
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {builder.getLhsTensor(), aBuf},
          {builder.getRhsTensor(), bBuf},
          {builder.getOutputTensor(), resultBuf},
      };

  // Execute graph
  builder.execute(handle, variantPack);

  // Read output buffer and verify against expected result
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(resultBuf->read(handle, result));
  REQUIRE(result == expectedResult);
}
