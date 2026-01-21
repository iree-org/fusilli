// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "pointwise_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<std::vector<int64_t>> &dims) {
  std::string name =
      std::format("pointwise_{}_dt{}", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (size_t i = 0; i < dims.size(); ++i) {
    name += std::format("_in{}", i);
    for (const auto &d : dims[i]) {
      name += std::format("_{}", d);
    }
  }
  return name;
};

TEST_CASE("Pointwise binary ops", "[pointwise][graph]") {
  const auto dims = std::vector<std::vector<int64_t>>{
      std::vector<int64_t>{2, 16, 64, 64},
      GENERATE(std::vector<int64_t>{2, 16, 64, 64},
               std::vector<int64_t>{1, 16, 1, 1})};

  const auto mode =
      GENERATE(PointwiseAttr::Mode::ADD, PointwiseAttr::Mode::DIV,
               PointwiseAttr::Mode::MUL, PointwiseAttr::Mode::SUB);

  auto executeAndVerify = [&]<typename T>(PointwiseBinaryGraphBuilder &builder,
                                          Handle &handle, DataType dt, T x0,
                                          T x1) {
    // Execute and get output buffer (output type same as input type)
    auto outBuf = builder.execute(handle, dt, x0, x1, dt, T(0));

    // Calculate reference value
    T y = 0;
    switch (builder.getMode()) {
    case PointwiseAttr::Mode::ADD: {
      y = x0 + x1;
      break;
    }
    case PointwiseAttr::Mode::DIV: {
      y = x0 / x1;
      break;
    }
    case PointwiseAttr::Mode::MUL: {
      y = x0 * x1;
      break;
    }
    case PointwiseAttr::Mode::SUB: {
      y = x0 - x1;
      break;
    }
    default:
      FAIL("Unsupported pointwise mode: "
           << PointwiseAttr::kModeToStr.at(builder.getMode()));
    }

    // Read output buffers and verify
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);
  };

  // Parameterize sample by backend and create device-specific handles.
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

  auto runTests = [&]<typename T>(DataType dt, T x0, T x1) {
    std::string name = generateName(mode, dt, dims);
    TensorAttr lhsTy = getTensorAttr(dt, dims[0]);
    TensorAttr rhsTy = getTensorAttr(dt, dims[1]);
    PointwiseBinaryGraphBuilder builder(name, dt, mode, lhsTy, rhsTy);
    builder.compile(handle);
    executeAndVerify(builder, handle, dt, x0, x1);
  };

  // int32
  runTests(DataType::Int32, int(-50), int(13));
  // fp16
  runTests(DataType::Half, half(-32.5f16), half(2.f16));
}
