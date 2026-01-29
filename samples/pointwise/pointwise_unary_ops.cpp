// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>
#include <utils.h>

#include "pointwise_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &dim) {
  std::string name =
      std::format("pointwise_{}_dt{}_in0", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (const auto &d : dim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Pointwise unary ops", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};

  const auto mode =
      GENERATE(PointwiseAttr::Mode::CEIL, PointwiseAttr::Mode::RELU_FWD,
               PointwiseAttr::Mode::SIGMOID_FWD, PointwiseAttr::Mode::TANH_FWD);

  auto executeAndVerify = [&]<typename T>(PointwiseUnaryGraphBuilder &builder,
                                          Handle &handle, DataType dt, T x) {
    // Execute and get output buffer (output type same as input type)
    auto outBuf = builder.execute(handle, dt, x, dt, T(0));

    // Calculate reference value
    T y = 0;
    switch (builder.getMode()) {
    case PointwiseAttr::Mode::RELU_FWD: {
      y = std::max(x, T(0));
      break;
    }
    case PointwiseAttr::Mode::SIGMOID_FWD: {
      double xD = static_cast<double>(x);
      y = T(1) / (T(1) + std::exp(-xD));
      break;
    }
    case PointwiseAttr::Mode::TANH_FWD: {
      double xD = static_cast<double>(x);
      y = std::tanh(xD);
      break;
    }
    case PointwiseAttr::Mode::CEIL: {
      double xD = static_cast<double>(x);
      y = std::ceil(xD);
      break;
    }
    default:
      FAIL("Unsupported pointwise mode: "
           << PointwiseAttr::kModeToStr.at(builder.getMode()));
    }

    // Read output buffers and verify
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));

    auto isClose = [](T lhs, T rhs) -> bool {
      if (std::is_floating_point<T>::value || std::is_same<T, half>::value) {
        return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs)) <
               1e-3;
      }
      return lhs == rhs;
    };

    for (auto val : result) {
      REQUIRE(isClose(val, y));
    }
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

  auto runTests = [&]<typename T>(DataType dt, T x) {
    std::string name = generateName(mode, dt, dim);
    TensorAttr inputTy = getTensorAttr(dt, dim);
    PointwiseUnaryGraphBuilder builder(name, dt, mode, inputTy);
    builder.compile(handle);
    executeAndVerify(builder, handle, dt, x);
  };

  // int32
  runTests(DataType::Int32, int(-128));
  // fp16
  runTests(DataType::Half, half(3.14));
}
