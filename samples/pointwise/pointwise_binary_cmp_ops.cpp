// Copyright 2026 Advanced Micro Devices, Inc.
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
                                const std::vector<int64_t> &lhsDims,
                                const std::vector<int64_t> &rhsDims) {
  std::string name =
      std::format("pointwise_{}_dt{}", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  name += std::format("_in0");
  for (const auto &d : lhsDims) {
    name += std::format("_{}", d);
  }
  name += std::format("_in1");
  for (const auto &d : rhsDims) {
    name += std::format("_{}", d);
  }
  return name;
};

void testPointwiseBinaryCmpOp(PointwiseAttr::Mode mode,
                              const std::vector<int64_t> &lhsDims,
                              const std::vector<int64_t> &rhsDims) {

  auto executeAndVerify = []<typename T>(PointwiseBinaryGraphBuilder &builder,
                                         Handle &handle, DataType dt, T x0,
                                         T x1) {
    // Execute and get output buffer
    auto outBuf = builder.execute(handle, dt, x0, x1, DataType::Boolean, false);

    // Calculate reference value
    bool y = 0;
    switch (builder.getMode()) {
    case PointwiseAttr::Mode::CMP_EQ: {
      y = (x0 == x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_LT: {
      y = (x0 < x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_LE: {
      y = (x0 <= x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_GT: {
      y = (x0 > x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_GE: {
      y = (x0 >= x1);
      break;
    }
    case PointwiseAttr::Mode::CMP_NEQ: {
      y = (x0 != x1);
      break;
    }
    default:
      FAIL("Unsupported pointwise mode: "
           << PointwiseAttr::kModeToStr.at(builder.getMode()));
    }

    // Read output buffers and verify
    std::vector<uint8_t> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
    for (auto val : result) {
      REQUIRE(val == y);
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
  // int32
  {
    std::string name = generateName(mode, DataType::Int32, lhsDims, rhsDims);
    TensorAttr lhsTy = getTensorAttr(DataType::Int32, lhsDims);
    TensorAttr rhsTy = getTensorAttr(DataType::Int32, rhsDims);
    PointwiseBinaryGraphBuilder builder(name, DataType::Int32, mode, lhsTy,
                                        rhsTy);
    builder.compile(handle);
    executeAndVerify(builder, handle, DataType::Int32, int(-50), int(-50));
    executeAndVerify(builder, handle, DataType::Int32, int(-50), int(-51));
    executeAndVerify(builder, handle, DataType::Int32, int(-51), int(-50));
    executeAndVerify(builder, handle, DataType::Int32, int(-51), int(-51));
  }
  // fp16
  {
    std::string name = generateName(mode, DataType::Half, lhsDims, rhsDims);
    TensorAttr lhsTy = getTensorAttr(DataType::Half, lhsDims);
    TensorAttr rhsTy = getTensorAttr(DataType::Half, rhsDims);
    PointwiseBinaryGraphBuilder builder(name, DataType::Half, mode, lhsTy,
                                        rhsTy);
    builder.compile(handle);
    executeAndVerify(builder, handle, DataType::Half, half(1.0), half(1.0));
    executeAndVerify(builder, handle, DataType::Half, half(1.0), half(1.1));
    executeAndVerify(builder, handle, DataType::Half, half(1.1), half(1.1));
    executeAndVerify(builder, handle, DataType::Half, half(1.1), half(1.0));
  }
}

TEST_CASE("Pointwise binary compare ops", "[pointwise][graph]") {
  std::vector<int64_t> lhsDims = {2, 16, 64, 64};
  std::vector<int64_t> rhsDims = GENERATE(std::vector<int64_t>{2, 16, 64, 64},
                                          std::vector<int64_t>{1, 16, 1, 1});

  const auto mode =
      GENERATE(PointwiseAttr::Mode::CMP_EQ, PointwiseAttr::Mode::CMP_LT,
               PointwiseAttr::Mode::CMP_LE, PointwiseAttr::Mode::CMP_GT,
               PointwiseAttr::Mode::CMP_GE, PointwiseAttr::Mode::CMP_NEQ);
  testPointwiseBinaryCmpOp(mode, lhsDims, rhsDims);
}
