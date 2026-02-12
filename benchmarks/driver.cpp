// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// For CLI11 Option Validators
const auto kIsNonNegativeInteger =
    CLI::Range(int64_t{0}, std::numeric_limits<int64_t>::max());
const auto kIsPositiveInteger =
    CLI::Range(int64_t{1}, std::numeric_limits<int64_t>::max());
// Note: for floating-point types, `std::numeric_limits<T>::min()` is the
// smallest *positive normalized* value. For integers, `min()` is the most
// negative value.
const auto kIsPositiveDouble = CLI::Range(std::numeric_limits<double>::min(),
                                          std::numeric_limits<double>::max());
const auto kIsValidLayout =
    CLI::IsMember({"NC", "NCH", "NHC", "NCHW", "NHWC", "NCDHW", "NDHWC"});
const auto kIsValidDataType = CLI::IsMember({"f32", "f16", "bf16"});

//===---------------------------------------------------------------------===//
// Option classes for organizing benchmark parameters
//===---------------------------------------------------------------------===//

struct ConvOptions {
  int64_t n, c, d, h, w, g, k, z, y, x;
  int64_t t, u, v, o, p, q, m, l, j, s;
  int64_t mode;
  std::string imageLayout, filterLayout, outputLayout;
  bool fp16{false};
  bool bf16{false};
  bool bias{false};
};

struct LayerNormOptions {
  std::string input;
  std::string type;
  std::string layout;
  int64_t forw;
  int64_t mode;
  float eps = 1e-5f;
};

struct MatmulOptions {
  int64_t m, n, k, b;
  std::string a_type;
  std::string b_type;
  std::string out_type;
  std::string bias_type;
  bool transA{false};
  bool transB{false};
  bool bias{false};
};

//===---------------------------------------------------------------------===//
// Helpers
//===---------------------------------------------------------------------===//

static std::pair<std::vector<int64_t>, std::vector<int64_t>>
getBiasDimsAndStride(int64_t spatialDim, int64_t k) {
  auto biasDims = (spatialDim == 2) ? std::vector<int64_t>{1, k, 1, 1}
                                    : std::vector<int64_t>{1, k, 1, 1, 1};
  auto biasStride = std::vector<int64_t>(biasDims.size(), 1);
  return {std::move(biasDims), std::move(biasStride)};
}

static std::vector<int64_t>
parseDimensionsFromString(const std::string &dimStr) {
  std::vector<int64_t> dims;
  size_t start = 0, end;
  while ((end = dimStr.find('x', start)) != std::string::npos) {
    dims.push_back(std::stoll(dimStr.substr(start, end - start)));
    start = end + 1;
  }
  if (start < dimStr.size())
    dims.push_back(std::stoll(dimStr.substr(start)));
  return dims;
}

//===---------------------------------------------------------------------===//
// Benchmark functions
//===---------------------------------------------------------------------===//

static ErrorObject benchmarkConvFprop(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#if defined(FUSILLI_ENABLE_AMDGPU)
  FUSILLI_ASSIGN_OR_RETURN(Handle handle,
                           Handle::create(Backend::AMDGPU, deviceId));
#else
  FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = opts.c / opts.g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims =
      (opts.s == 2)
          ? std::vector<int64_t>{opts.n, opts.c, opts.h, opts.w}
          : std::vector<int64_t>{opts.n, opts.c, opts.d, opts.h, opts.w};
  auto wDims = (opts.s == 2)
                   ? std::vector<int64_t>{opts.k, fc, opts.y, opts.x}
                   : std::vector<int64_t>{opts.k, fc, opts.z, opts.y, opts.x};
  FUSILLI_ASSIGN_OR_RETURN(auto xStride,
                           generateStrideFromLayout(xDims, opts.imageLayout));
  FUSILLI_ASSIGN_OR_RETURN(auto wStride,
                           generateStrideFromLayout(wDims, opts.filterLayout));
  auto convStride = (opts.s == 2)
                        ? std::vector<int64_t>{opts.u, opts.v}
                        : std::vector<int64_t>{opts.t, opts.u, opts.v};
  auto convPadding = (opts.s == 2)
                         ? std::vector<int64_t>{opts.p, opts.q}
                         : std::vector<int64_t>{opts.o, opts.p, opts.q};
  auto convDilation = (opts.s == 2)
                          ? std::vector<int64_t>{opts.l, opts.j}
                          : std::vector<int64_t>{opts.m, opts.l, opts.j};

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations of the benchmark driver
  // from polluting the same cache files leading to race conditions.
  auto graphName = std::format("benchmark_conv_fprop_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                               "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                               "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}_bias{}",
                               opts.n, opts.c, opts.d, opts.h, opts.w, opts.g,
                               opts.k, opts.z, opts.y, opts.x, opts.t, opts.u,
                               opts.v, opts.o, opts.p, opts.q, opts.m, opts.l,
                               opts.j, opts.s, opts.imageLayout,
                               opts.outputLayout, opts.filterLayout, opts.bias);
  graph.setName(graphName);

  // Types on the graph are kept at fp32 but we explicitly set
  // individual tensor types below based on configuration. These
  // types hence don't matter much and are used only to infer
  // missing type annotations on tensors.
  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto xT = graph.tensor(TensorAttr()
                             .setName("input")
                             .setDim(xDims)
                             .setStride(xStride)
                             .setDataType(convIOType));

  auto wT = graph.tensor(TensorAttr()
                             .setName("filter")
                             .setDim(wDims)
                             .setStride(wStride)
                             .setDataType(convIOType));

  auto convAttr = ConvFPropAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_fprop");

  auto yT = graph.convFProp(xT, wT, convAttr);
  yT->setDataType(convIOType);

  std::shared_ptr<TensorAttr> bT;
  if (opts.bias) {
    auto [biasDims, biasStride] = getBiasDimsAndStride(opts.s, opts.k);
    bT = graph.tensor(TensorAttr()
                          .setName("bias")
                          .setDim(biasDims)
                          .setStride(biasStride)
                          .setDataType(convIOType));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    yT = graph.pointwise(yT, bT, biasAttr);
    yT->setDataType(convIOType);
  }
  yT->setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate input, weight and output buffers.
  FUSILLI_ASSIGN_OR_RETURN(auto xBuf,
                           allocateBufferOfType(handle, xT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto wBuf,
                           allocateBufferOfType(handle, wT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto yBuf,
                           allocateBufferOfType(handle, yT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {wT, wBuf},
          {yT, yBuf},
      };

  if (opts.bias) {
    FUSILLI_ASSIGN_OR_RETURN(
        auto bBuf, allocateBufferOfType(handle, bT, convIOType, 1.0f));
    variantPack.insert({bT, bBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < iter; ++i)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkConvWGrad(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#if defined(FUSILLI_ENABLE_AMDGPU)
  FUSILLI_ASSIGN_OR_RETURN(Handle handle,
                           Handle::create(Backend::AMDGPU, deviceId));
#else
  FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = opts.c / opts.g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims =
      (opts.s == 2)
          ? std::vector<int64_t>{opts.n, opts.c, opts.h, opts.w}
          : std::vector<int64_t>{opts.n, opts.c, opts.d, opts.h, opts.w};
  auto wDims = (opts.s == 2)
                   ? std::vector<int64_t>{opts.k, fc, opts.y, opts.x}
                   : std::vector<int64_t>{opts.k, fc, opts.z, opts.y, opts.x};
  auto convStride = (opts.s == 2)
                        ? std::vector<int64_t>{opts.u, opts.v}
                        : std::vector<int64_t>{opts.t, opts.u, opts.v};
  auto convPadding = (opts.s == 2)
                         ? std::vector<int64_t>{opts.p, opts.q}
                         : std::vector<int64_t>{opts.o, opts.p, opts.q};
  auto convDilation = (opts.s == 2)
                          ? std::vector<int64_t>{opts.l, opts.j}
                          : std::vector<int64_t>{opts.m, opts.l, opts.j};

  // Calculate output dimensions (DY shape) using the same inference as forward
  auto dyDims = getConvInferredOutputShape(xDims, wDims, convDilation,
                                           convPadding, convStride);

  FUSILLI_ASSIGN_OR_RETURN(auto xStride,
                           generateStrideFromLayout(xDims, opts.imageLayout));
  FUSILLI_ASSIGN_OR_RETURN(auto dyStride,
                           generateStrideFromLayout(dyDims, opts.outputLayout));
  FUSILLI_ASSIGN_OR_RETURN(auto wStride,
                           generateStrideFromLayout(wDims, opts.filterLayout));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName = std::format("benchmark_conv_wgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                               "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                               "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}_bias{}",
                               opts.n, opts.c, opts.d, opts.h, opts.w, opts.g,
                               opts.k, opts.z, opts.y, opts.x, opts.t, opts.u,
                               opts.v, opts.o, opts.p, opts.q, opts.m, opts.l,
                               opts.j, opts.s, opts.imageLayout,
                               opts.outputLayout, opts.filterLayout, opts.bias);
  graph.setName(graphName);

  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto dyT = graph.tensor(
      TensorAttr().setName("dy").setDim(dyDims).setStride(dyStride).setDataType(
          convIOType));

  auto xT = graph.tensor(TensorAttr()
                             .setName("input")
                             .setDim(xDims)
                             .setStride(xStride)
                             .setDataType(convIOType));

  std::shared_ptr<TensorAttr> dbT;
  if (opts.bias) {
    auto [biasDims, biasStride] = getBiasDimsAndStride(opts.s, opts.k);
    auto reductionAttr = ReductionAttr()
                             .setMode(ReductionAttr::Mode::SUM)
                             .setName("bias_reduction");
    dbT = graph.reduction(dyT, reductionAttr);
    dbT->setDim(biasDims)
        .setStride(biasStride)
        .setOutput(true)
        .setDataType(convIOType);
  }

  auto convAttr = ConvWGradAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_wgrad");

  auto dwT = graph.convWGrad(dyT, xT, convAttr);
  dwT->setDim(wDims).setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate buffers.
  FUSILLI_ASSIGN_OR_RETURN(auto dyBuf,
                           allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto xBuf,
                           allocateBufferOfType(handle, xT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto dwBuf,
                           allocateBufferOfType(handle, dwT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dwT, dwBuf},
      };

  if (opts.bias) {
    FUSILLI_ASSIGN_OR_RETURN(
        auto dbBuf, allocateBufferOfType(handle, dbT, convIOType, 0.0f));
    variantPack.insert({dbT, dbBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < iter; ++i)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkConvDGrad(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#if defined(FUSILLI_ENABLE_AMDGPU)
  FUSILLI_ASSIGN_OR_RETURN(Handle handle,
                           Handle::create(Backend::AMDGPU, deviceId));
#else
  FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = opts.c / opts.g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims =
      (opts.s == 2)
          ? std::vector<int64_t>{opts.n, opts.c, opts.h, opts.w}
          : std::vector<int64_t>{opts.n, opts.c, opts.d, opts.h, opts.w};
  auto wDims = (opts.s == 2)
                   ? std::vector<int64_t>{opts.k, fc, opts.y, opts.x}
                   : std::vector<int64_t>{opts.k, fc, opts.z, opts.y, opts.x};
  auto convStride = (opts.s == 2)
                        ? std::vector<int64_t>{opts.u, opts.v}
                        : std::vector<int64_t>{opts.t, opts.u, opts.v};
  auto convPadding = (opts.s == 2)
                         ? std::vector<int64_t>{opts.p, opts.q}
                         : std::vector<int64_t>{opts.o, opts.p, opts.q};
  auto convDilation = (opts.s == 2)
                          ? std::vector<int64_t>{opts.l, opts.j}
                          : std::vector<int64_t>{opts.m, opts.l, opts.j};

  // Calculate output dimensions (DY shape) using the same inference as forward
  auto dyDims = getConvInferredOutputShape(xDims, wDims, convDilation,
                                           convPadding, convStride);

  FUSILLI_ASSIGN_OR_RETURN(auto xStride,
                           generateStrideFromLayout(xDims, opts.imageLayout));
  FUSILLI_ASSIGN_OR_RETURN(auto dyStride,
                           generateStrideFromLayout(dyDims, opts.outputLayout));
  FUSILLI_ASSIGN_OR_RETURN(auto wStride,
                           generateStrideFromLayout(wDims, opts.filterLayout));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName = std::format("benchmark_conv_dgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                               "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                               "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}_bias{}",
                               opts.n, opts.c, opts.d, opts.h, opts.w, opts.g,
                               opts.k, opts.z, opts.y, opts.x, opts.t, opts.u,
                               opts.v, opts.o, opts.p, opts.q, opts.m, opts.l,
                               opts.j, opts.s, opts.imageLayout,
                               opts.outputLayout, opts.filterLayout, opts.bias);
  graph.setName(graphName);

  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto dyT = graph.tensor(
      TensorAttr().setName("dy").setDim(dyDims).setStride(dyStride).setDataType(
          convIOType));

  auto wT = graph.tensor(TensorAttr()
                             .setName("filter")
                             .setDim(wDims)
                             .setStride(wStride)
                             .setDataType(convIOType));

  std::shared_ptr<TensorAttr> dbT;
  if (opts.bias) {
    auto [biasDims, biasStride] = getBiasDimsAndStride(opts.s, opts.k);
    auto reductionAttr = ReductionAttr()
                             .setMode(ReductionAttr::Mode::SUM)
                             .setName("bias_reduction");
    dbT = graph.reduction(dyT, reductionAttr);
    dbT->setDim(biasDims)
        .setStride(biasStride)
        .setOutput(true)
        .setDataType(convIOType);
  }

  auto convAttr = ConvDGradAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_dgrad");

  auto dxT = graph.convDGrad(dyT, wT, convAttr);
  dxT->setDim(xDims).setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate buffers.
  FUSILLI_ASSIGN_OR_RETURN(auto dyBuf,
                           allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto wBuf,
                           allocateBufferOfType(handle, wT, convIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto dxBuf,
                           allocateBufferOfType(handle, dxT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dxT, dxBuf},
      };

  if (opts.bias) {
    FUSILLI_ASSIGN_OR_RETURN(
        auto dbBuf, allocateBufferOfType(handle, dbT, convIOType, 0.0f));
    variantPack.insert({dbT, dbBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < iter; ++i)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkLayerNormFwd(const LayerNormOptions &opts,
                                         const std::vector<int64_t> &dims,
                                         DataType layernormIOType, int64_t iter,
                                         int64_t deviceId, bool dump) {
#if defined(FUSILLI_ENABLE_AMDGPU)
  FUSILLI_ASSIGN_OR_RETURN(Handle handle,
                           Handle::create(Backend::AMDGPU, deviceId));
#else
  FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif

  constexpr NormFwdPhase phase = NormFwdPhase::TRAINING;
  const bool withScaleBias = opts.mode == 1;

  auto xDims = dims;
  FUSILLI_ASSIGN_OR_RETURN(auto xStride,
                           generateStrideFromLayout(xDims, opts.layout));

  Graph graph;
  auto graphName = std::format(
      "benchmark_layernorm_input{}_forw{}_layout{}_type{}_mode{}_eps{}",
      opts.input, opts.forw, opts.layout, opts.type, opts.mode, opts.eps);
  graph.setName(graphName);

  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto xT = graph.tensor(
      TensorAttr().setName("x").setDim(xDims).setStride(xStride).setDataType(
          layernormIOType));

  // Shape and strides will be inferred later in inferPropertiesNode()
  std::shared_ptr<TensorAttr> sT = nullptr, bT = nullptr;
  if (withScaleBias) {
    sT = graph.tensor(
        TensorAttr().setName("scale").setDataType(layernormIOType));
    bT =
        graph.tensor(TensorAttr().setName("bias").setDataType(layernormIOType));
  }

  auto epsilonT = graph.tensor(TensorAttr(opts.eps));

  auto layernormAttr =
      LayernormAttr().setForwardPhase(phase).setEpsilon(epsilonT).setName(
          "layernorm_fwd");

  auto [yT, mT, vT] = graph.layernorm(xT, sT, bT, layernormAttr);
  yT->setName("y").setDataType(layernormIOType).setOutput(true);
  mT->setName("mean").setDataType(layernormIOType).setOutput(true);
  vT->setName("inv_variance").setDataType(layernormIOType).setOutput(true);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate input and output buffers.
  FUSILLI_ASSIGN_OR_RETURN(
      auto xBuf, allocateBufferOfType(handle, xT, layernormIOType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(
      auto yBuf, allocateBufferOfType(handle, yT, layernormIOType, 0.0f));
  FUSILLI_ASSIGN_OR_RETURN(
      auto mBuf, allocateBufferOfType(handle, mT, layernormIOType, 0.0f));
  FUSILLI_ASSIGN_OR_RETURN(
      auto vBuf, allocateBufferOfType(handle, vT, layernormIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{xT, xBuf}, {yT, yBuf}, {mT, mBuf}, {vT, vBuf}};

  if (withScaleBias) {
    FUSILLI_ASSIGN_OR_RETURN(
        auto sBuf, allocateBufferOfType(handle, sT, layernormIOType, 1.0f));
    variantPack.insert({sT, sBuf});

    FUSILLI_ASSIGN_OR_RETURN(
        auto bBuf, allocateBufferOfType(handle, bT, layernormIOType, 1.0f));
    variantPack.insert({bT, bBuf});
  }
  // Execute graph `iter` times.
  for (size_t i = 0; i < iter; ++i)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkMatmul(const MatmulOptions &opts, DataType aType,
                                   DataType bType, DataType outType,
                                   DataType biasType, int64_t iter,
                                   int64_t deviceId, bool dump) {
#if defined(FUSILLI_ENABLE_AMDGPU)
  FUSILLI_ASSIGN_OR_RETURN(Handle handle,
                           Handle::create(Backend::AMDGPU, deviceId));
#else
  FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif

  // Build attributes based on transpose flags and batch count.
  auto aDims = (opts.b > 1) ? std::vector<int64_t>{opts.b, opts.m, opts.k}
                            : std::vector<int64_t>{opts.m, opts.k};
  auto bDims = (opts.b > 1) ? std::vector<int64_t>{opts.b, opts.k, opts.n}
                            : std::vector<int64_t>{opts.k, opts.n};

  std::vector<int64_t> aStride, bStride;
  if (opts.b > 1) {
    // Batched matmul strides
    aStride = opts.transA ? std::vector<int64_t>{opts.m * opts.k, 1, opts.m}
                          : std::vector<int64_t>{opts.m * opts.k, opts.k, 1};
    bStride = opts.transB ? std::vector<int64_t>{opts.k * opts.n, 1, opts.k}
                          : std::vector<int64_t>{opts.k * opts.n, opts.n, 1};
  } else {
    // Non-batched matmul strides
    aStride = opts.transA ? std::vector<int64_t>{1, opts.m}
                          : std::vector<int64_t>{opts.k, 1};
    bStride = opts.transB ? std::vector<int64_t>{1, opts.k}
                          : std::vector<int64_t>{opts.n, 1};
  }

  Graph graph;
  auto graphName = std::format(
      "benchmark_matmul_b{}_m{}_n{}_k{}_transA{}_transB{}_bias{}_atype{}_"
      "btype{}_outtype{}",
      opts.b, opts.m, opts.n, opts.k, opts.transA, opts.transB, opts.bias,
      kDataTypeToMlirTypeAsm.at(aType), kDataTypeToMlirTypeAsm.at(bType),
      kDataTypeToMlirTypeAsm.at(outType));
  if (opts.bias) {
    graphName +=
        std::format("_biastype{}", kDataTypeToMlirTypeAsm.at(biasType));
  }
  graph.setName(graphName);

  // Types on the graph are kept at fp32 but we explicitly set
  // individual tensor types below based on configuration. These
  // types hence don't matter much and are used only to infer
  // missing type annotations on tensors.
  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto aT = graph.tensor(TensorAttr()
                             .setName("matrix_a")
                             .setDim(aDims)
                             .setStride(aStride)
                             .setDataType(aType));

  auto bT = graph.tensor(TensorAttr()
                             .setName("matrix_b")
                             .setDim(bDims)
                             .setStride(bStride)
                             .setDataType(bType));

  auto matmulAttr = MatmulAttr().setName("matmul");

  auto outT = graph.matmul(aT, bT, matmulAttr);
  // Use `biasType` as the intermediate type when bias is present,
  // and `outType` otherwise.
  outT->setDataType(opts.bias ? biasType : outType);

  std::shared_ptr<TensorAttr> biasT;
  if (opts.bias) {
    auto biasDims = (opts.b > 1) ? std::vector<int64_t>{1, 1, opts.n}
                                 : std::vector<int64_t>{1, opts.n};
    auto biasStride = generateStrideFromDim(
        biasDims, getContiguousStrideOrder(biasDims.size()));

    biasT = graph.tensor(TensorAttr()
                             .setName("bias")
                             .setDim(biasDims)
                             .setStride(biasStride)
                             .setDataType(biasType));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    outT = graph.pointwise(outT, biasT, biasAttr);
    outT->setDataType(outType);
  }
  outT->setOutput(true).setDataType(outType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate input, weight and output buffers.
  FUSILLI_ASSIGN_OR_RETURN(auto aBuf,
                           allocateBufferOfType(handle, aT, aType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto bBuf,
                           allocateBufferOfType(handle, bT, bType, 1.0f));
  FUSILLI_ASSIGN_OR_RETURN(auto outBuf,
                           allocateBufferOfType(handle, outT, outType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {outT, outBuf},
      };

  if (opts.bias) {
    FUSILLI_ASSIGN_OR_RETURN(
        auto biasBuf, allocateBufferOfType(handle, biasT, biasType, 1.0f));
    variantPack.insert({biasT, biasBuf});
  }

  // Execute graph `iter` times.
  for (size_t i = 0; i < iter; ++i)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

//===---------------------------------------------------------------------===//
// CLI registration functions
//===---------------------------------------------------------------------===//

static CLI::App *registerConvOptions(CLI::App &mainApp, ConvOptions &convOpts) {
  // Conv flags are kept in sync with MIOpen's ConvDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/conv_driver.hpp#L878
  CLI::App *convApp =
      mainApp.add_subcommand("conv", "Fusilli Benchmark Convolution");

  // convApp CLI Options - bind to ConvOptions members
  convApp
      ->add_option("--mode,-F", convOpts.mode,
                   "Conv mode: 1=forward, 2=data_grad, 4=weight_grad")
      ->required()
      ->check(CLI::IsMember({1, 2, 4}));
  convApp->add_option("--batchsize,-n", convOpts.n, "Input batch size")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_channels,-c", convOpts.c, "Input channels")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_d", convOpts.d, "Input depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_h,-H", convOpts.h, "Input height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_w,-W", convOpts.w, "Input width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--group_count,-g", convOpts.g, "Number of groups")
      ->default_val("1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--out_channels,-k", convOpts.k, "Output channels")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_d", convOpts.z, "Filter depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_h,-y", convOpts.y, "Filter height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_w,-x", convOpts.x, "Filter width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_d", convOpts.t, "Conv stride depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_h,-u", convOpts.u, "Conv stride height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_w,-v", convOpts.v, "Conv stride width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--pad_d", convOpts.o, "Conv padding depth")
      ->default_val("-1")
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--pad_h,-p", convOpts.p, "Conv padding height")
      ->required()
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--pad_w,-q", convOpts.q, "Conv padding width")
      ->required()
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--dilation_d", convOpts.m, "Conv dilation depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--dilation_h,-l", convOpts.l, "Conv dilation height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--dilation_w,-j", convOpts.j, "Conv dilation width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_layout", convOpts.imageLayout, "Input layout")
      ->required()
      ->check(kIsValidLayout);
  convApp->add_option("--fil_layout", convOpts.filterLayout, "Filter layout")
      ->required()
      ->check(kIsValidLayout);
  convApp->add_option("--out_layout", convOpts.outputLayout, "Output layout")
      ->required()
      ->check(kIsValidLayout);
  convApp
      ->add_option("--spatial_dim", convOpts.s,
                   "Number of spatial dimensions (2 for conv2d, 3 for conv3d)")
      ->required()
      ->check(CLI::IsMember({2, 3}));

  // convApp CLI Flags:
  auto *f1 = convApp->add_flag("--fp16", convOpts.fp16, "Run fp16 convolution");
  auto *f2 = convApp->add_flag("--bf16", convOpts.bf16, "Run bf16 convolution");
  // Can't specify both flags.
  f1->excludes(f2);
  convApp->add_flag("--bias,-b", convOpts.bias, "Run with bias");

  return convApp;
}

// Register LayerNorm options to CLI app
static CLI::App *registerLayerNormOptions(CLI::App &mainApp,
                                          LayerNormOptions &layerNormOpts) {
  // LayerNorm flags are kept in sync with MIOpen's LayerNormDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/layernorm_driver.hpp#L386
  CLI::App *layerNormApp = mainApp.add_subcommand(
      "layernorm", "Fusilli Benchmark Layer Normalization");

  // layerNormApp CLI Options - bind to LayerNormOptions members
  layerNormApp
      ->add_option(
          "--input,-X", layerNormOpts.input,
          "Input tensor dimensions. The shapes with rank 1 to 5 are supported.")
      ->required();
  layerNormApp
      ->add_option("--type,-t", layerNormOpts.type,
                   "LayerNorm data type (f32, f16, bf16)")
      ->required()
      ->check(kIsValidDataType);
  layerNormApp
      ->add_option(
          "--forw,-F", layerNormOpts.forw,
          "Kind of kernel to run: 1 - forward (training mode). Currently, "
          "only one kernel type is supported.")
      ->required()
      ->check(CLI::IsMember({1}));
  layerNormApp
      ->add_option("--mode,-m", layerNormOpts.mode,
                   "Elementwise affine mode (0), weight and bias mode (1). By "
                   "default, 0 is used.")
      ->default_val(0)
      ->check(CLI::IsMember({0, 1}));
  layerNormApp
      ->add_option("--layout,-l", layerNormOpts.layout, "Input/Output layout")
      ->required()
      ->check(kIsValidLayout);
  layerNormApp
      ->add_option("--eps,-e", layerNormOpts.eps, "Epsilon, 1e-5 by default")
      ->default_val(1e-5f)
      ->check(kIsPositiveDouble);

  return layerNormApp;
}

// Register matmul options to CLI app
static CLI::App *registerMatmulOptions(CLI::App &mainApp,
                                       MatmulOptions &matmulOpts) {
  CLI::App *matmulApp = mainApp.add_subcommand(
      "matmul", "Fusilli Benchmark Matrix Multiplication");

  // matmulApp CLI Options - bind to MatmulOptions members
  matmulApp->add_option("--m,-M", matmulOpts.m, "Matrix M dimension")
      ->required()
      ->check(kIsPositiveInteger);
  matmulApp->add_option("--n,-N", matmulOpts.n, "Matrix N dimension")
      ->required()
      ->check(kIsPositiveInteger);
  matmulApp->add_option("--k,-K", matmulOpts.k, "Matrix K dimension")
      ->required()
      ->check(kIsPositiveInteger);
  matmulApp
      ->add_option("--b,-B", matmulOpts.b, "Batch dimension for batched matmul")
      ->default_val("1")
      ->check(kIsPositiveInteger);
  matmulApp
      ->add_option("--a_type", matmulOpts.a_type,
                   "Matrix A data type (f32, f16, bf16)")
      ->required()
      ->check(kIsValidDataType);
  matmulApp
      ->add_option("--b_type", matmulOpts.b_type,
                   "Matrix B data type (f32, f16, bf16)")
      ->required()
      ->check(kIsValidDataType);
  matmulApp
      ->add_option("--out_type", matmulOpts.out_type,
                   "Result data type (f32, f16, bf16)")
      ->required()
      ->check(kIsValidDataType);
  matmulApp
      ->add_option("--bias_type", matmulOpts.bias_type,
                   "Bias data type (f32, f16, bf16)")
      ->check(kIsValidDataType);

  // matmulApp CLI Flags:
  matmulApp->add_flag("--transA", matmulOpts.transA, "Transpose matrix A");
  matmulApp->add_flag("--transB", matmulOpts.transB, "Transpose matrix B");
  matmulApp->add_flag(
      "--bias", matmulOpts.bias,
      "Add bias vector to result (after broadcasting to result shape)");

  return matmulApp;
}

// Validate and run convolution benchmark
static ErrorObject runConvBenchmark(const ConvOptions &convOpts, int64_t iter,
                                    int64_t deviceId, bool dump) {
  // Additional validation of convApp options (apart from default CLI checks)
  if (convOpts.s == 2) {
    // Reject 3D layouts for 2D conv
    FUSILLI_RETURN_ERROR_IF(
        convOpts.imageLayout.size() != 4 || convOpts.filterLayout.size() != 4 ||
            convOpts.outputLayout.size() != 4,
        ErrorCode::InvalidArgument,
        "Detected at least one invalid {input, filter, output} "
        "layout for 2D convolution.");
  }
  if (convOpts.s == 3) {
    // Reject 2D layouts for 3D conv
    FUSILLI_RETURN_ERROR_IF(
        convOpts.imageLayout.size() != 5 || convOpts.filterLayout.size() != 5 ||
            convOpts.outputLayout.size() != 5,
        ErrorCode::InvalidArgument,
        "Detected at least one invalid {input, filter, output} "
        "layout for 3D convolution.");
    // Reject default (sentinel) values for optional args in 3D conv
    FUSILLI_RETURN_ERROR_IF(
        convOpts.d == -1 || convOpts.z == -1 || convOpts.t == -1 ||
            convOpts.o == -1 || convOpts.m == -1,
        ErrorCode::InvalidArgument,
        "Detected at least one of {in_d, fil_d, conv_stride_d, "
        "pad_d, dilation_d} that was not set for 3D convolution.");
  }

  // Validation of group count
  FUSILLI_RETURN_ERROR_IF(
      convOpts.c % convOpts.g != 0 || convOpts.k % convOpts.g != 0,
      ErrorCode::InvalidArgument, "Detected invalid group count.");

  DataType convIOType;
  if (convOpts.fp16)
    convIOType = DataType::Half;
  else if (convOpts.bf16)
    convIOType = DataType::BFloat16;
  else
    // When unspecified, default to fp32 conv.
    convIOType = DataType::Float;

  ErrorObject status = ok();
  if (convOpts.mode == 1) {
    // Forward convolution
    status = benchmarkConvFprop(convOpts, convIOType, iter, deviceId, dump);
  } else if (convOpts.mode == 2) {
    // Data gradient
    status = benchmarkConvDGrad(convOpts, convIOType, iter, deviceId, dump);
  } else if (convOpts.mode == 4) {
    // Weight gradient
    status = benchmarkConvWGrad(convOpts, convIOType, iter, deviceId, dump);
  }

  FUSILLI_CHECK_ERROR(status);

  return ok();
}

// Validate and run layernorm benchmark
static ErrorObject runLayerNormBenchmark(const LayerNormOptions &layerNormOpts,
                                         int64_t iter, int64_t deviceId,
                                         bool dump) {
  // Parse dimensions string into vector
  auto dims = parseDimensionsFromString(layerNormOpts.input);
  FUSILLI_RETURN_ERROR_IF(std::any_of(dims.begin(), dims.end(),
                                      [](int64_t dim) { return dim <= 0; }),
                          ErrorCode::InvalidArgument,
                          "Invalid input dimensions: they must be positive");
  FUSILLI_RETURN_ERROR_IF(dims.size() < 2 || dims.size() > 5,
                          ErrorCode::InvalidArgument,
                          "Input dimensions must have rank between 2 and 5");

  // Validate layout
  FUSILLI_RETURN_ERROR_IF(
      dims.size() != layerNormOpts.layout.size(), ErrorCode::InvalidArgument,
      "Input dimensions and layout must have the same rank");

  // Parse data type strings using direct map lookup
  DataType type = kMlirTypeAsmToDataType.at(layerNormOpts.type);

  ErrorObject status =
      benchmarkLayerNormFwd(layerNormOpts, dims, type, iter, deviceId, dump);

  FUSILLI_CHECK_ERROR(status);

  return ok();
}

// Validate and run matmul benchmark
static ErrorObject runMatmulBenchmark(const MatmulOptions &matmulOpts,
                                      int64_t iter, int64_t deviceId,
                                      bool dump) {
  // Validate that bias_type is set when --bias is used
  FUSILLI_RETURN_ERROR_IF(
      matmulOpts.bias && matmulOpts.bias_type.empty(),
      ErrorCode::InvalidArgument,
      "bias_type must be specified when --bias flag is set");

  // Parse data type strings using direct map lookup
  DataType aType = kMlirTypeAsmToDataType.at(matmulOpts.a_type);
  DataType bType = kMlirTypeAsmToDataType.at(matmulOpts.b_type);
  DataType outType = kMlirTypeAsmToDataType.at(matmulOpts.out_type);
  DataType biasType = DataType::NotSet;
  if (matmulOpts.bias) {
    biasType = kMlirTypeAsmToDataType.at(matmulOpts.bias_type);
  }

  ErrorObject status = benchmarkMatmul(matmulOpts, aType, bType, outType,
                                       biasType, iter, deviceId, dump);

  FUSILLI_CHECK_ERROR(status);

  return ok();
}

//===---------------------------------------------------------------------===//
// Main function
//===---------------------------------------------------------------------===//

static int benchmark(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  // Create option objects
  ConvOptions convOpts;
  MatmulOptions matmulOpts;
  LayerNormOptions layerNormOpts;

  // Shared options between subcommands
  int64_t iter, deviceId;
  bool dump{false};

  // mainApp CLI Options - shared between subcommands
  mainApp.add_option("--iter,-i", iter, "Benchmark iterations")
      ->required()
      ->check(kIsPositiveInteger);
  mainApp
      .add_option("--device,-D", deviceId,
                  "AMDGPU Device ID (ignored for CPU backend)")
      ->default_val("0")
      ->check(kIsNonNegativeInteger);

  // mainApp CLI Flags:
  mainApp.add_flag("--dump,-d", dump,
                   "Dump compilation artifacts to disk at "
                   "'${FUSILLI_CACHE_DIR}/.cache/fusilli'. "
                   "When not set, it defaults to '${HOME}/.cache/fusilli'.");

  // Register subcommands
  CLI::App *convApp = registerConvOptions(mainApp, convOpts);
  CLI::App *matmulApp = registerMatmulOptions(mainApp, matmulOpts);
  CLI::App *layerNormApp = registerLayerNormOptions(mainApp, layerNormOpts);

  CLI11_PARSE(mainApp, argc, argv);

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    ErrorObject status = runConvBenchmark(convOpts, iter, deviceId, dump);
    if (isError(status)) {
      std::cerr << "Fusilli Conv Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  if (layerNormApp->parsed()) {
    ErrorObject status =
        runLayerNormBenchmark(layerNormOpts, iter, deviceId, dump);
    if (isError(status)) {
      std::cerr << "Fusilli LayerNorm Benchmark failed: " << status
                << std::endl;
      return 1;
    }
  }

  if (matmulApp->parsed()) {
    ErrorObject status = runMatmulBenchmark(matmulOpts, iter, deviceId, dump);
    if (isError(status)) {
      std::cerr << "Fusilli Matmul Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  try {
    return benchmark(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
    return 1;
  }
}
