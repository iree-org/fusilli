// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>

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
const auto kIsValidConvLayout =
    CLI::IsMember({"NCHW", "NHWC", "NCDHW", "NDHWC"});

// Option classes for organizing benchmark parameters
struct ConvOptions {
  int64_t n, c, d{-1}, h, w, g{1}, k, z{-1}, y, x;
  int64_t t{-1}, u, v, o{-1}, p, q, m{-1}, l, j, s;
  int64_t mode;
  std::string imageLayout, filterLayout, outputLayout;
  bool fp16{false};
  bool bf16{false};
  bool bias{false};
};

struct MatmulOptions {
  int64_t m, n, k;
  bool fp16{false};
  bool bf16{false};
  bool transA{false};
  bool transB{false};
};

static ErrorObject benchmarkConvFprop(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU, deviceId));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
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
  auto xStride =
      (opts.imageLayout == "NCHW" || opts.imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto wStride =
      (opts.filterLayout == "NCHW" || opts.filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));
  auto convStride = (opts.s == 2)
                        ? std::vector<int64_t>{opts.u, opts.v}
                        : std::vector<int64_t>{opts.t, opts.u, opts.v};
  auto convPadding = (opts.s == 2)
                         ? std::vector<int64_t>{opts.p, opts.q}
                         : std::vector<int64_t>{opts.o, opts.p, opts.q};
  auto convDilation = (opts.s == 2)
                          ? std::vector<int64_t>{opts.l, opts.j}
                          : std::vector<int64_t>{opts.m, opts.l, opts.j};
  auto biasDims = (opts.s == 2) ? std::vector<int64_t>{1, opts.k, 1, 1}
                                : std::vector<int64_t>{1, opts.k, 1, 1, 1};
  auto biasStride =
      (opts.imageLayout == "NCHW" || opts.imageLayout == "NCDHW")
          ? generateStrideFromDim(biasDims,
                                  getContiguousStrideOrder(biasDims.size()))
          : generateStrideFromDim(biasDims,
                                  getChannelsLastStrideOrder(biasDims.size()));

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
  auto xBuf = FUSILLI_TRY(allocateBufferOfType(handle, xT, convIOType, 1.0f));
  auto wBuf = FUSILLI_TRY(allocateBufferOfType(handle, wT, convIOType, 1.0f));
  auto yBuf = FUSILLI_TRY(allocateBufferOfType(handle, yT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {wT, wBuf},
          {yT, yBuf},
      };

  if (opts.bias) {
    auto bBuf = FUSILLI_TRY(allocateBufferOfType(handle, bT, convIOType, 1.0f));
    variantPack.insert({bT, bBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkMatmul(const MatmulOptions &opts,
                                   DataType matmulIOType, int64_t iter,
                                   int64_t deviceId, bool dump) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU, deviceId));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Build attributes based on transpose flags.
  auto aDims = std::vector<int64_t>{opts.m, opts.k};
  auto bDims = std::vector<int64_t>{opts.k, opts.n};
  auto aStride = opts.transA ? std::vector<int64_t>{1, opts.m}
                             : std::vector<int64_t>{opts.k, 1};
  auto bStride = opts.transB ? std::vector<int64_t>{1, opts.k}
                             : std::vector<int64_t>{opts.n, 1};

  Graph graph;
  auto graphName =
      std::format("benchmark_matmul_m{}_n{}_k{}_transA{}_transB{}_dtype{}",
                  opts.m, opts.n, opts.k, opts.transA, opts.transB,
                  kDataTypeToMlirTypeAsm.at(matmulIOType));
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
                             .setDataType(matmulIOType));

  auto bT = graph.tensor(TensorAttr()
                             .setName("matrix_b")
                             .setDim(bDims)
                             .setStride(bStride)
                             .setDataType(matmulIOType));

  auto matmulAttr = MatmulAttr().setName("matmul");

  auto cT = graph.matmul(aT, bT, matmulAttr);
  cT->setOutput(true).setDataType(matmulIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/!dump));

  // Allocate input, weight and output buffers.
  auto aBuf = FUSILLI_TRY(allocateBufferOfType(handle, aT, matmulIOType, 1.0f));
  auto bBuf = FUSILLI_TRY(allocateBufferOfType(handle, bT, matmulIOType, 1.0f));
  auto cBuf = FUSILLI_TRY(allocateBufferOfType(handle, cT, matmulIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {cT, cBuf},
      };

  // Execute graph `iter` times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkConvWGrad(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU, deviceId));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
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

  auto xStride =
      (opts.imageLayout == "NCHW" || opts.imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto dyStride = (opts.outputLayout == "NCHW" || opts.outputLayout == "NCDHW")
                      ? generateStrideFromDim(
                            dyDims, getContiguousStrideOrder(dyDims.size()))
                      : generateStrideFromDim(
                            dyDims, getChannelsLastStrideOrder(dyDims.size()));
  auto wStride =
      (opts.filterLayout == "NCHW" || opts.filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName = std::format(
      "benchmark_conv_wgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
      "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
      "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}",
      opts.n, opts.c, opts.d, opts.h, opts.w, opts.g, opts.k, opts.z, opts.y,
      opts.x, opts.t, opts.u, opts.v, opts.o, opts.p, opts.q, opts.m, opts.l,
      opts.j, opts.s, opts.imageLayout, opts.outputLayout, opts.filterLayout);
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
  auto dyBuf = FUSILLI_TRY(allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  auto xBuf = FUSILLI_TRY(allocateBufferOfType(handle, xT, convIOType, 1.0f));
  auto dwBuf = FUSILLI_TRY(allocateBufferOfType(handle, dwT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dwT, dwBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject benchmarkConvDGrad(const ConvOptions &opts,
                                      DataType convIOType, int64_t iter,
                                      int64_t deviceId, bool dump) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU, deviceId));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
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

  auto xStride =
      (opts.imageLayout == "NCHW" || opts.imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto dyStride = (opts.outputLayout == "NCHW" || opts.outputLayout == "NCDHW")
                      ? generateStrideFromDim(
                            dyDims, getContiguousStrideOrder(dyDims.size()))
                      : generateStrideFromDim(
                            dyDims, getChannelsLastStrideOrder(dyDims.size()));
  auto wStride =
      (opts.filterLayout == "NCHW" || opts.filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName = std::format(
      "benchmark_conv_dgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
      "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
      "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}",
      opts.n, opts.c, opts.d, opts.h, opts.w, opts.g, opts.k, opts.z, opts.y,
      opts.x, opts.t, opts.u, opts.v, opts.o, opts.p, opts.q, opts.m, opts.l,
      opts.j, opts.s, opts.imageLayout, opts.outputLayout, opts.filterLayout);
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
  auto dyBuf = FUSILLI_TRY(allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  auto wBuf = FUSILLI_TRY(allocateBufferOfType(handle, wT, convIOType, 1.0f));
  auto dxBuf = FUSILLI_TRY(allocateBufferOfType(handle, dxT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dxT, dxBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

// Register convolution options to CLI app
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
      ->check(kIsValidConvLayout);
  convApp->add_option("--fil_layout", convOpts.filterLayout, "Filter layout")
      ->required()
      ->check(kIsValidConvLayout);
  convApp->add_option("--out_layout", convOpts.outputLayout, "Output layout")
      ->required()
      ->check(kIsValidConvLayout);
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
  convApp->add_flag("--bias,-b", convOpts.bias,
                    "Run with bias (only for mode=1)");

  return convApp;
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

  // matmulApp CLI Flags:
  auto *mf1 = matmulApp->add_flag("--fp16", matmulOpts.fp16, "Run fp16 matmul");
  auto *mf2 = matmulApp->add_flag("--bf16", matmulOpts.bf16, "Run bf16 matmul");
  // Can't specify both flags.
  mf1->excludes(mf2);
  matmulApp->add_flag("--transA", matmulOpts.transA, "Transpose matrix A");
  matmulApp->add_flag("--transB", matmulOpts.transB, "Transpose matrix B");

  return matmulApp;
}

// Validate and run convolution benchmark
static int runConvBenchmark(const ConvOptions &convOpts, int64_t iter,
                            int64_t deviceId, bool dump) {
  // Additional validation of convApp options (apart from default CLI checks)
  if (convOpts.s == 2) {
    // Reject 3D layouts for 2D conv
    if (convOpts.imageLayout.size() != 4 || convOpts.filterLayout.size() != 4 ||
        convOpts.outputLayout.size() != 4) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 2D convolution."
                << std::endl;
      return 1;
    }
  }
  if (convOpts.s == 3) {
    // Reject 2D layouts for 3D conv
    if (convOpts.imageLayout.size() != 5 || convOpts.filterLayout.size() != 5 ||
        convOpts.outputLayout.size() != 5) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 3D convolution."
                << std::endl;
      return 1;
    }
    // Reject default (sentinel) values for optional args in 3D conv
    if (convOpts.d == -1 || convOpts.z == -1 || convOpts.t == -1 ||
        convOpts.o == -1 || convOpts.m == -1) {
      std::cerr << "Detected at least one of {in_d, fil_d, conv_stride_d, "
                   "pad_d, dilation_d} that was not set for 3D convolution."
                << std::endl;
      return 1;
    }
  }

  // Validation of group count
  if (convOpts.c % convOpts.g != 0 || convOpts.k % convOpts.g != 0) {
    std::cerr << "Detected invalid group count." << std::endl;
    return 1;
  }

  // Validate bias flag only works with forward mode
  if (convOpts.bias && convOpts.mode != 1) {
    std::cerr << "Bias flag (--bias) is only supported for forward "
                 "convolution (mode=1)."
              << std::endl;
    return 1;
  }

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

  if (isError(status)) {
    std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
    return 1;
  }

  return 0;
}

// Validate and run matmul benchmark
static int runMatmulBenchmark(const MatmulOptions &matmulOpts, int64_t iter,
                              int64_t deviceId, bool dump) {
  DataType matmulIOType;
  if (matmulOpts.fp16)
    matmulIOType = DataType::Half;
  else if (matmulOpts.bf16)
    matmulIOType = DataType::BFloat16;
  else
    // When unspecified, default to fp32 matmul.
    matmulIOType = DataType::Float;

  ErrorObject status =
      benchmarkMatmul(matmulOpts, matmulIOType, iter, deviceId, dump);

  if (isError(status)) {
    std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
    return 1;
  }

  return 0;
}

static int benchmark(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  // Create option objects
  ConvOptions convOpts;
  MatmulOptions matmulOpts;

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

  CLI11_PARSE(mainApp, argc, argv);

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    int result = runConvBenchmark(convOpts, iter, deviceId, dump);
    if (result != 0)
      return result;
  }

  if (matmulApp->parsed()) {
    int result = runMatmulBenchmark(matmulOpts, iter, deviceId, dump);
    if (result != 0)
      return result;
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
