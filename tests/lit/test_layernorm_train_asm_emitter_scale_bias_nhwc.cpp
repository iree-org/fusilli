// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} | iree-compile - --compile-to=input | \
// RUN:             FileCheck %s --check-prefix=LINALG-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%inv_variance_: !torch.tensor<[16,1,1,1],f32>, %mean_: !torch.tensor<[16,1,1,1],f32>, %y_: !torch.tensor<[16,64,32,128],f32>, %arg0_x: !torch.vtensor<[16,64,32,128],f32>, %arg1_scale: !torch.vtensor<[1,128,64,32],f32>, %arg2_bias: !torch.vtensor<[1,128,64,32],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %normalized_shape_val_0_layernorm_train = torch.constant.int 128
// TORCH-CHECK:       %normalized_shape_val_1_layernorm_train = torch.constant.int 64
// TORCH-CHECK:       %normalized_shape_val_2_layernorm_train = torch.constant.int 32
// TORCH-CHECK:       %normalized_shape_layernorm_train = torch.prim.ListConstruct %normalized_shape_val_0_layernorm_train, %normalized_shape_val_1_layernorm_train, %normalized_shape_val_2_layernorm_train : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %eps_layernorm_train = torch.constant.float 1.000000e-05
// TORCH-CHECK:       %permute_x_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_x_val_1_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_x_val_2_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_x_val_3_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_x_layernorm_train = torch.prim.ListConstruct %permute_x_val_0_layernorm_train, %permute_x_val_1_layernorm_train, %permute_x_val_2_layernorm_train, %permute_x_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_x_perm = torch.aten.permute %arg0_x, %permute_x_layernorm_train : !torch.vtensor<[16,64,32,128],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %permute_scale_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_scale_val_1_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_scale_val_2_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_scale_val_3_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_scale_layernorm_train = torch.prim.ListConstruct %permute_scale_val_0_layernorm_train, %permute_scale_val_1_layernorm_train, %permute_scale_val_2_layernorm_train, %permute_scale_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_scale_perm = torch.aten.permute %arg1_scale, %permute_scale_layernorm_train : !torch.vtensor<[1,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[1,128,64,32],f32>
// TORCH-CHECK:       %permute_bias_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_bias_val_1_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_bias_val_2_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_bias_val_3_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_bias_layernorm_train = torch.prim.ListConstruct %permute_bias_val_0_layernorm_train, %permute_bias_val_1_layernorm_train, %permute_bias_val_2_layernorm_train, %permute_bias_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg2_bias_perm = torch.aten.permute %arg2_bias, %permute_bias_layernorm_train : !torch.vtensor<[1,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[1,128,64,32],f32>
// TORCH-CHECK:       %y_perm, %mean_perm, %inv_variance_perm = torch.aten.native_layer_norm %arg0_x_perm, %normalized_shape_layernorm_train, %arg1_scale_perm, %arg2_bias_perm, %eps_layernorm_train : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int>, !torch.vtensor<[1,128,64,32],f32>, !torch.vtensor<[1,128,64,32],f32>, !torch.float -> !torch.vtensor<[16,128,64,32],f32>, !torch.vtensor<[16,1,1,1],f32>, !torch.vtensor<[16,1,1,1],f32>
// TORCH-CHECK:       %permute_y_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_y_val_1_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_y_val_2_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_y_val_3_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_y_layernorm_train = torch.prim.ListConstruct %permute_y_val_0_layernorm_train, %permute_y_val_1_layernorm_train, %permute_y_val_2_layernorm_train, %permute_y_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %y = torch.aten.permute %y_perm, %permute_y_layernorm_train : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,64,32,128],f32>
// TORCH-CHECK:       %permute_mean_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_mean_val_1_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_mean_val_2_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_mean_val_3_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_mean_layernorm_train = torch.prim.ListConstruct %permute_mean_val_0_layernorm_train, %permute_mean_val_1_layernorm_train, %permute_mean_val_2_layernorm_train, %permute_mean_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %mean = torch.aten.permute %mean_perm, %permute_mean_layernorm_train : !torch.vtensor<[16,1,1,1],f32>, !torch.list<int> -> !torch.vtensor<[16,1,1,1],f32>
// TORCH-CHECK:       %permute_inv_variance_val_0_layernorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_inv_variance_val_1_layernorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_inv_variance_val_2_layernorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_inv_variance_val_3_layernorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_inv_variance_layernorm_train = torch.prim.ListConstruct %permute_inv_variance_val_0_layernorm_train, %permute_inv_variance_val_1_layernorm_train, %permute_inv_variance_val_2_layernorm_train, %permute_inv_variance_val_3_layernorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %inv_variance = torch.aten.permute %inv_variance_perm, %permute_inv_variance_layernorm_train : !torch.vtensor<[16,1,1,1],f32>, !torch.list<int> -> !torch.vtensor<[16,1,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %inv_variance overwrites %inv_variance_ : !torch.vtensor<[16,1,1,1],f32>, !torch.tensor<[16,1,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %mean overwrites %mean_ : !torch.vtensor<[16,1,1,1],f32>, !torch.tensor<[16,1,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %y overwrites %y_ : !torch.vtensor<[16,64,32,128],f32>, !torch.tensor<[16,64,32,128],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, %[[ARG3:.+]]: !hal.buffer_view, %[[ARG4:.+]]: !hal.buffer_view, %[[ARG5:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// LINALG-CHECK:      %[[EPS:.*]] = arith.constant 9.99999974E-6 : f32
// LINALG-CHECK:      %[[NELEMS:.*]] = arith.constant 2.621440e+05 : f32
// LINALG-CHECK:      %[[INPUT:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG3]] : !hal.buffer_view -> tensor<16x64x32x128xf32>
// LINALG-CHECK:      %[[SCALE:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG4]] : !hal.buffer_view -> tensor<1x128x64x32xf32>
// LINALG-CHECK:      %[[BIAS:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG5]] : !hal.buffer_view -> tensor<1x128x64x32xf32>
// LINALG-CHECK:      %[[EMPTY1:.+]] = tensor.empty() : tensor<16x128x64x32xf32>
// LINALG-CHECK:      %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<16x64x32x128xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>) permutation = [0, 3, 1, 2]
// LINALG-CHECK:      %[[EMPTY2:.+]] = tensor.empty() : tensor<16x1x1x1xf32>
// LINALG-CHECK:      %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY2]] : tensor<16x1x1x1xf32>) -> tensor<16x1x1x1xf32>
// LINALG-CHECK:      %[[SUM:.+]] = linalg.generic {indexing_maps = [{{.+}}, {{.+}}], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%[[TRANSPOSED]] : tensor<16x128x64x32xf32>) outs(%[[FILL]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[MEAN:.+]] = linalg.generic {{{.+}}} ins(%[[SUM]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY2]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0, 1, 2, 3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
// LINALG-CHECK:      %[[BCAST_MEAN:.+]] = linalg.generic {{{.+}}} ins(%[[COLLAPSED]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[CENTERED:.+]] = linalg.generic {{{.+}}} ins(%[[TRANSPOSED]], %[[BCAST_MEAN]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[SQUARED:.+]] = linalg.generic {{{.+}}} ins(%[[CENTERED]], %[[CENTERED]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[VAR_SUM:.+]] = linalg.generic {{{.+}}} ins(%[[SQUARED]] : tensor<16x128x64x32xf32>) outs(%{{.+}} : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[VAR:.+]] = linalg.generic {{{.+}}} ins(%[[VAR_SUM]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY2]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[VAR_EPS:.+]] = linalg.generic {{{.+}}} ins(%[[VAR]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY2]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[RSQRT:.+]] = linalg.generic {{{.+}}} ins(%[[VAR_EPS]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY2]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[COLLAPSED2:.+]] = tensor.collapse_shape %[[RSQRT]] {{\[\[}}0, 1, 2, 3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
// LINALG-CHECK:      %[[BCAST_RSQRT:.+]] = linalg.generic {{{.+}}} ins(%[[COLLAPSED2]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[NORMALIZED:.+]] = linalg.generic {{{.+}}} ins(%[[CENTERED]], %[[BCAST_RSQRT]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[SCALED:.+]] = linalg.generic {{{.+}}} ins(%[[NORMALIZED]], %[[SCALE]] : tensor<16x128x64x32xf32>, tensor<1x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[BIASED:.+]] = linalg.generic {{{.+}}} ins(%[[SCALED]], %[[BIAS]] : tensor<16x128x64x32xf32>, tensor<1x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[EMPTY3:.+]] = tensor.empty() : tensor<16x64x32x128xf32>
// LINALG-CHECK:      %[[OUT:.+]] = linalg.transpose ins(%[[BIASED]] : tensor<16x128x64x32xf32>) outs(%[[EMPTY3]] : tensor<16x64x32x128xf32>) permutation = [0, 2, 3, 1]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[RSQRT]] : tensor<16x1x1x1xf32> to %[[ARG0]] : !hal.buffer_view
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[MEAN]] : tensor<16x1x1x1xf32> to %[[ARG1]] : !hal.buffer_view
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUT]] : tensor<16x64x32x128xf32> to %[[ARG2]] : !hal.buffer_view
// LINALG-CHECK:      %{{.+}}:3 = hal.tensor.barrier join(%{{.+}}, %{{.+}}, %{{.+}} : tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x64x32x128xf32>) => %{{.+}} : !hal.fence
//
// AMDGPU-STATS-CHECK: "dispatch-count": 2
// CPU-STATS-CHECK: "dispatch-count": 4
//
// clang-format on

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject
testLayernormInferAsmEmitterScaleBiasNhwc(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32;
  auto graph = std::make_shared<Graph>();
  graph->setName("layernorm_train_asm_emitter_scale_bias_nhwc");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto sT = graph->tensor(TensorAttr()
                              .setName("arg1_scale")
                              .setDim({1, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto bT = graph->tensor(TensorAttr()
                              .setName("arg2_bias")
                              .setDim({1, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto layernormAttr = LayernormAttr()
                           .setForwardPhase(NormFwdPhase::TRAINING)
                           .setEpsilon(epsilonT)
                           .setName("layernorm_train");

  auto [yT, mT, vT] = graph->layernorm(xT, sT, bT, layernormAttr);

  yT->setName("y").setDataType(DataType::Float).setOutput(true);
  mT->setName("mean").setDataType(DataType::Float).setOutput(true);
  vT->setName("inv_variance").setDataType(DataType::Float).setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testLayernormInferAsmEmitterScaleBiasNhwc(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
