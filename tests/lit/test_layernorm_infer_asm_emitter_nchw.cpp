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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,128,64,32],f32>, %arg0_x: !torch.vtensor<[16,128,64,32],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %normalized_shape_val_0_layernorm_infer = torch.constant.int 128
// TORCH-CHECK:       %normalized_shape_val_1_layernorm_infer = torch.constant.int 64
// TORCH-CHECK:       %normalized_shape_val_2_layernorm_infer = torch.constant.int 32
// TORCH-CHECK:       %normalized_shape_layernorm_infer = torch.prim.ListConstruct %normalized_shape_val_0_layernorm_infer, %normalized_shape_val_1_layernorm_infer, %normalized_shape_val_2_layernorm_infer : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %eps_layernorm_infer = torch.constant.float 1.000000e-05
// TORCH-CHECK:       %permute_x_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_x_val_1_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_x_val_2_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_x_val_3_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_x_layernorm_infer = torch.prim.ListConstruct %permute_x_val_0_layernorm_infer, %permute_x_val_1_layernorm_infer, %permute_x_val_2_layernorm_infer, %permute_x_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_x_layernorm_infer_perm = torch.aten.permute %arg0_x, %permute_x_layernorm_infer : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %none_scale_layernorm_infer = torch.constant.none
// TORCH-CHECK:       %none_bias_layernorm_infer = torch.constant.none
// TORCH-CHECK:       %cudnn_enable_layernorm_infer = torch.constant.bool false
// TORCH-CHECK:       %result_layernorm_infer_perm = torch.aten.layer_norm %arg0_x_layernorm_infer_perm, %normalized_shape_layernorm_infer, %none_scale_layernorm_infer, %none_bias_layernorm_infer, %eps_layernorm_infer, %cudnn_enable_layernorm_infer : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.float, !torch.bool -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %permute_y_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_y_val_1_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_y_val_2_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_y_val_3_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_y_layernorm_infer = torch.prim.ListConstruct %permute_y_val_0_layernorm_infer, %permute_y_val_1_layernorm_infer, %permute_y_val_2_layernorm_infer, %permute_y_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_layernorm_infer_perm, %permute_y_layernorm_infer : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,128,64,32],f32>, !torch.tensor<[16,128,64,32],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// LINALG-CHECK:      %[[EPS:.*]] = arith.constant 9.99999974E-6 : f32
// LINALG-CHECK:      %[[NELEMS:.*]] = arith.constant 2.621440e+05 : f32
// LINALG-CHECK:      %[[INPUT:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x128x64x32xf32>
// LINALG-CHECK:      %[[EMPTY0:.+]] = tensor.empty() : tensor<16x1x1x1xf32>
// LINALG-CHECK:      %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<16x1x1x1xf32>) -> tensor<16x1x1x1xf32>
// LINALG-CHECK:      %[[SUM:.+]] = linalg.generic {indexing_maps = [{{.+}}, {{.+}}], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%[[INPUT]] : tensor<16x128x64x32xf32>) outs(%[[FILL]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[MEAN:.+]] = linalg.generic {{{.+}}} ins(%[[SUM]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY0]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[EMPTY1:.+]] = tensor.empty() : tensor<16x128x64x32xf32>
// LINALG-CHECK:      %[[COLLAPSED:.+]] = tensor.collapse_shape %{{.+}} {{\[\[}}0, 1, 2, 3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
// LINALG-CHECK:      %[[BCAST_MEAN:.+]] = linalg.generic {{{.+}}} ins(%[[COLLAPSED]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[CENTERED:.+]] = linalg.generic {{{.+}}} ins(%[[INPUT]], %[[BCAST_MEAN]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[SQUARED:.+]] = linalg.generic {{{.+}}} ins(%[[CENTERED]], %[[CENTERED]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[VAR_SUM:.+]] = linalg.generic {{{.+}}} ins(%[[SQUARED]] : tensor<16x128x64x32xf32>) outs(%{{.+}} : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[VAR:.+]] = linalg.generic {{{.+}}} ins(%[[VAR_SUM]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY0]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[VAR_EPS:.+]] = linalg.generic {{{.+}}} ins(%[[VAR]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY0]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[RSQRT:.+]] = linalg.generic {{{.+}}} ins(%[[VAR_EPS]] : tensor<16x1x1x1xf32>) outs(%[[EMPTY0]] : tensor<16x1x1x1xf32>)
// LINALG-CHECK:      %[[COLLAPSED2:.+]] = tensor.collapse_shape %[[RSQRT]] {{\[\[}}0, 1, 2, 3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
// LINALG-CHECK:      %[[BCAST_RSQRT:.+]] = linalg.generic {{{.+}}} ins(%[[COLLAPSED2]] : tensor<16xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %[[OUT:.+]] = linalg.generic {{{.+}}} ins(%[[CENTERED]], %[[BCAST_RSQRT]] : tensor<16x128x64x32xf32>, tensor<16x128x64x32xf32>) outs(%[[EMPTY1]] : tensor<16x128x64x32xf32>)
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUT]] : tensor<16x128x64x32xf32> to %[[ARG0]] : !hal.buffer_view
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 2
//
// clang-format on

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testLayernormInferAsmEmitterNchw(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32;
  auto graph = std::make_shared<Graph>();
  graph->setName("layernorm_infer_asm_emitter_nchw");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto layernormAttr = LayernormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setName("layernorm_infer");

  auto [yT, mT, vT] = graph->layernorm(xT, nullptr, nullptr, layernormAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::AMDGPU));
#else
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testLayernormInferAsmEmitterNchw(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
