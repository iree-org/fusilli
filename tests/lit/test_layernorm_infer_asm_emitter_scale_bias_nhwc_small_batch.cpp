// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[2,128,128,3],f32>, %arg0_x: !torch.vtensor<[2,128,128,3],f32>, %arg1_scale: !torch.vtensor<[1,128,128,3],f32>, %arg2_bias: !torch.vtensor<[1,128,128,3],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// Graph-level scalar constant emission for epsilon:
// TORCH-CHECK:       %layernorm_infer_EPSILON = torch.vtensor.literal(dense<0x3727C5AC> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// TORCH-CHECK:       %normalized_shape_val_0_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %normalized_shape_val_1_layernorm_infer = torch.constant.int 128
// TORCH-CHECK:       %normalized_shape_val_2_layernorm_infer = torch.constant.int 128
// TORCH-CHECK:       %normalized_shape_layernorm_infer = torch.prim.ListConstruct %normalized_shape_val_0_layernorm_infer, %normalized_shape_val_1_layernorm_infer, %normalized_shape_val_2_layernorm_infer : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %eps_layernorm_infer = torch.aten.item %layernorm_infer_EPSILON : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %permute_x_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_x_val_1_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_x_val_2_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_x_val_3_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_x_layernorm_infer = torch.prim.ListConstruct %permute_x_val_0_layernorm_infer, %permute_x_val_1_layernorm_infer, %permute_x_val_2_layernorm_infer, %permute_x_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_x_layernorm_infer_perm = torch.aten.permute %arg0_x, %permute_x_layernorm_infer : !torch.vtensor<[2,128,128,3],f32>, !torch.list<int> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       %permute_scale_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_scale_val_1_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_scale_val_2_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_scale_val_3_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_scale_layernorm_infer = torch.prim.ListConstruct %permute_scale_val_0_layernorm_infer, %permute_scale_val_1_layernorm_infer, %permute_scale_val_2_layernorm_infer, %permute_scale_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_scale_layernorm_infer_perm = torch.aten.permute %arg1_scale, %permute_scale_layernorm_infer : !torch.vtensor<[1,128,128,3],f32>, !torch.list<int> -> !torch.vtensor<[1,3,128,128],f32>
// TORCH-CHECK:       %permute_bias_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_bias_val_1_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_bias_val_2_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_bias_val_3_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_bias_layernorm_infer = torch.prim.ListConstruct %permute_bias_val_0_layernorm_infer, %permute_bias_val_1_layernorm_infer, %permute_bias_val_2_layernorm_infer, %permute_bias_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg2_bias_layernorm_infer_perm = torch.aten.permute %arg2_bias, %permute_bias_layernorm_infer : !torch.vtensor<[1,128,128,3],f32>, !torch.list<int> -> !torch.vtensor<[1,3,128,128],f32>
// TORCH-CHECK:       %cudnn_enable_layernorm_infer = torch.constant.bool false
// TORCH-CHECK:       %result_layernorm_infer_perm = torch.aten.layer_norm %arg0_x_layernorm_infer_perm, %normalized_shape_layernorm_infer, %arg1_scale_layernorm_infer_perm, %arg2_bias_layernorm_infer_perm, %eps_layernorm_infer, %cudnn_enable_layernorm_infer : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int>, !torch.vtensor<[1,3,128,128],f32>, !torch.vtensor<[1,3,128,128],f32>, !torch.float, !torch.bool -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       %permute_y_val_0_layernorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_y_val_1_layernorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_y_val_2_layernorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_y_val_3_layernorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_y_layernorm_infer = torch.prim.ListConstruct %permute_y_val_0_layernorm_infer, %permute_y_val_1_layernorm_infer, %permute_y_val_2_layernorm_infer, %permute_y_val_3_layernorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_layernorm_infer_perm, %permute_y_layernorm_infer : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int> -> !torch.vtensor<[2,128,128,3],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[2,128,128,3],f32>, !torch.tensor<[2,128,128,3],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "transient-memory-size": 0
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "transient-memory-size": 0
// CPU-STATS-CHECK: "dispatch-count": 2
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject
testLayernormInferAsmEmitterScaleBiasNhwcSmallBatch(const std::string &mode) {
  int64_t n = 2, c = 3, h = 128, w = 128;
  auto graph = std::make_shared<Graph>();
  graph->setName("layernorm_infer_asm_emitter_scale_bias_nhwc_small_batch");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto sT = graph->tensor(TensorAttr()
                              .setName("arg1_scale")
                              .setDim({1, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto bT = graph->tensor(TensorAttr()
                              .setName("arg2_bias")
                              .setDim({1, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto layernormAttr = LayernormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setName("layernorm_infer");

  auto [yT, mT, vT] = graph->layernorm(xT, sT, bT, layernormAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(kDefaultBackend));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testLayernormInferAsmEmitterScaleBiasNhwcSmallBatch(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
