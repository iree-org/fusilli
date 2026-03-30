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
// TORCH-CHECK:     func.func @main(%batchnorm_infer_Y_: !torch.tensor<[4,16,8,8],f32>, %batchnorm_infer_MEAN: !torch.vtensor<[1,16,1,1],f32>, %batchnorm_infer_VAR: !torch.vtensor<[1,16,1,1],f32>, %batchnorm_infer_X: !torch.vtensor<[4,16,8,8],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %batchnorm_infer_EPSILON = torch.vtensor.literal(dense<
// TORCH-CHECK:       %batchnorm_infer_MOMENTUM = torch.vtensor.literal(dense<
// TORCH-CHECK:       %eps_batchnorm_infer = torch.aten.item %batchnorm_infer_EPSILON : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %momentum_batchnorm_infer = torch.aten.item %batchnorm_infer_MOMENTUM : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %permute_x_val_0_batchnorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_x_val_1_batchnorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_x_val_2_batchnorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_x_val_3_batchnorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_x_batchnorm_infer = torch.prim.ListConstruct %permute_x_val_0_batchnorm_infer, %permute_x_val_1_batchnorm_infer, %permute_x_val_2_batchnorm_infer, %permute_x_val_3_batchnorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %batchnorm_infer_X_batchnorm_infer_perm = torch.aten.permute %batchnorm_infer_X, %permute_x_batchnorm_infer : !torch.vtensor<[4,16,8,8],f32>, !torch.list<int> -> !torch.vtensor<[4,16,8,8],f32>
// TORCH-CHECK:       %none_scale_batchnorm_infer = torch.constant.none
// TORCH-CHECK:       %none_bias_batchnorm_infer = torch.constant.none
// TORCH-CHECK:       %flat_start_mean_batchnorm_infer = torch.constant.int 0
// TORCH-CHECK:       %flat_end_mean_batchnorm_infer = torch.constant.int -1
// TORCH-CHECK:       %batchnorm_infer_MEAN_batchnorm_infer_collapsed = torch.aten.flatten.using_ints %batchnorm_infer_MEAN, %flat_start_mean_batchnorm_infer, %flat_end_mean_batchnorm_infer : !torch.vtensor<[1,16,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[16],f32>
// TORCH-CHECK:       %flat_start_var_batchnorm_infer = torch.constant.int 0
// TORCH-CHECK:       %flat_end_var_batchnorm_infer = torch.constant.int -1
// TORCH-CHECK:       %batchnorm_infer_VAR_batchnorm_infer_collapsed = torch.aten.flatten.using_ints %batchnorm_infer_VAR, %flat_start_var_batchnorm_infer, %flat_end_var_batchnorm_infer : !torch.vtensor<[1,16,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[16],f32>
// TORCH-CHECK:       %training_batchnorm_infer = torch.constant.bool false
// TORCH-CHECK:       %batchnorm_infer_Y_batchnorm_infer_perm, %_infer_saved_mean_batchnorm_infer_perm, %_infer_saved_invstd_batchnorm_infer_perm = torch.aten.native_batch_norm %batchnorm_infer_X_batchnorm_infer_perm, %none_scale_batchnorm_infer, %none_bias_batchnorm_infer, %batchnorm_infer_MEAN_batchnorm_infer_collapsed, %batchnorm_infer_VAR_batchnorm_infer_collapsed, %training_batchnorm_infer, %momentum_batchnorm_infer, %eps_batchnorm_infer : !torch.vtensor<[4,16,8,8],f32>, !torch.none, !torch.none, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.bool, !torch.float, !torch.float -> !torch.vtensor<[4,16,8,8],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>
// TORCH-CHECK:       %permute_y_val_0_batchnorm_infer = torch.constant.int 0
// TORCH-CHECK:       %permute_y_val_1_batchnorm_infer = torch.constant.int 1
// TORCH-CHECK:       %permute_y_val_2_batchnorm_infer = torch.constant.int 2
// TORCH-CHECK:       %permute_y_val_3_batchnorm_infer = torch.constant.int 3
// TORCH-CHECK:       %permute_y_batchnorm_infer = torch.prim.ListConstruct %permute_y_val_0_batchnorm_infer, %permute_y_val_1_batchnorm_infer, %permute_y_val_2_batchnorm_infer, %permute_y_val_3_batchnorm_infer : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %batchnorm_infer_Y = torch.aten.permute %batchnorm_infer_Y_batchnorm_infer_perm, %permute_y_batchnorm_infer : !torch.vtensor<[4,16,8,8],f32>, !torch.list<int> -> !torch.vtensor<[4,16,8,8],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_infer_Y overwrites %batchnorm_infer_Y_ : !torch.vtensor<[4,16,8,8],f32>, !torch.tensor<[4,16,8,8],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "transient-memory-size": 0
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "transient-memory-size": 0
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testBatchnormInferAsmEmitterNchw(const std::string &mode) {
  int64_t n = 4, c = 16, h = 8, w = 8;
  auto graph = std::make_shared<Graph>();
  graph->setName("batchnorm_infer_asm_emitter_nchw");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("batchnorm_infer_X")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto meanT = graph->tensor(TensorAttr()
                                 .setName("batchnorm_infer_MEAN")
                                 .setDim({1, c, 1, 1})
                                 .setStride({c, 1, 1, 1}));

  auto varT = graph->tensor(TensorAttr()
                                .setName("batchnorm_infer_VAR")
                                .setDim({1, c, 1, 1})
                                .setStride({c, 1, 1, 1}));

  auto epsilonT =
      graph->tensor(TensorAttr(1e-5f).setName("batchnorm_infer_EPSILON"));
  auto momentumT =
      graph->tensor(TensorAttr(0.1f).setName("batchnorm_infer_MOMENTUM"));

  auto batchnormAttr = BatchnormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setMomentum(momentumT)
                           .setName("batchnorm_infer");

  auto [yT, smT, sivT] =
      graph->batchnorm(xT, nullptr, nullptr, meanT, varT, batchnormAttr);

  yT->setName("batchnorm_infer_Y").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
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

  auto status = testBatchnormInferAsmEmitterNchw(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
