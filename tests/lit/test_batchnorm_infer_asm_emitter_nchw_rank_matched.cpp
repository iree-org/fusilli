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
// Verifies that rank-matched [1,C,1,1] channel tensors are collapsed to 1D
// [C] before being passed to native_batch_norm.
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(
// TORCH-CHECK:       %none_scale_bn_infer_rm = torch.constant.none
// TORCH-CHECK:       %none_bias_bn_infer_rm = torch.constant.none
// TORCH-CHECK:       %flat_start_mean_bn_infer_rm = torch.constant.int 0
// TORCH-CHECK:       %flat_end_mean_bn_infer_rm = torch.constant.int -1
// TORCH-CHECK:       %bn_infer_rm_MEAN_bn_infer_rm_collapsed = torch.aten.flatten.using_ints %bn_infer_rm_MEAN, %flat_start_mean_bn_infer_rm, %flat_end_mean_bn_infer_rm : !torch.vtensor<[1,16,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[16],f32>
// TORCH-CHECK:       %flat_start_var_bn_infer_rm = torch.constant.int 0
// TORCH-CHECK:       %flat_end_var_bn_infer_rm = torch.constant.int -1
// TORCH-CHECK:       %bn_infer_rm_VAR_bn_infer_rm_collapsed = torch.aten.flatten.using_ints %bn_infer_rm_VAR, %flat_start_var_bn_infer_rm, %flat_end_var_bn_infer_rm : !torch.vtensor<[1,16,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[16],f32>
// TORCH-CHECK:       %training_bn_infer_rm = torch.constant.bool false
// TORCH-CHECK:       %bn_infer_rm_Y_bn_infer_rm_perm, %_infer_saved_mean_bn_infer_rm_perm, %_infer_saved_invstd_bn_infer_rm_perm = torch.aten.native_batch_norm %bn_infer_rm_X_bn_infer_rm_perm, %none_scale_bn_infer_rm, %none_bias_bn_infer_rm, %bn_infer_rm_MEAN_bn_infer_rm_collapsed, %bn_infer_rm_VAR_bn_infer_rm_collapsed, %training_bn_infer_rm, %momentum_bn_infer_rm, %eps_bn_infer_rm : !torch.vtensor<[4,16,8,8],f32>, !torch.none, !torch.none, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.bool, !torch.float, !torch.float -> !torch.vtensor<[4,16,8,8],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>
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

static ErrorObject
testBatchnormInferAsmEmitterNchwRankMatched(const std::string &mode) {
  int64_t n = 4, c = 16, h = 8, w = 8;
  auto graph = std::make_shared<Graph>();
  graph->setName("batchnorm_infer_asm_emitter_nchw_rank_matched");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("bn_infer_rm_X")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  // Rank-matched [1, C, 1, 1] channel tensors.
  auto meanT = graph->tensor(TensorAttr()
                                 .setName("bn_infer_rm_MEAN")
                                 .setDim({1, c, 1, 1})
                                 .setStride({c, 1, 1, 1}));

  auto varT = graph->tensor(TensorAttr()
                                .setName("bn_infer_rm_VAR")
                                .setDim({1, c, 1, 1})
                                .setStride({c, 1, 1, 1}));

  auto epsilonT =
      graph->tensor(TensorAttr(1e-5f).setName("bn_infer_rm_EPSILON"));
  auto momentumT =
      graph->tensor(TensorAttr(0.1f).setName("bn_infer_rm_MOMENTUM"));

  auto batchnormAttr = BatchnormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setMomentum(momentumT)
                           .setName("bn_infer_rm");

  auto [yT, smT, sivT] =
      graph->batchnorm(xT, nullptr, nullptr, meanT, varT, batchnormAttr);

  yT->setName("bn_infer_rm_Y").setOutput(true);

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

  auto status = testBatchnormInferAsmEmitterNchwRankMatched(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
