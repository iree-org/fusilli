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
// Verifies that omitting the momentum tensor causes the emitter to emit a
// torch.constant.float with PyTorch's default value (0.1) instead of an
// torch.aten.item extraction, and that no MOMENTUM vtensor.literal appears.
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(
// TORCH-CHECK-NOT:   %bn_no_mom_MOMENTUM = torch.vtensor.literal
// TORCH-CHECK:       %eps_bn_no_mom = torch.aten.item %bn_no_mom_EPSILON : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %momentum_bn_no_mom = torch.constant.float 1.000000e-01
// TORCH-CHECK:       %training_bn_no_mom = torch.constant.bool false
// TORCH-CHECK:       %bn_no_mom_Y_bn_no_mom_perm, %_infer_saved_mean_bn_no_mom_perm, %_infer_saved_invstd_bn_no_mom_perm = torch.aten.native_batch_norm {{.*}}, %momentum_bn_no_mom, %eps_bn_no_mom : {{.*}}
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
testBatchnormInferAsmEmitterNchwNoMomentum(const std::string &mode) {
  int64_t n = 4, c = 16, h = 8, w = 8;
  auto graph = std::make_shared<Graph>();
  graph->setName("batchnorm_infer_asm_emitter_nchw_no_momentum");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("bn_no_mom_X")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto meanT = graph->tensor(
      TensorAttr().setName("bn_no_mom_MEAN").setDim({c}).setStride({1}));

  auto varT = graph->tensor(
      TensorAttr().setName("bn_no_mom_VAR").setDim({c}).setStride({1}));

  auto epsilonT = graph->tensor(TensorAttr(1e-5f).setName("bn_no_mom_EPSILON"));

  // No momentum tensor — the emitter should use the PyTorch default (0.1).
  auto batchnormAttr = BatchnormAttr()
                           .setForwardPhase(NormFwdPhase::INFERENCE)
                           .setEpsilon(epsilonT)
                           .setName("bn_no_mom");

  auto [yT, smT, sivT] =
      graph->batchnorm(xT, nullptr, nullptr, meanT, varT, batchnormAttr);

  yT->setName("bn_no_mom_Y").setOutput(true);

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

  auto status = testBatchnormInferAsmEmitterNchwNoMomentum(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
