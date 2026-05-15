// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verifies that an RMSNorm with a single-dim trailing-suffix scale
// (scale=[1,1,1,W] for x=[N,C,H,W]) emits torch.aten.rms_norm with
// normalized_shape=[W] (only the last dim of x), exercising the narrowest
// valid trailing suffix under the cuDNN/hipDNN rule.

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main({{.*}}) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %normalized_shape_val_0_rmsnorm_infer = torch.constant.int 32
// TORCH-CHECK:       %normalized_shape_rmsnorm_infer = torch.prim.ListConstruct %normalized_shape_val_0_rmsnorm_infer : (!torch.int) -> !torch.list<int>
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
testRmsnormInferAsmEmitterPartialSuffixW(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32;
  auto graph = std::make_shared<Graph>();
  graph->setName("rmsnorm_infer_asm_emitter_partial_suffix_w");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  // scale=[1, 1, 1, W]: trailing match is just W → expect
  // normalized_shape=[W].
  auto scaleT = graph->tensor(TensorAttr()
                                  .setName("arg0_scale")
                                  .setDim({1, 1, 1, w})
                                  .setStride({w, w, w, 1}));

  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto rmsnormAttr = RmsnormAttr()
                         .setForwardPhase(NormFwdPhase::INFERENCE)
                         .setEpsilon(epsilonT)
                         .setName("rmsnorm_infer");

  auto [yT, rT] = graph->rmsnorm(xT, scaleT, rmsnormAttr);

  yT->setName("result").setOutput(true);

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

  auto status = testRmsnormInferAsmEmitterPartialSuffixW(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
