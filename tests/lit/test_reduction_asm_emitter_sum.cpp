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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,256,1,1],f32>, %arg0_input: !torch.vtensor<[16,256,64,64],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_X_val_0_reduction_sum = torch.constant.int 0
// TORCH-CHECK:       %permute_X_val_1_reduction_sum = torch.constant.int 1
// TORCH-CHECK:       %permute_X_val_2_reduction_sum = torch.constant.int 2
// TORCH-CHECK:       %permute_X_val_3_reduction_sum = torch.constant.int 3
// TORCH-CHECK:       %permute_X_reduction_sum = torch.prim.ListConstruct %permute_X_val_0_reduction_sum, %permute_X_val_1_reduction_sum, %permute_X_val_2_reduction_sum, %permute_X_val_3_reduction_sum : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_input_reduction_sum_perm = torch.aten.permute %arg0_input, %permute_X_reduction_sum : !torch.vtensor<[16,256,64,64],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,64],f32>
// TORCH-CHECK:       %reduction_dims_val_0_reduction_sum = torch.constant.int 2
// TORCH-CHECK:       %reduction_dims_val_1_reduction_sum = torch.constant.int 3
// TORCH-CHECK:       %reduction_dims_reduction_sum = torch.prim.ListConstruct %reduction_dims_val_0_reduction_sum, %reduction_dims_val_1_reduction_sum : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %keepdim_reduction_sum = torch.constant.bool true
// TORCH-CHECK:       %dtype_reduction_sum = torch.constant.none
// TORCH-CHECK:       %result_reduction_sum_perm = torch.aten.sum.dim_IntList %arg0_input_reduction_sum_perm, %reduction_dims_reduction_sum, %keepdim_reduction_sum, %dtype_reduction_sum : !torch.vtensor<[16,256,64,64],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[16,256,1,1],f32>
// TORCH-CHECK:       %permute_Y_val_0_reduction_sum = torch.constant.int 0
// TORCH-CHECK:       %permute_Y_val_1_reduction_sum = torch.constant.int 1
// TORCH-CHECK:       %permute_Y_val_2_reduction_sum = torch.constant.int 2
// TORCH-CHECK:       %permute_Y_val_3_reduction_sum = torch.constant.int 3
// TORCH-CHECK:       %permute_Y_reduction_sum = torch.prim.ListConstruct %permute_Y_val_0_reduction_sum, %permute_Y_val_1_reduction_sum, %permute_Y_val_2_reduction_sum, %permute_Y_val_3_reduction_sum : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_reduction_sum_perm, %permute_Y_reduction_sum : !torch.vtensor<[16,256,1,1],f32>, !torch.list<int> -> !torch.vtensor<[16,256,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,256,1,1],f32>, !torch.tensor<[16,256,1,1],f32>
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

static ErrorObject testReductionAsmEmitterSum(const std::string &mode) {
  int64_t d0 = 16, d1 = 256, d2 = 64, d3 = 64;
  auto graph = std::make_shared<Graph>();
  graph->setName("reduction_asm_emitter_sum");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_input")
                              .setDim({d0, d1, d2, d3})
                              .setStride({d1 * d2 * d3, d2 * d3, d3, 1}));
  auto reductionAttr = ReductionAttr()
                           .setMode(ReductionAttr::Mode::SUM)
                           .setName("reduction_sum");
  auto yT = graph->reduction(xT, reductionAttr);
  yT->setDim({d0, d1, 1, 1}).setStride({d1, 1, 1, 1});
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

  auto status = testReductionAsmEmitterSum(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
