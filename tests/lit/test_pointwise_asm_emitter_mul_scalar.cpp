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
// Scalar IN_1 is NOT in the func.func signature — only result_ and arg0.
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[2,3,128,128],f32>, %arg0: !torch.vtensor<[2,3,128,128],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_IN_0_val_0_pointwise_mul = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_0_val_1_pointwise_mul = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_0_val_2_pointwise_mul = torch.constant.int 2
// TORCH-CHECK:       %permute_IN_0_val_3_pointwise_mul = torch.constant.int 3
// TORCH-CHECK:       %permute_IN_0_pointwise_mul = torch.prim.ListConstruct %permute_IN_0_val_0_pointwise_mul, %permute_IN_0_val_1_pointwise_mul, %permute_IN_0_val_2_pointwise_mul, %permute_IN_0_val_3_pointwise_mul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_pointwise_mul_perm = torch.aten.permute %arg0, %permute_IN_0_pointwise_mul : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int> -> !torch.vtensor<[2,3,128,128],f32>
// Scalar emitted as torch.vtensor.literal constant:
// TORCH-CHECK:       %alpha_pointwise_mul_perm = torch.vtensor.literal(dense<2.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// MUL with scalar operand type !torch.vtensor<[1],f32>:
// TORCH-CHECK:       %result_pointwise_mul_perm = torch.aten.mul.Tensor %arg0_pointwise_mul_perm, %alpha_pointwise_mul_perm : !torch.vtensor<[2,3,128,128],f32>, !torch.vtensor<[1],f32> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       %permute_OUT_0_val_0_pointwise_mul = torch.constant.int 0
// TORCH-CHECK:       %permute_OUT_0_val_1_pointwise_mul = torch.constant.int 1
// TORCH-CHECK:       %permute_OUT_0_val_2_pointwise_mul = torch.constant.int 2
// TORCH-CHECK:       %permute_OUT_0_val_3_pointwise_mul = torch.constant.int 3
// TORCH-CHECK:       %permute_OUT_0_pointwise_mul = torch.prim.ListConstruct %permute_OUT_0_val_0_pointwise_mul, %permute_OUT_0_val_1_pointwise_mul, %permute_OUT_0_val_2_pointwise_mul, %permute_OUT_0_val_3_pointwise_mul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_pointwise_mul_perm, %permute_OUT_0_pointwise_mul : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[2,3,128,128],f32>, !torch.tensor<[2,3,128,128],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include "utils.h"
#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace fusilli;

static ErrorObject testPointwiseMulScalarAsmEmitter(const std::string &mode) {
  auto graph = std::make_shared<Graph>();
  graph->setName("pointwise_asm_emitter_mul_scalar");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  std::vector<int64_t> dims = {2, 3, 128, 128};
  auto xT = createTestTensor("arg0", dims, graph.get());

  auto alphaT = graph->tensor(TensorAttr(2.0f));
  alphaT->setName("alpha");

  auto pointwiseAttr = PointwiseAttr()
                           .setMode(PointwiseAttr::Mode::MUL)
                           .setName("pointwise_mul");

  auto yT = graph->pointwise(xT, alphaT, pointwiseAttr);
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

  auto status = testPointwiseMulScalarAsmEmitter(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
