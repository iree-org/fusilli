// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// TODO: Enable after torch.aten.native_rms_norm is supported by IREE.
// disabled: %{TEST_EXE} | iree-opt --verify-roundtrip
// disabled: %{TEST_EXE} stats | FileCheck %s
// --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%inv_rms_: !torch.tensor<[16,1,1,1],f32>, %y_: !torch.tensor<[16,64,32,128],f32>, %arg0_scale: !torch.vtensor<[1,64,32,128],f32>, %arg0_x: !torch.vtensor<[16,64,32,128],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// Graph-level scalar constant emission for epsilon:
// TORCH-CHECK:       %rmsnorm_train_EPSILON = torch.vtensor.literal(dense<0x3727C5AC> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// TORCH-CHECK:       %normalized_shape_val_0_rmsnorm_train = torch.constant.int 128
// TORCH-CHECK:       %normalized_shape_val_1_rmsnorm_train = torch.constant.int 64
// TORCH-CHECK:       %normalized_shape_val_2_rmsnorm_train = torch.constant.int 32
// TORCH-CHECK:       %normalized_shape_rmsnorm_train = torch.prim.ListConstruct %normalized_shape_val_0_rmsnorm_train, %normalized_shape_val_1_rmsnorm_train, %normalized_shape_val_2_rmsnorm_train : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %eps_rmsnorm_train = torch.aten.item %rmsnorm_train_EPSILON : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %permute_x_val_0_rmsnorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_x_val_1_rmsnorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_x_val_2_rmsnorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_x_val_3_rmsnorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_x_rmsnorm_train = torch.prim.ListConstruct %permute_x_val_0_rmsnorm_train, %permute_x_val_1_rmsnorm_train, %permute_x_val_2_rmsnorm_train, %permute_x_val_3_rmsnorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_x_rmsnorm_train_perm = torch.aten.permute %arg0_x, %permute_x_rmsnorm_train : !torch.vtensor<[16,64,32,128],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %permute_scale_val_0_rmsnorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_scale_val_1_rmsnorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_scale_val_2_rmsnorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_scale_val_3_rmsnorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_scale_rmsnorm_train = torch.prim.ListConstruct %permute_scale_val_0_rmsnorm_train, %permute_scale_val_1_rmsnorm_train, %permute_scale_val_2_rmsnorm_train, %permute_scale_val_3_rmsnorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_scale_rmsnorm_train_perm = torch.aten.permute %arg0_scale, %permute_scale_rmsnorm_train : !torch.vtensor<[1,64,32,128],f32>, !torch.list<int> -> !torch.vtensor<[1,128,64,32],f32>
// TORCH-CHECK:       %y_rmsnorm_train_perm, %inv_rms_rmsnorm_train_perm = torch.aten.native_rms_norm %arg0_x_rmsnorm_train_perm, %normalized_shape_rmsnorm_train, %arg0_scale_rmsnorm_train_perm, %eps_rmsnorm_train : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int>, !torch.vtensor<[1,128,64,32],f32>, !torch.float -> !torch.vtensor<[16,128,64,32],f32>, !torch.vtensor<[16,1,1,1],f32>
// TORCH-CHECK:       %permute_y_val_0_rmsnorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_y_val_1_rmsnorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_y_val_2_rmsnorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_y_val_3_rmsnorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_y_rmsnorm_train = torch.prim.ListConstruct %permute_y_val_0_rmsnorm_train, %permute_y_val_1_rmsnorm_train, %permute_y_val_2_rmsnorm_train, %permute_y_val_3_rmsnorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %y = torch.aten.permute %y_rmsnorm_train_perm, %permute_y_rmsnorm_train : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,64,32,128],f32>
// TORCH-CHECK:       %permute_inv_rms_val_0_rmsnorm_train = torch.constant.int 0
// TORCH-CHECK:       %permute_inv_rms_val_1_rmsnorm_train = torch.constant.int 1
// TORCH-CHECK:       %permute_inv_rms_val_2_rmsnorm_train = torch.constant.int 2
// TORCH-CHECK:       %permute_inv_rms_val_3_rmsnorm_train = torch.constant.int 3
// TORCH-CHECK:       %permute_inv_rms_rmsnorm_train = torch.prim.ListConstruct %permute_inv_rms_val_0_rmsnorm_train, %permute_inv_rms_val_1_rmsnorm_train, %permute_inv_rms_val_2_rmsnorm_train, %permute_inv_rms_val_3_rmsnorm_train : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %inv_rms = torch.aten.permute %inv_rms_rmsnorm_train_perm, %permute_inv_rms_rmsnorm_train : !torch.vtensor<[16,1,1,1],f32>, !torch.list<int> -> !torch.vtensor<[16,1,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %inv_rms overwrites %inv_rms_ : !torch.vtensor<[16,1,1,1],f32>, !torch.tensor<[16,1,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %y overwrites %y_ : !torch.vtensor<[16,64,32,128],f32>, !torch.tensor<[16,64,32,128],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
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
testRmsnormTrainAsmEmitterScaleNhwc(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32;
  auto graph = std::make_shared<Graph>();
  graph->setName("rmsnorm_train_asm_emitter_scale_nhwc");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto scaleT = graph->tensor(TensorAttr().setName("arg0_scale"));

  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto rmsnormAttr = RmsnormAttr()
                         .setForwardPhase(NormFwdPhase::TRAINING)
                         .setEpsilon(epsilonT)
                         .setName("rmsnorm_train");

  auto [yT, rT] = graph->rmsnorm(xT, scaleT, rmsnormAttr);

  yT->setName("y").setDataType(DataType::Float).setOutput(true);
  rT->setName("inv_rms").setDataType(DataType::Float).setOutput(true);

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

  auto status = testRmsnormTrainAsmEmitterScaleNhwc(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
