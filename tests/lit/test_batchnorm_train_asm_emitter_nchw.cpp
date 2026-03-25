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
// TORCH-CHECK:     func.func @main(%batchnorm_train_SAVED_INV_VARIANCE_: !torch.tensor<[16],f32>, %batchnorm_train_SAVED_MEAN_: !torch.tensor<[16],f32>, %batchnorm_train_Y_: !torch.tensor<[4,16,8,8],f32>, %batchnorm_train_BIAS: !torch.vtensor<[16],f32>, %batchnorm_train_SCALE: !torch.vtensor<[16],f32>, %batchnorm_train_X: !torch.vtensor<[4,16,8,8],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %batchnorm_train_EPSILON = torch.vtensor.literal(dense<
// TORCH-CHECK:       %batchnorm_train_MOMENTUM = torch.vtensor.literal(dense<
// TORCH-CHECK:       %eps_batchnorm_train = torch.aten.item %batchnorm_train_EPSILON : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %momentum_batchnorm_train = torch.aten.item %batchnorm_train_MOMENTUM : !torch.vtensor<[1],f32> -> !torch.float
// TORCH-CHECK:       %batchnorm_train_X_batchnorm_train_perm = torch.aten.permute %batchnorm_train_X, %permute_x_batchnorm_train : !torch.vtensor<[4,16,8,8],f32>, !torch.list<int> -> !torch.vtensor<[4,16,8,8],f32>
// TORCH-CHECK:       %none_mean_batchnorm_train = torch.constant.none
// TORCH-CHECK:       %none_var_batchnorm_train = torch.constant.none
// TORCH-CHECK:       %training_batchnorm_train = torch.constant.bool true
// TORCH-CHECK:       %batchnorm_train_Y_batchnorm_train_perm, %batchnorm_train_SAVED_MEAN_batchnorm_train_perm, %batchnorm_train_SAVED_INV_VARIANCE_batchnorm_train_perm = torch.aten.native_batch_norm %batchnorm_train_X_batchnorm_train_perm, %batchnorm_train_SCALE, %batchnorm_train_BIAS, %none_mean_batchnorm_train, %none_var_batchnorm_train, %training_batchnorm_train, %momentum_batchnorm_train, %eps_batchnorm_train : !torch.vtensor<[4,16,8,8],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>, !torch.none, !torch.none, !torch.bool, !torch.float, !torch.float -> !torch.vtensor<[4,16,8,8],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],f32>
// TORCH-CHECK:       %batchnorm_train_Y = torch.aten.permute %batchnorm_train_Y_batchnorm_train_perm
// TORCH-CHECK:       %batchnorm_train_SAVED_MEAN = torch.aten.permute %batchnorm_train_SAVED_MEAN_batchnorm_train_perm
// TORCH-CHECK:       %batchnorm_train_SAVED_INV_VARIANCE = torch.aten.permute %batchnorm_train_SAVED_INV_VARIANCE_batchnorm_train_perm
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_train_SAVED_INV_VARIANCE overwrites %batchnorm_train_SAVED_INV_VARIANCE_ : !torch.vtensor<[16],f32>, !torch.tensor<[16],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_train_SAVED_MEAN overwrites %batchnorm_train_SAVED_MEAN_ : !torch.vtensor<[16],f32>, !torch.tensor<[16],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_train_Y overwrites %batchnorm_train_Y_ : !torch.vtensor<[4,16,8,8],f32>, !torch.tensor<[4,16,8,8],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "transient-memory-size": 0
// AMDGPU-STATS-CHECK: "dispatch-count": 6
// CPU-STATS-CHECK: "transient-memory-size": 0
// CPU-STATS-CHECK: "dispatch-count": 4
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testBatchnormTrainAsmEmitterNchw(const std::string &mode) {
  int64_t n = 4, c = 16, h = 8, w = 8;
  auto graph = std::make_shared<Graph>();
  graph->setName("batchnorm_train_asm_emitter_nchw");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("batchnorm_train_X")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto scaleT = graph->tensor(
      TensorAttr().setName("batchnorm_train_SCALE").setDim({c}).setStride({1}));

  auto biasT = graph->tensor(
      TensorAttr().setName("batchnorm_train_BIAS").setDim({c}).setStride({1}));

  auto epsilonT =
      graph->tensor(TensorAttr(1e-5f).setName("batchnorm_train_EPSILON"));
  auto momentumT =
      graph->tensor(TensorAttr(0.1f).setName("batchnorm_train_MOMENTUM"));

  auto batchnormAttr = BatchnormAttr()
                           .setForwardPhase(NormFwdPhase::TRAINING)
                           .setEpsilon(epsilonT)
                           .setMomentum(momentumT)
                           .setName("batchnorm_train");

  // Training: no running stats, scale and bias provided.
  auto [yT, smT, sivT] =
      graph->batchnorm(xT, scaleT, biasT, nullptr, nullptr, batchnormAttr);

  yT->setName("batchnorm_train_Y").setDataType(DataType::Float).setOutput(true);
  smT->setName("batchnorm_train_SAVED_MEAN")
      .setDataType(DataType::Float)
      .setOutput(true);
  sivT->setName("batchnorm_train_SAVED_INV_VARIANCE")
      .setDataType(DataType::Float)
      .setOutput(true);

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

  auto status = testBatchnormTrainAsmEmitterNchw(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
