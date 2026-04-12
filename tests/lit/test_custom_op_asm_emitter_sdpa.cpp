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
// TORCH-CHECK:       module @module {
// TORCH-CHECK:         func.func private @sdpa(
// TORCH-CHECK:             %arg0: !torch.vtensor<[1,8,64,64],f16>,
// TORCH-CHECK:             %arg1: !torch.vtensor<[1,8,64,64],f16>,
// TORCH-CHECK:             %arg2: !torch.vtensor<[1,8,64,64],f16>)
// TORCH-CHECK:             -> !torch.vtensor<[1,8,64,64],f16> {
// TORCH-CHECK:           %none_mask = torch.constant.none
// TORCH-CHECK:           %dropout = torch.constant.float 0.000000e+00
// TORCH-CHECK:           %is_causal = torch.constant.bool false
// TORCH-CHECK:           %scale = torch.constant.none
// TORCH-CHECK:           %enable_gqa = torch.constant.bool false
// TORCH-CHECK:           %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
// TORCH-CHECK:               %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
// TORCH-CHECK:               !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>,
// TORCH-CHECK:               !torch.vtensor<[1,8,64,64],f16>, !torch.none, !torch.float, !torch.bool,
// TORCH-CHECK:               !torch.none, !torch.bool -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           return %0 : !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:         }
//
//
// TORCH-CHECK:         func.func @main(%sdpa_OUT_0_: !torch.tensor<[1,8,64,64],f16>, %k: !torch.vtensor<[1,8,64,64],f16>, %q: !torch.vtensor<[1,8,64,64],f16>, %v: !torch.vtensor<[1,8,64,64],f16>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:           %permute_IN_0_val_0_sdpa_i0 = torch.constant.int 0
// TORCH-CHECK:           %permute_IN_0_val_1_sdpa_i0 = torch.constant.int 1
// TORCH-CHECK:           %permute_IN_0_val_2_sdpa_i0 = torch.constant.int 2
// TORCH-CHECK:           %permute_IN_0_val_3_sdpa_i0 = torch.constant.int 3
// TORCH-CHECK:           %permute_IN_0_sdpa_i0 = torch.prim.ListConstruct %permute_IN_0_val_0_sdpa_i0, %permute_IN_0_val_1_sdpa_i0, %permute_IN_0_val_2_sdpa_i0, %permute_IN_0_val_3_sdpa_i0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:           %q_sdpa_i0_perm = torch.aten.permute %q, %permute_IN_0_sdpa_i0 : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           %permute_IN_1_val_0_sdpa_i1 = torch.constant.int 0
// TORCH-CHECK:           %permute_IN_1_val_1_sdpa_i1 = torch.constant.int 1
// TORCH-CHECK:           %permute_IN_1_val_2_sdpa_i1 = torch.constant.int 2
// TORCH-CHECK:           %permute_IN_1_val_3_sdpa_i1 = torch.constant.int 3
// TORCH-CHECK:           %permute_IN_1_sdpa_i1 = torch.prim.ListConstruct %permute_IN_1_val_0_sdpa_i1, %permute_IN_1_val_1_sdpa_i1, %permute_IN_1_val_2_sdpa_i1, %permute_IN_1_val_3_sdpa_i1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:           %k_sdpa_i1_perm = torch.aten.permute %k, %permute_IN_1_sdpa_i1 : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           %permute_IN_2_val_0_sdpa_i2 = torch.constant.int 0
// TORCH-CHECK:           %permute_IN_2_val_1_sdpa_i2 = torch.constant.int 1
// TORCH-CHECK:           %permute_IN_2_val_2_sdpa_i2 = torch.constant.int 2
// TORCH-CHECK:           %permute_IN_2_val_3_sdpa_i2 = torch.constant.int 3
// TORCH-CHECK:           %permute_IN_2_sdpa_i2 = torch.prim.ListConstruct %permute_IN_2_val_0_sdpa_i2, %permute_IN_2_val_1_sdpa_i2, %permute_IN_2_val_2_sdpa_i2, %permute_IN_2_val_3_sdpa_i2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:           %v_sdpa_i2_perm = torch.aten.permute %v, %permute_IN_2_sdpa_i2 : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           %sdpa_OUT_0_sdpa_perm = func.call @sdpa(%q_sdpa_i0_perm, %k_sdpa_i1_perm, %v_sdpa_i2_perm) : (!torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>) -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           %permute_OUT_0_val_0_sdpa = torch.constant.int 0
// TORCH-CHECK:           %permute_OUT_0_val_1_sdpa = torch.constant.int 1
// TORCH-CHECK:           %permute_OUT_0_val_2_sdpa = torch.constant.int 2
// TORCH-CHECK:           %permute_OUT_0_val_3_sdpa = torch.constant.int 3
// TORCH-CHECK:           %permute_OUT_0_sdpa = torch.prim.ListConstruct %permute_OUT_0_val_0_sdpa, %permute_OUT_0_val_1_sdpa, %permute_OUT_0_val_2_sdpa, %permute_OUT_0_val_3_sdpa : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:           %sdpa_OUT_0 = torch.aten.permute %sdpa_OUT_0_sdpa_perm, %permute_OUT_0_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:           torch.overwrite.tensor.contents %sdpa_OUT_0 overwrites %sdpa_OUT_0_ : !torch.vtensor<[1,8,64,64],f16>, !torch.tensor<[1,8,64,64],f16>
// TORCH-CHECK:           return
// TORCH-CHECK:         }
// TORCH-CHECK:       }
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
#include <vector>

using namespace fusilli;

static ErrorObject testCustomOpAsmEmitterSdpa(const std::string &mode) {
  auto graph = std::make_shared<Graph>();
  graph->setName("custom_op_asm_emitter_sdpa").setIODataType(DataType::Half);

  std::vector<int64_t> dim = {1, 8, 64, 64};
  auto stride =
      generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));

  auto q = graph->tensor(
      TensorAttr().setName("q").setDim(dim).setStride(stride).setDataType(
          DataType::Half));
  auto k = graph->tensor(
      TensorAttr().setName("k").setDim(dim).setStride(stride).setDataType(
          DataType::Half));
  auto v = graph->tensor(
      TensorAttr().setName("v").setDim(dim).setStride(stride).setDataType(
          DataType::Half));

  std::string sdpaMlir = R"mlir(
  func.func private @{FUNC_NAME}(
      %arg0: {IN0_TYPE},
      %arg1: {IN1_TYPE},
      %arg2: {IN2_TYPE})
      -> {OUT0_TYPE} {
    %none_mask = torch.constant.none
    %dropout = torch.constant.float 0.000000e+00
    %is_causal = torch.constant.bool false
    %scale = torch.constant.none
    %enable_gqa = torch.constant.bool false
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        {IN0_TYPE}, {IN1_TYPE},
        {IN2_TYPE}, !torch.none, !torch.float, !torch.bool,
        !torch.none, !torch.bool -> {OUT0_TYPE}
    return %0 : {OUT0_TYPE}
  }
)mlir";

  CustomOpAttr sdpaAttr;
  sdpaAttr.setName("sdpa").setMlir(sdpaMlir).setNumOutputs(1);

  auto outs = graph->customOp({q, k, v}, sdpaAttr);
  outs[0]
      ->setDim(dim)
      .setStride(stride)
      .setDataType(DataType::Half)
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

  auto status = testCustomOpAsmEmitterSdpa(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
