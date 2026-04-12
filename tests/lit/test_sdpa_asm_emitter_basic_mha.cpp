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
// TORCH-CHECK:     func.func @main(%sdpa_O_: !torch.tensor<[1,8,64,64],f16>, %k: !torch.vtensor<[1,8,64,64],f16>, %q: !torch.vtensor<[1,8,64,64],f16>, %v: !torch.vtensor<[1,8,64,64],f16>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_Q_val_0_sdpa = torch.constant.int 0
// TORCH-CHECK:       %permute_Q_val_1_sdpa = torch.constant.int 1
// TORCH-CHECK:       %permute_Q_val_2_sdpa = torch.constant.int 2
// TORCH-CHECK:       %permute_Q_val_3_sdpa = torch.constant.int 3
// TORCH-CHECK:       %permute_Q_sdpa = torch.prim.ListConstruct %permute_Q_val_0_sdpa, %permute_Q_val_1_sdpa, %permute_Q_val_2_sdpa, %permute_Q_val_3_sdpa : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %q_sdpa_perm = torch.aten.permute %q, %permute_Q_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:       %permute_K_val_0_sdpa = torch.constant.int 0
// TORCH-CHECK:       %permute_K_val_1_sdpa = torch.constant.int 1
// TORCH-CHECK:       %permute_K_val_2_sdpa = torch.constant.int 2
// TORCH-CHECK:       %permute_K_val_3_sdpa = torch.constant.int 3
// TORCH-CHECK:       %permute_K_sdpa = torch.prim.ListConstruct %permute_K_val_0_sdpa, %permute_K_val_1_sdpa, %permute_K_val_2_sdpa, %permute_K_val_3_sdpa : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %k_sdpa_perm = torch.aten.permute %k, %permute_K_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:       %permute_V_val_0_sdpa = torch.constant.int 0
// TORCH-CHECK:       %permute_V_val_1_sdpa = torch.constant.int 1
// TORCH-CHECK:       %permute_V_val_2_sdpa = torch.constant.int 2
// TORCH-CHECK:       %permute_V_val_3_sdpa = torch.constant.int 3
// TORCH-CHECK:       %permute_V_sdpa = torch.prim.ListConstruct %permute_V_val_0_sdpa, %permute_V_val_1_sdpa, %permute_V_val_2_sdpa, %permute_V_val_3_sdpa : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %v_sdpa_perm = torch.aten.permute %v, %permute_V_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:       %none_mask_sdpa = torch.constant.none
// TORCH-CHECK:       %dropout_sdpa = torch.constant.float 0.000000e+00
// TORCH-CHECK:       %is_causal_sdpa = torch.constant.bool false
// TORCH-CHECK:       %scale_sdpa = torch.constant.none
// TORCH-CHECK:       %enable_gqa_sdpa = torch.constant.bool false
// TORCH-CHECK:       %sdpa_O_sdpa_perm = torch.aten.scaled_dot_product_attention %q_sdpa_perm, %k_sdpa_perm, %v_sdpa_perm, %none_mask_sdpa, %dropout_sdpa, %is_causal_sdpa, %scale_sdpa, %enable_gqa_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>, !torch.vtensor<[1,8,64,64],f16>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:       %permute_O_val_0_sdpa = torch.constant.int 0
// TORCH-CHECK:       %permute_O_val_1_sdpa = torch.constant.int 1
// TORCH-CHECK:       %permute_O_val_2_sdpa = torch.constant.int 2
// TORCH-CHECK:       %permute_O_val_3_sdpa = torch.constant.int 3
// TORCH-CHECK:       %permute_O_sdpa = torch.prim.ListConstruct %permute_O_val_0_sdpa, %permute_O_val_1_sdpa, %permute_O_val_2_sdpa, %permute_O_val_3_sdpa : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %sdpa_O = torch.aten.permute %sdpa_O_sdpa_perm, %permute_O_sdpa : !torch.vtensor<[1,8,64,64],f16>, !torch.list<int> -> !torch.vtensor<[1,8,64,64],f16>
// TORCH-CHECK:       torch.overwrite.tensor.contents %sdpa_O overwrites %sdpa_O_ : !torch.vtensor<[1,8,64,64],f16>, !torch.tensor<[1,8,64,64],f16>
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
#include <vector>

using namespace fusilli;

static ErrorObject testSdpaAsmEmitterBasicMha(const std::string &mode) {
  auto graph = std::make_shared<Graph>();
  graph->setName("sdpa_asm_emitter_basic_mha").setIODataType(DataType::Half);

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

  auto sdpaAttr = SdpaAttr().setName("sdpa");
  auto o = graph->sdpa(q, k, v, /*mask=*/nullptr, sdpaAttr);
  o->setDim(dim).setStride(stride).setDataType(DataType::Half).setOutput(true);

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

  auto status = testSdpaAsmEmitterBasicMha(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
