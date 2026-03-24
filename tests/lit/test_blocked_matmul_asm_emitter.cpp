// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} | iree-compile - \
// RUN:     --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-llvmcpu-target-cpu=host \
// RUN:     --iree-torch-externalize-transients \
// RUN:     --iree-torch-enable-shape-refinement \
// RUN:     --compile-to=flow | \
// RUN:     FileCheck %s --check-prefix=FLOW-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// Logical matmul: [128, 64] @ [64, 256] -> [128, 256]
// Tile sizes: M1=8, N1=8, K1=4
//   LHS logical [M0=16, K0=16, M1=8, K1=4], physical same (contiguous)
//   RHS logical [K0=16, N0=32, K1=4, N1=8], physical [N0=32, K0=16, N1=8, K1=4] (transposed)
//   OUT          [M0=16, N0=32, M1=8, N1=8]
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%{{.+}}: !torch.tensor<[16,32,8,8],f32>, %{{.+}}: !torch.vtensor<[16,16,8,4],f32>, %{{.+}}: !torch.vtensor<[32,16,8,4],f32>)
// TORCH-CHECK:       torch_c.to_builtin_tensor
// TORCH-CHECK:       torch_c.to_builtin_tensor
// TORCH-CHECK:       linalg.mmt4d
// TORCH-CHECK:       torch_c.from_builtin_tensor
// TORCH-CHECK:       torch.overwrite.tensor.contents
//
// FLOW-CHECK:      linalg.mmt4d
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
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

static ErrorObject testBlockedMatmulAsmEmitter(const std::string &mode) {
  // Logical matmul: [128, 64] @ [64, 256] -> [128, 256]
  // Tile sizes: m1=8, n1=8, k1=4
  int64_t m0 = 16, k0 = 16, m1 = 8, k1 = 4;
  int64_t n0 = 32, n1 = 8;

  auto graph = std::make_shared<Graph>();
  graph->setName("blocked_matmul_mmt4d");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // LHS: logical [m0, k0, m1, k1], contiguous (row-major)
  auto lhsT = graph->tensor(TensorAttr()
                                .setName("arg0_lhs")
                                .setDim({m0, k0, m1, k1})
                                .setStride({k0 * m1 * k1, m1 * k1, k1, 1}));

  // RHS: logical [k0, n0, k1, n1], physical [n0, k0, n1, k1] (transposed)
  // Strides encode the physical layout: dim order in memory is [n0, k0, n1, k1]
  // stride[0] (k0) = n1 * k1 (k0 moves within an n0-block)
  // stride[1] (n0) = k0 * n1 * k1 (n0 is outermost)
  // stride[2] (k1) = 1 (k1 is innermost)
  // stride[3] (n1) = k1 (n1 comes before k1 innermost)
  auto rhsT = graph->tensor(TensorAttr()
                                .setName("arg1_rhs")
                                .setDim({k0, n0, k1, n1})
                                .setStride({n1 * k1, k0 * n1 * k1, 1, k1}));

  auto bmAttr = BlockedMatmulAttr().setName("blocked_matmul");

  auto outT = graph->blockedMatmul(lhsT, rhsT, bmAttr);
  outT->setName("result").setOutput(true);

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

  auto status = testBlockedMatmulAsmEmitter(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
