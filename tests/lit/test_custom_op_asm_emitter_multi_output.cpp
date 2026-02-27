// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies multi-output custom op ASM emission:
//   - Module-scope function with multiple return types
//   - %name:N multi-result func.call syntax
//   - #0 and #1 result indexing in dynamic-to-static casts

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @my_split(%arg0: !torch.vtensor<[?],f32>)
// CHECK:             -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) {
// CHECK:           return %arg0, %arg0 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>
// CHECK:         }
// CHECK:         func.func @main(
// CHECK-SAME:      %{{[^:]*}}: !torch.tensor<[4],f32>
// CHECK-SAME:      %{{[^:]*}}: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f32>
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[4],f32> to !torch.vtensor<[?],f32>
// CHECK:           %{{.*}}:2 = func.call @my_split(%{{.*}}) : (!torch.vtensor<[?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>)
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}}#0 : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}}#1 : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
// CHECK:           torch.overwrite.tensor.contents %{{.*}} overwrites %{{.*}}
// CHECK:           torch.overwrite.tensor.contents %{{.*}} overwrites %{{.*}}
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include <iostream>
#include <string>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("custom_op_asm_emitter_multi_output")
      .setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  std::string splitMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>)
      -> (!torch.vtensor<[?],{OUT0_DTYPE}>, !torch.vtensor<[?],{OUT1_DTYPE}>) {
    return %arg0, %arg0 : !torch.vtensor<[?],{OUT0_DTYPE}>, !torch.vtensor<[?],{OUT1_DTYPE}>
  }
)";

  CustomOpAttr splitAttr;
  splitAttr.setName("my_split").setMlir(splitMlir).setNumOutputs(2);

  auto outs = g.customOp({a}, splitAttr);
  outs[0]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);
  outs[1]
      ->setDim({4})
      .setStride({1})
      .setDataType(DataType::Float)
      .setOutput(true);

  auto status = g.validate();
  if (isError(status)) {
    std::cerr << "Validation failed: " << status << std::endl;
    return 1;
  }

  auto asmOrErr = g.emitAsm();
  if (isError(asmOrErr)) {
    std::cerr << "ASM emission failed: " << asmOrErr << std::endl;
    return 1;
  }

  std::cout << *asmOrErr << std::endl;
  return 0;
}
