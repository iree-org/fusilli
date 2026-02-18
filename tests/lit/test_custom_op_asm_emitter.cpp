// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies basic custom op ASM emission with placeholder resolution:
//   - {FUNC_NAME} resolved to node name
//   - {IN0_DTYPE}, {IN1_DTYPE}, {OUT0_DTYPE} resolved to f32
//   - Module-scope function definition present
//   - func.call to the custom function
//   - Static-to-dynamic casts and output overwrite

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @my_add(%arg0: !torch.vtensor<[?],f32>,
// CHECK:                                    %arg1: !torch.vtensor<[?],f32>)
// CHECK:                                    -> !torch.vtensor<[?],f32> {
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
// CHECK:               : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int
// CHECK:               -> !torch.vtensor<[?],f32>
// CHECK:           return %0 : !torch.vtensor<[?],f32>
// CHECK:         }
// CHECK:         func.func @main(
// CHECK-SAME:      %{{.*}}: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f32>
// CHECK-SAME:      %b: !torch.vtensor<[4],f32>
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[4],f32> to !torch.vtensor<[?],f32>
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[4],f32> to !torch.vtensor<[?],f32>
// CHECK:           %{{.*}} = func.call @my_add(%{{.*}}, %{{.*}}) : (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32>
// CHECK:           %{{.*}} = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
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
  g.setName("custom_op_asm_emitter").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b =
      g.tensor(TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  std::string addMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>,
                                   %arg1: !torch.vtensor<[?],{IN1_DTYPE}>)
                                   -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : !torch.vtensor<[?],{IN0_DTYPE}>, !torch.vtensor<[?],{IN1_DTYPE}>, !torch.int
        -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";

  CustomOpAttr addAttr;
  addAttr.setName("my_add").setMlir(addMlir).setNumOutputs(1);

  auto outs = g.customOp(addAttr, a, b);
  outs[0]
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
