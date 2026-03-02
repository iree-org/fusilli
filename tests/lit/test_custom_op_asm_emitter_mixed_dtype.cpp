// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies mixed dtype custom op ASM emission with two chained ops:
//   - identity_f16: f16 input, f16 output (passthrough)
//   - cast_to_f32: f16 input, f32 output (dtype change via
//   convert_element_type)
//   - Dtype placeholders correctly resolved for each op
//   - Multiple module-scope definitions properly separated

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @identity_f16(%arg0: !torch.vtensor<[?],f16>)
// CHECK:             -> !torch.vtensor<[?],f16> {
// CHECK:           return %arg0 : !torch.vtensor<[?],f16>
// CHECK:         }
// CHECK:         func.func private @cast_to_f32(%arg0: !torch.vtensor<[?],f16>)
// CHECK:             -> !torch.vtensor<[?],f32> {
// CHECK:           torch.prims.convert_element_type
// CHECK:           return %{{.*}} : !torch.vtensor<[?],f32>
// CHECK:         }
// CHECK:         func.func @main(
// CHECK-SAME:      %{{.*}}: !torch.tensor<[4],f32>
// CHECK-SAME:      %a: !torch.vtensor<[4],f16>
// CHECK:           func.call @identity_f16
// CHECK:           func.call @cast_to_f32
// CHECK:           torch.overwrite.tensor.contents %{{.*}} overwrites %{{.*}}
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <iostream>
#include <string>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("custom_op_asm_emitter_mixed_dtype").setIODataType(DataType::Float);

  auto a =
      g.tensor(TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Half));

  std::string identityMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>)
      -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    return %arg0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";

  std::string castMlir = R"(
  func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>)
      -> !torch.vtensor<[?],{OUT0_DTYPE}> {
    %int6 = torch.constant.int 6
    %0 = torch.prims.convert_element_type %arg0, %int6
        : !torch.vtensor<[?],{IN0_DTYPE}>, !torch.int -> !torch.vtensor<[?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?],{OUT0_DTYPE}>
  }
)";

  // First op: identity_f16 (f16 -> f16)
  CustomOpAttr identityAttr;
  identityAttr.setName("identity_f16").setMlir(identityMlir).setNumOutputs(1);

  auto outs1 = g.customOp({a}, identityAttr);
  outs1[0]->setDim({4}).setStride({1}).setDataType(DataType::Half);

  // Second op: cast_to_f32 (f16 -> f32)
  CustomOpAttr castAttr;
  castAttr.setName("cast_to_f32").setMlir(castMlir).setNumOutputs(1);

  auto outs2 = g.customOp({outs1[0]}, castAttr);
  outs2[0]
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

  auto indentErr = checkMlirIndentation(*asmOrErr);
  if (isError(indentErr)) {
    std::cerr << "Indentation check failed: " << indentErr << std::endl;
    return 1;
  }

  std::cout << *asmOrErr << std::endl;
  return 0;
}
