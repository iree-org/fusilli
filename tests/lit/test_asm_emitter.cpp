// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | FileCheck %s

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace fusilli;

static void testGetListOfIntOpsAsm() {
  std::vector<int64_t> vals{1, 2, 3};
  std::string prefix = "stride";
  std::string suffix = "conv";
  std::string asmStr = getListOfIntOpsAsm(vals, prefix, suffix);

  // clang-format off
  // CHECK:  %stride_val_0_conv = torch.constant.int 1
  // CHECK:  %stride_val_1_conv = torch.constant.int 2
  // CHECK:  %stride_val_2_conv = torch.constant.int 3
  // CHECK:  %stride_conv = torch.prim.ListConstruct %stride_val_0_conv, %stride_val_1_conv, %stride_val_2_conv : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // clang-format on
  std::cout << asmStr << std::endl;
}

static void testGetTensorTypeAsm() {
  TensorAttr t1;
  t1.setName("tensor1")
      .setDataType(DataType::Float)
      .setDim({2, 3})
      .setStride({3, 1});

  // CHECK:  !torch.vtensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.vtensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/true)
            << std::endl;

  TensorAttr t2;
  t2.setName("tensor2")
      .setDataType(DataType::Float)
      .setDim({2, 3, 4})
      .setStride({12, 1, 3});

  // CHECK:  !torch.vtensor<[2,4,3],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.tensor<[2,4,3],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.vtensor<[2,3,4],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3,4],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/true)
            << std::endl;

  TensorAttr scalar(2.0f);
  scalar.setName("alpha");

  // CHECK:  !torch.vtensor<[1],f32>
  std::cout << scalar.getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true)
            << std::endl;
}

static void testGetValueNameAsm() {
  TensorAttr t;
  t.setName("foo_Bar::X0").setDataType(DataType::Float).setDim({1});

  // CHECK:  %foo_BarX0
  std::cout << t.getValueNameAsm(/*isOutputAliased=*/false) << std::endl;

  // CHECK:  %foo_BarX0_
  std::cout << t.getValueNameAsm(/*isOutputAliased=*/true) << std::endl;
}

static void testGetScalarConstantAsm() {
  // Float scalar.
  auto floatScalar = std::make_shared<TensorAttr>(2.0f);
  floatScalar->setName("alpha_f32");

  // clang-format off
  // CHECK: %alpha_f32 = torch.vtensor.literal(dense<0x40000000> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  // clang-format on
  std::cout << getScalarConstantAsm(floatScalar) << std::endl;

  // Double scalar.
  auto doubleScalar = std::make_shared<TensorAttr>(3.14);
  doubleScalar->setName("alpha_f64");

  // clang-format off
  // CHECK: %alpha_f64 = torch.vtensor.literal(dense<0x40091EB851EB851F> : tensor<1xf64>) : !torch.vtensor<[1],f64>
  // clang-format on
  std::cout << getScalarConstantAsm(doubleScalar) << std::endl;

  // Int32 scalar.
  auto int32Scalar = std::make_shared<TensorAttr>(int32_t(42));
  int32Scalar->setName("alpha_i32");

  // clang-format off
  // CHECK: %alpha_i32 = torch.vtensor.literal(dense<0x0000002A> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  // clang-format on
  std::cout << getScalarConstantAsm(int32Scalar) << std::endl;

  // Int64 scalar.
  auto int64Scalar = std::make_shared<TensorAttr>(int64_t(-7));
  int64Scalar->setName("alpha_i64");

  // clang-format off
  // CHECK: %alpha_i64 = torch.vtensor.literal(dense<0xFFFFFFFFFFFFFFF9> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
  // clang-format on
  std::cout << getScalarConstantAsm(int64Scalar) << std::endl;
}

static void testBuildTensorTypeStr() {
  // CHECK: !torch.vtensor<[8,2,64,128],f32>
  std::cout << buildTensorTypeStr({8, 2, 64, 128}, DataType::Float)
            << std::endl;

  // CHECK: !torch.vtensor<[1,1,64,128],bf16>
  std::cout << buildTensorTypeStr({1, 1, 64, 128}, DataType::BFloat16)
            << std::endl;

  // CHECK: !torch.tensor<[3,4],f16>
  std::cout << buildTensorTypeStr({3, 4}, DataType::Half,
                                  /*isValueTensor=*/false)
            << std::endl;
}

static void testGetTensorTypeAsmBroadcast() {
  // Broadcast tensor: physical dims should show 1 for broadcast dims.
  TensorAttr t;
  t.setName("Q")
      .setDataType(DataType::BFloat16)
      .setDim({8, 2, 64, 128})
      .setStride({0, 0, 128, 1});

  // Physical dims: broadcast dims collapse to 1
  // CHECK: !torch.vtensor<[1,1,64,128],bf16>
  std::cout << t.getTensorTypeAsm(/*isValueTensor=*/true,
                                  /*useLogicalDims=*/false)
            << std::endl;

  // Logical dims: full shape
  // CHECK: !torch.vtensor<[8,2,64,128],bf16>
  std::cout << t.getTensorTypeAsm(/*isValueTensor=*/true,
                                  /*useLogicalDims=*/true)
            << std::endl;
}

static void testGetLayoutConversionOpsAsmBroadcast() {
  auto tensor = std::make_shared<TensorAttr>();
  tensor->setName("Q")
      .setDataType(DataType::BFloat16)
      .setDim({8, 2, 64, 128})
      .setStride({0, 0, 128, 1});

  // Physical→logical for a broadcast tensor: permute + expand.
  // clang-format off
  // CHECK: %Q_op_i0_perm_unexpanded = torch.aten.permute %Q, %permute_IN_0_op_i0 : !torch.vtensor<[1,1,64,128],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,64,128],bf16>
  // CHECK: %expand_size_permute_IN_0_val_0_op_i0 = torch.constant.int 8
  // CHECK: %expand_size_permute_IN_0_val_1_op_i0 = torch.constant.int 2
  // CHECK: %expand_size_permute_IN_0_val_2_op_i0 = torch.constant.int 64
  // CHECK: %expand_size_permute_IN_0_val_3_op_i0 = torch.constant.int 128
  // CHECK: %expand_size_permute_IN_0_op_i0 = torch.prim.ListConstruct %expand_size_permute_IN_0_val_0_op_i0, %expand_size_permute_IN_0_val_1_op_i0, %expand_size_permute_IN_0_val_2_op_i0, %expand_size_permute_IN_0_val_3_op_i0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %expand_implicit_permute_IN_0_op_i0 = torch.constant.bool false
  // CHECK: %Q_op_i0_perm = torch.aten.expand %Q_op_i0_perm_unexpanded, %expand_size_permute_IN_0_op_i0, %expand_implicit_permute_IN_0_op_i0 : !torch.vtensor<[1,1,64,128],bf16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[8,2,64,128],bf16>
  // clang-format on
  std::cout << getLayoutConversionOpsAsm(tensor, "permute_IN_0", "op_i0",
                                         /*isInput=*/true)
            << std::endl;
}

static void testGetLayoutConversionOpsAsmBroadcastPermuted() {
  // Broadcast + non-trivial permute: inner dims are transposed.
  auto tensor = std::make_shared<TensorAttr>();
  tensor->setName("K")
      .setDataType(DataType::BFloat16)
      .setDim({8, 2, 64, 128})
      .setStride({0, 0, 1, 64});

  // Physical shape is [1,1,128,64] (permuted), unexpanded logical is
  // [1,1,64,128]. The permute swaps dims 2 and 3, then expand fills broadcast.
  // clang-format off
  // CHECK: %K_op_i1_perm_unexpanded = torch.aten.permute %K, %permute_IN_1_op_i1 : !torch.vtensor<[1,1,128,64],bf16>, !torch.list<int> -> !torch.vtensor<[1,1,64,128],bf16>
  // CHECK: %K_op_i1_perm = torch.aten.expand %K_op_i1_perm_unexpanded, %expand_size_permute_IN_1_op_i1, %expand_implicit_permute_IN_1_op_i1 : !torch.vtensor<[1,1,64,128],bf16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[8,2,64,128],bf16>
  // clang-format on
  std::cout << getLayoutConversionOpsAsm(tensor, "permute_IN_1", "op_i1",
                                         /*isInput=*/true)
            << std::endl;
}

int main() {
  testGetListOfIntOpsAsm();
  testGetTensorTypeAsm();
  testGetValueNameAsm();
  testGetScalarConstantAsm();
  testBuildTensorTypeStr();
  testGetTensorTypeAsmBroadcast();
  testGetLayoutConversionOpsAsmBroadcast();
  testGetLayoutConversionOpsAsmBroadcastPermuted();
  return 0;
}
