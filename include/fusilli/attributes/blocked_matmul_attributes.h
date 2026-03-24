// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// blocked matmul nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_BLOCKED_MATMUL_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_BLOCKED_MATMUL_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace fusilli {

class BlockedMatmulAttr : public AttributesCRTP<BlockedMatmulAttr> {
public:
  // Names for Tensor Inputs and Outputs.
  enum class InputNames : uint8_t { LHS, RHS };
  enum class OutputNames : uint8_t { RESULT };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BlockedMatmulAttr, InputNames, LHS)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BlockedMatmulAttr, InputNames, RHS)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(BlockedMatmulAttr, OutputNames, RESULT)

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, LHS)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, RHS)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, RESULT)
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_BLOCKED_MATMUL_ATTRIBUTES_H
