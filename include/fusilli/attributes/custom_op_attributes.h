// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the `CustomOpAttr` class definition for custom operation
// attributes. Unlike standard op attributes that use `AttributesCRTP`, this is
// a standalone class since custom ops have variable-length I/O that doesn't
// fit the enum-keyed map pattern.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_CUSTOM_OP_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_CUSTOM_OP_ATTRIBUTES_H

#include <string>

namespace fusilli {

class CustomOpAttr {
public:
  const std::string &getName() const { return name_; }
  CustomOpAttr &setName(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  /// Returns the MLIR template string (may contain unresolved placeholders).
  const std::string &getMlir() const { return mlir_; }

  /// Sets the MLIR function definition for this custom op.
  ///
  /// The string may contain placeholders that are resolved at emission time
  /// by `CustomOpNode::resolveMlirPlaceholders()`:
  ///
  ///   {FUNC_NAME}   — replaced with the node's unique name (from setName()).
  ///   {IN0_DTYPE}   — replaced with input 0's MLIR element type (e.g., "f32").
  ///   {OUT0_DTYPE}  — replaced with output 0's MLIR element type.
  ///
  /// Example:
  ///   func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>,
  ///                                    %arg1: !torch.vtensor<[?],{IN1_DTYPE}>)
  ///                                    -> !torch.vtensor<[?],{OUT0_DTYPE}> {
  ///     ...
  ///   }
  ///
  /// If no placeholders are present, the string is emitted verbatim.
  CustomOpAttr &setMlir(std::string mlir) {
    mlir_ = std::move(mlir);
    return *this;
  }

  uint64_t getNumOutputs() const { return numOutputs_; }
  CustomOpAttr &setNumOutputs(uint64_t numOutputs) {
    numOutputs_ = numOutputs;
    return *this;
  }

private:
  std::string name_;
  std::string mlir_;
  uint64_t numOutputs_ = 0;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_CUSTOM_OP_ATTRIBUTES_H
