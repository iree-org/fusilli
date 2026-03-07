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
  // Setters:
  CustomOpAttr &setName(const std::string &name) {
    name_ = name;
    return *this;
  }

  // Sets the MLIR function definition for this custom op.
  //
  // IMPORTANT: The MLIR function must be written in terms of logical
  // dimensions, not physical (storage) dimensions. The emitter automatically
  // inserts permute ops around the custom op call to convert between the
  // physical layout of graph tensors and the logical layout expected by the
  // custom function. For example, if a tensor has dim={4,8} with transposed
  // stride={1,4} (physical layout [8,4]), the emitter permutes it to logical
  // [4,8] before passing it to the custom function. The custom MLIR should
  // therefore use [4,8], not [8,4].
  //
  // The string may contain placeholders that are resolved at emission time
  // by `CustomOpNode::resolveMlirPlaceholders()`:
  //
  //   {FUNC_NAME}   — replaced with the node's unique name (from setName()).
  //   {IN0_DTYPE}   — replaced with input 0's MLIR element type (e.g., "f32").
  //   {OUT0_DTYPE}  — replaced with output 0's MLIR element type.
  //
  // By default, shapes in the MLIR should use dynamic placeholders [?] so
  // the function works for any tensor size. If the MLIR uses concrete static
  // shapes instead (e.g., [4,8]), call `setIsStatic(true)` so the emitter
  // uses matching static types in the surrounding casts and func.call
  // signature.
  //
  // Example (dynamic):
  //   func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[?],{IN0_DTYPE}>,
  //                                    %arg1: !torch.vtensor<[?],{IN1_DTYPE}>)
  //                                    -> !torch.vtensor<[?],{OUT0_DTYPE}> {
  //     ...
  //   }
  //
  // Example (static, requires setIsStatic(true)):
  //   func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[4,8],{IN0_DTYPE}>)
  //                                    -> !torch.vtensor<[4,8],{OUT0_DTYPE}> {
  //     ...
  //   }
  //
  // If no placeholders are present, the string is emitted verbatim.
  CustomOpAttr &setMlir(const std::string &mlir) {
    mlir_ = mlir;
    return *this;
  }

  CustomOpAttr &setNumOutputs(size_t numOutputs) {
    numOutputs_ = numOutputs;
    return *this;
  }

  // When true, the user-provided MLIR already has static shapes baked in.
  // The emitter will use static types in casts and func.call signatures,
  // making the casts identity no-ops that the compiler eliminates.
  CustomOpAttr &setIsStatic(bool isStatic) {
    isStatic_ = isStatic;
    return *this;
  }

  // Getters:
  const std::string &getName() const { return name_; }

  // Returns the MLIR template string (may contain unresolved placeholders).
  const std::string &getMlir() const { return mlir_; }

  size_t getNumOutputs() const { return numOutputs_; }

  bool isStatic() const { return isStatic_; }

private:
  std::string name_;
  std::string mlir_;
  size_t numOutputs_ = 0;
  bool isStatic_ = false;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_CUSTOM_OP_ATTRIBUTES_H
