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
  // The MLIR string must define a `func.func` whose signature matches the
  // `func.call` the emitter generates:
  //
  //   Name:    @<setName()>
  //   Inputs:  static logical value tensor types
  //              (e.g., !torch.vtensor<[4,8],f32>)
  //   Outputs: static logical value tensor types
  //
  // The following placeholders are resolved at emission time by
  // `CustomOpNode::resolveMlirPlaceholders()` and can be used to build a
  // matching signature without hardcoding values:
  //
  //   {FUNC_NAME}  — the node name (from setName())
  //   {IN0_DTYPE}  — input 0's element type (e.g., "f32")
  //   {OUT0_DTYPE} — output 0's element type
  //   {IN0_TYPE}   — input 0's full tensor type
  //                   (e.g., "!torch.vtensor<[4,8],f32>")
  //   {OUT0_TYPE}  — output 0's full tensor type
  //   {IN0_DIM0}   — input 0's logical dimension 0 (e.g., "4")
  //   {OUT0_DIM0}  — output 0's logical dimension 0
  //
  // All indices are 0-based and generalize to any input/output/dimension
  // count (e.g., {IN2_TYPE}, {OUT1_DIM3}).
  //
  // Example using placeholders:
  //   func.func private @{FUNC_NAME}(%arg0: {IN0_TYPE},
  //                                    %arg1: {IN1_TYPE})
  //                                    -> {OUT0_TYPE} {
  //     ...
  //   }
  //
  // Example with hardcoded static types (no placeholders except name):
  //   func.func private @{FUNC_NAME}(%arg0: !torch.vtensor<[4,8],f32>,
  //                                    %arg1: !torch.vtensor<[4,8],f32>)
  //                                    -> !torch.vtensor<[4,8],f32> {
  //     ...
  //   }
  CustomOpAttr &setMlir(const std::string &mlir) {
    mlir_ = mlir;
    return *this;
  }

  CustomOpAttr &setNumOutputs(size_t numOutputs) {
    numOutputs_ = numOutputs;
    return *this;
  }

  // Getters:
  const std::string &getName() const { return name_; }

  // Returns the MLIR template string (may contain unresolved placeholders).
  const std::string &getMlir() const { return mlir_; }

  size_t getNumOutputs() const { return numOutputs_; }

private:
  std::string name_;
  std::string mlir_;
  size_t numOutputs_ = 0;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_CUSTOM_OP_ATTRIBUTES_H
