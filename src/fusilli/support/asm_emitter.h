// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains declarations for all the MLIR assembly generation methods
// on the `Graph`, `TensorAttr`, `INode` and derived node classes. It is meant
// to be a common place for all things ASM emitter related to make maintenance
// and future improvements easier.
//
// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from C++20) to implement a simple templating system for generating MLIR
// assembly code. This could be made better with a jinja2-like templating
// system but for now this gets us mostly what we need.
//
// Caution: An important foot-gun with `std::format` is to forget to double the
// brace for a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}`
// to disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_ASM_EMITTER_H
#define FUSILLI_SUPPORT_ASM_EMITTER_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/graph.h"

#include <memory>
#include <string>
#include <vector>

namespace fusilli {

// Given a vector of ints, returns the MLIR assembly for the
// `torch.constant.int` ops for each int value and the
// `torch.prim.ListConstruct` op wrapping these into a single
// value.
std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                               const std::string &prefix,
                               const std::string &suffix);

// Emits permute ops for a tensor in MLIR assembly format.
std::string getPermuteOpsAsm(const std::shared_ptr<TensorAttr> &tensor,
                             const std::string &prefix,
                             const std::string &suffix, bool isInput);

// Emits a scalar TensorAttr as a constant tensor literal in MLIR assembly.
std::string getScalarConstantAsm(const std::shared_ptr<TensorAttr> &tensor);

} // namespace fusilli

#endif // FUSILLI_SUPPORT_ASM_EMITTER_H
