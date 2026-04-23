// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the MLIR assembly
// generation methods on the `Graph`, `TensorAttr`, `INode` and derived node
// classes. It is meant to be a common place for all things ASM emitter related
// to make maintenance and future improvements easier.
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
#include "fusilli/attributes/types.h"
#include "fusilli/external/torch_types.h"
#include "fusilli/graph/graph.h"
#include "fusilli/node/batchnorm_node.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/custom_op_node.h"
#include "fusilli/node/layernorm_node.h"
#include "fusilli/node/pointwise_node.h"
#include "fusilli/node/rmsnorm_node.h"
#include "fusilli/node/sdpa_node.h"
#include "fusilli/support/extras.h"

#include <bit> // C++20
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <format> // C++20
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fusilli {

// Given a vector of ints, returns the MLIR assembly for the
// `torch.constant.int` ops for each int value and the
// `torch.prim.ListConstruct` op wrapping these into a single
// value.
//
// For example if `getListOfIntOpsAsm` is called on these inputs:
//    listOfInts: {1, 2}
//    prefix: "stride"
//    suffix: "conv"
//
// It generates the following MLIR assembly:
//
//   %stride_val_0_conv = torch.constant.int 1
//   %stride_val_1_conv = torch.constant.int 2
//   %stride_conv = torch.prim.ListConstruct
//          %stride_val_0_conv, %stride_val_1_conv :
//              (!torch.int, !torch.int) -> !torch.list<int>
//
// The prefix is generally what attribute this refers to (e.g.
// padding, stride, dilation etc.) and the suffix is the node's
// unique name (for SSA disambiguation).
inline std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                                      const std::string &prefix,
                                      const std::string &suffix) {
  std::ostringstream oss;
  std::vector<std::string> ssaValueNames;

  // Emit `torch.constant.int` ops for each int value.
  for (size_t i = 0; i < listOfInts.size(); ++i) {
    std::string ssaValueName =
        "%" + prefix + "_val_" + std::to_string(i) + "_" + suffix;
    oss << ssaValueName << " = torch.constant.int " << listOfInts[i]
        << "\n    ";
    ssaValueNames.push_back(ssaValueName);
  }

  // Emit the ListConstruct op.
  oss << "%" + prefix + "_" + suffix << " = torch.prim.ListConstruct ";
  // %val_0, %val_1, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](const std::string &name) { oss << name; },
      // between_fn:
      [&] { oss << ", "; });
  oss << " : (";
  // !torch.int, !torch.int, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](const std::string &name) { oss << "!torch.int"; },
      // between_fn:
      [&] { oss << ", "; });
  oss << ") -> !torch.list<int>\n";

  return oss.str();
}

// Builds a tensor type string from explicit dims and dtype, without requiring
// a TensorAttr. Used when the type differs from what a TensorAttr stores
// (e.g., unexpanded dims for broadcast tensors).
inline std::string buildTensorTypeStr(const std::vector<int64_t> &dims,
                                      DataType dtype,
                                      bool isValueTensor = true) {
  std::ostringstream oss;
  oss << (isValueTensor ? "!torch.vtensor<[" : "!torch.tensor<[");
  interleave(
      dims.begin(), dims.end(), [&](int64_t dim) { oss << dim; },
      [&] { oss << ","; });
  oss << "]," << kDataTypeToMlirTypeAsm.at(dtype) << ">";
  return oss.str();
}

// ---------------------------------------------------------------------------
// Torch IR constant helpers
//
// Emit `torch.constant.*` SSA declarations as strings. Each returns a
// single line of the form `%<name>_<suffix> = torch.constant.<kind> <val>`.
// ---------------------------------------------------------------------------

inline std::string torchNoneAsm(std::string_view name,
                                std::string_view suffix) {
  return std::format("%{}_{} = torch.constant.none", name, suffix);
}

inline std::string torchBoolAsm(std::string_view name, std::string_view suffix,
                                bool value) {
  return std::format("%{}_{} = torch.constant.bool {}", name, suffix,
                     value ? "true" : "false");
}

inline std::string torchFloatAsm(std::string_view name, std::string_view suffix,
                                 float value) {
  return std::format("%{}_{} = torch.constant.float {:e}", name, suffix, value);
}

inline std::string torchIntAsm(std::string_view name, std::string_view suffix,
                               int64_t value) {
  return std::format("%{}_{} = torch.constant.int {}", name, suffix, value);
}

// Emits layout conversion ops (permute + broadcast expand if needed) for a
// tensor in MLIR assembly format. Handles both directions:
//
// When isInput=true (physical → logical):
//   - Permute from physical to (unexpanded-)logical order
//   - If broadcast: expand from unexpanded to full logical shape
//   - Operand: {name}, Result: {name}_{suffix}_perm
//
// When isInput=false (logical → physical):
//   - Permute from logical to physical order
//   - Operand: {name}_{suffix}_perm, Result: {name}
//
// The suffix is used to ensure unique SSA names when the same tensor is used
// by multiple different operations in a graph.
inline std::string
getLayoutConversionOpsAsm(const std::shared_ptr<TensorAttr> &tensor,
                          const std::string &prefix, const std::string &suffix,
                          bool isInput,
                          const std::string &operandOverride = "") {
  std::ostringstream oss;
  bool hasBroadcast = isInput && tensor->hasBroadcastDims();

  // Permute
  std::vector<int64_t> permuteOrder =
      isInput ? tensor->getPhysicalToLogicalPermuteOrder()
              : tensor->getLogicalToPhysicalPermuteOrder();

  oss << getListOfIntOpsAsm(permuteOrder, prefix, suffix);

  std::string permuteResultName =
      tensor->getValueNameAsm() +
      (isInput ? "_" + suffix + (hasBroadcast ? "_perm_unexpanded" : "_perm")
               : "");
  // An operandOverride (e.g., multi-result "%base#i") replaces the default.
  std::string permuteOperandName =
      operandOverride.empty()
          ? tensor->getValueNameAsm() + (isInput ? "" : "_" + suffix + "_perm")
          : operandOverride;

  std::string permuteFromType = tensor->getTensorTypeAsm(
      /*isValueTensor=*/true, /*useLogicalDims=*/!isInput);
  std::string permuteToType =
      hasBroadcast ? buildTensorTypeStr(tensor->getUnexpandedDim(),
                                        tensor->getDataType())
                   : tensor->getTensorTypeAsm(
                         /*isValueTensor=*/true, /*useLogicalDims=*/isInput);

  constexpr std::string_view permuteSchema = R"(
    {0} = torch.aten.permute {1}, {2} : {3}, !torch.list<int> -> {4}
  )";
  oss << std::format(permuteSchema, permuteResultName, permuteOperandName,
                     "%" + prefix + "_" + suffix, permuteFromType,
                     permuteToType);
  if (!hasBroadcast) {
    return oss.str();
  }

  // Expand broadcast dims (input direction only)
  std::string expandSizePrefix = "expand_size_" + prefix;
  std::string expandImplicitName = "%expand_implicit_" + prefix + "_" + suffix;
  oss << "    "
      << getListOfIntOpsAsm(tensor->getDim(), expandSizePrefix, suffix);
  oss << expandImplicitName << " = torch.constant.bool false\n    ";

  std::string expandResult = tensor->getValueNameAsm() + "_" + suffix + "_perm";
  std::string expandFromType =
      buildTensorTypeStr(tensor->getUnexpandedDim(), tensor->getDataType());
  std::string expandToType = tensor->getTensorTypeAsm(
      /*isValueTensor=*/true, /*useLogicalDims=*/true);

  constexpr std::string_view expandSchema =
      R"({0} = torch.aten.expand {1}, {2}, {3} : {4}, !torch.list<int>, !torch.bool -> {5}
  )";
  oss << std::format(expandSchema, expandResult, permuteResultName,
                     "%" + expandSizePrefix + "_" + suffix, expandImplicitName,
                     expandFromType, expandToType);
  return oss.str();
}

// Emits a scalar TensorAttr as a constant tensor literal in MLIR assembly.
// The result SSA name is the tensor's value name (e.g. %alpha).
inline std::string
getScalarConstantAsm(const std::shared_ptr<TensorAttr> &tensor) {
  assert(tensor->isScalar() && tensor->getScalarValue().has_value() &&
         "getScalarConstantAsm called with non-scalar tensor");
  std::string resultName = tensor->getValueNameAsm();
  std::string mlirType = kDataTypeToMlirTypeAsm.at(tensor->getDataType());
  std::string resultType = tensor->getTensorTypeAsm(/*isValueTensor=*/true,
                                                    /*useLogicalDims=*/true);
  // std::visit generates a compile time switch statement executing lambda
  // instantiation per variant alternative.
  // Example output:
  //   float  1.0f  → dense<0x3F800000>   : tensor<1xf32>
  //   double 1.0   → dense<0x3FF0000000000000> : tensor<1xf64>
  //   int32  42    → dense<0x0000002A>   : tensor<1xi32>
  //   int64  42L   → dense<0x000000000000002A> : tensor<1xi64>
  constexpr std::string_view schema = R"(
    {0} = torch.vtensor.literal(dense<0x{1}> : tensor<1x{2}>) : {3}
)";
  return std::visit(
      [&](auto val) -> std::string {
        using UInt = std::conditional_t<sizeof(val) == 4, uint32_t, uint64_t>;
        // {:0{}X} -> hex format with runtime width:
        //   - 0  pad with zeros
        //   - {} runtime width (sizeof(val)*2: 4 bytes→8, 8 bytes→16)
        //   - X  uppercase hex
        return std::format(schema,
                           resultName, // {0}
                           std::format("{:0{}X}", std::bit_cast<UInt>(val),
                                       sizeof(val) * 2), // {1}
                           mlirType,                     // {2}
                           resultType                    // {3}
        );
      },
      tensor->getScalarValue()
          .value()); // std::variant<int64_t, int32_t, float, double>
}

// Emits a `torch.aten.item` op to extract a scalar float from a tensor.
// The result SSA name is `%<prefix>_<suffix>`.
//
// For example:
//   getScalarItemOpsAsm("eps", epsilonTensor, "bn0")
//
// generates:
//   %eps_bn0 = torch.aten.item %bn0_EPSILON : !torch.vtensor<[1],f32> ->
//   !torch.float
inline std::string getScalarItemOpsAsm(const std::string &prefix,
                                       const std::shared_ptr<TensorAttr> &t,
                                       const std::string &suffix) {
  std::string tensorName = t->getValueNameAsm();
  std::string tensorType =
      t->getTensorTypeAsm(/*isValueTensor=*/true, /*useLogicalDims=*/true);
  constexpr std::string_view schema = R"(
    %{0}_{1} = torch.aten.item {2} : {3} -> !torch.float
)";
  return std::format(schema,
                     prefix,     // {0}
                     suffix,     // {1}
                     tensorName, // {2}
                     tensorType  // {3}
  );
}

//===----------------------------------------------------------------------===//
//
// TensorAttr ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits a ranked tensor type in MLIR assembly representation.
//
// This expects ranked tensors as we blanket generate a `!torch.vtensor` (or
// `!torch.tensor` if mutable) type. The caller is responsible to check for
// this.
//
// Example:
//
//    TensorAttr t;
//    t.setName("tensor")
//      .setDataType(DataType::Float)
//      .setDim({2, 3, 4})
//      .setStride({12, 1, 3})
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/true)
//        --> "!torch.vtensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/true)
//        --> "!torch.tensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/false)
//        --> "!torch.vtensor<[2,4,3],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/false)
//        --> "!torch.tensor<[2,4,3],f32>"
//
// Scalars (dim={1}, stride={1}) also work through this path:
//
//    TensorAttr s(2.0f);
//    s.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/true)
//        --> "!torch.vtensor<[1],f32>"
inline std::string TensorAttr::getTensorTypeAsm(bool isValueTensor,
                                                bool useLogicalDims) const {
  assert(!getDim().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty dims");
  assert(!getStride().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty strides");
  assert(getDataType() != DataType::NotSet &&
         "TensorAttr::getTensorTypeAsm expects a valid data type");

  std::ostringstream oss;
  oss << (isValueTensor ? "!torch.vtensor<[" : "!torch.tensor<[");

  std::vector<int64_t> dims = useLogicalDims ? getDim() : getPhysicalDim();

  interleave(
      dims.begin(), dims.end(),
      [&](int64_t dim) { oss << std::to_string(dim); },
      // between_fn:
      [&] { oss << ","; });
  oss << "],";
  oss << kDataTypeToMlirTypeAsm.at(getDataType());
  oss << ">";
  return oss.str();
}

// Emits an MLIR SSA value name starting with the `%` sigil based off the
// TensorAttr name but only using alphanumeric / underscore [A-Za-z0-9_]
// characters.
//
// `foo_Bar::X0` becomes `%foo_BarX0` if `isOutputAliased=false`.
// `foo_Bar::X0` becomes `%foo_BarX0_` if `isOutputAliased=true`.
inline std::string TensorAttr::getValueNameAsm(bool isOutputAliased) const {
  assert(!getName().empty() &&
         "TensorAttr name must not be empty for `getValueNameAsm`");

  std::string filtered = getName();
  std::erase_if(filtered, // C++20
                [](unsigned char c) { return !(std::isalnum(c) || c == '_'); });
  return "%" + filtered + (isOutputAliased ? "_" : "");
}

//===----------------------------------------------------------------------===//
//
// Graph ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits Graph's operand names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main(..., {}) -> ...
// with
//      "%arg0_image: !torch.vtensor<[16,128,64,64],f32>,
//       %arg1_filter: !torch.vtensor<[256,128,1,1],f32>"
//
// Order of operands is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphInputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphInputsSorted_.begin(), fullGraphInputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << input->getValueNameAsm() << ": " << input->getTensorTypeAsm();
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        // We only use the tensor inputs and not scalar (constants) as those
        // wouldn't be part of the main func.func signature but embedded as
        // constants in the IR.
        return input->isScalar();
      });
  return oss.str();
}

// Emits Graph's result names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main({}, ...) -> ...
// with
//      "%result: !torch.tensor<[16,256,64,64],f32>
//
// Order of results is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphOutputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getResultNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getValueNameAsm(/*isOutputAliased=*/true) << ": "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });
  return oss.str();
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePreAsm() const {
  // Collect module-scope declarations from sub-nodes
  // (e.g., custom op function definitions).
  std::ostringstream moduleScopeOss;
  collectModuleScopeAsm(moduleScopeOss);

  constexpr std::string_view schema = R"(
module @module {{
  {0}
  func.func @main({1}, {2}) attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  std::string output = std::format(schema,
                                   moduleScopeOss.str(),        // {0}
                                   getResultNamesAndTypesAsm(), // {1}
                                   getOperandNamesAndTypesAsm() // {2}
  );

  // Emit scalar constants (`torch.vtensor.literal`) for all scalar graph inputs
  // at the top of the function body.
  for (const auto &input : fullGraphInputsSorted_) {
    if (input->isScalar())
      output += getScalarConstantAsm(input);
  }

  return output;
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the post-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePostAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << "torch.overwrite.tensor.contents "
            << output->getValueNameAsm(/*isOutputAliased=*/false)
            << " overwrites "
            << output->getValueNameAsm(/*isOutputAliased=*/true) << " : "
            << output->getTensorTypeAsm(/*isValueTensor=*/true) << ", "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << "\n    "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });

  constexpr std::string_view schema = R"(
    {0}

    return
  }}
}}
  )";

  std::string output = std::format(schema,
                                   oss.str() // {0}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvFPropNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvFPropNode's operand names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution {}, ...
// with
//      "%arg0_image_conv_fprop_perm, %arg1_filter_conv_fprop_perm"
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string ConvFPropNode::getOperandNamesAsm() const {
  std::string suffix = convFPropAttr.getName();
  return convFPropAttr.getX()->getValueNameAsm() + "_" + suffix + "_perm" +
         ", " + convFPropAttr.getW()->getValueNameAsm() + "_" + suffix +
         "_perm";
}

// Emits ConvFPropNode's operand types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... : {}, ...
// with
//      "!torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>"
inline std::string ConvFPropNode::getOperandTypesAsm() const {
  return convFPropAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true) +
         ", " +
         convFPropAttr.getW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Emits ConvFPropNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution ...
// with
//      "%result_conv_fprop_perm"
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string ConvFPropNode::getResultNamesAsm() const {
  return convFPropAttr.getY()->getValueNameAsm() + "_" +
         convFPropAttr.getName() + "_perm";
}

// Emits ConvFPropNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... -> {}
// with
//      "!torch.vtensor<[16,256,64,64],f32>"
inline std::string ConvFPropNode::getResultTypesAsm() const {
  return convFPropAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Get groups in MLIR assembly format.
inline std::string ConvFPropNode::getGroupOpsAsm() const {
  constexpr size_t channelsIdx = 1;
  int64_t inChannels = convFPropAttr.getX()->getDim()[channelsIdx];
  int64_t filterChannels = convFPropAttr.getW()->getDim()[channelsIdx];
  int64_t groupCount = inChannels / filterChannels;

  return torchIntAsm("groups", convFPropAttr.getName(), groupCount);
}

// Get strides in MLIR assembly format.
inline std::string ConvFPropNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convFPropAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvFPropNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convFPropAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvFPropNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convFPropAttr.getName());
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string ConvFPropNode::emitNodePreAsm() const {
  // `torch.aten.convolution` signature from GeneratedTorchOps.td
  // https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
  //
  //  def Torch_AtenConvolutionOp : Torch_Op<"aten.convolution", [
  //    ...
  //    let summary = "Generated op for `aten::convolution : (Tensor, Tensor,
  //    Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)`"; let
  //    arguments = (ins
  //      AnyTorchTensorType:$input,
  //      AnyTorchTensorType:$weight,
  //      AnyTorchOptionalTensorType:$bias,
  //      AnyTorchListOfTorchIntType:$stride,
  //      AnyTorchListOfTorchIntType:$padding,
  //      AnyTorchListOfTorchIntType:$dilation,
  //      Torch_BoolType:$transposed,
  //      AnyTorchListOfTorchIntType:$output_padding,
  //      Torch_IntType:$groups
  //    );
  //    let results = (outs
  //      AnyTorchOptionalTensorType:$result
  //    );
  //   ...
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7} = torch.aten.convolution {8}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0} : {9}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> {10}
    {11}
    )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvFPropAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convFPropAttr.getName();
  std::string permuteX = getLayoutConversionOpsAsm(
      convFPropAttr.getX(), "permute_X", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteW = getLayoutConversionOpsAsm(
      convFPropAttr.getW(), "permute_W", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteY = getLayoutConversionOpsAsm(
      convFPropAttr.getY(), "permute_Y", uniqueSSASuffix, /*isInput=*/false);

  std::string output = std::format(schema,
                                   uniqueSSASuffix,      // {0}
                                   getGroupOpsAsm(),     // {1}
                                   getStrideOpsAsm(),    // {2}
                                   getPaddingOpsAsm(),   // {3}
                                   getDilationOpsAsm(),  // {4}
                                   permuteX,             // {5}
                                   permuteW,             // {6}
                                   getResultNamesAsm(),  // {7}
                                   getOperandNamesAsm(), // {8}
                                   getOperandTypesAsm(), // {9}
                                   getResultTypesAsm(),  // {10}
                                   permuteY              // {11}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvWGradNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvWGradNode's operand names in MLIR assembly format.
// torch.aten.convolution_backward has a fixed op signature that takes 3 main
// args (dy, x, and w). The empty tensor (%empty_w_{suffix}) is required by
// torch.aten.convolution_backward for the w arg even when calculating weight
// gradient.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string ConvWGradNode::getOperandNamesAsm() const {
  std::string suffix = convWGradAttr.getName();
  return convWGradAttr.getDY()->getValueNameAsm() + "_" + suffix + "_perm" +
         ", " + convWGradAttr.getX()->getValueNameAsm() + "_" + suffix +
         "_perm" + ", %empty_w_" + suffix;
}

// Emits ConvWGradNode's operand types in MLIR assembly format.
// Note: An operand for W is required by torch.aten.convolution_backward even
// when calculating weight gradient, so it's included after the DY and X types.
inline std::string ConvWGradNode::getOperandTypesAsm() const {
  return convWGradAttr.getDY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convWGradAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true) +
         ", " +
         convWGradAttr.getDW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Emits ConvWGradNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution_backward ...
// with
//      "%result_wgrad_perm"
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string ConvWGradNode::getResultNamesAsm() const {
  return convWGradAttr.getDW()->getValueNameAsm() + "_" +
         convWGradAttr.getName() + "_perm";
}

// Emits ConvWGradNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution_backward ... -> {}
// with
//      "!torch.vtensor<[256,128,1,1],f32>"
inline std::string ConvWGradNode::getResultTypesAsm() const {
  return convWGradAttr.getDW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Get groups in MLIR assembly format.
inline std::string ConvWGradNode::getGroupOpsAsm() const {
  constexpr size_t channelsIdx = 1;
  int64_t inChannels = convWGradAttr.getX()->getDim()[channelsIdx];
  int64_t filterChannels = convWGradAttr.getDW()->getDim()[channelsIdx];
  int64_t groupCount = inChannels / filterChannels;

  return torchIntAsm("groups", convWGradAttr.getName(), groupCount);
}

// Get strides in MLIR assembly format.
inline std::string ConvWGradNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convWGradAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvWGradNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convWGradAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvWGradNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convWGradAttr.getName());
}

// `torch.aten.convolution_backward` requires an input for the weight even when
// calculating the gradient of the weight. Create an empty tensor with the same
// dimensions as the weight tensor.
inline std::string ConvWGradNode::getPermuteEmptyWOpsAsm() const {
  std::ostringstream oss;
  std::string prefix = "empty_DW";
  std::string suffix = convWGradAttr.getName();
  std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();

  oss << getListOfIntOpsAsm(dwT->getDim(), prefix, suffix);

  // Use `torch.aten.empty.memory_format` to create an empty tensor. It is the
  // simplest op to create a new tensor without having a pre-existing one
  // (then `torch.aten.empty_like` could be used).
  constexpr std::string_view schema = R"(
    %none_DW_{0} = torch.constant.none
    %dtype_DW_{0} = torch.constant.int {3}
    %empty_w_{0} = torch.aten.empty.memory_format {1}, %dtype_DW_{0}, %none_DW_{0}, %none_DW_{0}, %none_DW_{0}, %none_DW_{0} : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> {2}
  )";

  torch_upstream::ScalarType dataType =
      kDataTypeToTorchType.at(dwT->getDataType());
  std::string output =
      std::format(schema,
                  suffix,                      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dwT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  std::to_string(static_cast<int>(dataType))      // {3}
      );

  return oss.str() + output;
}

inline std::string ConvWGradNode::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7}
    %true_{0} = torch.constant.bool true
    %false_{0} = torch.constant.bool false
    %output_mask_{0} = torch.prim.ListConstruct %false_{0}, %true_{0}, %false_{0} : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
    %grad_input_{0}, {8}, %grad_bias_{0} = torch.aten.convolution_backward {9}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0}, %output_mask_{0} : {10}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, {11}, !torch.none
    {12}
    )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvWGradAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convWGradAttr.getName();
  std::string permuteDY = getLayoutConversionOpsAsm(
      convWGradAttr.getDY(), "permute_DY", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteX = getLayoutConversionOpsAsm(
      convWGradAttr.getX(), "permute_X", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteDW = getLayoutConversionOpsAsm(
      convWGradAttr.getDW(), "permute_DW", uniqueSSASuffix, /*isInput=*/false);

  std::string output = std::format(schema,
                                   uniqueSSASuffix,          // {0}
                                   getGroupOpsAsm(),         // {1}
                                   getStrideOpsAsm(),        // {2}
                                   getPaddingOpsAsm(),       // {3}
                                   getDilationOpsAsm(),      // {4}
                                   permuteDY,                // {5}
                                   permuteX,                 // {6}
                                   getPermuteEmptyWOpsAsm(), // {7}
                                   getResultNamesAsm(),      // {8}
                                   getOperandNamesAsm(),     // {9}
                                   getOperandTypesAsm(),     // {10}
                                   getResultTypesAsm(),      // {11}
                                   permuteDW                 // {12}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvDGradNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvDGradNode's operand names in MLIR assembly format.
// torch.aten.convolution_backward has a fixed op signature that takes 3 main
// args (dy, x, and w). The empty tensor (%empty_x_{suffix}) is required by
// torch.aten.convolution_backward for the x arg even when calculating data
// gradient, so it's included between DY and W operands.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string ConvDGradNode::getOperandNamesAsm() const {
  std::string suffix = convDGradAttr.getName();
  return convDGradAttr.getDY()->getValueNameAsm() + "_" + suffix + "_perm" +
         ", %empty_x_" + suffix + ", " +
         convDGradAttr.getW()->getValueNameAsm() + "_" + suffix + "_perm";
}

// Emits ConvDGradNode's operand types in MLIR assembly format.
// Note: An operand for X is required by torch.aten.convolution_backward even
// when calculating data gradient, so it's included between DY and W operands.
inline std::string ConvDGradNode::getOperandTypesAsm() const {
  return convDGradAttr.getDY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convDGradAttr.getDX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convDGradAttr.getW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Emits ConvDGradNode's result names in MLIR assembly format.
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string ConvDGradNode::getResultNamesAsm() const {
  return convDGradAttr.getDX()->getValueNameAsm() + "_" +
         convDGradAttr.getName() + "_perm";
}

// Emits ConvDGradNode's result types in MLIR assembly format.
inline std::string ConvDGradNode::getResultTypesAsm() const {
  return convDGradAttr.getDX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Get groups in MLIR assembly format.
inline std::string ConvDGradNode::getGroupOpsAsm() const {
  constexpr size_t channelsIdx = 1;
  int64_t inChannels = convDGradAttr.getDX()->getDim()[channelsIdx];
  int64_t filterChannels = convDGradAttr.getW()->getDim()[channelsIdx];
  int64_t groupCount = inChannels / filterChannels;

  return torchIntAsm("groups", convDGradAttr.getName(), groupCount);
}

// Get strides in MLIR assembly format.
inline std::string ConvDGradNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convDGradAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvDGradNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convDGradAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvDGradNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convDGradAttr.getName());
}

// `torch.aten.convolution_backward` requires an input for the image even when
// calculating the gradient of the image. Create an empty tensor with the same
// dimensions as DX.
inline std::string ConvDGradNode::getPermuteEmptyXOpsAsm() const {
  std::ostringstream oss;
  std::string prefix = "empty_DX";
  std::string suffix = convDGradAttr.getName();
  std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();

  oss << getListOfIntOpsAsm(dxT->getDim(), prefix, suffix);

  // Use `torch.aten.empty.memory_format` to create an empty tensor. It is the
  // simplest op to create a new tensor without having a pre-existing one
  // (then `torch.aten.empty_like` could be used).
  constexpr std::string_view schema = R"(
    %none_DX_{0} = torch.constant.none
    %dtype_DX_{0} = torch.constant.int {3}
    %empty_x_{0} = torch.aten.empty.memory_format {1}, %dtype_DX_{0}, %none_DX_{0}, %none_DX_{0}, %none_DX_{0}, %none_DX_{0} : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> {2}
  )";

  torch_upstream::ScalarType dataType =
      kDataTypeToTorchType.at(dxT->getDataType());
  std::string output =
      std::format(schema,
                  suffix,                      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dxT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  std::to_string(static_cast<int>(dataType))      // {3}
      );

  return oss.str() + output;
}

inline std::string ConvDGradNode::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7}
    %true_{0} = torch.constant.bool true
    %false_{0} = torch.constant.bool false
    %output_mask_{0} = torch.prim.ListConstruct %true_{0}, %false_{0}, %false_{0} : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
    {8}, %grad_weight_{0}, %grad_bias_{0} = torch.aten.convolution_backward {9}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0}, %output_mask_{0} : {10}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> {11}, !torch.none, !torch.none
    {12}
  )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvDGradAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convDGradAttr.getName();
  std::string permuteDY = getLayoutConversionOpsAsm(
      convDGradAttr.getDY(), "permute_DY", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteW = getLayoutConversionOpsAsm(
      convDGradAttr.getW(), "permute_W", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteDX = getLayoutConversionOpsAsm(
      convDGradAttr.getDX(), "permute_DX", uniqueSSASuffix, /*isInput=*/false);

  std::string output = std::format(schema,
                                   uniqueSSASuffix,          // {0}
                                   getGroupOpsAsm(),         // {1}
                                   getStrideOpsAsm(),        // {2}
                                   getPaddingOpsAsm(),       // {3}
                                   getDilationOpsAsm(),      // {4}
                                   permuteDY,                // {5}
                                   permuteW,                 // {6}
                                   getPermuteEmptyXOpsAsm(), // {7}
                                   getResultNamesAsm(),      // {8}
                                   getOperandNamesAsm(),     // {9}
                                   getOperandTypesAsm(),     // {10}
                                   getResultTypesAsm(),      // {11}
                                   permuteDX                 // {12}
  );
  return output;
}

//===----------------------------------------------------------------------===//
//
// BatchNormNode ASM Emitter Methods
//
// Both inference and training modes emit `torch.aten.native_batch_norm`, which
// always returns three results: (output, saved_mean, saved_invstd). For
// inference, training=false and the last two results are discarded. For
// training, training=true and all three results are named outputs.
//
// Example (inference, NCHW [4,16,8,8] input, no scale/bias):
//
//   %eps_bn = torch.aten.item %EPSILON : !torch.vtensor<[1],f32> ->
//   !torch.float %momentum_bn = torch.aten.item %MOMENTUM :
//   !torch.vtensor<[1],f32> -> !torch.float %permute_x_val_0_bn =
//   torch.constant.int 0
//   ...
//   %X_bn_perm = torch.aten.permute %X, %permute_x_bn : ...
//   %none_scale_bn = torch.constant.none
//   %none_bias_bn = torch.constant.none
//   %training_bn = torch.constant.bool false
//   %Y_bn_perm, %_infer_saved_mean_bn_perm, %_infer_saved_invstd_bn_perm =
//       torch.aten.native_batch_norm %X_bn_perm, %none_scale_bn, %none_bias_bn,
//           %MEAN, %VAR, %training_bn, %momentum_bn, %eps_bn : ... -> ...
//   %Y = torch.aten.permute %Y_bn_perm, %permute_y_bn : ...
//
//===----------------------------------------------------------------------===//

// Emits BatchNormNode's operand names in MLIR assembly format.
//
// The operand order for torch.aten.batch_norm / torch.aten.native_batch_norm:
//   input, weight?, bias?, running_mean?, running_var?, training, momentum, eps
//   (batch_norm also takes cudnn_enabled)
//
// For 1D tensors (scale, bias, mean, var) no permutation is applied; they are
// referenced directly by their SSA value name.
inline std::string BatchNormNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = batchnormAttr.getName();

  // Input X (permuted to logical NCHW).
  oss << batchnormAttr.getX()->getValueNameAsm() << "_" << suffix << "_perm, ";

  // Optional scale / bias (1D, referenced directly).
  auto getOptional1DName = [&](const std::shared_ptr<TensorAttr> &t,
                               const std::string &name) -> std::string {
    return t ? t->getValueNameAsm() + ", "
             : "%none_" + name + "_" + suffix + ", ";
  };
  oss << getOptional1DName(batchnormAttr.getSCALE(), "scale");
  oss << getOptional1DName(batchnormAttr.getBIAS(), "bias");
  oss << getOptional1DName(batchnormAttr.getMEAN(), "mean");
  oss << getOptional1DName(batchnormAttr.getVAR(), "var");

  oss << "%training_" << suffix << ", ";
  oss << "%momentum_" << suffix << ", ";
  oss << "%eps_" << suffix;

  return oss.str();
}

// Emits BatchNormNode's operand types in MLIR assembly format.
inline std::string BatchNormNode::getOperandTypesAsm() const {
  std::ostringstream oss;

  // Input X type (logical dims).
  oss << batchnormAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true)
      << ", ";

  // Optional scale / bias / running mean / var types.
  auto getOptional1DType =
      [&](const std::shared_ptr<TensorAttr> &t) -> std::string {
    return t ? t->getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
             : "!torch.none";
  };
  oss << getOptional1DType(batchnormAttr.getSCALE()) << ", ";
  oss << getOptional1DType(batchnormAttr.getBIAS()) << ", ";
  oss << getOptional1DType(batchnormAttr.getMEAN()) << ", ";
  oss << getOptional1DType(batchnormAttr.getVAR()) << ", ";

  oss << "!torch.bool, ";
  oss << "!torch.float, ";
  oss << "!torch.float";

  return oss.str();
}

// Emits BatchNormNode's result names in MLIR assembly format.
//
// Both inference and training use torch.aten.native_batch_norm which always
// returns three tensors: (output, saved_mean, saved_invstd).
// For training, all three are named after the corresponding output tensors.
// For inference, the last two are discarded placeholder names.
inline std::string BatchNormNode::getResultNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = batchnormAttr.getName();

  oss << batchnormAttr.getY()->getValueNameAsm() << "_" << suffix << "_perm";

  if (isTrainingForwardPhase()) {
    oss << ", ";
    oss << batchnormAttr.getSAVED_MEAN()->getValueNameAsm() << "_" << suffix
        << "_perm" << ", ";
    oss << batchnormAttr.getSAVED_INV_VARIANCE()->getValueNameAsm() << "_"
        << suffix << "_perm";
  } else {
    // Inference: native_batch_norm still returns 3 tensors; discard last two.
    oss << ", %_infer_saved_mean_" << suffix << "_perm";
    oss << ", %_infer_saved_invstd_" << suffix << "_perm";
  }

  return oss.str();
}

// Emits BatchNormNode's result types in MLIR assembly format.
//
// Both inference and training emit three result types because
// torch.aten.native_batch_norm always returns (output, saved_mean,
// saved_invstd).
inline std::string BatchNormNode::getResultTypesAsm() const {
  std::ostringstream oss;
  oss << batchnormAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);

  if (isTrainingForwardPhase()) {
    oss << ", ";
    oss << batchnormAttr.getSAVED_MEAN()->getTensorTypeAsm(
               /*isValueTensor=*/true,
               /*useLogicalDims=*/true)
        << ", ";
    oss << batchnormAttr.getSAVED_INV_VARIANCE()->getTensorTypeAsm(
        /*isValueTensor=*/true,
        /*useLogicalDims=*/true);
  } else {
    // Inference: use MEAN/VAR types for the two discarded native_batch_norm
    // outputs (saved_mean and saved_invstd are the same shape as running
    // stats).
    oss << ", ";
    oss << batchnormAttr.getMEAN()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                     /*useLogicalDims=*/true)
        << ", ";
    oss << batchnormAttr.getVAR()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                    /*useLogicalDims=*/true);
  }

  return oss.str();
}

// Get epsilon extraction op in MLIR assembly format.
inline std::string BatchNormNode::getEpsilonOpsAsm() const {
  return getScalarItemOpsAsm("eps", batchnormAttr.getEpsilon(),
                             batchnormAttr.getName());
}

// Get momentum extraction op in MLIR assembly format.
inline std::string BatchNormNode::getMomentumOpsAsm() const {
  return getScalarItemOpsAsm("momentum", batchnormAttr.getMomentum(),
                             batchnormAttr.getName());
}

// Emits the MLIR assembly for the BatchNormNode.
//
// Both inference and training use `torch.aten.native_batch_norm` (three
// outputs). For inference, training=false and the last two outputs (saved_mean,
// saved_invstd) are discarded placeholders.
inline std::string BatchNormNode::emitNodePreAsm() const {
  std::string suffix = batchnormAttr.getName();

  std::string permuteX = getLayoutConversionOpsAsm(
      batchnormAttr.getX(), "permute_x", suffix, /*isInput=*/true);
  std::string permuteY = getLayoutConversionOpsAsm(
      batchnormAttr.getY(), "permute_y", suffix, /*isInput=*/false);

  // Emit "none" declarations for optional 1D inputs that are not provided.
  // Returns empty string when tensor is present (nothing to emit),
  // or a `torch.constant.none` decl (no leading spaces; schema provides
  // indent).
  auto getNoneOrEmpty = [&](const std::shared_ptr<TensorAttr> &t,
                            const std::string &name) -> std::string {
    if (t)
      return "";
    return torchNoneAsm("none_" + name, suffix);
  };

  std::string scaleNone = getNoneOrEmpty(batchnormAttr.getSCALE(), "scale");
  std::string biasNone = getNoneOrEmpty(batchnormAttr.getBIAS(), "bias");
  std::string meanNone = getNoneOrEmpty(batchnormAttr.getMEAN(), "mean");
  std::string varNone = getNoneOrEmpty(batchnormAttr.getVAR(), "var");

  std::string permuteSavedMean =
      isTrainingForwardPhase()
          ? getLayoutConversionOpsAsm(batchnormAttr.getSAVED_MEAN(),
                                      "permute_saved_mean", suffix,
                                      /*isInput=*/false)
          : "";
  std::string permuteSavedInvVar =
      isTrainingForwardPhase()
          ? getLayoutConversionOpsAsm(batchnormAttr.getSAVED_INV_VARIANCE(),
                                      "permute_saved_inv_variance", suffix,
                                      /*isInput=*/false)
          : "";
  std::string trainingStr = isTrainingForwardPhase() ? "true" : "false";

  // Each optional 1D operand slot ({4}-{7}) is on its own schema line.
  // When the tensor is provided, the slot is empty (line becomes blank,
  // which the indentation checker skips). When not provided, the slot
  // holds a torch.constant.none decl. For inference, {13} and {14} are empty
  // strings, producing blank lines.
  constexpr std::string_view schema = R"(
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7}
    %training_{0} = torch.constant.bool {15}
    {8} = torch.aten.native_batch_norm {9} : {10} -> {11}
    {12}
    {13}
    {14}
    )";

  return std::format(schema,
                     suffix,               // {0}
                     getEpsilonOpsAsm(),   // {1}
                     getMomentumOpsAsm(),  // {2}
                     permuteX,             // {3}
                     scaleNone,            // {4}
                     biasNone,             // {5}
                     meanNone,             // {6}
                     varNone,              // {7}
                     getResultNamesAsm(),  // {8}
                     getOperandNamesAsm(), // {9}
                     getOperandTypesAsm(), // {10}
                     getResultTypesAsm(),  // {11}
                     permuteY,             // {12}
                     permuteSavedMean,     // {13}
                     permuteSavedInvVar,   // {14}
                     trainingStr           // {15}
  );
}

//===----------------------------------------------------------------------===//
//
// LayerNormNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits LayerNormNode's operand names in MLIR assembly format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string LayerNormNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = layernormAttr.getName();

  oss << layernormAttr.getX()->getValueNameAsm() << "_" << suffix << "_perm, ";
  oss << "%normalized_shape_" << suffix << ", ";

  auto getOptionalOperandNameAsm = [&](const std::shared_ptr<TensorAttr> &t,
                                       const std::string &name) {
    return t ? t->getValueNameAsm() + "_" + suffix + "_perm, "
             : "%none_" + name + "_" + suffix + ", ";
  };

  oss << getOptionalOperandNameAsm(layernormAttr.getSCALE(), "scale");
  oss << getOptionalOperandNameAsm(layernormAttr.getBIAS(), "bias");
  oss << "%eps_" << suffix;

  return oss.str();
}

// Emits LayerNormNode's operand types in MLIR assembly format.
inline std::string LayerNormNode::getOperandTypesAsm() const {
  std::ostringstream oss;

  oss << layernormAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true)
      << ", ";
  oss << "!torch.list<int>" << ", ";

  auto getOptionalOperandTypeAsm = [&](const std::shared_ptr<TensorAttr> &t) {
    return t ? t->getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
             : "!torch.none";
  };

  oss << getOptionalOperandTypeAsm(layernormAttr.getSCALE()) << ", ";
  oss << getOptionalOperandTypeAsm(layernormAttr.getBIAS()) << ", ";
  oss << "!torch.float";

  return oss.str();
}

// Emits LayerNormNode's result names in MLIR assembly format.
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string LayerNormNode::getResultNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = layernormAttr.getName();

  oss << layernormAttr.getY()->getValueNameAsm() << "_" << suffix << "_perm";

  if (isTrainingForwardPhase()) {
    oss << ", ";
    oss << layernormAttr.getMEAN()->getValueNameAsm() << "_" << suffix
        << "_perm" << ", ";
    oss << layernormAttr.getINV_VARIANCE()->getValueNameAsm() << "_" << suffix
        << "_perm";
  }

  return oss.str();
}

// Emits LayerNormNode's result types in MLIR assembly format.
inline std::string LayerNormNode::getResultTypesAsm() const {
  std::ostringstream oss;
  oss << layernormAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);

  if (isTrainingForwardPhase()) {
    oss << ", ";
    oss << layernormAttr.getMEAN()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                     /*useLogicalDims=*/true)
        << ", ";
    oss << layernormAttr.getINV_VARIANCE()->getTensorTypeAsm(
        /*isValueTensor=*/true,
        /*useLogicalDims=*/true);
  }

  return oss.str();
}

// Get normalized_shape list construction ops in MLIR assembly format.
// normalized_shape is the dimensions to normalize over (typically all dims
// except batch).
inline std::string LayerNormNode::getNormalizedShapeOpsAsm() const {
  return getListOfIntOpsAsm(getNormalizedShape(), /*prefix=*/"normalized_shape",
                            /*suffix=*/layernormAttr.getName());
}

// Get epsilon extraction op in MLIR assembly format. The scalar constant
// `torch.vtensor.literal` is emitted once at graph level
// (Graph::emitNodePreAsm). Here we extract the float value with
// `torch.aten.item` for use with `torch.aten.layer_norm` which expects
// `!torch.float`.
inline std::string LayerNormNode::getEpsilonOpsAsm() const {
  return getScalarItemOpsAsm("eps", layernormAttr.getEpsilon(),
                             layernormAttr.getName());
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string LayerNormNode::emitNodePreAsm() const {
  std::string uniqueSSASuffix = layernormAttr.getName();
  std::string permuteX = getLayoutConversionOpsAsm(
      layernormAttr.getX(), "permute_x", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteY = getLayoutConversionOpsAsm(
      layernormAttr.getY(), "permute_y", uniqueSSASuffix, /*isInput=*/false);
  std::string permuteScale =
      layernormAttr.getSCALE()
          ? getLayoutConversionOpsAsm(layernormAttr.getSCALE(), "permute_scale",
                                      uniqueSSASuffix, /*isInput=*/true)
          : torchNoneAsm("none_scale", uniqueSSASuffix);
  std::string permuteBias =
      layernormAttr.getBIAS()
          ? getLayoutConversionOpsAsm(layernormAttr.getBIAS(), "permute_bias",
                                      uniqueSSASuffix, /*isInput=*/true)
          : torchNoneAsm("none_bias", uniqueSSASuffix);

  if (isTrainingForwardPhase()) {
    std::string permuteMean =
        getLayoutConversionOpsAsm(layernormAttr.getMEAN(), "permute_mean",
                                  uniqueSSASuffix, /*isInput=*/false);
    std::string permuteInvVariance = getLayoutConversionOpsAsm(
        layernormAttr.getINV_VARIANCE(), "permute_inv_variance",
        uniqueSSASuffix, /*isInput=*/false);

    constexpr std::string_view schema = R"(
    {0}
    {1}
    {2}
    {3}
    {4}
    {5} = torch.aten.native_layer_norm {6} : {7} -> {8}
    {9}
    {10}
    {11}
    )";

    return std::format(schema,
                       getNormalizedShapeOpsAsm(), // {0}
                       getEpsilonOpsAsm(),         // {1}
                       permuteX,                   // {2}
                       permuteScale,               // {3}
                       permuteBias,                // {4}
                       getResultNamesAsm(),        // {5}
                       getOperandNamesAsm(),       // {6}
                       getOperandTypesAsm(),       // {7}
                       getResultTypesAsm(),        // {8}
                       permuteY,                   // {9}
                       permuteMean,                // {10}
                       permuteInvVariance          // {11}
    );
  }

  constexpr std::string_view schema = R"(
    {1}
    {2}
    {3}
    {4}
    {5}
    %cudnn_enable_{0} = torch.constant.bool false
    {6} = torch.aten.layer_norm {7}, %cudnn_enable_{0} : {8}, !torch.bool -> {9}
    {10}
  )";

  return std::format(schema, uniqueSSASuffix,    // {0}
                     getNormalizedShapeOpsAsm(), // {1}
                     getEpsilonOpsAsm(),         // {2}
                     permuteX,                   // {3}
                     permuteScale,               // {4}
                     permuteBias,                // {5}
                     getResultNamesAsm(),        // {6}
                     getOperandNamesAsm(),       // {7}
                     getOperandTypesAsm(),       // {8}
                     getResultTypesAsm(),        // {9}
                     permuteY                    // {10}
  );
}

//===----------------------------------------------------------------------===//
//
// RmsNormNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits RmsNormNode's operand names in MLIR assembly format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string RmsNormNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = rmsnormAttr.getName();

  oss << rmsnormAttr.getX()->getValueNameAsm() << "_" << suffix << "_perm, ";
  oss << "%normalized_shape_" << suffix << ", ";

  auto sT = rmsnormAttr.getSCALE();
  oss << (sT ? sT->getValueNameAsm() + "_" + suffix + "_perm, "
             : "%none_scale_" + suffix + ", ");
  oss << "%eps_" << suffix;

  return oss.str();
}

// Emits RmsNormNode's operand types in MLIR assembly format.
inline std::string RmsNormNode::getOperandTypesAsm() const {
  std::ostringstream oss;

  oss << rmsnormAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                              /*useLogicalDims=*/true)
      << ", ";
  oss << "!torch.list<int>" << ", ";

  auto sT = rmsnormAttr.getSCALE();
  oss << (sT ? sT->getTensorTypeAsm(/*isValueTensor=*/true,
                                    /*useLogicalDims=*/true)
             : "!torch.none")
      << ", ";
  oss << "!torch.float";

  return oss.str();
}

// Emits RmsNormNode's result names in MLIR assembly format.
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string RmsNormNode::getResultNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = rmsnormAttr.getName();

  oss << rmsnormAttr.getY()->getValueNameAsm() << "_" << suffix << "_perm";

  return oss.str();
}

// Emits RmsNormNode's result types in MLIR assembly format.
inline std::string RmsNormNode::getResultTypesAsm() const {
  std::ostringstream oss;
  oss << rmsnormAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                              /*useLogicalDims=*/true);

  return oss.str();
}

// Get normalized_shape list construction ops in MLIR assembly format.
// normalized_shape is the dimensions to normalize over (typically all dims
// except batch).
inline std::string RmsNormNode::getNormalizedShapeOpsAsm() const {
  return getListOfIntOpsAsm(getNormalizedShape(), /*prefix=*/"normalized_shape",
                            /*suffix=*/rmsnormAttr.getName());
}

// Get epsilon extraction op in MLIR assembly format. The scalar constant
// `torch.vtensor.literal` is emitted once at graph level
// (Graph::emitNodePreAsm). Here we extract the float value with
// `torch.aten.item` for use with `torch.aten.rms_norm` which expects
// `!torch.float`.
inline std::string RmsNormNode::getEpsilonOpsAsm() const {
  return getScalarItemOpsAsm("eps", rmsnormAttr.getEpsilon(),
                             rmsnormAttr.getName());
}

// Emits MLIR assembly for inference-mode RmsNorm. Training mode ASM emission
// is not yet supported as torch-mlir does not lower the training variant.
inline std::string RmsNormNode::emitNodePreAsm() const {
  assert(!isTrainingForwardPhase() &&
         "RmsNorm training mode ASM emission is not yet supported");

  std::string uniqueSSASuffix = rmsnormAttr.getName();
  std::string permuteX = getLayoutConversionOpsAsm(
      rmsnormAttr.getX(), "permute_x", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteY = getLayoutConversionOpsAsm(
      rmsnormAttr.getY(), "permute_y", uniqueSSASuffix, /*isInput=*/false);
  std::string permuteScale =
      rmsnormAttr.getSCALE()
          ? getLayoutConversionOpsAsm(rmsnormAttr.getSCALE(), "permute_scale",
                                      uniqueSSASuffix, /*isInput=*/true)
          : torchNoneAsm("none_scale", uniqueSSASuffix);

  constexpr std::string_view schema = R"(
    {0}
    {1}
    {2}
    {3}
    {4} = torch.aten.rms_norm {5} : {6} -> {7}
    {8}
  )";

  return std::format(schema, getNormalizedShapeOpsAsm(), // {0}
                     getEpsilonOpsAsm(),                 // {1}
                     permuteX,                           // {2}
                     permuteScale,                       // {3}
                     getResultNamesAsm(),                // {4}
                     getOperandNamesAsm(),               // {5}
                     getOperandTypesAsm(),               // {6}
                     getResultTypesAsm(),                // {7}
                     permuteY                            // {8}
  );
}

//===----------------------------------------------------------------------===//
//
// MatmulNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits MatmulNode's operand names in MLIR assembly format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations.
inline std::string MatmulNode::getOperandNamesAsm() const {
  std::string suffix = matmulAttr.getName();
  return matmulAttr.getA()->getValueNameAsm() + "_" + suffix + "_perm" + ", " +
         matmulAttr.getB()->getValueNameAsm() + "_" + suffix + "_perm";
}

// Emits MatmulNode's operand types in MLIR assembly format.
inline std::string MatmulNode::getOperandTypesAsm() const {
  return matmulAttr.getA()->getTensorTypeAsm(/*isValueTensor=*/true,
                                             /*useLogicalDims=*/true) +
         ", " +
         matmulAttr.getB()->getTensorTypeAsm(/*isValueTensor=*/true,
                                             /*useLogicalDims=*/true);
}

// Emits MatmulNode's result names in MLIR assembly format.
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations. This intermediate result
// is then used by the output permute.
inline std::string MatmulNode::getResultNamesAsm() const {
  return matmulAttr.getC()->getValueNameAsm() + "_" + matmulAttr.getName() +
         "_perm";
}

// Emits MatmulNode's result types in MLIR assembly format.
inline std::string MatmulNode::getResultTypesAsm() const {
  return matmulAttr.getC()->getTensorTypeAsm(/*isValueTensor=*/true,
                                             /*useLogicalDims=*/true);
}

inline std::string MatmulNode::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
    {0}
    {1}
    {2} = torch.aten.matmul {3} : {4} -> {5}
    {6}
  )";

  std::string uniqueSSASuffix = matmulAttr.getName();
  std::string permuteA = getLayoutConversionOpsAsm(
      matmulAttr.getA(), "permute_A", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteB = getLayoutConversionOpsAsm(
      matmulAttr.getB(), "permute_B", uniqueSSASuffix, /*isInput=*/true);
  std::string permuteC = getLayoutConversionOpsAsm(
      matmulAttr.getC(), "permute_C", uniqueSSASuffix, /*isInput=*/false);

  std::string output = std::format(schema,
                                   permuteA,             // {0}
                                   permuteB,             // {1}
                                   getResultNamesAsm(),  // {2}
                                   getOperandNamesAsm(), // {3}
                                   getOperandTypesAsm(), // {4}
                                   getResultTypesAsm(),  // {5}
                                   permuteC              // {6}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// PointwiseNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits the SSA name for a single PointwiseNode input in MLIR assembly
// format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations in a graph.
inline std::string PointwiseNode::getInputNameAsm(size_t inputIndex) const {
  const auto &in = pointwiseAttr.inputs.at(
      static_cast<PointwiseAttr::InputNames>(inputIndex));
  return in->getValueNameAsm() + "_" + pointwiseAttr.getName() + "_perm";
}

// Emits the type for a single PointwiseNode input in MLIR assembly format.
inline std::string PointwiseNode::getInputTypeAsm(size_t inputIndex) const {
  const auto &in = pointwiseAttr.inputs.at(
      static_cast<PointwiseAttr::InputNames>(inputIndex));
  return in->getTensorTypeAsm(/*isValueTensor=*/true, /*useLogicalDims=*/true);
}

// Emits PointwiseNode's operand names in MLIR assembly format.
inline std::string PointwiseNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  oss << getInputNameAsm(0);
  if (pointwiseAttr.getIN_1())
    oss << ", " << getInputNameAsm(1);
  if (pointwiseAttr.getIN_2())
    oss << ", " << getInputNameAsm(2);
  return oss.str();
}

// Emits PointwiseNode's operand types in MLIR assembly format.
inline std::string PointwiseNode::getOperandTypesAsm() const {
  std::ostringstream oss;
  oss << getInputTypeAsm(0);
  if (pointwiseAttr.getIN_1())
    oss << ", " << getInputTypeAsm(1);
  if (pointwiseAttr.getIN_2())
    oss << ", " << getInputTypeAsm(2);
  return oss.str();
}

// Emits PointwiseNode's result names in MLIR assembly format.
//
// The unique suffix and "_perm" are included to ensure SSA uniqueness when
// the same tensor is used by multiple operations in a graph.
inline std::string PointwiseNode::getResultNamesAsm() const {
  return pointwiseAttr.getOUT_0()->getValueNameAsm() + "_" +
         pointwiseAttr.getName() + "_perm";
}

// Emits PointwiseNode's result types in MLIR assembly format.
inline std::string PointwiseNode::getResultTypesAsm() const {
  return pointwiseAttr.getOUT_0()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                    /*useLogicalDims=*/true);
}

// Emits PointwiseNode's result names and types in MLIR assembly format.
inline std::string PointwiseNode::getResultNamesAndTypesAsm() const {
  return getResultNamesAsm() + ": " + getResultTypesAsm();
}

#define FUSILLI_DECLARE_UNARY_POINTWISE_EMITTER(PWOP, SCHEMA, OPIR)            \
  case PointwiseAttr::Mode::PWOP: {                                            \
    return std::format(SCHEMA, permuteIN0,   /* {0} */                         \
                       getResultNamesAsm(),  /* {1} */                         \
                       getOperandNamesAsm(), /* {2} */                         \
                       getOperandTypesAsm(), /* {3} */                         \
                       getResultTypesAsm(),  /* {4} */                         \
                       permuteOUT0,          /* {5} */                         \
                       #OPIR,                /* {6} */                         \
                       getName()             /* {7} */                         \
    );                                                                         \
  }

#define FUSILLI_DECLARE_BINARY_POINTWISE_EMITTER(PWOP, SCHEMA, OPIR)           \
  case PointwiseAttr::Mode::PWOP: {                                            \
    return std::format(SCHEMA, permuteIN0,   /* {0} */                         \
                       permuteIN1,           /* {1} */                         \
                       getResultNamesAsm(),  /* {2} */                         \
                       getOperandNamesAsm(), /* {3} */                         \
                       getOperandTypesAsm(), /* {4} */                         \
                       getResultTypesAsm(),  /* {5} */                         \
                       permuteOUT0,          /* {6} */                         \
                       #OPIR,                /* {7} */                         \
                       getName()             /* {8} */                         \
    );                                                                         \
  }

inline std::string PointwiseNode::emitNodePreAsm() const {
  std::string uniqueSSASuffix = pointwiseAttr.getName();

  // Generate permute operations for inputs and output using the standard
  // getLayoutConversionOpsAsm() with unique suffixes to prevent SSA
  // redefinitions when multiple operations use the same tensors.
  std::string permuteIN0 = getLayoutConversionOpsAsm(
      pointwiseAttr.getIN_0(), "permute_IN_0", uniqueSSASuffix,
      /*isInput=*/true);
  std::string permuteIN1 =
      pointwiseAttr.getIN_1()
          ? getLayoutConversionOpsAsm(pointwiseAttr.getIN_1(), "permute_IN_1",
                                      uniqueSSASuffix, /*isInput=*/true)
          : "";
  std::string permuteOUT0 =
      getLayoutConversionOpsAsm(pointwiseAttr.getOUT_0(), "permute_OUT_0",
                                uniqueSSASuffix, /*isInput=*/false);

  constexpr std::string_view kUnaryTorchSchema = R"(
    {0}
    {1} = {6} {2} : {3} -> {4}
    {5}
)";

  constexpr std::string_view kBinaryTorchSchema = R"(
    {0}
    {1}
    {2} = {7} {3} : {4} -> {5}
    {6}
)";

  constexpr std::string_view kSubAddSchema = R"(
    {0}
    {1}
    %alpha_{8} = torch.constant.int 1
    {2} = {7} {3}, %alpha_{8} : {4}, !torch.int -> {5}
    {6}
)";

  constexpr std::string_view kAddSquareSchema = R"(
    {0}
    {1}
    %add_square_sq_{8} = torch.aten.mul.Tensor {3}, {3} : {4}, {4} -> {4}
    %alpha_{8} = torch.constant.int 1
    {5} = torch.aten.add.Tensor {2}, %add_square_sq_{8}, %alpha_{8} : {9}, !torch.int -> {6}
    {7}
)";

  constexpr std::string_view kIdentitySchema = R"(
    {0}
    %none_{7} = torch.constant.none
    {1} = {6} {2}, %none_{7} : {3}, !torch.none -> {4}
    {5}
)";

  constexpr std::string_view kEluSchema = R"(
    {0}
    %elu_alpha_{7} = torch.constant.float {8:e}
    %elu_scale_{7} = torch.constant.float 1.000000e+00
    %elu_input_scale_{7} = torch.constant.float 1.000000e+00
    {1} = {6} {2}, %elu_alpha_{7}, %elu_scale_{7}, %elu_input_scale_{7} : {3}, !torch.float, !torch.float, !torch.float -> {4}
    {5}
)";

  constexpr std::string_view kGeluSchema = R"(
    {0}
    %gelu_approximate_{7} = torch.constant.str "{8}"
    {1} = {6} {2}, %gelu_approximate_{7} : {3}, !torch.str -> {4}
    {5}
)";

  constexpr std::string_view kSoftplusSchema = R"(
    {0}
    %softplus_beta_{7} = torch.constant.float {8:e}
    %softplus_threshold_{7} = torch.constant.float {9:e}
    {1} = {6} {2}, %softplus_beta_{7}, %softplus_threshold_{7} : {3}, !torch.float, !torch.float -> {4}
    {5}
)";

#define FUSILLI_DECLARE_UNARY_TORCH_EMITTER(PWOP, OPIR)                        \
  FUSILLI_DECLARE_UNARY_POINTWISE_EMITTER(PWOP, kUnaryTorchSchema, OPIR)
#define FUSILLI_DECLARE_BINARY_TORCH_EMITTER(PWOP, OPIR)                       \
  FUSILLI_DECLARE_BINARY_POINTWISE_EMITTER(PWOP, kBinaryTorchSchema, OPIR)
#define FUSILLI_DECLARE_SUB_ADD_TORCH_EMITTER(PWOP, OPIR)                      \
  FUSILLI_DECLARE_BINARY_POINTWISE_EMITTER(PWOP, kSubAddSchema, OPIR)

  switch (pointwiseAttr.getMode()) {
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(ABS, torch.aten.abs)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(CEIL, torch.aten.ceil)
  case PointwiseAttr::Mode::ELU_FWD: {
    return std::format(kEluSchema, permuteIN0,     /* {0} */
                       getResultNamesAsm(),        /* {1} */
                       getOperandNamesAsm(),       /* {2} */
                       getOperandTypesAsm(),       /* {3} */
                       getResultTypesAsm(),        /* {4} */
                       permuteOUT0,                /* {5} */
                       "torch.aten.elu",           /* {6} */
                       getName(),                  /* {7} */
                       pointwiseAttr.getEluAlpha() /* {8} */
    );
  }
  case PointwiseAttr::Mode::GELU_FWD:
  case PointwiseAttr::Mode::GELU_APPROX_TANH_FWD: {
    const char *approx =
        pointwiseAttr.getMode() == PointwiseAttr::Mode::GELU_FWD ? "none"
                                                                 : "tanh";
    return std::format(kGeluSchema, permuteIN0, /* {0} */
                       getResultNamesAsm(),     /* {1} */
                       getOperandNamesAsm(),    /* {2} */
                       getOperandTypesAsm(),    /* {3} */
                       getResultTypesAsm(),     /* {4} */
                       permuteOUT0,             /* {5} */
                       "torch.aten.gelu",       /* {6} */
                       getName(),               /* {7} */
                       approx                   /* {8} */
    );
  }
  case PointwiseAttr::Mode::SOFTPLUS_FWD: {
    return std::format(kSoftplusSchema, permuteIN0,         /* {0} */
                       getResultNamesAsm(),                 /* {1} */
                       getOperandNamesAsm(),                /* {2} */
                       getOperandTypesAsm(),                /* {3} */
                       getResultTypesAsm(),                 /* {4} */
                       permuteOUT0,                         /* {5} */
                       "torch.aten.softplus",               /* {6} */
                       getName(),                           /* {7} */
                       pointwiseAttr.getSoftplusBeta(),     /* {8} */
                       pointwiseAttr.getSoftplusThreshold() /* {9} */
    );
  }
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(SWISH_FWD, torch.aten.silu)
    FUSILLI_DECLARE_UNARY_POINTWISE_EMITTER(IDENTITY, kIdentitySchema,
                                            torch.aten.clone)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(ERF, torch.aten.erf)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(EXP, torch.aten.exp)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(FLOOR, torch.aten.floor)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(LOG, torch.aten.log)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(LOGICAL_NOT, torch.aten.logical_not)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(NEG, torch.aten.neg)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(RECIPROCAL, torch.aten.reciprocal)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(RELU_FWD, torch.aten.relu)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(RSQRT, torch.aten.rsqrt)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(SIGMOID_FWD, torch.aten.sigmoid)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(SIN, torch.aten.sin)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(SQRT, torch.aten.sqrt)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(TANH_FWD, torch.aten.tanh)
    FUSILLI_DECLARE_UNARY_TORCH_EMITTER(TAN, torch.aten.tan)

    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_EQ, torch.aten.eq.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_LT, torch.aten.lt.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_LE, torch.aten.le.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_GT, torch.aten.gt.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_GE, torch.aten.ge.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(CMP_NEQ, torch.aten.ne.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(DIV, torch.aten.div.Tensor)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(LOGICAL_AND, torch.aten.logical_and)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(LOGICAL_OR, torch.aten.logical_or)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(MAX_OP, torch.aten.maximum)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(MIN_OP, torch.aten.minimum)
    FUSILLI_DECLARE_BINARY_TORCH_EMITTER(MUL, torch.aten.mul.Tensor)

    FUSILLI_DECLARE_SUB_ADD_TORCH_EMITTER(ADD, torch.aten.add.Tensor)
    FUSILLI_DECLARE_SUB_ADD_TORCH_EMITTER(SUB, torch.aten.sub.Tensor)

  case PointwiseAttr::Mode::ADD_SQUARE: {
    return std::format(kAddSquareSchema, permuteIN0, /* {0} */
                       permuteIN1,                   /* {1} */
                       getInputNameAsm(0),           /* {2} */
                       getInputNameAsm(1),           /* {3} */
                       getInputTypeAsm(1),           /* {4} */
                       getResultNamesAsm(),          /* {5} */
                       getResultTypesAsm(),          /* {6} */
                       permuteOUT0,                  /* {7} */
                       getName(),                    /* {8} */
                       getOperandTypesAsm()          /* {9} */
    );
  }

  default:
    assert(false && "Unsupported pointwise mode");
    return "";
  }
}

#undef FUSILLI_DECLARE_UNARY_POINTWISE_EMITTER
#undef FUSILLI_DECLARE_BINARY_POINTWISE_EMITTER
#undef FUSILLI_DECLARE_UNARY_TORCH_EMITTER
#undef FUSILLI_DECLARE_BINARY_TORCH_EMITTER
#undef FUSILLI_DECLARE_SUB_ADD_TORCH_EMITTER

//===----------------------------------------------------------------------===//
//
// ReductionNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ReductionNode's operand names in MLIR assembly format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations in a graph.
inline std::string ReductionNode::getOperandNamesAsm() const {
  const auto &x = reductionAttr.getX();
  std::string suffix = reductionAttr.getName();
  return x->getValueNameAsm() + "_" + suffix + "_perm";
}

// Emits ReductionNode's operand types in MLIR assembly format.
inline std::string ReductionNode::getOperandTypesAsm() const {
  const auto &x = reductionAttr.getX();
  return x->getTensorTypeAsm(/*isValueTensor=*/true, /*useLogicalDims=*/true);
}

// Emits ReductionNode's result names in MLIR assembly format.
inline std::string ReductionNode::getResultNamesAsm() const {
  return reductionAttr.getY()->getValueNameAsm();
}

// Emits ReductionNode's result types in MLIR assembly format.
inline std::string ReductionNode::getResultTypesAsm() const {
  return reductionAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

inline std::string ReductionNode::emitNodePreAsm() const {
  const auto &xT = reductionAttr.getX();
  const auto &yT = reductionAttr.getY();

  // Get which dimensions to reduce (validated in postValidateNode)
  std::vector<int64_t> reductionDims = getReductionDims();

  // Emit the reduction dimension list
  std::ostringstream dimListOss;
  std::string suffix = reductionAttr.getName();
  dimListOss << getListOfIntOpsAsm(reductionDims, "reduction_dims", suffix);

  std::string permuteX =
      getLayoutConversionOpsAsm(xT, "permute_X", suffix, /*isInput=*/true);
  std::string permuteY =
      getLayoutConversionOpsAsm(yT, "permute_Y", suffix, /*isInput=*/false);

  constexpr std::string_view kKeepdimReductionSchema = R"(
    {0}
    {1}
    %keepdim_{2} = torch.constant.bool true
    {3}_{2}_perm = {8} {4}, %reduction_dims_{2}, %keepdim_{2} : {5}, !torch.list<int>, !torch.bool -> {6}
    {7}
    )";

  constexpr std::string_view kKeepdimDtypeReductionSchema = R"(
    {0}
    {1}
    %keepdim_{2} = torch.constant.bool true
    %dtype_{2} = torch.constant.none
    {3}_{2}_perm = {8} {4}, %reduction_dims_{2}, %keepdim_{2}, %dtype_{2} : {5}, !torch.list<int>, !torch.bool, !torch.none -> {6}
    {7}
    )";

#define FUSILLI_DECLARE_REDUCTION_EMITTER(MODE, SCHEMA, OPIR)                  \
  case ReductionAttr::Mode::MODE: {                                            \
    return std::format(SCHEMA, permuteX,     /* {0} */                         \
                       dimListOss.str(),     /* {1} */                         \
                       suffix,               /* {2} */                         \
                       getResultNamesAsm(),  /* {3} */                         \
                       getOperandNamesAsm(), /* {4} */                         \
                       getOperandTypesAsm(), /* {5} */                         \
                       getResultTypesAsm(),  /* {6} */                         \
                       permuteY,             /* {7} */                         \
                       #OPIR                 /* {8} */                         \
    );                                                                         \
  }

#define FUSILLI_DECLARE_KEEPDIM_REDUCTION_EMITTER(MODE, OPIR)                  \
  FUSILLI_DECLARE_REDUCTION_EMITTER(MODE, kKeepdimReductionSchema, OPIR)

#define FUSILLI_DECLARE_KEEPDIM_DTYPE_REDUCTION_EMITTER(MODE, OPIR)            \
  FUSILLI_DECLARE_REDUCTION_EMITTER(MODE, kKeepdimDtypeReductionSchema, OPIR)

  switch (reductionAttr.getMode()) {
    FUSILLI_DECLARE_KEEPDIM_DTYPE_REDUCTION_EMITTER(SUM,
                                                    torch.aten.sum.dim_IntList)
    FUSILLI_DECLARE_KEEPDIM_REDUCTION_EMITTER(MIN, torch.aten.amin)
    FUSILLI_DECLARE_KEEPDIM_REDUCTION_EMITTER(MAX, torch.aten.amax)
  default:
    assert(false && "Unsupported reduction mode");
    return "";
  }
}

#undef FUSILLI_DECLARE_REDUCTION_EMITTER
#undef FUSILLI_DECLARE_KEEPDIM_REDUCTION_EMITTER
#undef FUSILLI_DECLARE_KEEPDIM_DTYPE_REDUCTION_EMITTER

//===----------------------------------------------------------------------===//
//
// SdpaNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits SdpaNode's operand names in MLIR assembly format.
//
// The unique suffix is included to ensure SSA uniqueness when the same
// tensor is used by multiple operations in a graph.
inline std::string SdpaNode::getOperandNamesAsm() const {
  std::string suffix = sdpaAttr.getName();
  std::ostringstream oss;
  oss << sdpaAttr.getQ()->getValueNameAsm() << "_" << suffix << "_perm, "
      << sdpaAttr.getK()->getValueNameAsm() << "_" << suffix << "_perm, "
      << sdpaAttr.getV()->getValueNameAsm() << "_" << suffix << "_perm";
  if (sdpaAttr.getMASK())
    oss << ", " << sdpaAttr.getMASK()->getValueNameAsm() << "_" << suffix
        << "_perm";
  else
    oss << ", %none_mask_" << suffix;
  return oss.str();
}

// Emits SdpaNode's operand types in MLIR assembly format.
inline std::string SdpaNode::getOperandTypesAsm() const {
  std::ostringstream oss;
  oss << sdpaAttr.getQ()->getTensorTypeAsm(/*isValueTensor=*/true,
                                           /*useLogicalDims=*/true)
      << ", "
      << sdpaAttr.getK()->getTensorTypeAsm(/*isValueTensor=*/true,
                                           /*useLogicalDims=*/true)
      << ", "
      << sdpaAttr.getV()->getTensorTypeAsm(/*isValueTensor=*/true,
                                           /*useLogicalDims=*/true)
      << ", ";
  if (sdpaAttr.getMASK())
    oss << sdpaAttr.getMASK()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
  else
    oss << "!torch.none";
  return oss.str();
}

// Emits SdpaNode's result names in MLIR assembly format.
inline std::string SdpaNode::getResultNamesAsm() const {
  return sdpaAttr.getO()->getValueNameAsm() + "_" + sdpaAttr.getName() +
         "_perm";
}

// Emits SdpaNode's result types in MLIR assembly format.
inline std::string SdpaNode::getResultTypesAsm() const {
  return sdpaAttr.getO()->getTensorTypeAsm(/*isValueTensor=*/true,
                                           /*useLogicalDims=*/true);
}

// Emits the dropout probability constant.
inline std::string SdpaNode::getDropoutOpsAsm() const {
  return torchFloatAsm("dropout", sdpaAttr.getName(), sdpaAttr.getDropout());
}

// Emits the is_causal boolean constant.
inline std::string SdpaNode::getIsCausalOpsAsm() const {
  return torchBoolAsm("is_causal", sdpaAttr.getName(), sdpaAttr.getIsCausal());
}

// Emits the scale constant (float or none).
inline std::string SdpaNode::getScaleOpsAsm() const {
  std::string suffix = sdpaAttr.getName();
  if (sdpaAttr.getScale().has_value())
    return torchFloatAsm("scale", suffix, *sdpaAttr.getScale());
  return torchNoneAsm("scale", suffix);
}

// Emits the enable_gqa boolean constant.
inline std::string SdpaNode::getEnableGqaOpsAsm() const {
  return torchBoolAsm("enable_gqa", sdpaAttr.getName(),
                      sdpaAttr.getEnableGqa());
}

inline std::string SdpaNode::emitNodePreAsm() const {
  std::string suffix = sdpaAttr.getName();

  // Permute inputs.
  std::string permuteQ = getLayoutConversionOpsAsm(sdpaAttr.getQ(), "permute_Q",
                                                   suffix, /*isInput=*/true);
  std::string permuteK = getLayoutConversionOpsAsm(sdpaAttr.getK(), "permute_K",
                                                   suffix, /*isInput=*/true);
  std::string permuteV = getLayoutConversionOpsAsm(sdpaAttr.getV(), "permute_V",
                                                   suffix, /*isInput=*/true);

  std::string mask;
  if (sdpaAttr.getMASK())
    mask = getLayoutConversionOpsAsm(sdpaAttr.getMASK(), "permute_mask", suffix,
                                     /*isInput=*/true);
  else
    mask = torchNoneAsm("none_mask", suffix);

  // Permute output.
  std::string permuteO = getLayoutConversionOpsAsm(sdpaAttr.getO(), "permute_O",
                                                   suffix, /*isInput=*/false);

  std::string operandNames = getOperandNamesAsm() + ", %dropout_" + suffix +
                             ", %is_causal_" + suffix + ", %scale_" + suffix +
                             ", %enable_gqa_" + suffix;

  // Scale type for the MLIR signature.
  std::string scaleType =
      sdpaAttr.getScale().has_value() ? "!torch.float" : "!torch.none";

  constexpr std::string_view schema = R"(
    {0}
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7}
    {8} = torch.aten.scaled_dot_product_attention {9} : {10}, !torch.float, !torch.bool, {11}, !torch.bool -> {12}
    {13}
  )";

  return std::format(schema,
                     permuteQ,             // {0}
                     permuteK,             // {1}
                     permuteV,             // {2}
                     mask,                 // {3}
                     getDropoutOpsAsm(),   // {4}
                     getIsCausalOpsAsm(),  // {5}
                     getScaleOpsAsm(),     // {6}
                     getEnableGqaOpsAsm(), // {7}
                     getResultNamesAsm(),  // {8}
                     operandNames,         // {9}
                     getOperandTypesAsm(), // {10}
                     scaleType,            // {11}
                     getResultTypesAsm(),  // {12}
                     permuteO              // {13}
  );
}

//===----------------------------------------------------------------------===//
//
// CustomOpNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

inline void replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), to);
    pos += to.length();
  }
}

inline std::string CustomOpNode::resolveMlirPlaceholders() const {
  std::string mlir = customOpAttr.getMlir();
  replaceAll(mlir, "{FUNC_NAME}", customOpAttr.getName());
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string iStr = std::to_string(i);
    replaceAll(mlir, "{IN" + iStr + "_DTYPE}",
               kDataTypeToMlirTypeAsm.at(inputs[i]->getDataType()));
    replaceAll(mlir, "{IN" + iStr + "_TYPE}",
               inputs[i]->getTensorTypeAsm(/*isValueTensor=*/true,
                                           /*useLogicalDims=*/true));
    const auto &dims = inputs[i]->getDim();
    for (size_t d = 0; d < dims.size(); ++d)
      replaceAll(mlir, "{IN" + iStr + "_DIM" + std::to_string(d) + "}",
                 std::to_string(dims[d]));
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string iStr = std::to_string(i);
    replaceAll(mlir, "{OUT" + iStr + "_DTYPE}",
               kDataTypeToMlirTypeAsm.at(outputs[i]->getDataType()));
    replaceAll(mlir, "{OUT" + iStr + "_TYPE}",
               outputs[i]->getTensorTypeAsm(/*isValueTensor=*/true,
                                            /*useLogicalDims=*/true));
    const auto &dims = outputs[i]->getDim();
    for (size_t d = 0; d < dims.size(); ++d)
      replaceAll(mlir, "{OUT" + iStr + "_DIM" + std::to_string(d) + "}",
                 std::to_string(dims[d]));
  }
  return mlir;
}

inline std::string CustomOpNode::emitModuleScopeAsm() const {
  std::string mlir = resolveMlirPlaceholders();
  if (!mlir.empty() && mlir.back() != '\n')
    mlir += '\n';
  return mlir;
}

// Emits CustomOpNode's call operand names in MLIR assembly format.
// These are the permuted input values passed directly to func.call.
inline std::string CustomOpNode::getCallOperandNamesAsm() const {
  std::ostringstream oss;
  std::string suffix = customOpAttr.getName();
  size_t idx = 0;
  interleave(
      inputs.begin(), inputs.end(),
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << std::format("{}_{}_i{}_perm", input->getValueNameAsm(), suffix,
                           idx++);
      },
      [&] { oss << ", "; });
  return oss.str();
}

// Emits CustomOpNode's call operand types in MLIR assembly format.
// Uses static logical tensor types (matching the func.func signature).
inline std::string CustomOpNode::getCallOperandTypesAsm() const {
  std::ostringstream oss;
  interleave(
      inputs.begin(), inputs.end(),
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << input->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true);
      },
      [&] { oss << ", "; });
  return oss.str();
}

// Emits CustomOpNode's call result names in MLIR assembly format.
// For single output: %name_suffix_perm (feeds directly into output permute)
// For multi-output: %name_suffix_res:N (individual results via %base#i)
inline std::string CustomOpNode::getCallResultNamesAsm() const {
  std::string suffix = customOpAttr.getName();
  if (outputs.size() == 1)
    return std::format("{}_{}_perm", outputs[0]->getValueNameAsm(), suffix);
  return std::format("{}_{}_res:{}", outputs[0]->getValueNameAsm(), suffix,
                     outputs.size());
}

// Emits CustomOpNode's call result types in MLIR assembly format.
// Uses static logical tensor types (matching the func.func signature).
inline std::string CustomOpNode::getCallResultTypesAsm() const {
  std::ostringstream oss;
  interleave(
      outputs.begin(), outputs.end(),
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true);
      },
      [&] { oss << ", "; });
  return oss.str();
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for the CustomOpNode. It generates:
//   1. Convert inputs physical -> logical (permute + expand if broadcast)
//   2. func.call to the custom function
//   3. Convert outputs logical -> physical (permute)
inline std::string CustomOpNode::emitNodePreAsm() const {
  std::ostringstream oss;
  std::string suffix = customOpAttr.getName();

  // 1. For each input: layout conversion (physical → logical).
  // Use per-input indexed suffix to ensure unique SSA names when the same
  // tensor appears in multiple input slots (e.g., g.customOp({A, A}, attr)).
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string inputSuffix = suffix + "_i" + std::to_string(i);
    std::string convPrefix = "permute_IN_" + std::to_string(i);
    oss << "\n    "
        << getLayoutConversionOpsAsm(inputs[i], convPrefix, inputSuffix,
                                     /*isInput=*/true);
  }

  // 2. func.call — use the node name as the callee (matches {FUNC_NAME}
  // resolved in the module-scope definition).
  std::string resultTypes = getCallResultTypesAsm();
  if (outputs.size() > 1)
    resultTypes = "(" + resultTypes + ")";

  constexpr std::string_view kCallSchema = R"(
    {0} = func.call @{1}({2}) : ({3}) -> {4})";
  oss << std::format(kCallSchema,
                     getCallResultNamesAsm(),  // {0}
                     customOpAttr.getName(),   // {1}
                     getCallOperandNamesAsm(), // {2}
                     getCallOperandTypesAsm(), // {3}
                     resultTypes               // {4}
  );

  // 3. For each output: layout conversion (logical → physical).
  // For multi-output, func.call produces %base:N and individual results
  // are accessed via %base#0, %base#1, etc., so we pass the #i name as
  // the operand override to the layout conversion.
  std::string multiResultBase =
      outputs[0]->getValueNameAsm() + "_" + suffix + "_res";
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string permutePrefix = "permute_OUT_" + std::to_string(i);
    std::string operandOverride;
    if (outputs.size() > 1)
      operandOverride = multiResultBase + "#" + std::to_string(i);
    oss << "\n    "
        << getLayoutConversionOpsAsm(outputs[i], permutePrefix, suffix,
                                     /*isInput=*/false, operandOverride);
  }

  return oss.str();
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_ASM_EMITTER_H
