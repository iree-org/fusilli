// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the custom operation node
// `CustomOpNode`. A custom op node allows embedding user-provided MLIR
// functions within a regular Graph, composable alongside built-in ops.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_CUSTOM_OP_NODE_H
#define FUSILLI_NODE_CUSTOM_OP_NODE_H

#include "fusilli/attributes/custom_op_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

class CustomOpNode : public NodeCRTP<CustomOpNode> {
public:
  CustomOpAttr customOpAttr;
  std::vector<std::shared_ptr<TensorAttr>> inputs;
  std::vector<std::shared_ptr<TensorAttr>> outputs;

  CustomOpNode(CustomOpAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), customOpAttr(std::move(attr)) {}

  // ASM emitter methods (definitions in asm_emitter.h).
  ErrorOr<std::string> emitModuleScopeAsm() const override final;
  ErrorOr<std::string> emitNodePreAsm() const override final;

  // ASM emission helpers (definitions in asm_emitter.h).
  std::string getCallOperandNamesAsm() const;
  std::string getCallOperandTypesAsm() const;
  std::string getCallResultNamesAsm() const;
  std::string getCallResultTypesAsm() const;

  const std::string &getName() const override final {
    return customOpAttr.getName();
  }
  Type getType() const override final { return Type::Custom; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating CustomOpNode '"
                           << customOpAttr.getName() << "'");

    FUSILLI_RETURN_ERROR_IF(customOpAttr.getMlir().empty(),
                            ErrorCode::AttributeNotSet,
                            "CustomOp MLIR not set");

    FUSILLI_RETURN_ERROR_IF(inputs.empty(), ErrorCode::AttributeNotSet,
                            "CustomOp inputs not set");

    FUSILLI_RETURN_ERROR_IF(outputs.empty(), ErrorCode::AttributeNotSet,
                            "CustomOp outputs not set");

    for (size_t i = 0; i < inputs.size(); ++i) {
      FUSILLI_RETURN_ERROR_IF(!inputs[i], ErrorCode::AttributeNotSet,
                              "CustomOp input " + std::to_string(i) +
                                  " is null");
      FUSILLI_RETURN_ERROR_IF(
          inputs[i]->isScalar(), ErrorCode::InvalidAttribute,
          "CustomOp input " + std::to_string(i) + " is scalar (not supported)");
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      FUSILLI_RETURN_ERROR_IF(outputs[i]->isScalar(),
                              ErrorCode::InvalidAttribute,
                              "CustomOp output " + std::to_string(i) +
                                  " is scalar (not supported)");
    }

    return ok();
  }

  // Resolves all placeholders in the MLIR template:
  //   {FUNC_NAME}                    — node name
  //   {IN<i>_DTYPE}/{OUT<i>_DTYPE}   — element type (e.g., "f32")
  //   {IN<i>_TYPE}/{OUT<i>_TYPE}     — full value tensor type
  //   {IN<i>_DIM<j>}/{OUT<i>_DIM<j>} — single logical dimension
  //
  // Definition in asm_emitter.h (needs getTensorTypeAsm()).
  std::string resolveMlirPlaceholders() const;

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for CustomOpNode '"
                           << customOpAttr.getName() << "'");

    // Fill datatypes from context for inputs that need it.
    for (auto &input : inputs)
      input->fillFromContext(context);

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_CUSTOM_OP_NODE_H
