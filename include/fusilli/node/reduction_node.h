// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the reduction nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_REDUCTION_NODE_H
#define FUSILLI_NODE_REDUCTION_NODE_H

#include "fusilli/attributes/reduction_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fusilli {

class ReductionNode : public NodeCRTP<ReductionNode> {
public:
  ReductionAttr reductionAttr;

  ReductionNode(ReductionAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), reductionAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;

  // Returns the list of dimension indices that are reduced.
  // A dimension is reduced if Y[i] == 1 and X[i] > 1.
  std::vector<int64_t> getReductionDims() const {
    std::vector<int64_t> reductionDims;
    const auto &xDim = reductionAttr.getX()->getDim();
    const auto &yDim = reductionAttr.getY()->getDim();
    for (size_t i = 0; i < xDim.size(); ++i) {
      if (yDim[i] == 1 && xDim[i] > 1)
        reductionDims.push_back(static_cast<int64_t>(i));
    }
    return reductionDims;
  }

  const std::string &getName() const override final {
    return reductionAttr.getName();
  }
  Type getType() const override final { return Type::Reduction; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating ReductionNode '"
                           << reductionAttr.getName() << "'");
    FUSILLI_RETURN_ERROR_IF(
        reductionAttr.getMode() == ReductionAttr::Mode::NOT_SET,
        ErrorCode::AttributeNotSet, "Reduction mode not set");

    // Validate input X exists
    FUSILLI_RETURN_ERROR_IF(!reductionAttr.getX(), ErrorCode::AttributeNotSet,
                            "Reduction operation requires X input");

    // Validate output Y exists
    FUSILLI_RETURN_ERROR_IF(!reductionAttr.getY(), ErrorCode::AttributeNotSet,
                            "Reduction operation requires Y output");

    // Validate that X dimensions are set
    FUSILLI_RETURN_ERROR_IF(reductionAttr.getX()->getDim().empty(),
                            ErrorCode::AttributeNotSet,
                            "Reduction input X dimensions not set");
    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for ReductionNode '"
                           << reductionAttr.getName() << "'");

    // Fill missing properties from context (including data types)
    reductionAttr.fillFromContext(context);

    // If Y dimensions are not set, default to same shape as X (no reduction)
    // User must explicitly set which dimensions to reduce by setting output
    // dims
    const auto &xTensor = reductionAttr.getX();
    const auto &yTensor = reductionAttr.getY();
    if (yTensor->getDim().empty()) {
      yTensor->setDim(xTensor->getDim());
    }

    if (yTensor->getStride().empty()) {
      // Compute stride for output based on its dimensions
      yTensor->setStride(generateStrideFromDim(
          yTensor->getDim(),
          getContiguousStrideOrder(yTensor->getDim().size())));
    }
    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating ReductionNode '"
                           << reductionAttr.getName() << "'");

    // Validate that input and output have the same rank
    const auto &xTensor = reductionAttr.getX();
    const auto &yTensor = reductionAttr.getY();
    FUSILLI_RETURN_ERROR_IF(
        xTensor->getDim().size() != yTensor->getDim().size(),
        ErrorCode::InvalidAttribute,
        "Reduction input and output must have the same rank");

    // Validate reduction dimensions - if Y[i] differs from X[i], Y[i] must be 1
    const auto &xDim = xTensor->getDim();
    const auto &yDim = yTensor->getDim();
    for (size_t i = 0; i < xDim.size(); ++i) {
      FUSILLI_RETURN_ERROR_IF(
          yDim[i] != 1 && yDim[i] != xDim[i], ErrorCode::InvalidAttribute,
          "Reduction output dimension " + std::to_string(i) +
              " must be 1 or match input dimension");
    }

    // Validate that at least one dimension is being reduced
    FUSILLI_RETURN_ERROR_IF(getReductionDims().empty(),
                            ErrorCode::InvalidAttribute,
                            "Reduction requires at least one dimension to "
                            "reduce (Y[i] == 1 where X[i] > 1)");
    return ok();
  }
};
} // namespace fusilli

#endif // FUSILLI_NODE_REDUCTION_NODE_H
