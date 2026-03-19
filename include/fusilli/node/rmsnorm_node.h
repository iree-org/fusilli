// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the RMS normalization node
// `RmsNormNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_RMSNORM_NODE_H
#define FUSILLI_NODE_RMSNORM_NODE_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/rmsnorm_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/node/norm_utils.h"
#include "fusilli/support/logging.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// RMS normalization node.
//===----------------------------------------------------------------------===//

class RmsNormNode : public NodeCRTP<RmsNormNode> {
public:
  RmsnormAttr rmsnormAttr;

  RmsNormNode(RmsnormAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), rmsnormAttr(std::move(attr)) {}

  // ASM emitter methods (inference mode only).
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getNormalizedShapeOpsAsm() const;
  std::string getEpsilonOpsAsm() const;

  const std::string &getName() const override final {
    return rmsnormAttr.getName();
  }
  Type getType() const override final { return Type::RmsNorm; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating RmsNormNode '"
                           << rmsnormAttr.getName() << "'");

    FUSILLI_RETURN_ERROR_IF(
        rmsnormAttr.getForwardPhase() == NormFwdPhase::NOT_SET,
        ErrorCode::AttributeNotSet, "RmsNorm forward phase not set");

    std::shared_ptr<TensorAttr> xT = rmsnormAttr.getX();
    std::shared_ptr<TensorAttr> sT = rmsnormAttr.getSCALE();
    std::shared_ptr<TensorAttr> yT = rmsnormAttr.getY();
    std::shared_ptr<TensorAttr> rT = rmsnormAttr.getINV_RMS();

    // Ensure mandatory input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "RmsNorm input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!yT, ErrorCode::AttributeNotSet,
                            "RmsNorm output tensor Y not set");

    // Shape and layout checks on input tensor.
    size_t xRank = xT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(
        xRank < 2, ErrorCode::InvalidAttribute,
        "RmsNorm input tensor X must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(!xT->isContiguous() && !xT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + xT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    // Shape and layout checks on scale tensor.
    // If scale tensor's dims/strides are not set, they will be inferred in
    // inferPropertiesNode().
    if (sT) {
      if (!sT->getDim().empty()) {
        FUSILLI_RETURN_ERROR_IF(
            sT->getDim() != norm_utils::getScaleBiasDim(xT->getDim()),
            ErrorCode::InvalidAttribute,
            "RmsNorm input tensor SCALE must have shape as "
            "tensor X with single batch");
      }

      if (!sT->getStride().empty()) {
        FUSILLI_RETURN_ERROR_IF(
            !sT->isContiguous() && !sT->isChannelsLast(),
            ErrorCode::NotImplemented,
            "Tensor '" + sT->getName() +
                "' is neither contiguous nor channels-last as "
                "defined by its stride");
      }
    }

    // Output tensor checks for training and inference forward phases.
    if (isTrainingForwardPhase()) {
      FUSILLI_RETURN_ERROR_IF(!rT, ErrorCode::AttributeNotSet,
                              "RmsNorm output tensor INV_RMS not set");
    } else {
      FUSILLI_RETURN_ERROR_IF(
          rT, ErrorCode::InvalidAttribute,
          "RmsNorm output tensor INV_RMS should not be set");
    }

    // Epsilon checks.
    std::shared_ptr<TensorAttr> eT = rmsnormAttr.getEpsilon();
    FUSILLI_RETURN_ERROR_IF(!eT, ErrorCode::AttributeNotSet,
                            "RmsNorm epsilon not set");
    FUSILLI_RETURN_ERROR_IF(!eT->isScalar(), ErrorCode::InvalidAttribute,
                            "RmsNorm epsilon must be a scalar constant");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for RmsNormNode '"
                           << rmsnormAttr.getName() << "'");

    rmsnormAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> xT = rmsnormAttr.getX();
    std::shared_ptr<TensorAttr> yT = rmsnormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Infer shape and stride of input SCALE tensor if they're not set.
    std::shared_ptr<TensorAttr> sT = rmsnormAttr.getSCALE();
    if (sT) {
      norm_utils::inferScaleBiasDimAndStride(sT, xDim, xT->getStride());
    }

    // Infer shape and stride of output Y tensor.
    // When stride is unspecified, preserve the stride order of xT.
    norm_utils::inferDimAndStride(yT, xDim, xT->getStride());

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] =
          norm_utils::getTrainingForwardOutputDimAndStride(xDim);

      // Infer shape and stride of output INV_RMS tensor.
      std::shared_ptr<TensorAttr> rT = rmsnormAttr.getINV_RMS();
      norm_utils::inferDimAndStride(rT, dim, stride);
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating RmsNormNode '"
                           << rmsnormAttr.getName() << "'");

    std::shared_ptr<TensorAttr> xT = rmsnormAttr.getX();
    std::shared_ptr<TensorAttr> yT = rmsnormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Shape check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(
        xDim != yT->getDim(), ErrorCode::InvalidAttribute,
        "RmsNorm output Y tensor must have the same shape as input X tensor");

    // Layout check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] =
          norm_utils::getTrainingForwardOutputDimAndStride(xDim);

      std::shared_ptr<TensorAttr> rT = rmsnormAttr.getINV_RMS();

      // Shape check for output INV_RMS tensor
      FUSILLI_RETURN_ERROR_IF(
          dim != rT->getDim(), ErrorCode::InvalidAttribute,
          "RmsNorm output INV_RMS tensor must have shape [B, 1, ..., 1] with "
          "rank equal to input X tensor's rank, and batch dimension equal "
          "to input X tensor's batch dimension");
      // Stride check for output INV_RMS tensor
      FUSILLI_RETURN_ERROR_IF(
          stride != rT->getStride(), ErrorCode::InvalidAttribute,
          "RmsNorm output INV_RMS tensor must have unit strides");
    }

    FUSILLI_RETURN_ERROR_IF(
        isTrainingForwardPhase(), ErrorCode::NotImplemented,
        "RmsNorm training mode is not yet supported: torch-mlir does not "
        "lower the training variant");

    return ok();
  }

private:
  inline bool isTrainingForwardPhase() const {
    return rmsnormAttr.getForwardPhase() == NormFwdPhase::TRAINING;
  }

  std::vector<int64_t> getNormalizedShape() const {
    return norm_utils::getNormalizedShape(rmsnormAttr.getX()->getDim());
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_RMSNORM_NODE_H
