// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the batch normalization node
// `BatchNormNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_BATCHNORM_NODE_H
#define FUSILLI_NODE_BATCHNORM_NODE_H

#include "fusilli/attributes/batchnorm_attributes.h"
#include "fusilli/attributes/common.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// Batch normalization node.
//
// Batch norm normalizes over the (N, H, W, ...) dimensions independently for
// each channel C. The input X has logical shape [N, C, *] where C is at
// dimension 1 in logical (NCHW) order.
//
// Scale (gamma), bias (beta), running mean, and running variance are all 1D
// tensors of shape [C].
//
// Inference: requires running MEAN and VAR; outputs Y only.
// Training:  running MEAN and VAR are optional; outputs Y, SAVED_MEAN, and
//            SAVED_INV_VARIANCE.
//===----------------------------------------------------------------------===//

class BatchNormNode : public NodeCRTP<BatchNormNode> {
public:
  BatchnormAttr batchnormAttr;

  BatchNormNode(BatchnormAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), batchnormAttr(std::move(attr)) {}

  // ASM emitter methods.
  ErrorOr<std::string> emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getEpsilonOpsAsm() const;
  std::string getMomentumOpsAsm() const;

  const std::string &getName() const override final {
    return batchnormAttr.getName();
  }
  Type getType() const override final { return Type::BatchNorm; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating BatchNormNode '"
                           << batchnormAttr.getName() << "'");

    FUSILLI_RETURN_ERROR_IF(
        batchnormAttr.getForwardPhase() == NormFwdPhase::NOT_SET,
        ErrorCode::AttributeNotSet, "BatchNorm forward phase not set");

    std::shared_ptr<TensorAttr> xT = batchnormAttr.getX();
    std::shared_ptr<TensorAttr> yT = batchnormAttr.getY();

    // Ensure mandatory input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "BatchNorm input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!yT, ErrorCode::AttributeNotSet,
                            "BatchNorm output tensor Y not set");

    // Shape and layout checks on input tensor.
    size_t xRank = xT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(
        xRank < 2, ErrorCode::InvalidAttribute,
        "BatchNorm input tensor X must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(!xT->isContiguous() && !xT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + xT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    int64_t c = getChannelDim();

    // Validate optional 1D inputs (scale, bias, mean, var) if provided.
    auto validate1DTensor = [&](const std::shared_ptr<TensorAttr> &t,
                                const std::string &name) -> ErrorObject {
      if (!t)
        return ok();
      if (!t->getDim().empty()) {
        FUSILLI_RETURN_ERROR_IF(
            t->getDim().size() != 1 || t->getDim()[0] != c,
            ErrorCode::InvalidAttribute,
            "BatchNorm tensor " + name +
                " must be 1D with size equal to channel dimension C");
      }
      return ok();
    };

    FUSILLI_CHECK_ERROR(validate1DTensor(batchnormAttr.getSCALE(), "SCALE"));
    FUSILLI_CHECK_ERROR(validate1DTensor(batchnormAttr.getBIAS(), "BIAS"));
    FUSILLI_CHECK_ERROR(validate1DTensor(batchnormAttr.getMEAN(), "MEAN"));
    FUSILLI_CHECK_ERROR(validate1DTensor(batchnormAttr.getVAR(), "VAR"));

    // Inference requires running mean and variance.
    if (isInferenceForwardPhase()) {
      FUSILLI_RETURN_ERROR_IF(!batchnormAttr.getMEAN(),
                              ErrorCode::AttributeNotSet,
                              "BatchNorm inference requires running MEAN");
      FUSILLI_RETURN_ERROR_IF(!batchnormAttr.getVAR(),
                              ErrorCode::AttributeNotSet,
                              "BatchNorm inference requires running VAR");
      FUSILLI_RETURN_ERROR_IF(
          batchnormAttr.getSAVED_MEAN(), ErrorCode::InvalidAttribute,
          "BatchNorm SAVED_MEAN should not be set in inference mode");
      FUSILLI_RETURN_ERROR_IF(
          batchnormAttr.getSAVED_INV_VARIANCE(), ErrorCode::InvalidAttribute,
          "BatchNorm SAVED_INV_VARIANCE should not be set in inference mode");
    } else {
      // Training requires saved statistics outputs.
      FUSILLI_RETURN_ERROR_IF(!batchnormAttr.getSAVED_MEAN(),
                              ErrorCode::AttributeNotSet,
                              "BatchNorm training requires SAVED_MEAN output");
      FUSILLI_RETURN_ERROR_IF(
          !batchnormAttr.getSAVED_INV_VARIANCE(), ErrorCode::AttributeNotSet,
          "BatchNorm training requires SAVED_INV_VARIANCE output");
    }

    // Epsilon checks.
    std::shared_ptr<TensorAttr> eT = batchnormAttr.getEpsilon();
    FUSILLI_RETURN_ERROR_IF(!eT, ErrorCode::AttributeNotSet,
                            "BatchNorm epsilon not set");
    FUSILLI_RETURN_ERROR_IF(!eT->isScalar(), ErrorCode::InvalidAttribute,
                            "BatchNorm epsilon must be a scalar constant");

    // Momentum checks.
    std::shared_ptr<TensorAttr> mT = batchnormAttr.getMomentum();
    FUSILLI_RETURN_ERROR_IF(!mT, ErrorCode::AttributeNotSet,
                            "BatchNorm momentum not set");
    FUSILLI_RETURN_ERROR_IF(!mT->isScalar(), ErrorCode::InvalidAttribute,
                            "BatchNorm momentum must be a scalar constant");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for BatchNormNode '"
                           << batchnormAttr.getName() << "'");

    batchnormAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> xT = batchnormAttr.getX();
    std::shared_ptr<TensorAttr> yT = batchnormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();
    const std::vector<int64_t> channel1DDim = {xDim[1]};
    const std::vector<int64_t> channel1DStride = {1};

#define INFER_1D_TENSOR(TENSOR)                                                \
  if (TENSOR->getDim().empty())                                                \
    TENSOR->setDim(channel1DDim);                                              \
  if (TENSOR->getStride().empty())                                             \
    TENSOR->setStride(channel1DStride);

    // Infer 1D channel tensors.
    std::shared_ptr<TensorAttr> sT = batchnormAttr.getSCALE();
    if (sT) {
      INFER_1D_TENSOR(sT);
    }
    std::shared_ptr<TensorAttr> bT = batchnormAttr.getBIAS();
    if (bT) {
      INFER_1D_TENSOR(bT);
    }
    std::shared_ptr<TensorAttr> meanT = batchnormAttr.getMEAN();
    if (meanT) {
      INFER_1D_TENSOR(meanT);
    }
    std::shared_ptr<TensorAttr> varT = batchnormAttr.getVAR();
    if (varT) {
      INFER_1D_TENSOR(varT);
    }

    // Infer shape and stride of output Y tensor (same as X).
    if (yT->getDim().empty())
      yT->setDim(xDim);
    if (yT->getStride().empty())
      yT->setStride(xT->getStride());

    // Infer saved statistics shapes for training.
    if (isTrainingForwardPhase()) {
      std::shared_ptr<TensorAttr> smT = batchnormAttr.getSAVED_MEAN();
      INFER_1D_TENSOR(smT);
      std::shared_ptr<TensorAttr> sivT = batchnormAttr.getSAVED_INV_VARIANCE();
      INFER_1D_TENSOR(sivT);
    }

#undef INFER_1D_TENSOR

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating BatchNormNode '"
                           << batchnormAttr.getName() << "'");

    std::shared_ptr<TensorAttr> xT = batchnormAttr.getX();
    std::shared_ptr<TensorAttr> yT = batchnormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();
    const std::vector<int64_t> expectedCDim = {xDim[1]};

    // Shape check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(
        xDim != yT->getDim(), ErrorCode::InvalidAttribute,
        "BatchNorm output Y tensor must have the same shape as input X tensor");

    // Layout check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    // Shape checks for 1D channel tensors.
    auto check1DShape = [&](const std::shared_ptr<TensorAttr> &t,
                            const std::string &name) -> ErrorObject {
      if (!t)
        return ok();
      FUSILLI_RETURN_ERROR_IF(
          t->getDim() != expectedCDim, ErrorCode::InvalidAttribute,
          "BatchNorm tensor " + name +
              " must be 1D with size equal to channel dimension C");
      FUSILLI_RETURN_ERROR_IF(t->getStride() != std::vector<int64_t>{1},
                              ErrorCode::InvalidAttribute,
                              "BatchNorm tensor " + name +
                                  " must have unit stride");
      return ok();
    };

    FUSILLI_CHECK_ERROR(check1DShape(batchnormAttr.getSCALE(), "SCALE"));
    FUSILLI_CHECK_ERROR(check1DShape(batchnormAttr.getBIAS(), "BIAS"));
    FUSILLI_CHECK_ERROR(check1DShape(batchnormAttr.getMEAN(), "MEAN"));
    FUSILLI_CHECK_ERROR(check1DShape(batchnormAttr.getVAR(), "VAR"));

    if (isTrainingForwardPhase()) {
      FUSILLI_CHECK_ERROR(
          check1DShape(batchnormAttr.getSAVED_MEAN(), "SAVED_MEAN"));
      FUSILLI_CHECK_ERROR(check1DShape(batchnormAttr.getSAVED_INV_VARIANCE(),
                                       "SAVED_INV_VARIANCE"));
    }

    return ok();
  }

private:
  inline bool isInferenceForwardPhase() const {
    return batchnormAttr.getForwardPhase() == NormFwdPhase::INFERENCE;
  }

  inline bool isTrainingForwardPhase() const {
    return batchnormAttr.getForwardPhase() == NormFwdPhase::TRAINING;
  }

  // Returns the channel dimension count (C = dim[1] in logical NCHW order).
  int64_t getChannelDim() const { return batchnormAttr.getX()->getDim()[1]; }
};

} // namespace fusilli

#endif // FUSILLI_NODE_BATCHNORM_NODE_H
