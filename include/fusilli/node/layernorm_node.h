// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the layer normalization node
// `LayerNormNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_LAYERNORM_NODE_H
#define FUSILLI_NODE_LAYERNORM_NODE_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/layernorm_attributes.h"
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
// Layer normalization node.
//===----------------------------------------------------------------------===//

class LayerNormNode : public NodeCRTP<LayerNormNode> {
public:
  LayernormAttr layernormAttr;

  LayerNormNode(LayernormAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), layernormAttr(std::move(attr)) {}

  // ASM emitter methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getNormalizedShapeOpsAsm() const;
  std::string getEpsilonOpsAsm() const;

  const std::string &getName() const override final {
    return layernormAttr.getName();
  }
  Type getType() const override final { return Type::LayerNorm; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating LayerNormNode '"
                           << layernormAttr.getName() << "'");

    FUSILLI_RETURN_ERROR_IF(
        layernormAttr.getForwardPhase() == NormFwdPhase::NOT_SET,
        ErrorCode::AttributeNotSet, "LayerNorm forward phase not set");

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> sT = layernormAttr.getSCALE();
    std::shared_ptr<TensorAttr> bT = layernormAttr.getBIAS();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();
    std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
    std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();

    // Ensure mandatory input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "LayerNorm input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!yT, ErrorCode::AttributeNotSet,
                            "LayerNorm output tensor Y not set");

    // Shape and layout checks on input tensor.
    size_t xRank = xT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(
        xRank < 2, ErrorCode::InvalidAttribute,
        "LayerNorm input tensor X must have a rank of at least 2");
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
            "LayerNorm input tensor SCALE must have shape as "
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

    // Shape and layout checks on bias tensor.
    // If scale tensor's dims/strides are not set, they will be inferred in
    // inferPropertiesNode().
    if (bT) {
      if (!bT->getDim().empty()) {
        FUSILLI_RETURN_ERROR_IF(
            bT->getDim() != norm_utils::getScaleBiasDim(xT->getDim()),
            ErrorCode::InvalidAttribute,
            "LayerNorm input tensor BIAS must have shape as "
            "tensor X with single batch");
      }

      if (!bT->getStride().empty()) {
        FUSILLI_RETURN_ERROR_IF(
            !bT->isContiguous() && !bT->isChannelsLast(),
            ErrorCode::NotImplemented,
            "Tensor '" + bT->getName() +
                "' is neither contiguous nor channels-last as "
                "defined by its stride");
      }
    }

    // Output tensor checks for training and inference forward phases.
    if (isTrainingForwardPhase()) {
      FUSILLI_RETURN_ERROR_IF(!mT, ErrorCode::AttributeNotSet,
                              "LayerNorm output tensor MEAN not set");
      FUSILLI_RETURN_ERROR_IF(!vT, ErrorCode::AttributeNotSet,
                              "LayerNorm output tensor INV_VARIANCE not set");
    } else {
      FUSILLI_RETURN_ERROR_IF(mT, ErrorCode::InvalidAttribute,
                              "LayerNorm output tensor MEAN should not be set");
      FUSILLI_RETURN_ERROR_IF(
          vT, ErrorCode::InvalidAttribute,
          "LayerNorm output tensor INV_VARIANCE should not be set");
    }

    // Epsilon checks.
    std::shared_ptr<TensorAttr> eT = layernormAttr.getEpsilon();
    FUSILLI_RETURN_ERROR_IF(!eT, ErrorCode::AttributeNotSet,
                            "LayerNorm epsilon not set");
    FUSILLI_RETURN_ERROR_IF(!eT->isScalar(), ErrorCode::InvalidAttribute,
                            "LayerNorm epsilon must be a scalar constant");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for LayerNormNode '"
                           << layernormAttr.getName() << "'");

    layernormAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Infer shape and stride of input SCALE tensor if they're not set.
    std::shared_ptr<TensorAttr> sT = layernormAttr.getSCALE();
    if (sT) {
      norm_utils::inferScaleBiasDimAndStride(sT, xDim, xT->getStride());
    }

    // Infer shape and stride of input BIAS tensor if they're not set.
    std::shared_ptr<TensorAttr> bT = layernormAttr.getBIAS();
    if (bT) {
      norm_utils::inferScaleBiasDimAndStride(bT, xDim, xT->getStride());
    }

    // Infer shape and stride of output Y tensor.
    // When stride is unspecified, preserve the stride order of xT.
    norm_utils::inferDimAndStride(yT, xDim, xT->getStride());

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] =
          norm_utils::getTrainingForwardOutputDimAndStride(xDim);

      // Infer shape and stride of output MEAN tensor.
      std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
      norm_utils::inferDimAndStride(mT, dim, stride);

      // Infer shape and stride of output INV_VARIANCE tensor.
      std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();
      norm_utils::inferDimAndStride(vT, dim, stride);
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating LayerNormNode '"
                           << layernormAttr.getName() << "'");

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Shape check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(
        xDim != yT->getDim(), ErrorCode::InvalidAttribute,
        "LayerNorm output Y tensor must have the same shape as input X tensor");

    // Layout check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] =
          norm_utils::getTrainingForwardOutputDimAndStride(xDim);

      std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
      std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();

      // Shape check for output MEAN tensor
      FUSILLI_RETURN_ERROR_IF(
          dim != mT->getDim(), ErrorCode::InvalidAttribute,
          "Layernorm output MEAN tensor must have shape [B, 1, ..., 1] with "
          "rank equal to input X tensor's rank, and batch dimension equal "
          "to input X tensor's batch dimension");
      // Shape check for output INV_VARIANCE tensor
      FUSILLI_RETURN_ERROR_IF(
          dim != vT->getDim(), ErrorCode::InvalidAttribute,
          "LayerNorm output INV_VARIANCE tensor must have "
          "shape [B, 1, ..., 1] with  rank equal to "
          "input X tensor's rank, and batch dimension equal "
          "to input X tensor's batch dimension");
      // Stride check for output MEAN tensor
      FUSILLI_RETURN_ERROR_IF(
          stride != mT->getStride(), ErrorCode::InvalidAttribute,
          "LayerNorm output MEAN tensor must have unit strides");
      // Stride check for output INV_VARIANCE tensor
      FUSILLI_RETURN_ERROR_IF(
          stride != vT->getStride(), ErrorCode::InvalidAttribute,
          "LayerNorm output INV_VARIANCE tensor must have unit strides");
    }

    return ok();
  }

private:
  inline bool isTrainingForwardPhase() const {
    return layernormAttr.getForwardPhase() == NormFwdPhase::TRAINING;
  }

  std::vector<int64_t> getNormalizedShape() const {
    return norm_utils::getNormalizedShape(layernormAttr.getX()->getDim());
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_LAYERNORM_NODE_H
