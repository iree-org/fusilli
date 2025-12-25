// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the layer normalization node
// `LayernormNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_LAYERNORM_NODE_H
#define FUSILLI_NODE_LAYERNORM_NODE_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/layernorm_attributes.h"
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
// Layer normalization node.
//===----------------------------------------------------------------------===//

class LayernormNode : public NodeCRTP<LayernormNode> {
public:
  LayernormAttr layernormAttr;

  LayernormNode(LayernormAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), layernormAttr(std::move(attr)) {}

  const std::string &getName() const override final {
    return layernormAttr.getName();
  }
  Type getType() const override final { return Type::Layernorm; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating LayernormNode '"
                           << layernormAttr.getName() << "'");

    FUSILLI_RETURN_ERROR_IF(
        layernormAttr.getForwardPhase() == NormFwdPhase::NOT_SET,
        ErrorCode::AttributeNotSet, "Layernorm forward phase not set");

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> sT = layernormAttr.getSCALE();
    std::shared_ptr<TensorAttr> bT = layernormAttr.getBIAS();
    std::shared_ptr<TensorAttr> eT = layernormAttr.getEPSILON();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();
    std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
    std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();

    // Ensure mandatory input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "Layernorm input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!eT, ErrorCode::AttributeNotSet,
                            "Layernorm input tensor EPSILON not set");
    FUSILLI_RETURN_ERROR_IF(!yT, ErrorCode::AttributeNotSet,
                            "Layernorm output tensor Y not set");

    // Shape and layout checks on input tensor.
    size_t xRank = xT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(
        xRank < 2, ErrorCode::InvalidAttribute,
        "Layernorm input tensor X must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(!xT->isContiguous() && !xT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + xT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    // Shape and layout checks on scale tensor.
    if (sT) {
      std::vector<int64_t> expectedDim = xT->getDim();
      expectedDim[0] = 1;
      FUSILLI_RETURN_ERROR_IF(sT->getDim() != expectedDim,
                              ErrorCode::InvalidAttribute,
                              "Layernorm input tensor SCALE must have shape as "
                              "tensor X with single batch");

      FUSILLI_RETURN_ERROR_IF(
          !sT->isContiguous() && !sT->isChannelsLast(),
          ErrorCode::NotImplemented,
          "Tensor '" + sT->getName() +
              "' is neither contiguous nor channels-last as "
              "defined by its stride");
    }

    // Shape and layout checks on bias tensor.
    if (bT) {
      std::vector<int64_t> expectedDim = xT->getDim();
      expectedDim[0] = 1;
      FUSILLI_RETURN_ERROR_IF(bT->getDim() != expectedDim,
                              ErrorCode::InvalidAttribute,
                              "Layernorm input tensor BIAS must have shape as "
                              "tensor X with single batch");
      FUSILLI_RETURN_ERROR_IF(
          !bT->isContiguous() && !bT->isChannelsLast(),
          ErrorCode::NotImplemented,
          "Tensor '" + bT->getName() +
              "' is neither contiguous nor channels-last as "
              "defined by its stride");
    }

    // Epsilon should be set and be constant scalar.
    FUSILLI_RETURN_ERROR_IF(
        !eT->isScalar(), ErrorCode::InvalidAttribute,
        "Layernorm input tensor EPSILON must be a constant scalar");

    // Output tensor checks for training and inference forward phases.
    if (isTrainingForwardPhase()) {
      FUSILLI_RETURN_ERROR_IF(!mT, ErrorCode::AttributeNotSet,
                              "Layernorm output tensor MEAN not set");
      FUSILLI_RETURN_ERROR_IF(!vT, ErrorCode::AttributeNotSet,
                              "Layernorm output tensor INV_VARIANCE not set");
    } else {
      FUSILLI_RETURN_ERROR_IF(mT, ErrorCode::InvalidAttribute,
                              "Layernorm output tensor MEAN should not be set");
      FUSILLI_RETURN_ERROR_IF(
          vT, ErrorCode::InvalidAttribute,
          "Layernorm output tensor INV_VARIANCE should not be set");
    }

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for LayernormNode '"
                           << layernormAttr.getName() << "'");

    layernormAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Infer shape of output Y tensor.
    if (yT->getDim().empty()) {
      yT->setDim(xDim);
    }

    // Infer stride of output Y tensor.
    if (yT->getStride().empty()) {
      // When unspecified, preserve the stride order of xT (input tensor).
      yT->setStride(xT->getStride());
    }

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

      std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
      std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();

      // Infer shape of output MEAN tensor.
      if (mT->getDim().empty()) {
        mT->setDim(dim);
      }
      // Infer shape of output INV_VARIANCE tensor.
      if (vT->getDim().empty()) {
        vT->setDim(dim);
      }
      // Infer stride of output MEAN tensor.
      if (mT->getStride().empty()) {
        mT->setStride(stride);
      }
      // Infer stride of output INV_VARIANCE tensor.
      if (vT->getStride().empty()) {
        vT->setStride(stride);
      }
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating LayernormNode '"
                           << layernormAttr.getName() << "'");

    std::shared_ptr<TensorAttr> xT = layernormAttr.getX();
    std::shared_ptr<TensorAttr> yT = layernormAttr.getY();

    const std::vector<int64_t> &xDim = xT->getDim();

    // Shape check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(
        xDim != yT->getDim(), ErrorCode::InvalidAttribute,
        "Layernorm output Y tensor must have the same shape as input X tensor");

    // Layout check for output Y tensor.
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

      std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
      std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();

      // Shape check for output MEAN tensor
      FUSILLI_RETURN_ERROR_IF(
          dim != mT->getDim(), ErrorCode::InvalidAttribute,
          "Layernorm output MEAN tensor must have shape [B, 1, ..., 1] with "
          "rank equal to shape rank of input X tensor and batch dimension "
          "equal to "
          "input X tensor batch dimension");
      // Shape check for output INV_VARIANCE tensor
      FUSILLI_RETURN_ERROR_IF(dim != vT->getDim(), ErrorCode::InvalidAttribute,
                              "Layernorm output INV_VARIANCE tensor must have "
                              "shape [B, 1, ..., 1] with "
                              "rank equal to shape rank of input X tensor and "
                              "batch dimension equal to "
                              "input X tensor batch dimension");
      // Stride check for output MEAN tensor
      FUSILLI_RETURN_ERROR_IF(
          stride != mT->getStride(), ErrorCode::InvalidAttribute,
          "Layernorm output MEAN tensor must have unit strides");
      // Stride check for output INV_VARIANCE tensor
      FUSILLI_RETURN_ERROR_IF(
          stride != vT->getStride(), ErrorCode::InvalidAttribute,
          "Layernorm output INV_VARIANCE tensor must have unit strides");
    }

    return ok();
  }

private:
  inline bool isTrainingForwardPhase() const {
    return layernormAttr.getForwardPhase() == NormFwdPhase::TRAINING;
  }

  std::pair<std::vector<int64_t>, std::vector<int64_t>>
  getTrainingForwardOutputDimAndStride(const std::vector<int64_t> &xDim) const {
    // The MEAN and INV_VARIANCE tensors have shape [B, 1, ..., 1]
    std::vector<int64_t> dim(xDim.size(), 1);
    dim[0] = xDim[0];

    // Since MEAN and INV_VARIANCE tensors have shape [B, 1, ..., 1],
    // strides are always equal to [1, 1, ..., 1] for both contiguous and
    // channels-last layouts.
    std::vector<int64_t> stride(dim.size(), 1);
    return {dim, stride};
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_LAYERNORM_NODE_H
