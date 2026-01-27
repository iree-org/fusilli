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
            sT->getDim() != getScaleBiasDim(xT->getDim()),
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
            bT->getDim() != getScaleBiasDim(xT->getDim()),
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

#define INFER_TENSOR_DIM_AND_STRIDE(TENSOR, DIM, STRIDE)                       \
  if (TENSOR->getDim().empty())                                                \
    TENSOR->setDim(DIM);                                                       \
  if (TENSOR->getStride().empty())                                             \
    TENSOR->setStride(STRIDE);

    // Infer shape and stride of input SCALE tensor if they're not set.
    std::shared_ptr<TensorAttr> sT = layernormAttr.getSCALE();
    if (sT) {
      INFER_TENSOR_DIM_AND_STRIDE(
          sT, getScaleBiasDim(xDim),
          getScaleBiasStride(sT->getDim(), xT->getStride()));
    }

    // Infer shape and stride of input BIAS tensor if they're not set.
    std::shared_ptr<TensorAttr> bT = layernormAttr.getBIAS();
    if (bT) {
      INFER_TENSOR_DIM_AND_STRIDE(
          bT, getScaleBiasDim(xDim),
          getScaleBiasStride(bT->getDim(), xT->getStride()));
    }

    // Infer shape and stride of output Y tensor.
    // When stride is unspecified, preserve the stride order of xT.
    INFER_TENSOR_DIM_AND_STRIDE(yT, xDim, xT->getStride());

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

      // Infer shape and stride of output MEAN tensor.
      std::shared_ptr<TensorAttr> mT = layernormAttr.getMEAN();
      INFER_TENSOR_DIM_AND_STRIDE(mT, dim, stride);

      // Infer shape and stride of output INV_VARIANCE tensor.
      std::shared_ptr<TensorAttr> vT = layernormAttr.getINV_VARIANCE();
      INFER_TENSOR_DIM_AND_STRIDE(vT, dim, stride);
    }
#undef INFER_TENSOR_DIM_AND_STRIDE

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
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

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

  // Returns the shape over which normalization is applied:
  // the input tensor's shape excluding the batch dimension (dim 0),
  // as normalization is computed independently for each sample in the batch.
  std::vector<int64_t> getNormalizedShape() const {
    const std::vector<int64_t> &xDim = layernormAttr.getX()->getDim();
    std::vector<int64_t> normalizedShape(xDim.cbegin() + 1, xDim.cend());
    return normalizedShape;
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

  std::vector<int64_t> getScaleBiasDim(const std::vector<int64_t> &xDim) const {
    // The SCALE and BIAS tensors are input X tensor's dims with single batch.
    auto dim = xDim;
    dim[0] = 1;
    return dim;
  }

  std::vector<int64_t>
  getScaleBiasStride(const std::vector<int64_t> &scaleBiasDim,
                     const std::vector<int64_t> &xStride) const {
    // The SCALE and BIAS tensors have stride based on input stride order.
    const auto strideOrder =
        generateStrideOrderPreservingFormat(xStride, scaleBiasDim.size());
    return generateStrideFromDim(scaleBiasDim, strideOrder);
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_LAYERNORM_NODE_H
