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

  // ASM emitter methods.
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
        FUSILLI_RETURN_ERROR_IF(sT->getDim() != getScaleDim(xT->getDim()),
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

#define INFER_TENSOR_DIM_AND_STRIDE(TENSOR, DIM, STRIDE)                       \
  if (TENSOR->getDim().empty())                                                \
    TENSOR->setDim(DIM);                                                       \
  if (TENSOR->getStride().empty())                                             \
    TENSOR->setStride(STRIDE);

    // Infer shape and stride of input SCALE tensor if they're not set.
    std::shared_ptr<TensorAttr> sT = rmsnormAttr.getSCALE();
    if (sT) {
      INFER_TENSOR_DIM_AND_STRIDE(
          sT, getScaleDim(xDim), getScaleStride(sT->getDim(), xT->getStride()));
    }

    // Infer shape and stride of output Y tensor.
    // When stride is unspecified, preserve the stride order of xT.
    INFER_TENSOR_DIM_AND_STRIDE(yT, xDim, xT->getStride());

    if (isTrainingForwardPhase()) {
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

      // Infer shape and stride of output INV_RMS tensor.
      std::shared_ptr<TensorAttr> rT = rmsnormAttr.getINV_RMS();
      INFER_TENSOR_DIM_AND_STRIDE(rT, dim, stride);
    }
#undef INFER_TENSOR_DIM_AND_STRIDE

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
      const auto &[dim, stride] = getTrainingForwardOutputDimAndStride(xDim);

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

    return ok();
  }

private:
  inline bool isTrainingForwardPhase() const {
    return rmsnormAttr.getForwardPhase() == NormFwdPhase::TRAINING;
  }

  // Returns the shape over which normalization is applied:
  // the input tensor's shape excluding the batch dimension (dim 0),
  // as normalization is computed independently for each sample in the batch.
  std::vector<int64_t> getNormalizedShape() const {
    const std::vector<int64_t> &xDim = rmsnormAttr.getX()->getDim();
    std::vector<int64_t> normalizedShape(xDim.cbegin() + 1, xDim.cend());
    return normalizedShape;
  }

  std::pair<std::vector<int64_t>, std::vector<int64_t>>
  getTrainingForwardOutputDimAndStride(const std::vector<int64_t> &xDim) const {
    // The INV_RMS tensor has shape [B, 1, ..., 1]
    std::vector<int64_t> dim(xDim.size(), 1);
    dim[0] = xDim[0];

    // Since INV_RMS tensor has shape [B, 1, ..., 1],
    // strides are always equal to [1, 1, ..., 1] for both contiguous and
    // channels-last layouts.
    std::vector<int64_t> stride(dim.size(), 1);
    return {dim, stride};
  }

  std::vector<int64_t> getScaleDim(const std::vector<int64_t> &xDim) const {
    // The SCALE tensor is input X tensor's dims with single batch.
    auto dim = xDim;
    dim[0] = 1;
    return dim;
  }

  std::vector<int64_t>
  getScaleStride(const std::vector<int64_t> &scaleDim,
                 const std::vector<int64_t> &xStride) const {
    // The SCALE tensor has stride based on input stride order.
    const auto strideOrder =
        generateStrideOrderPreservingFormat(xStride, scaleDim.size());
    return generateStrideFromDim(scaleDim, strideOrder);
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_RMSNORM_NODE_H
