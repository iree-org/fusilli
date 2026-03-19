// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Shared utilities for normalization nodes (LayerNorm, RmsNorm, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_NORM_UTILS_H
#define FUSILLI_NODE_NORM_UTILS_H

#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace fusilli::norm_utils {

// Returns the shape over which normalization is applied:
// the input tensor's shape excluding the batch dimension,
// as normalization is computed independently for each sample in the batch.
inline std::vector<int64_t> getNormalizedShape(const std::vector<int64_t> &xDim,
                                               size_t batchDim = 0) {
  std::vector<int64_t> shape;
  shape.reserve(xDim.size() - 1);
  for (size_t i = 0; i < xDim.size(); ++i) {
    if (i != batchDim)
      shape.push_back(xDim[i]);
  }
  return shape;
}

// Returns [1, ..., B, ..., 1] dim and unit strides for training forward outputs
// (e.g. MEAN, INV_VARIANCE, INV_RMS), where only the batch dimension is
// preserved from xDim.
inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
getTrainingForwardOutputDimAndStride(const std::vector<int64_t> &xDim,
                                     size_t batchDim = 0) {
  std::vector<int64_t> dim(xDim.size(), 1);
  dim[batchDim] = xDim[batchDim];
  std::vector<int64_t> stride =
      generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));
  return {dim, stride};
}

// Returns the expected shape for scale (and bias) tensors:
// input X tensor's dims with single batch.
inline std::vector<int64_t> getScaleBiasDim(const std::vector<int64_t> &xDim,
                                            size_t batchDim = 0) {
  auto dim = xDim;
  dim[batchDim] = 1;
  return dim;
}

// Returns the stride for scale (and bias) tensors based on input stride order.
inline std::vector<int64_t>
getScaleBiasStride(const std::vector<int64_t> &scaleBiasDim,
                   const std::vector<int64_t> &xStride) {
  const auto strideOrder =
      generateStrideOrderPreservingFormat(xStride, scaleBiasDim.size());
  return generateStrideFromDim(scaleBiasDim, strideOrder);
}

// Infers dim of a tensor if not already set.
inline void inferDim(std::shared_ptr<TensorAttr> &tensor,
                     const std::vector<int64_t> &dim) {
  if (tensor->getDim().empty())
    tensor->setDim(dim);
}

// Infers stride of a tensor if not already set.
inline void inferStride(std::shared_ptr<TensorAttr> &tensor,
                        const std::vector<int64_t> &stride) {
  if (tensor->getStride().empty())
    tensor->setStride(stride);
}

// Infers dim and stride of a tensor if they are not already set.
inline void inferDimAndStride(std::shared_ptr<TensorAttr> &tensor,
                              const std::vector<int64_t> &dim,
                              const std::vector<int64_t> &stride) {
  inferDim(tensor, dim);
  inferStride(tensor, stride);
}

// Infers dim and stride of a scale/bias tensor.
// Stride depends on the tensor's dim (which may be just-inferred), so dim
// must be set before computing stride.
inline void inferScaleBiasDimAndStride(std::shared_ptr<TensorAttr> &tensor,
                                       const std::vector<int64_t> &xDim,
                                       const std::vector<int64_t> &xStride) {
  inferDim(tensor, getScaleBiasDim(xDim));
  inferStride(tensor, getScaleBiasStride(tensor->getDim(), xStride));
}

} // namespace fusilli::norm_utils

#endif // FUSILLI_NODE_NORM_UTILS_H
