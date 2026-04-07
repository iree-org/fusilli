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

// Returns the sizes at the given dimension indices. Used to extract the
// normalized shape from the input tensor. For example, for input
// [N, C, H, W] with reductionDims = {1, 2, 3} (LayerNorm), returns
// [C, H, W]. With reductionDims = {2, 3} (RMSNorm), returns [H, W].
inline std::vector<int64_t>
getNormalizedShape(const std::vector<int64_t> &xDim,
                   const std::vector<size_t> &reductionDims) {
  std::vector<int64_t> shape;
  shape.reserve(reductionDims.size());
  for (size_t i : reductionDims) {
    shape.push_back(xDim[i]);
  }
  return shape;
}

// Returns dim and unit strides for training forward outputs (e.g. MEAN,
// INV_VARIANCE, INV_RMS), where the reduction dimensions are collapsed to 1
// and all other dimensions are preserved from xDim. For example, for input
// [N, C, H, W] with reductionDims = {1, 2, 3} (LayerNorm), returns
// [N, 1, 1, 1]. With reductionDims = {2, 3} (RMSNorm), returns [N, C, 1, 1].
inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
getTrainingForwardOutputDimAndStride(const std::vector<int64_t> &xDim,
                                     const std::vector<size_t> &reductionDims) {
  std::vector<int64_t> dim = xDim;
  for (size_t i : reductionDims) {
    dim[i] = 1;
  }
  std::vector<int64_t> stride =
      generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()));
  return {dim, stride};
}

// Returns the expected shape for scale (and bias) tensors. The result has
// the same rank as xDim, with dimensions at positions in batchDims and
// reductionDims set to 1, and all other dimensions preserved from xDim.
// The caller decides which dims to collapse:
//   - LayerNorm:  getScaleBiasDim(xDim, {0}, {})      -> [1, C, H, W]
//   - RMSNorm:    getScaleBiasDim(xDim, {0}, {2, 3})  -> [1, C, 1, 1]
inline std::vector<int64_t>
getScaleBiasDim(const std::vector<int64_t> &xDim,
                const std::vector<size_t> &batchDims,
                const std::vector<size_t> &reductionDims) {
  std::vector<int64_t> dim = xDim;
  for (size_t i : batchDims) {
    dim[i] = 1;
  }
  for (size_t i : reductionDims) {
    dim[i] = 1;
  }
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

// Infers dim and stride of a scale/bias tensor. See getScaleBiasDim for
// the meaning of batchDims and reductionDims. Stride depends on the
// tensor's dim (which may be just-inferred), so dim must be set before
// computing stride.
inline void inferScaleBiasDimAndStride(
    std::shared_ptr<TensorAttr> &tensor, const std::vector<int64_t> &xDim,
    const std::vector<int64_t> &xStride, const std::vector<size_t> &batchDims,
    const std::vector<size_t> &reductionDims) {
  inferDim(tensor, getScaleBiasDim(xDim, batchDims, reductionDims));
  inferStride(tensor, getScaleBiasStride(tensor->getDim(), xStride));
}

} // namespace fusilli::norm_utils

#endif // FUSILLI_NODE_NORM_UTILS_H
