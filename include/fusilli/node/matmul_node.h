// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the matrix multiplication node
// `MatmulNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_MATMUL_NODE_H
#define FUSILLI_NODE_MATMUL_NODE_H

#include "fusilli/attributes/matmul_attributes.h"
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
// Helper functions for matrix multiplication nodes.
//===----------------------------------------------------------------------===//

// Infer the output shape of a matrix multiplication operation from the input
// shapes. For matrices A [..., M, K] and B [..., K, N], the output is [..., M,
// N].
inline std::vector<int64_t>
getMatmulInferredOutputShape(const std::vector<int64_t> &aDim,
                             const std::vector<int64_t> &bDim) {
  constexpr int64_t kNonBatchRank = 2;
  size_t rank = aDim.size();
  assert(rank == bDim.size() && "Input tensors must have the same rank");
  assert(rank >= kNonBatchRank && "Input tensors must have rank >= 2");

  std::vector<int64_t> cDim(rank);

  // Handle batch dimensions (broadcast if necessary)
  size_t batchDims = rank - kNonBatchRank;
  for (size_t i = 0; i < batchDims; ++i) {
    int64_t aDimVal = aDim[i];
    int64_t bDimVal = bDim[i];
    // Use the maximum of the two dimensions (broadcasting rule)
    assert((aDimVal % bDimVal == 0 || bDimVal % aDimVal == 0) &&
           "Incompatible dimensions for broadcasting");
    cDim[i] = std::max(aDimVal, bDimVal);
  }

  // Matrix dimensions: M from A, N from B
  cDim[rank - 2] = aDim[rank - 2]; // M
  cDim[rank - 1] = bDim[rank - 1]; // N

  return cDim;
}

//===----------------------------------------------------------------------===//
// Matrix multiplication node.
//===----------------------------------------------------------------------===//

class MatmulNode : public NodeCRTP<MatmulNode> {
public:
  MatmulAttr matmulAttr;

  MatmulNode(MatmulAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), matmulAttr(std::move(attr)) {}

  const std::string &getName() const override final {
    return matmulAttr.getName();
  }
  Type getType() const override final { return Type::Matmul; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating MatmulNode '"
                           << matmulAttr.getName() << "'");

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    // Ensure input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!aT, ErrorCode::AttributeNotSet,
                            "Matmul input tensor A not set");
    FUSILLI_RETURN_ERROR_IF(!bT, ErrorCode::AttributeNotSet,
                            "Matmul input tensor B not set");
    FUSILLI_RETURN_ERROR_IF(!cT, ErrorCode::AttributeNotSet,
                            "Matmul output tensor C not set");

    size_t aRank = aT->getDim().size();
    size_t bRank = bT->getDim().size();

    // Rank checks on input tensors (must be at least rank 2).
    constexpr int64_t kNonBatchRank = 2;
    FUSILLI_RETURN_ERROR_IF(
        aRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul input tensor A must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(
        bRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul input tensor B must have a rank of at least 2");

    // Check that input tensors have the same rank.
    FUSILLI_RETURN_ERROR_IF(
        aRank != bRank, ErrorCode::InvalidAttribute,
        "Matmul input tensors A and B must have the same rank: A has rank=" +
            std::to_string(aRank) + ", B has rank=" + std::to_string(bRank));

    // Check that inner dimensions match (K dimension).
    const std::vector<int64_t> &aDim = aT->getDim();
    const std::vector<int64_t> &bDim = bT->getDim();
    int64_t aK = aDim[aRank - 1]; // Last dimension of A
    int64_t bK = bDim[bRank - 2]; // Second-to-last dimension of B

    FUSILLI_RETURN_ERROR_IF(
        aK != bK, ErrorCode::InvalidAttribute,
        "Matmul input tensors A and B have incompatible inner dimensions (K): "
        "A has K=" +
            std::to_string(aK) + ", B has K=" + std::to_string(bK));

    // Check that batch dimensions are broadcastable.
    // Since both inputs have the same rank, we can directly compare batch dims.
    size_t batchDims = aRank - kNonBatchRank;
    for (size_t i = 0; i < batchDims; ++i) {
      int64_t aDimVal = aDim[i];
      int64_t bDimVal = bDim[i];
      FUSILLI_RETURN_ERROR_IF(
          !(aDimVal % bDimVal == 0 || bDimVal % aDimVal == 0),
          ErrorCode::InvalidAttribute,
          "Matmul input tensors A and B have incompatible batch dimensions for "
          "broadcasting at index " +
              std::to_string(i) + ": A has dim=" + std::to_string(aDimVal) +
              ", B has dim=" + std::to_string(bDimVal));
    }

    FUSILLI_CHECK_ERROR(checkBatchDims(aT, "A"));
    FUSILLI_CHECK_ERROR(checkBatchDims(bT, "B"));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for MatmulNode '"
                           << matmulAttr.getName() << "'");

    matmulAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    const std::vector<int64_t> &aDim = aT->getDim();
    const std::vector<int64_t> &bDim = bT->getDim();

    const std::vector<int64_t> &cDim = cT->getDim();
    const std::vector<int64_t> &cStride = cT->getStride();

    // Infer shape of output tensor.
    if (cDim.empty())
      cT->setDim(getMatmulInferredOutputShape(aDim, bDim));

    // Output stride is contiguous (row-major) when unspecified.
    if (cStride.empty()) {
      cT->setStride(
          generateStrideFromDim(cDim, getContiguousStrideOrder(cDim.size())));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating MatmulNode '"
                           << matmulAttr.getName() << "'");

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    size_t cRank = cT->getDim().size();

    // Rank checks
    constexpr int64_t kNonBatchRank = 2;
    FUSILLI_RETURN_ERROR_IF(
        cRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul output tensor C must have a rank of at least 2");

    FUSILLI_RETURN_ERROR_IF(
        cT->getDim() !=
            getMatmulInferredOutputShape(aT->getDim(), bT->getDim()),
        ErrorCode::InvalidAttribute,
        "Matmul output tensor C dimensions do not match the expected shapes "
        "inferred based on the input dimensions");
    FUSILLI_CHECK_ERROR(checkBatchDims(cT, "C"));
    return ok();
  }

private:
  // Check that batch dimensions are outermost and non-transposed.
  // This is equivalent to checking that perm[i] == i for all batch dims.
  ErrorObject checkBatchDims(const std::shared_ptr<TensorAttr> &tensor,
                             const std::string &name) const {
    constexpr int64_t kNonBatchRank = 2;
    size_t batchDims = tensor->getDim().size() - kNonBatchRank;
    std::vector<int64_t> perm = tensor->getLogicalToPhysicalPermuteOrder();
    for (size_t i = 0; i < batchDims; ++i) {
      FUSILLI_RETURN_ERROR_IF(
          perm[i] != static_cast<int64_t>(i), ErrorCode::InvalidAttribute,
          "Matmul tensor " + name +
              " has batch dimensions that are not outermost or are "
              "transposed");
    }
    return ok();
  };
};

} // namespace fusilli

#endif // FUSILLI_NODE_MATMUL_NODE_H
