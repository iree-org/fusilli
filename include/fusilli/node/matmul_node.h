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

#include <algorithm>
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
inline ErrorOr<std::vector<int64_t>>
tryGetMatmulInferredOutputShape(const std::vector<int64_t> &aDim,
                                const std::vector<int64_t> &bDim) {
  constexpr size_t kNonBatchRank = 2;
  size_t aRank = aDim.size();
  size_t bRank = bDim.size();
  FUSILLI_RETURN_ERROR_IF(aRank < kNonBatchRank || bRank < kNonBatchRank,
                          ErrorCode::InvalidAttribute,
                          "Matmul input tensors must have rank >= 2");

  std::vector<int64_t> aBatchDim(aDim.begin(), aDim.end() - kNonBatchRank);
  std::vector<int64_t> bBatchDim(bDim.begin(), bDim.end() - kNonBatchRank);
  size_t cBatchRank = std::max(aBatchDim.size(), bBatchDim.size());
  std::vector<int64_t> cDim(cBatchRank + kNonBatchRank, 1);

  // Broadcast batch dimensions using PyTorch/NumPy right-aligned semantics.
  for (size_t offset = 0; offset < cBatchRank; ++offset) {
    int64_t aDimVal = offset < aBatchDim.size()
                          ? aBatchDim[aBatchDim.size() - 1 - offset]
                          : 1;
    int64_t bDimVal = offset < bBatchDim.size()
                          ? bBatchDim[bBatchDim.size() - 1 - offset]
                          : 1;
    FUSILLI_RETURN_ERROR_IF(
        aDimVal != bDimVal && aDimVal != 1 && bDimVal != 1,
        ErrorCode::InvalidAttribute,
        "Matmul input tensors A and B have incompatible batch dimensions for "
        "broadcasting at right-aligned batch index " +
            std::to_string(cBatchRank - 1 - offset) + ": A has dim=" +
            std::to_string(aDimVal) + ", B has dim=" + std::to_string(bDimVal));
    cDim[cBatchRank - 1 - offset] = std::max<int64_t>(aDimVal, bDimVal);
  }

  // Matrix dimensions: M from A, N from B
  cDim[cBatchRank] = aDim[aRank - 2];     // M
  cDim[cBatchRank + 1] = bDim[bRank - 1]; // N

  return ok(std::move(cDim));
}

inline std::vector<int64_t>
getMatmulInferredOutputShape(const std::vector<int64_t> &aDim,
                             const std::vector<int64_t> &bDim) {
  auto cDim = tryGetMatmulInferredOutputShape(aDim, bDim);
  assert(isOk(cDim) && "Invalid matmul input dimensions");
  return *cDim;
}

//===----------------------------------------------------------------------===//
// Matrix multiplication node.
//===----------------------------------------------------------------------===//

class MatmulNode : public NodeCRTP<MatmulNode> {
public:
  MatmulAttr matmulAttr;

  MatmulNode(MatmulAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), matmulAttr(std::move(attr)) {}

  // ASM emitter methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;

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
    constexpr size_t kNonBatchRank = 2;
    FUSILLI_RETURN_ERROR_IF(
        aRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul input tensor A must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(
        bRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul input tensor B must have a rank of at least 2");

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
    FUSILLI_ASSIGN_OR_RETURN(auto inferredCDim,
                             tryGetMatmulInferredOutputShape(aDim, bDim));
    (void)inferredCDim;

    FUSILLI_CHECK_ERROR(checkBatchDims(aT, "A"));
    FUSILLI_CHECK_ERROR(checkBatchDims(bT, "B"));

    // Check for mixed precision matmuls (inputs with differing element types).
    // Due to torch-mlir MLIR constraints, when element types differ:
    // - Both LHS and RHS must have rank 3 (single batch dim)
    // - The batch dim must be exactly equal (no broadcast)
    //
    // PyTorch does not allow differing input element types for any of the
    // matmul variants. However, torch-mlir breaks conformity with pytorch in
    // the case of `torch.bmm`. So, we need to be sure that `torch.matmul` will
    // lower to `torch.bmm` in the cases of mixed precision.
    if (aT->getDataType() != bT->getDataType()) {
      constexpr int64_t kMixedPrecisionRequiredRank = 3;
      FUSILLI_RETURN_ERROR_IF(
          aRank != kMixedPrecisionRequiredRank ||
              bRank != kMixedPrecisionRequiredRank,
          ErrorCode::InvalidAttribute,
          "Mixed precision matmul is only supported when input tensors A and B "
          "are of rank 3 (single batch dim): A has rank=" +
              std::to_string(aRank) + ", B has rank=" + std::to_string(bRank));
      FUSILLI_RETURN_ERROR_IF(
          aDim[0] != bDim[0], ErrorCode::InvalidAttribute,
          "Mixed precision matmul input tensors A and B must have exactly "
          "equal batch dimensions (no broadcast): A has batch dim=" +
              std::to_string(aDim[0]) +
              ", B has batch dim=" + std::to_string(bDim[0]));
    }

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

    const std::vector<int64_t> &cStride = cT->getStride();

    // Infer shape of output tensor.
    if (cT->getDim().empty()) {
      FUSILLI_ASSIGN_OR_RETURN(auto inferredCDim,
                               tryGetMatmulInferredOutputShape(aDim, bDim));
      cT->setDim(inferredCDim);
    }

    // Output stride is contiguous (row-major) when unspecified.
    if (cStride.empty()) {
      const std::vector<int64_t> &cDim = cT->getDim();
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
    constexpr size_t kNonBatchRank = 2;
    FUSILLI_RETURN_ERROR_IF(
        cRank < kNonBatchRank, ErrorCode::InvalidAttribute,
        "Matmul output tensor C must have a rank of at least 2");

    FUSILLI_ASSIGN_OR_RETURN(
        auto inferredCDim,
        tryGetMatmulInferredOutputShape(aT->getDim(), bT->getDim()));
    FUSILLI_RETURN_ERROR_IF(cT->getDim() != inferredCDim,
                            ErrorCode::InvalidAttribute,
                            "Matmul output tensor C dimensions do not match "
                            "the expected shapes inferred based on the input "
                            "dimensions");
    FUSILLI_CHECK_ERROR(checkBatchDims(cT, "C"));
    return ok();
  }

private:
  // Check that batch dimensions are outermost and non-transposed.
  // This is equivalent to checking that perm[i] == i for all batch dims.
  ErrorObject checkBatchDims(const std::shared_ptr<TensorAttr> &tensor,
                             const std::string &name) const {
    constexpr size_t kNonBatchRank = 2;
    size_t batchDims = tensor->getDim().size() - kNonBatchRank;
    std::vector<int64_t> perm = tensor->getLogicalToPhysicalPermuteOrder();
    for (size_t i = 0; i < batchDims; ++i) {
      FUSILLI_RETURN_ERROR_IF(
          perm[i] != static_cast<int64_t>(i), ErrorCode::InvalidAttribute,
          "Matmul tensor " + name + " has batch dimensions that are " +
              "not outermost or are transposed");
    }
    return ok();
  };
};

} // namespace fusilli

#endif // FUSILLI_NODE_MATMUL_NODE_H
