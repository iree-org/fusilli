// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the blocked matmul node
// `BlockedMatmulNode`.
//
// Blocked matmul operates on 4D tiled tensors:
//   LHS logical: [M0, K0, M1, K1]
//   RHS logical: [K0, N0, K1, N1]
//   OUT:         [M0, N0, M1, N1]
//
// When RHS is specified with transposed strides (physical layout
// [N0, K0, N1, K1]), this lowers to `linalg.mmt4d`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_BLOCKED_MATMUL_NODE_H
#define FUSILLI_NODE_BLOCKED_MATMUL_NODE_H

#include "fusilli/attributes/blocked_matmul_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// Helper functions for blocked matmul nodes.
//===----------------------------------------------------------------------===//

// Infer the output shape of a blocked matmul operation.
//   LHS [M0, K0, M1, K1] x RHS [K0, N0, K1, N1] -> OUT [M0, N0, M1, N1]
inline std::vector<int64_t>
getBlockedMatmulInferredOutputShape(const std::vector<int64_t> &lhsDim,
                                    const std::vector<int64_t> &rhsDim) {
  assert(lhsDim.size() == 4 && "LHS must be rank 4");
  assert(rhsDim.size() == 4 && "RHS must be rank 4");
  return {lhsDim[0], rhsDim[1], lhsDim[2], rhsDim[3]};
}

//===----------------------------------------------------------------------===//
// Blocked matmul node.
//===----------------------------------------------------------------------===//

class BlockedMatmulNode : public NodeCRTP<BlockedMatmulNode> {
public:
  BlockedMatmulAttr blockedMatmulAttr;

  BlockedMatmulNode(BlockedMatmulAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), blockedMatmulAttr(std::move(attr)) {}

  // ASM emitter methods.
  std::string emitNodePreAsm() const override final;

  const std::string &getName() const override final {
    return blockedMatmulAttr.getName();
  }
  Type getType() const override final { return Type::BlockedMatmul; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating BlockedMatmulNode '"
                           << blockedMatmulAttr.getName() << "'");

    auto lhsT = blockedMatmulAttr.getLHS();
    auto rhsT = blockedMatmulAttr.getRHS();
    auto outT = blockedMatmulAttr.getRESULT();

    FUSILLI_RETURN_ERROR_IF(!lhsT, ErrorCode::AttributeNotSet,
                            "BlockedMatmul input tensor LHS not set");
    FUSILLI_RETURN_ERROR_IF(!rhsT, ErrorCode::AttributeNotSet,
                            "BlockedMatmul input tensor RHS not set");
    FUSILLI_RETURN_ERROR_IF(!outT, ErrorCode::AttributeNotSet,
                            "BlockedMatmul output tensor OUT not set");

    size_t lhsRank = lhsT->getDim().size();
    size_t rhsRank = rhsT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(lhsRank != 4, ErrorCode::InvalidAttribute,
                            "BlockedMatmul LHS must have rank 4, got " +
                                std::to_string(lhsRank));
    FUSILLI_RETURN_ERROR_IF(rhsRank != 4, ErrorCode::InvalidAttribute,
                            "BlockedMatmul RHS must have rank 4, got " +
                                std::to_string(rhsRank));

    // K dimensions must match:
    //   LHS logical [M0, K0, M1, K1], RHS logical [K0, N0, K1, N1]
    //   LHS[1] == RHS[0] (K0) and LHS[3] == RHS[2] (K1)
    const auto &lhsDim = lhsT->getDim();
    const auto &rhsDim = rhsT->getDim();
    FUSILLI_RETURN_ERROR_IF(
        lhsDim[1] != rhsDim[0], ErrorCode::InvalidAttribute,
        "BlockedMatmul K0 mismatch: LHS[1]=" + std::to_string(lhsDim[1]) +
            ", RHS[0]=" + std::to_string(rhsDim[0]));
    FUSILLI_RETURN_ERROR_IF(
        lhsDim[3] != rhsDim[2], ErrorCode::InvalidAttribute,
        "BlockedMatmul K1 mismatch: LHS[3]=" + std::to_string(lhsDim[3]) +
            ", RHS[2]=" + std::to_string(rhsDim[2]));

    // RHS must be transposed: logical [K0, N0, K1, N1] must have physical
    // layout [N0, K0, N1, K1] for linalg.mmt4d. This corresponds to
    // logical-to-physical permutation [1, 0, 3, 2].
    std::vector<int64_t> rhsPerm = rhsT->getLogicalToPhysicalPermuteOrder();
    std::vector<int64_t> expectedPerm = {1, 0, 3, 2};
    FUSILLI_RETURN_ERROR_IF(
        rhsPerm != expectedPerm, ErrorCode::NotImplemented,
        "BlockedMatmul only supports RHS with transposed physical layout "
        "[N0, K0, N1, K1]. Non-transposed RHS is not yet supported");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for BlockedMatmulNode '"
                           << blockedMatmulAttr.getName() << "'");

    blockedMatmulAttr.fillFromContext(context);

    auto lhsT = blockedMatmulAttr.getLHS();
    auto rhsT = blockedMatmulAttr.getRHS();
    auto outT = blockedMatmulAttr.getRESULT();

    const auto &outDim = outT->getDim();
    const auto &outStride = outT->getStride();

    if (outDim.empty())
      outT->setDim(
          getBlockedMatmulInferredOutputShape(lhsT->getDim(), rhsT->getDim()));

    if (outStride.empty()) {
      outT->setStride(generateStrideFromDim(
          outT->getDim(), getContiguousStrideOrder(outT->getDim().size())));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating BlockedMatmulNode '"
                           << blockedMatmulAttr.getName() << "'");

    auto outT = blockedMatmulAttr.getRESULT();
    FUSILLI_RETURN_ERROR_IF(outT->getDim().size() != 4,
                            ErrorCode::InvalidAttribute,
                            "BlockedMatmul OUT must have rank 4");

    FUSILLI_RETURN_ERROR_IF(
        outT->getDim() != getBlockedMatmulInferredOutputShape(
                              blockedMatmulAttr.getLHS()->getDim(),
                              blockedMatmulAttr.getRHS()->getDim()),
        ErrorCode::InvalidAttribute,
        "BlockedMatmul OUT dimensions do not match expected shape");

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_BLOCKED_MATMUL_NODE_H
