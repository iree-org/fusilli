// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the scaled dot-product attention (SDPA)
// node `SdpaNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_SDPA_NODE_H
#define FUSILLI_NODE_SDPA_NODE_H

#include "fusilli/attributes/sdpa_attributes.h"
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
// Scaled dot-product attention node.
//===----------------------------------------------------------------------===//

class SdpaNode : public NodeCRTP<SdpaNode> {
public:
  SdpaAttr sdpaAttr;

  SdpaNode(SdpaAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), sdpaAttr(std::move(attr)) {}

  // ASM emitter methods.
  std::string emitNodePreAsm() const override final;
  std::string getResultNamesAsm() const;

  const std::string &getName() const override final {
    return sdpaAttr.getName();
  }
  Type getType() const override final { return Type::Sdpa; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating SdpaNode '"
                           << sdpaAttr.getName() << "'");

    std::shared_ptr<TensorAttr> qT = sdpaAttr.getQ();
    std::shared_ptr<TensorAttr> kT = sdpaAttr.getK();
    std::shared_ptr<TensorAttr> vT = sdpaAttr.getV();
    std::shared_ptr<TensorAttr> oT = sdpaAttr.getO();
    std::shared_ptr<TensorAttr> maskT = sdpaAttr.getMASK();

    // Ensure mandatory input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!qT, ErrorCode::AttributeNotSet,
                            "SDPA input tensor Q not set");
    FUSILLI_RETURN_ERROR_IF(!kT, ErrorCode::AttributeNotSet,
                            "SDPA input tensor K not set");
    FUSILLI_RETURN_ERROR_IF(!vT, ErrorCode::AttributeNotSet,
                            "SDPA input tensor V not set");
    FUSILLI_RETURN_ERROR_IF(!oT, ErrorCode::AttributeNotSet,
                            "SDPA output tensor O not set");

    // Rank checks: all tensors must be rank 4 [batch, heads, seq_len,
    // head_dim].
    constexpr size_t kRequiredRank = 4;
    FUSILLI_RETURN_ERROR_IF(
        qT->getDim().size() != kRequiredRank, ErrorCode::InvalidAttribute,
        "SDPA input tensor Q must be rank 4 [batch, heads, seq_len, head_dim]");
    FUSILLI_RETURN_ERROR_IF(
        kT->getDim().size() != kRequiredRank, ErrorCode::InvalidAttribute,
        "SDPA input tensor K must be rank 4 [batch, heads, seq_len, head_dim]");
    FUSILLI_RETURN_ERROR_IF(
        vT->getDim().size() != kRequiredRank, ErrorCode::InvalidAttribute,
        "SDPA input tensor V must be rank 4 [batch, heads, seq_len, head_dim]");

    const std::vector<int64_t> &qDim = qT->getDim();
    const std::vector<int64_t> &kDim = kT->getDim();
    const std::vector<int64_t> &vDim = vT->getDim();

    // Batch dimension must match across Q, K, V.
    FUSILLI_RETURN_ERROR_IF(
        qDim[0] != kDim[0] || qDim[0] != vDim[0], ErrorCode::InvalidAttribute,
        "SDPA input tensors Q, K, V must have matching batch dimension");

    // Head dimension must match across Q and K.
    FUSILLI_RETURN_ERROR_IF(
        qDim[3] != kDim[3], ErrorCode::InvalidAttribute,
        "SDPA input tensors Q and K must have matching head_dim");

    // K and V must have matching sequence length and heads.
    FUSILLI_RETURN_ERROR_IF(
        kDim[1] != vDim[1], ErrorCode::InvalidAttribute,
        "SDPA input tensors K and V must have matching heads dimension");
    FUSILLI_RETURN_ERROR_IF(
        kDim[2] != vDim[2], ErrorCode::InvalidAttribute,
        "SDPA input tensors K and V must have matching sequence length");

    // Head count validation.
    int64_t headsQ = qDim[1];
    int64_t headsKV = kDim[1];
    if (sdpaAttr.getEnableGqa()) {
      FUSILLI_RETURN_ERROR_IF(
          headsQ % headsKV != 0, ErrorCode::InvalidAttribute,
          "SDPA with GQA requires Q heads (" + std::to_string(headsQ) +
              ") to be a multiple of KV heads (" + std::to_string(headsKV) +
              ")");
    } else {
      FUSILLI_RETURN_ERROR_IF(
          headsQ != headsKV, ErrorCode::InvalidAttribute,
          "SDPA without GQA requires Q heads (" + std::to_string(headsQ) +
              ") to equal KV heads (" + std::to_string(headsKV) + ")");
    }

    // Mask and is_causal are mutually exclusive.
    FUSILLI_RETURN_ERROR_IF(
        maskT && sdpaAttr.getIsCausal(), ErrorCode::InvalidAttribute,
        "SDPA attention mask and is_causal are mutually exclusive");

    // Mask rank and shape checks.
    if (maskT) {
      FUSILLI_RETURN_ERROR_IF(maskT->getDim().size() != kRequiredRank,
                              ErrorCode::InvalidAttribute,
                              "SDPA attention mask must be rank 4");

      const std::vector<int64_t> &maskDim = maskT->getDim();
      FUSILLI_RETURN_ERROR_IF(
          maskDim[0] != qDim[0], ErrorCode::InvalidAttribute,
          "SDPA attention mask must have matching batch dimension");
      FUSILLI_RETURN_ERROR_IF(
          maskDim[1] != 1 && maskDim[1] != qDim[1],
          ErrorCode::InvalidAttribute,
          "SDPA attention mask heads dimension must be 1 or match Q heads");
      FUSILLI_RETURN_ERROR_IF(
          maskDim[2] != qDim[2], ErrorCode::InvalidAttribute,
          "SDPA attention mask must have sequence length matching Q");
      FUSILLI_RETURN_ERROR_IF(
          maskDim[3] != kDim[2], ErrorCode::InvalidAttribute,
          "SDPA attention mask must have sequence length matching K");
    }

    // Dropout range check.
    float dropout = sdpaAttr.getDropout();
    FUSILLI_RETURN_ERROR_IF(dropout < 0.0f || dropout >= 1.0f,
                            ErrorCode::InvalidAttribute,
                            "SDPA dropout probability must be in [0, 1)");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for SdpaNode '"
                           << sdpaAttr.getName() << "'");

    sdpaAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> qT = sdpaAttr.getQ();
    std::shared_ptr<TensorAttr> vT = sdpaAttr.getV();
    std::shared_ptr<TensorAttr> oT = sdpaAttr.getO();

    const std::vector<int64_t> &qDim = qT->getDim();
    const std::vector<int64_t> &vDim = vT->getDim();

    // Output O shape: [batch, headsQ, seqQ, headDim]
    // headDim comes from V's last dimension.
    std::vector<int64_t> oDim = {qDim[0], qDim[1], qDim[2], vDim[3]};

    if (oT->getDim().empty())
      oT->setDim(oDim);

    // Output stride is contiguous (row-major) when unspecified.
    if (oT->getStride().empty()) {
      oT->setStride(
          generateStrideFromDim(oDim, getContiguousStrideOrder(oDim.size())));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating SdpaNode '"
                           << sdpaAttr.getName() << "'");

    std::shared_ptr<TensorAttr> qT = sdpaAttr.getQ();
    std::shared_ptr<TensorAttr> vT = sdpaAttr.getV();
    std::shared_ptr<TensorAttr> oT = sdpaAttr.getO();

    const std::vector<int64_t> &qDim = qT->getDim();
    const std::vector<int64_t> &vDim = vT->getDim();

    // Expected output shape: [batch, headsQ, seqQ, headDim]
    std::vector<int64_t> expectedDim = {qDim[0], qDim[1], qDim[2], vDim[3]};

    FUSILLI_RETURN_ERROR_IF(
        oT->getDim() != expectedDim, ErrorCode::InvalidAttribute,
        "SDPA output tensor O dimensions do not match expected shape "
        "[batch, headsQ, seqQ, headDim]");

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_SDPA_NODE_H
