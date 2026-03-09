// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_SDPA_SDPA_UTILS_H
#define FUSILLI_SAMPLES_SDPA_SDPA_UTILS_H

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// SDPA MLIR templates for torch.aten.scaled_dot_product_attention.
//
// Templates are stored as R-string literals so the MLIR structure is
// directly readable in source. Standard CustomOp placeholders
// ({FUNC_NAME}, {IN<i>_DTYPE}, {OUT0_DTYPE}) are resolved by
// CustomOpNode::resolveMlirPlaceholders(). Scalar placeholders
// ({DROPOUT_P}, {IS_CAUSAL}, {SCALE_CONST}, {SCALE_TYPE}, {ENABLE_GQA})
// are resolved by buildSdpaMlir().
//
// Tensor rank is hardcoded to 4 ([batch, heads, seq_len, head_dim]).

// SDPA template: 3 tensor inputs (Q, K, V), attention mask is none.
// Positional args: {0}=DROPOUT_P, {1}=IS_CAUSAL, {2}=SCALE_CONST,
//                  {3}=SCALE_TYPE, {4}=ENABLE_GQA
// clang-format off
static constexpr std::string_view kSdpaNoMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>,
      %arg1: !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
      %arg2: !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>)
      -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}> {{
    %none_mask = torch.constant.none
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
        !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>, !torch.none, !torch.float, !torch.bool,
        {3}, !torch.bool -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
    return %0 : !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
  }}
)mlir";

// SDPA template: 4 tensor inputs (Q, K, V, attn_mask).
// Positional args: same as kSdpaNoMask.
static constexpr std::string_view kSdpaWithMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>,
      %arg1: !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
      %arg2: !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>,
      %arg3: !torch.vtensor<[?,?,?,?],{{IN3_DTYPE}}>)
      -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}> {{
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %arg3, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
        !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN3_DTYPE}}>,
        !torch.float, !torch.bool,
        {3}, !torch.bool -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
    return %0 : !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
  }}
)mlir";
// clang-format on

/// Builds the MLIR template for torch.aten.scaled_dot_product_attention.
///
/// Selects the appropriate R-string template (with/without attn_mask) and
/// resolves scalar placeholders. Standard CustomOp dtype/name placeholders
/// are left for CustomOpNode to resolve at emission time.
///
/// When hasAttnMask is false: 3 tensor inputs (Q=IN0, K=IN1, V=IN2).
/// When hasAttnMask is true:  4 tensor inputs (Q=IN0, K=IN1, V=IN2, mask=IN3).
static std::string buildSdpaMlir(bool hasAttnMask = false,
                                 float dropoutP = 0.0f, bool isCausal = false,
                                 std::optional<float> scale = std::nullopt,
                                 bool enableGqa = false) {
  // Compute raw values for scalar placeholders.
  std::string dropoutStr = std::format("{:e}", dropoutP);
  std::string isCausalStr = isCausal ? "true" : "false";
  std::string scaleConstStr =
      scale.has_value() ? std::format("torch.constant.float {:e}", *scale)
                        : "torch.constant.none";
  std::string scaleTypeStr = scale.has_value() ? "!torch.float" : "!torch.none";
  std::string enableGqaStr = enableGqa ? "true" : "false";

  // Resolve all scalar placeholders in a single format pass.
  return std::vformat(hasAttnMask ? kSdpaWithMask : kSdpaNoMask,
                      std::make_format_args(dropoutStr,    // {0} DROPOUT_P
                                            isCausalStr,   // {1} IS_CAUSAL
                                            scaleConstStr, // {2} SCALE_CONST
                                            scaleTypeStr,  // {3} SCALE_TYPE
                                            enableGqaStr   // {4} ENABLE_GQA
                                            ));
}

// CPU reference implementation of scaled dot-product attention.
// Computes SDPA in float precision for numerical verification against the GPU.
// Layout: [batch, heads, seq_len, head_dim] contiguous.
static std::vector<float>
referenceSdpa(float qVal, float kVal, float vVal, float maskVal, int64_t batch,
              int64_t headsQ, int64_t headsKV, int64_t seqQ, int64_t seqKV,
              int64_t headDim, bool isCausal, std::optional<float> scale,
              bool enableGqa, bool hasAttnMask) {
  float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(headDim)));
  int64_t outSize = batch * headsQ * seqQ * headDim;
  std::vector<float> out(outSize);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t hq = 0; hq < headsQ; ++hq) {
      // Map query head to KV head (identity for MHA, grouped for GQA).
      int64_t hkv = enableGqa ? hq / (headsQ / headsKV) : hq;

      for (int64_t sq = 0; sq < seqQ; ++sq) {
        // Compute attention scores: dot(Q[b,hq,sq,:], K[b,hkv,sk,:]) * scale.
        // Since Q and K are constant-filled, dot product = qVal * kVal *
        // headDim for every (sq, sk) pair. We still compute per-element to
        // handle causal/mask variations correctly.
        std::vector<float> scores(seqKV);
        for (int64_t sk = 0; sk < seqKV; ++sk) {
          float dot = static_cast<float>(headDim) * qVal * kVal;
          scores[sk] = dot * s;

          // Apply additive attention mask (broadcast head dim = 1).
          if (hasAttnMask)
            scores[sk] += maskVal;

          // Causal: mask future positions to -inf.
          if (isCausal && sk > sq)
            scores[sk] = -std::numeric_limits<float>::infinity();
        }

        // Softmax over scores.
        float maxScore = *std::max_element(scores.begin(), scores.end());
        float sumExp = 0.0f;
        for (int64_t sk = 0; sk < seqKV; ++sk) {
          scores[sk] = std::exp(scores[sk] - maxScore);
          sumExp += scores[sk];
        }
        for (int64_t sk = 0; sk < seqKV; ++sk)
          scores[sk] /= sumExp;

        // Output: weighted sum of V rows.
        // V[b, hkv, sk, d] = vVal for all elements.
        for (int64_t d = 0; d < headDim; ++d) {
          float val = 0.0f;
          for (int64_t sk = 0; sk < seqKV; ++sk)
            val += scores[sk] * vVal;
          out[((b * headsQ + hq) * seqQ + sq) * headDim + d] = val;
        }
      }
    }
  }
  return out;
}

// Build a graph that runs scaled dot-product attention on Q, K, V tensors.
// Shape convention: [batch, heads, seq_len, head_dim].
static void executeSdpa(Handle &handle, DataType dt, int64_t batch,
                        int64_t headsQ, int64_t headsKV, int64_t seqQ,
                        int64_t seqKV, int64_t headDim, bool isCausal = false,
                        std::optional<float> scale = std::nullopt,
                        bool enableGqa = false, bool hasAttnMask = false,
                        float dropoutP = 0.0f) {
  // attn_mask and is_causal are mutually exclusive: is_causal internally
  // applies a causal mask, making an explicit mask contradictory.
  REQUIRE(!(hasAttnMask && isCausal));

  if (enableGqa) {
    // GQA constraint: query heads must be a multiple of KV heads.
    REQUIRE(headsQ % headsKV == 0);
  } else {
    // Standard MHA: query and KV head counts must match.
    REQUIRE(headsQ == headsKV);
  }

  auto graph = std::make_shared<Graph>();
  graph
      ->setName(std::format(
          "sdpa_b{}hq{}hkv{}sq{}skv{}d{}{}{}{}{}{}", batch, headsQ, headsKV,
          seqQ, seqKV, headDim, isCausal ? "_causal" : "",
          hasAttnMask ? "_mask" : "", enableGqa ? "_gqa" : "",
          scale.has_value() ? std::format("_scale{:g}", *scale) : "",
          dropoutP > 0.0f ? std::format("_dropout{:g}", dropoutP) : ""))
      .setIODataType(dt)
      .setIntermediateDataType(dt);

  // Q: [batch, headsQ, seqQ, headDim]
  std::vector<int64_t> qDim = {batch, headsQ, seqQ, headDim};
  auto qStride =
      generateStrideFromDim(qDim, getContiguousStrideOrder(qDim.size()));
  auto qT =
      graph->tensor(TensorAttr().setName("q").setDim(qDim).setStride(qStride));

  // K: [batch, headsKV, seqKV, headDim]
  std::vector<int64_t> kDim = {batch, headsKV, seqKV, headDim};
  auto kStride =
      generateStrideFromDim(kDim, getContiguousStrideOrder(kDim.size()));
  auto kT =
      graph->tensor(TensorAttr().setName("k").setDim(kDim).setStride(kStride));

  // V: [batch, headsKV, seqKV, headDim]
  std::vector<int64_t> vDim = {batch, headsKV, seqKV, headDim};
  auto vStride =
      generateStrideFromDim(vDim, getContiguousStrideOrder(vDim.size()));
  auto vT =
      graph->tensor(TensorAttr().setName("v").setDim(vDim).setStride(vStride));

  // Attention mask: [batch, 1, seqQ, seqKV] — broadcast across heads.
  // Float masks are additive (attn_scores += mask), consistent with the
  // PyTorch reference: attn_bias = attn_mask + attn_bias.
  std::shared_ptr<TensorAttr> maskT;
  if (hasAttnMask) {
    std::vector<int64_t> maskDim = {batch, 1, seqQ, seqKV};
    auto maskStride = generateStrideFromDim(
        maskDim, getContiguousStrideOrder(maskDim.size()));
    maskT = graph->tensor(
        TensorAttr().setName("mask").setDim(maskDim).setStride(maskStride));
  }

  // Build the MLIR template with the given scalar parameters.
  std::string sdpaMlir =
      buildSdpaMlir(hasAttnMask, dropoutP, isCausal, scale, enableGqa);

  CustomOpAttr sdpaAttr;
  sdpaAttr.setName("sdpa").setMlir(sdpaMlir).setNumOutputs(1);

  std::vector<std::shared_ptr<TensorAttr>> inputs = {qT, kT, vT};
  if (hasAttnMask)
    inputs.push_back(maskT);

  auto outs = graph->customOp(inputs, sdpaAttr);

  // Output: [batch, headsQ, seqQ, headDim]
  std::vector<int64_t> outDim = {batch, headsQ, seqQ, headDim};
  auto outStride =
      generateStrideFromDim(outDim, getContiguousStrideOrder(outDim.size()));
  outs[0]->setDim(outDim).setStride(outStride).setDataType(dt).setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  FUSILLI_REQUIRE_ASSIGN(auto qBuf, allocateBufferOfType(handle, qT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto kBuf, allocateBufferOfType(handle, kT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto vBuf, allocateBufferOfType(handle, vT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto outBuf,
                         allocateBufferOfType(handle, outs[0], dt, 0.0));

  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{qT, qBuf}, {kT, kBuf}, {vT, vBuf}, {outs[0], outBuf}};

  if (hasAttnMask) {
    FUSILLI_REQUIRE_ASSIGN(auto maskBuf,
                           allocateBufferOfType(handle, maskT, dt, -1.0));
    variantPack[maskT] = maskBuf;
  }

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  // Read back output and verify against CPU reference.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
  REQUIRE(result.size() ==
          static_cast<size_t>(batch * headsQ * seqQ * headDim));

  // Skip numerical verification when dropout is enabled since it introduces
  // non-deterministic masking that the CPU reference cannot reproduce.
  if (dropoutP > 0.0f)
    return;

  constexpr float kInitQ = 0.01f;
  constexpr float kInitK = 0.01f;
  constexpr float kInitV = 0.01f;
  constexpr float kInitMask = -1.0f;

  auto expected = referenceSdpa(kInitQ, kInitK, kInitV, kInitMask, batch,
                                headsQ, headsKV, seqQ, seqKV, headDim, isCausal,
                                scale, enableGqa, hasAttnMask);

  // f16 has ~3 decimal digits of precision; use a tolerance that accounts
  // for accumulation error across the softmax and weighted-sum steps.
  constexpr float kTolerance = 1e-2f;
  for (size_t i = 0; i < result.size(); ++i) {
    float actual = static_cast<float>(result[i]);
    INFO("index " << i << ": actual=" << actual << " expected=" << expected[i]);
    REQUIRE(std::abs(actual - expected[i]) < kTolerance);
  }
}

#endif // FUSILLI_SAMPLES_SDPA_SDPA_UTILS_H
