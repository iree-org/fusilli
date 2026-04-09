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
#include <unordered_map>
#include <vector>

using namespace fusilli;

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

// Build a graph that runs scaled dot-product attention on Q, K, V tensors
// using the built-in SDPA op.
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

  std::string causalSuffix = isCausal ? "_causal" : "";
  std::string maskSuffix = hasAttnMask ? "_mask" : "";
  std::string gqaSuffix = enableGqa ? "_gqa" : "";
  std::string scaleSuffix =
      scale.has_value() ? std::format("_scale{:g}", *scale) : "";
  std::string dropoutSuffix =
      dropoutP > 0.0f ? std::format("_dropout{:g}", dropoutP) : "";

  auto graph = std::make_shared<Graph>();
  graph
      ->setName(std::format("sdpa_b{}hq{}hkv{}sq{}skv{}d{}{}{}{}{}{}", batch,
                            headsQ, headsKV, seqQ, seqKV, headDim, causalSuffix,
                            maskSuffix, gqaSuffix, scaleSuffix, dropoutSuffix))
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

  // Build the SDPA op using the built-in API.
  SdpaAttr sdpaAttr;
  sdpaAttr.setName("sdpa")
      .setDropout(dropoutP)
      .setIsCausal(isCausal)
      .setScale(scale)
      .setEnableGqa(enableGqa);

  auto oT = graph->sdpa(qT, kT, vT, maskT, sdpaAttr);
  oT->setOutput(true);

  FUSILLI_REQUIRE_OK(graph->validate());
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  FUSILLI_REQUIRE_ASSIGN(auto qBuf, allocateBufferOfType(handle, qT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto kBuf, allocateBufferOfType(handle, kT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto vBuf, allocateBufferOfType(handle, vT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto outBuf,
                         allocateBufferOfType(handle, oT, dt, 0.0));

  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{qT, qBuf}, {kT, kBuf}, {vT, vBuf}, {oT, outBuf}};

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
