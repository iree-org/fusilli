// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "sdpa_utils.h"
#include "utils.h"

#include <catch2/catch_message.hpp>
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

std::string buildSdpaMlir(bool hasAttnMask, float dropoutP, bool isCausal,
                          std::optional<float> scale, bool enableGqa) {
  std::string dropoutStr = std::format("{:e}", dropoutP);
  std::string isCausalStr = isCausal ? "true" : "false";
  std::string scaleConstStr =
      scale.has_value() ? std::format("torch.constant.float {:e}", *scale)
                        : "torch.constant.none";
  std::string scaleTypeStr = scale.has_value() ? "!torch.float" : "!torch.none";
  std::string enableGqaStr = enableGqa ? "true" : "false";

  return std::vformat(hasAttnMask ? kSdpaWithMask : kSdpaNoMask,
                      std::make_format_args(dropoutStr,    // {0} DROPOUT_P
                                            isCausalStr,   // {1} IS_CAUSAL
                                            scaleConstStr, // {2} SCALE_CONST
                                            scaleTypeStr,  // {3} SCALE_TYPE
                                            enableGqaStr   // {4} ENABLE_GQA
                                            ));
}

std::vector<float> referenceSdpa(float qVal, float kVal, float vVal,
                                 float maskVal, int64_t batch, int64_t headsQ,
                                 int64_t headsK, int64_t headsV, int64_t seqQ,
                                 int64_t seqKV, int64_t headDim, bool isCausal,
                                 std::optional<float> scale, bool enableGqa,
                                 bool hasAttnMask) {
  float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(headDim)));
  int64_t outSize = batch * headsQ * seqQ * headDim;
  std::vector<float> out(outSize);

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t hq = 0; hq < headsQ; ++hq) {
      // Map query head to K/V heads independently (identity for MHA,
      // grouped for GQA). K and V may have different head counts.
      // Not used in computation because tensors are constant-filled, but
      // kept for documentation of the actual head mapping:
      //   hk = enableGqa ? hq / (headsQ / headsK) : hq
      //   hv = enableGqa ? hq / (headsQ / headsV) : hq

      for (int64_t sq = 0; sq < seqQ; ++sq) {
        // Compute attention scores: dot(Q[b,hq,sq,:], K[b,hk,sk,:]) * scale.
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
        // V[b, hv, sk, d] = vVal for all elements.
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

// ---------------------------------------------------------------------------
// Shared setup and verification helpers
// ---------------------------------------------------------------------------

namespace {

struct SdpaTestContext {
  std::shared_ptr<Graph> graph;
  std::shared_ptr<TensorAttr> qT, kT, vT, maskT;
};

SdpaTestContext setupSdpaGraph(DataType dt, int64_t batch, int64_t headsQ,
                               int64_t headsK, int64_t headsV, int64_t seqQ,
                               int64_t seqKV, int64_t headDim, bool isCausal,
                               bool enableGqa, bool hasAttnMask, float dropoutP,
                               std::optional<float> scale,
                               std::string_view namePrefix) {
  REQUIRE(!(hasAttnMask && isCausal));
  if (enableGqa) {
    REQUIRE(headsQ % headsK == 0);
    REQUIRE(headsQ % headsV == 0);
  } else {
    REQUIRE(headsQ == headsK);
    REQUIRE(headsQ == headsV);
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
      ->setName(std::format("{}_b{}hq{}hk{}hv{}sq{}skv{}d{}{}{}{}{}{}",
                            namePrefix, batch, headsQ, headsK, headsV, seqQ,
                            seqKV, headDim, causalSuffix, maskSuffix, gqaSuffix,
                            scaleSuffix, dropoutSuffix))
      .setIODataType(dt)
      .setIntermediateDataType(dt);

  // Q: [batch, headsQ, seqQ, headDim]
  std::vector<int64_t> qDim = {batch, headsQ, seqQ, headDim};
  auto qStride =
      generateStrideFromDim(qDim, getContiguousStrideOrder(qDim.size()));
  auto qT =
      graph->tensor(TensorAttr().setName("q").setDim(qDim).setStride(qStride));

  // K: [batch, headsK, seqKV, headDim]
  std::vector<int64_t> kDim = {batch, headsK, seqKV, headDim};
  auto kStride =
      generateStrideFromDim(kDim, getContiguousStrideOrder(kDim.size()));
  auto kT =
      graph->tensor(TensorAttr().setName("k").setDim(kDim).setStride(kStride));

  // V: [batch, headsV, seqKV, headDim]
  std::vector<int64_t> vDim = {batch, headsV, seqKV, headDim};
  auto vStride =
      generateStrideFromDim(vDim, getContiguousStrideOrder(vDim.size()));
  auto vT =
      graph->tensor(TensorAttr().setName("v").setDim(vDim).setStride(vStride));

  // Attention mask: [batch, 1, seqQ, seqKV] — broadcast across heads.
  std::shared_ptr<TensorAttr> maskT;
  if (hasAttnMask) {
    std::vector<int64_t> maskDim = {batch, 1, seqQ, seqKV};
    auto maskStride = generateStrideFromDim(
        maskDim, getContiguousStrideOrder(maskDim.size()));
    maskT = graph->tensor(
        TensorAttr().setName("mask").setDim(maskDim).setStride(maskStride));
  }

  return {graph, qT, kT, vT, maskT};
}

void executeAndVerify(Handle &handle, const SdpaTestContext &ctx,
                      const std::shared_ptr<TensorAttr> &oT, DataType dt,
                      int64_t batch, int64_t headsQ, int64_t headsK,
                      int64_t headsV, int64_t seqQ, int64_t seqKV,
                      int64_t headDim, bool isCausal,
                      std::optional<float> scale, bool enableGqa,
                      bool hasAttnMask, float dropoutP) {
  FUSILLI_REQUIRE_OK(ctx.graph->validate());
  FUSILLI_REQUIRE_OK(ctx.graph->compile(handle, /*remove=*/true));

  FUSILLI_REQUIRE_ASSIGN(auto qBuf,
                         allocateBufferOfType(handle, ctx.qT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto kBuf,
                         allocateBufferOfType(handle, ctx.kT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto vBuf,
                         allocateBufferOfType(handle, ctx.vT, dt, 0.01));
  FUSILLI_REQUIRE_ASSIGN(auto outBuf,
                         allocateBufferOfType(handle, oT, dt, 0.0));

  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {ctx.qT, qBuf}, {ctx.kT, kBuf}, {ctx.vT, vBuf}, {oT, outBuf}};

  if (hasAttnMask) {
    FUSILLI_REQUIRE_ASSIGN(auto maskBuf,
                           allocateBufferOfType(handle, ctx.maskT, dt, -1.0));
    variantPack[ctx.maskT] = maskBuf;
  }

  FUSILLI_REQUIRE_ASSIGN(
      auto workspace, allocateWorkspace(handle, ctx.graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(ctx.graph->execute(handle, variantPack, workspace));

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
                                headsQ, headsK, headsV, seqQ, seqKV, headDim,
                                isCausal, scale, enableGqa, hasAttnMask);

  // f16 has ~3 decimal digits of precision; use a tolerance that accounts
  // for accumulation error across the softmax and weighted-sum steps.
  constexpr float kTolerance = 1e-2f;
  for (size_t i = 0; i < result.size(); ++i) {
    float actual = static_cast<float>(result[i]);
    INFO("index " << i << ": actual=" << actual << " expected=" << expected[i]);
    REQUIRE(std::abs(actual - expected[i]) < kTolerance);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void executeSdpa(Handle &handle, DataType dt, int64_t batch, int64_t headsQ,
                 int64_t headsK, int64_t headsV, int64_t seqQ, int64_t seqKV,
                 int64_t headDim, bool isCausal, std::optional<float> scale,
                 bool enableGqa, bool hasAttnMask, float dropoutP) {
  auto ctx =
      setupSdpaGraph(dt, batch, headsQ, headsK, headsV, seqQ, seqKV, headDim,
                     isCausal, enableGqa, hasAttnMask, dropoutP, scale, "sdpa");

  SdpaAttr sdpaAttr;
  sdpaAttr.setName("sdpa")
      .setDropout(dropoutP)
      .setIsCausal(isCausal)
      .setScale(scale)
      .setEnableGqa(enableGqa);

  auto oT = ctx.graph->sdpa(ctx.qT, ctx.kT, ctx.vT, ctx.maskT, sdpaAttr);
  oT->setOutput(true);

  executeAndVerify(handle, ctx, oT, dt, batch, headsQ, headsK, headsV, seqQ,
                   seqKV, headDim, isCausal, scale, enableGqa, hasAttnMask,
                   dropoutP);
}

void executeSdpaCustomOp(Handle &handle, DataType dt, int64_t batch,
                         int64_t headsQ, int64_t headsK, int64_t headsV,
                         int64_t seqQ, int64_t seqKV, int64_t headDim,
                         bool isCausal, std::optional<float> scale,
                         bool enableGqa, bool hasAttnMask, float dropoutP) {
  auto ctx = setupSdpaGraph(dt, batch, headsQ, headsK, headsV, seqQ, seqKV,
                            headDim, isCausal, enableGqa, hasAttnMask, dropoutP,
                            scale, "sdpa_custom_op");

  std::string sdpaMlir =
      buildSdpaMlir(hasAttnMask, dropoutP, isCausal, scale, enableGqa);

  CustomOpAttr sdpaAttr;
  sdpaAttr.setName("sdpa").setMlir(sdpaMlir).setNumOutputs(1);

  std::vector<std::shared_ptr<TensorAttr>> inputs = {ctx.qT, ctx.kT, ctx.vT};
  if (hasAttnMask)
    inputs.push_back(ctx.maskT);

  auto outs = ctx.graph->customOp(inputs, sdpaAttr);

  // Output: [batch, headsQ, seqQ, headDim]
  std::vector<int64_t> outDim = {batch, headsQ, seqQ, headDim};
  auto outStride =
      generateStrideFromDim(outDim, getContiguousStrideOrder(outDim.size()));
  outs[0]->setDim(outDim).setStride(outStride).setDataType(dt).setOutput(true);

  executeAndVerify(handle, ctx, outs[0], dt, batch, headsQ, headsK, headsV,
                   seqQ, seqKV, headDim, isCausal, scale, enableGqa,
                   hasAttnMask, dropoutP);
}
