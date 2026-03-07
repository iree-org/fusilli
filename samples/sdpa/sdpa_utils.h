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

#include <cstddef>
#include <cstdint>
#include <format>
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
// clang-format off
static constexpr std::string_view kSdpaNoMask = R"mlir(
  func.func private @{FUNC_NAME}(
      %arg0: !torch.vtensor<[?,?,?,?],{IN0_DTYPE}>,
      %arg1: !torch.vtensor<[?,?,?,?],{IN1_DTYPE}>,
      %arg2: !torch.vtensor<[?,?,?,?],{IN2_DTYPE}>)
      -> !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}> {
    %none_mask = torch.constant.none
    %dropout = torch.constant.float {DROPOUT_P}
    %is_causal = torch.constant.bool {IS_CAUSAL}
    %scale = {SCALE_CONST}
    %enable_gqa = torch.constant.bool {ENABLE_GQA}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{IN0_DTYPE}>, !torch.vtensor<[?,?,?,?],{IN1_DTYPE}>,
        !torch.vtensor<[?,?,?,?],{IN2_DTYPE}>, !torch.none, !torch.float, !torch.bool,
        {SCALE_TYPE}, !torch.bool -> !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}>
  }
)mlir";

// SDPA template: 4 tensor inputs (Q, K, V, attn_mask).
static constexpr std::string_view kSdpaWithMask = R"mlir(
  func.func private @{FUNC_NAME}(
      %arg0: !torch.vtensor<[?,?,?,?],{IN0_DTYPE}>,
      %arg1: !torch.vtensor<[?,?,?,?],{IN1_DTYPE}>,
      %arg2: !torch.vtensor<[?,?,?,?],{IN2_DTYPE}>,
      %arg3: !torch.vtensor<[?,?,?,?],{IN3_DTYPE}>)
      -> !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}> {
    %dropout = torch.constant.float {DROPOUT_P}
    %is_causal = torch.constant.bool {IS_CAUSAL}
    %scale = {SCALE_CONST}
    %enable_gqa = torch.constant.bool {ENABLE_GQA}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %arg3, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{IN0_DTYPE}>, !torch.vtensor<[?,?,?,?],{IN1_DTYPE}>,
        !torch.vtensor<[?,?,?,?],{IN2_DTYPE}>, !torch.vtensor<[?,?,?,?],{IN3_DTYPE}>,
        !torch.float, !torch.bool,
        {SCALE_TYPE}, !torch.bool -> !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}>
    return %0 : !torch.vtensor<[?,?,?,?],{OUT0_DTYPE}>
  }
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
  std::string mlir(hasAttnMask ? kSdpaWithMask : kSdpaNoMask);

  // Resolve scalar placeholders.
  auto replace = [&mlir](std::string_view placeholder,
                         const std::string &value) {
    auto pos = mlir.find(placeholder);
    if (pos != std::string::npos)
      mlir.replace(pos, placeholder.size(), value);
  };

  replace("{DROPOUT_P}", std::format("{:e}", dropoutP));
  replace("{IS_CAUSAL}", isCausal ? "true" : "false");
  replace("{ENABLE_GQA}", enableGqa ? "true" : "false");

  if (scale.has_value()) {
    replace("{SCALE_CONST}", std::format("torch.constant.float {:e}", *scale));
    replace("{SCALE_TYPE}", "!torch.float");
  } else {
    replace("{SCALE_CONST}", "torch.constant.none");
    replace("{SCALE_TYPE}", "!torch.none");
  }

  return mlir;
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

  // Read back output to verify execution completed successfully.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
  REQUIRE(result.size() ==
          static_cast<size_t>(batch * headsQ * seqQ * headDim));
}

#endif // FUSILLI_SAMPLES_SDPA_SDPA_UTILS_H
