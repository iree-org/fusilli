// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_SDPA_UTILS_H
#define FUSILLI_SAMPLES_SDPA_UTILS_H

#include <fusilli.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace fusilli;

// SDPA MLIR templates for torch.aten.scaled_dot_product_attention.
//
// Templates are stored as R-string literals so the MLIR structure is
// directly readable in source. Standard CustomOp placeholders
// ({FUNC_NAME}, {IN<i>_TYPE}, {OUT0_TYPE}) are resolved by
// CustomOpNode::resolveMlirPlaceholders(). Scalar placeholders
// ({DROPOUT_P}, {IS_CAUSAL}, {SCALE_CONST}, {SCALE_TYPE}, {ENABLE_GQA})
// are resolved by buildSdpaMlir().

// SDPA template: 3 tensor inputs (Q, K, V), attention mask is none.
// Positional args: {0}=DROPOUT_P, {1}=IS_CAUSAL, {2}=SCALE_CONST,
//                  {3}=SCALE_TYPE, {4}=ENABLE_GQA
// clang-format off
inline constexpr std::string_view kSdpaNoMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: {{IN0_TYPE}},
      %arg1: {{IN1_TYPE}},
      %arg2: {{IN2_TYPE}})
      -> {{OUT0_TYPE}} {{
    %none_mask = torch.constant.none
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        {{IN0_TYPE}}, {{IN1_TYPE}},
        {{IN2_TYPE}}, !torch.none, !torch.float, !torch.bool,
        {3}, !torch.bool -> {{OUT0_TYPE}}
    return %0 : {{OUT0_TYPE}}
  }}
)mlir";

// SDPA template: 4 tensor inputs (Q, K, V, attn_mask).
// Positional args: same as kSdpaNoMask.
inline constexpr std::string_view kSdpaWithMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: {{IN0_TYPE}},
      %arg1: {{IN1_TYPE}},
      %arg2: {{IN2_TYPE}},
      %arg3: {{IN3_TYPE}})
      -> {{OUT0_TYPE}} {{
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %arg3, %dropout, %is_causal, %scale, %enable_gqa :
        {{IN0_TYPE}}, {{IN1_TYPE}},
        {{IN2_TYPE}}, {{IN3_TYPE}},
        !torch.float, !torch.bool,
        {3}, !torch.bool -> {{OUT0_TYPE}}
    return %0 : {{OUT0_TYPE}}
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
std::string buildSdpaMlir(bool hasAttnMask = false, float dropoutP = 0.0f,
                          bool isCausal = false,
                          std::optional<float> scale = std::nullopt,
                          bool enableGqa = false);

/// CPU reference implementation of scaled dot-product attention.
/// Computes SDPA in float precision for numerical verification against the GPU.
/// Layout: [batch, heads, seq_len, head_dim] contiguous.
std::vector<float> referenceSdpa(float qVal, float kVal, float vVal,
                                 float maskVal, int64_t batch, int64_t headsQ,
                                 int64_t headsKV, int64_t seqQ, int64_t seqKV,
                                 int64_t headDim, bool isCausal,
                                 std::optional<float> scale, bool enableGqa,
                                 bool hasAttnMask);

/// Build and execute SDPA using the built-in graph API.
/// Shape convention: [batch, heads, seq_len, head_dim].
void executeSdpa(Handle &handle, DataType dt, int64_t batch, int64_t headsQ,
                 int64_t headsKV, int64_t seqQ, int64_t seqKV, int64_t headDim,
                 bool isCausal = false,
                 std::optional<float> scale = std::nullopt,
                 bool enableGqa = false, bool hasAttnMask = false,
                 float dropoutP = 0.0f);

/// Build and execute SDPA using the custom op graph API with MLIR templates.
/// Shape convention: [batch, heads, seq_len, head_dim].
void executeSdpaCustomOp(Handle &handle, DataType dt, int64_t batch,
                         int64_t headsQ, int64_t headsKV, int64_t seqQ,
                         int64_t seqKV, int64_t headDim, bool isCausal = false,
                         std::optional<float> scale = std::nullopt,
                         bool enableGqa = false, bool hasAttnMask = false,
                         float dropoutP = 0.0f);

#endif // FUSILLI_SAMPLES_SDPA_UTILS_H
