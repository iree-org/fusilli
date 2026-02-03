// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_LAYERNORM_LAYERNORM_UTILS_H
#define FUSILLI_SAMPLES_LAYERNORM_LAYERNORM_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace fusilli::layernorm_utils {

// Generates input and expected output tensors for LayerNorm node with inference
// mode. Uses a two-value pattern per batch (x0, x1) to produce analytically
// computable normalized outputs, enabling deterministic correctness
// verification.
// Returns: input tensor values, expected output tensor values for
// these input values.
inline std::tuple<std::vector<float>, std::vector<float>>
generateIOTensorsForInferForward(int64_t n, int64_t c, int64_t h, int64_t w,
                                 float scale, float bias, float eps) {
  // For each batch b, we fill:
  //   - first half of elements with x0 = 2*b
  //   - second half of elements with x1 = 2*b + 2
  //
  // Layer norm formula: y = scale * (x - mean) / sqrt(variance + eps) + bias
  // With two distinct values x0, x1:
  //   y0 = scale * (-1 / sqrt(1 + eps)) + bias
  //   y1 = scale * (1 / sqrt(1 + eps)) + bias
  const float div = std::sqrt(1.0f + eps);
  const float y0 = scale * (-1.0f / div) + bias;
  const float y1 = scale * (1.0f / div) + bias;

  const size_t size = n * c * h * w;
  std::vector<float> inputVals(size, 0.0f);
  std::vector<float> expectedVals(size, 0.0f);
  for (int64_t b = 0; b < n; ++b) {
    const float x0 = 2.0f * static_cast<float>(b);
    const float x1 = x0 + 2.0f;

    int64_t start = b * c * h * w;
    int64_t interm = start + c * h * w / 2;
    int64_t end = interm + c * h * w / 2;

    std::fill(inputVals.begin() + start, inputVals.begin() + interm, x0);
    std::fill(inputVals.begin() + interm, inputVals.begin() + end, x1);

    std::fill(expectedVals.begin() + start, expectedVals.begin() + interm, y0);
    std::fill(expectedVals.begin() + interm, expectedVals.begin() + end, y1);
  }
  return std::make_pair(inputVals, expectedVals);
}

// Generates input and expected output tensors for LayerNorm node with training
// mode. Extends the inference generator by also computing expected per-batch
// mean and inverse variance outputs.
// Returns: input values, expected output values, expected means,
// expected inv variances for these input values.
inline std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
                  std::vector<float>>
generateIOTensorsForTrainForward(int64_t n, int64_t c, int64_t h, int64_t w,
                                 float scale, float bias, float eps) {
  auto [inputVals, expectedVals] =
      generateIOTensorsForInferForward(n, c, h, w, scale, bias, eps);

  std::vector<float> expectedMeans(n, 0.0f);
  std::vector<float> expectedVariances(n, 1.0f);
  for (int64_t b = 0; b < n; ++b) {
    int64_t start = b * c * h * w;
    int64_t interm = start + c * h * w / 2;

    expectedMeans[b] = (inputVals[start] + inputVals[interm]) / 2.0f;
  }
  return std::make_tuple(inputVals, expectedVals, expectedMeans,
                         expectedVariances);
}

} // namespace fusilli::layernorm_utils

#endif // FUSILLI_SAMPLES_LAYERNORM_LAYERNORM_UTILS_H
