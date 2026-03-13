// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_RMSNORM_RMSNORM_UTILS_H
#define FUSILLI_SAMPLES_RMSNORM_RMSNORM_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace fusilli::rmsnorm_utils {

// Generates input and expected output tensors for RmsNorm node with inference
// mode. Uses a two-value pattern per batch (x0, x1) to produce analytically
// computable normalized outputs, enabling deterministic correctness
// verification.
// Returns: input tensor values, expected output tensor values for
// these input values.
inline std::tuple<std::vector<float>, std::vector<float>>
generateIOTensorsForInferForward(int64_t n, int64_t c, int64_t h, int64_t w,
                                 float scale, float eps) {
  // For each batch b, we fill:
  //   - first half of elements with x0 = 2*b + 1
  //   - second half of elements with x1 = 2*b + 3
  //
  // RMS norm formula: y = scale * x / sqrt(mean(x^2) + eps)
  // With two distinct values x0, x1 each filling half the elements:
  //   mean(x^2) = (x0^2 + x1^2) / 2
  //   rms = sqrt(mean(x^2) + eps)
  //   y0 = scale * x0 / rms
  //   y1 = scale * x1 / rms
  const size_t size = n * c * h * w;
  std::vector<float> inputVals(size, 0.0f);
  std::vector<float> expectedVals(size, 0.0f);
  for (int64_t b = 0; b < n; ++b) {
    // Use odd values to avoid x0=0 which would make y0 always 0.
    const float x0 = 2.0f * static_cast<float>(b) + 1.0f;
    const float x1 = x0 + 2.0f;

    const float meanSq = (x0 * x0 + x1 * x1) / 2.0f;
    const float rms = std::sqrt(meanSq + eps);
    const float y0 = scale * x0 / rms;
    const float y1 = scale * x1 / rms;

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

// Generates input and expected output tensors for RmsNorm node with training
// mode. Extends the inference generator by also computing expected per-batch
// inverse RMS outputs.
// Returns: input values, expected output values, expected inv_rms
// values for these input values.
inline std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
generateIOTensorsForTrainForward(int64_t n, int64_t c, int64_t h, int64_t w,
                                 float scale, float eps) {
  auto [inputVals, expectedVals] =
      generateIOTensorsForInferForward(n, c, h, w, scale, eps);

  std::vector<float> expectedInvRms(n, 0.0f);
  for (int64_t b = 0; b < n; ++b) {
    int64_t start = b * c * h * w;
    int64_t interm = start + c * h * w / 2;

    const float x0 = inputVals[start];
    const float x1 = inputVals[interm];
    const float meanSq = (x0 * x0 + x1 * x1) / 2.0f;
    expectedInvRms[b] = 1.0f / std::sqrt(meanSq + eps);
  }
  return std::make_tuple(inputVals, expectedVals, expectedInvRms);
}

} // namespace fusilli::rmsnorm_utils

#endif // FUSILLI_SAMPLES_RMSNORM_RMSNORM_UTILS_H
