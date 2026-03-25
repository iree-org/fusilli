// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_BATCHNORM_BATCHNORM_UTILS_H
#define FUSILLI_SAMPLES_BATCHNORM_BATCHNORM_UTILS_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace fusilli::batchnorm_utils {

// Generates input and expected output tensors for BatchNorm inference in NCHW
// physical memory order.
//
// Fill pattern: x[n, c, h, w] = float(c + 1)  (uniform per channel, differs
// across channels). Running statistics are set to mean=0 and var=1 for all
// channels, so the normalisation formula reduces to:
//
//   y = scale * (c + 1) / sqrt(1 + eps) + bias
//
// Returns: inputVals (NCHW flat), runningMeanVals [C], runningVarVals [C],
//          expectedVals (NCHW flat).
inline std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
                  std::vector<float>>
generateNchwForInferForward(int64_t n, int64_t c, int64_t h, int64_t w,
                            float scale, float bias, float eps) {
  const size_t totalSize = static_cast<size_t>(n * c * h * w);
  std::vector<float> inputVals(totalSize), meanVals(c, 0.0f),
      varVals(c, 1.0f), expectedVals(totalSize);

  const float invStd = 1.0f / std::sqrt(1.0f + eps);

  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ci = 0; ci < c; ++ci) {
      const float xVal = static_cast<float>(ci + 1);
      const float yVal = scale * xVal * invStd + bias;
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          int64_t idx = ni * c * h * w + ci * h * w + hi * w + wi;
          inputVals[idx] = xVal;
          expectedVals[idx] = yVal;
        }
      }
    }
  }
  return {inputVals, meanVals, varVals, expectedVals};
}

// Generates input and expected output tensors for BatchNorm training in NCHW
// physical memory order.
//
// Fill pattern: for each channel, the N*H*W elements are split in half. The
// first half (spatial index < N*H*W/2) is filled with -1, the second with +1.
// This produces analytically exact per-channel batch statistics:
//
//   batch_mean[c] = 0,    batch_var[c] = 1
//   saved_mean[c] = 0,    saved_inv_var[c] = 1 / sqrt(1 + eps)
//   y = scale * x / sqrt(1 + eps) + bias
//
// N*H*W must be even.
//
// Returns: inputVals (NCHW flat), expectedVals (NCHW flat),
//          savedMeanVals [C], savedInvVarVals [C].
inline std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
                  std::vector<float>>
generateNchwForTrainForward(int64_t n, int64_t c, int64_t h, int64_t w,
                            float scale, float bias, float eps) {
  assert(n * h * w % 2 == 0 && "n * h * w must be even for two-value pattern");

  const size_t totalSize = static_cast<size_t>(n * c * h * w);
  std::vector<float> inputVals(totalSize), expectedVals(totalSize);

  const float invStd = 1.0f / std::sqrt(1.0f + eps);
  const int64_t halfNHW = n * h * w / 2;

  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ci = 0; ci < c; ++ci) {
      for (int64_t hi = 0; hi < h; ++hi) {
        for (int64_t wi = 0; wi < w; ++wi) {
          int64_t spatialIdx = ni * h * w + hi * w + wi;
          float xVal = (spatialIdx < halfNHW) ? -1.0f : 1.0f;
          int64_t idx = ni * c * h * w + ci * h * w + hi * w + wi;
          inputVals[idx] = xVal;
          expectedVals[idx] = scale * xVal * invStd + bias;
        }
      }
    }
  }

  std::vector<float> savedMeanVals(c, 0.0f);
  std::vector<float> savedInvVarVals(c, invStd);

  return {inputVals, expectedVals, savedMeanVals, savedInvVarVals};
}

} // namespace fusilli::batchnorm_utils

#endif // FUSILLI_SAMPLES_BATCHNORM_BATCHNORM_UTILS_H
