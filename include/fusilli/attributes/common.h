// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the common utilities for different node attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_COMMON_H
#define FUSILLI_ATTRIBUTES_COMMON_H

#include <cstdint>
#include <string>
#include <unordered_map>

namespace fusilli {

// Phases of the normalization forward nodes.
enum class NormFwdPhase : uint8_t {
  NOT_SET,

  TRAINING,
  INFERENCE
};

inline const std::unordered_map<NormFwdPhase, std::string> kNormFwdPhaseToStr =
    {
        {NormFwdPhase::NOT_SET, "NOT_SET"},
        {NormFwdPhase::TRAINING, "TRAINING"},
        {NormFwdPhase::INFERENCE, "INFERENCE"},
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_COMMON_H
