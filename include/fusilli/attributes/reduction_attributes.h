// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// reduction nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_REDUCTION_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_REDUCTION_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace fusilli {

#define FUSILLI_REDUCTION_MODES(OP)                                            \
  OP(NOT_SET)                                                                  \
  OP(SUM)                                                                      \
  /* OP(ADD), */                                                               \
  /* OP(MUL), */                                                               \
  OP(MIN)                                                                      \
  OP(MAX)                                                                      \
  /* OP(AMAX), */                                                              \
  /* OP(AVG), */                                                               \
  /* OP(NORM1), */                                                             \
  /* OP(NORM2), */                                                             \
  /* OP(MUL_NO_ZEROS) */

class ReductionAttr : public AttributesCRTP<ReductionAttr> {
public:
  // Names for Tensor Inputs and Outputs. Reduction has a single input.
  enum class InputNames : uint8_t { X };
  enum class OutputNames : uint8_t { Y };

  enum class Mode : uint8_t {
#define FUSILLI_REDUCTION_MODE_ENUM(mode) mode,
    FUSILLI_REDUCTION_MODES(FUSILLI_REDUCTION_MODE_ENUM)
#undef FUSILLI_REDUCTION_MODE_ENUM
  };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(ReductionAttr, InputNames, X)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(ReductionAttr, OutputNames, Y)

  ReductionAttr &setMode(Mode mode) {
    mode_ = mode;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, X)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, Y)

  Mode getMode() const { return mode_; }

  // Utilities for reduction modes.
  static const std::unordered_map<Mode, std::string> kModeToStr;

private:
  Mode mode_ = Mode::NOT_SET;
};

inline const std::unordered_map<ReductionAttr::Mode, std::string>
#define FUSILLI_REDUCTION_MODE_MAP(mode) {ReductionAttr::Mode::mode, #mode},
    ReductionAttr::kModeToStr = {
        FUSILLI_REDUCTION_MODES(FUSILLI_REDUCTION_MODE_MAP)};
#undef FUSILLI_REDUCTION_MODE_MAP

#undef FUSILLI_REDUCTION_MODES

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_REDUCTION_ATTRIBUTES_H
