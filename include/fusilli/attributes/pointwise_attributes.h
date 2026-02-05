// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// pointwise nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace fusilli {

#define FUSILLI_POINTWISE_OPS(OP)                                              \
  OP(ADD)                                                                      \
  /* OP(ADD_SQUARE)  */                                                        \
  /* OP(BINARY_SELECT)  */                                                     \
  OP(CEIL)                                                                     \
  OP(CMP_EQ)                                                                   \
  OP(CMP_GE)                                                                   \
  OP(CMP_GT)                                                                   \
  OP(CMP_LE)                                                                   \
  OP(CMP_LT)                                                                   \
  OP(CMP_NEQ)                                                                  \
  OP(DIV)                                                                      \
  /* OP(ELU_BWD) */                                                            \
  /* OP(ELU_FWD) */                                                            \
  /* OP(ERF)   */                                                              \
  /* OP(EXP)   */                                                              \
  /* OP(FLOOR)  */                                                             \
  /* OP(GELU_APPROX_TANH_BWD)  */                                              \
  /* OP(GELU_APPROX_TANH_FWD)  */                                              \
  /* OP(GELU_BWD) */                                                           \
  /* OP(GELU_FWD) */                                                           \
  /* OP(GEN_INDEX) */                                                          \
  /* OP(IDENTITY)  */                                                          \
  /* OP(LOG) */                                                                \
  /* OP(LOGICAL_AND) */                                                        \
  /* OP(LOGICAL_NOT) */                                                        \
  /* OP(LOGICAL_OR) */                                                         \
  /* OP(MAX_OP) */                                                             \
  /* OP(MIN_OP) */                                                             \
  OP(MUL)                                                                      \
  /* OP(NEG) */                                                                \
  /* OP(RECIPROCAL) */                                                         \
  /* OP(RELU_BWD) */                                                           \
  OP(RELU_FWD)                                                                 \
  /* OP(RSQRT) */                                                              \
  /* OP(SIGMOID_BWD) */                                                        \
  OP(SIGMOID_FWD)                                                              \
  /* OP(SIN) */                                                                \
  /* OP(SOFTPLUS_BWD) */                                                       \
  /* OP(SOFTPLUS_FWD) */                                                       \
  /* OP(SQRT) */                                                               \
  OP(SUB)                                                                      \
  /* OP(SWISH_BWD) */                                                          \
  /* OP(SWISH_FWD) */                                                          \
  /* OP(TAN) */                                                                \
  /* OP(TANH_BWD) */                                                           \
  OP(TANH_FWD)

class PointwiseAttr : public AttributesCRTP<PointwiseAttr> {
public:
  // Names for Tensor Inputs and Outputs. Pointwise can have a maximum of three
  // inputs.
  enum class InputNames : uint8_t { IN_0, IN_1, IN_2 };
  enum class OutputNames : uint8_t { OUT_0 };

#define FUSILLI_POINTWISE_DECLARE(OP) OP,
  enum class Mode : uint8_t {
    NOT_SET,
    FUSILLI_POINTWISE_OPS(FUSILLI_POINTWISE_DECLARE)
  };
#undef FUSILLI_POINTWISE_DECLARE

  std::map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(PointwiseAttr, OutputNames, OUT_0)

  PointwiseAttr &setMode(Mode mode) {
    mode_ = mode;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, OUT_0)

  Mode getMode() const { return mode_; }

  // Utilities for pointwise modes.
  static const std::unordered_map<Mode, std::string> kModeToStr;
  static const std::unordered_map<PointwiseAttr::Mode, int>
      kModeToRequiredInputCount;

private:
  Mode mode_ = Mode::NOT_SET;
};

#define FUSILLI_DECLARE_STRINGIFY_POINTWISE_MODE(mode)                         \
  {PointwiseAttr::Mode::mode, #mode},

inline const std::unordered_map<PointwiseAttr::Mode, std::string>
    PointwiseAttr::kModeToStr = {
        FUSILLI_POINTWISE_OPS(FUSILLI_DECLARE_STRINGIFY_POINTWISE_MODE)};
#undef FUSILLI_DECLARE_STRINGIFY_POINTWISE_MODE

inline const std::unordered_map<PointwiseAttr::Mode, int>
    PointwiseAttr::kModeToRequiredInputCount = {
        {PointwiseAttr::Mode::ADD, 2},
        {PointwiseAttr::Mode::CEIL, 1},
        {PointwiseAttr::Mode::CMP_EQ, 2},
        {PointwiseAttr::Mode::CMP_LT, 2},
        {PointwiseAttr::Mode::CMP_LE, 2},
        {PointwiseAttr::Mode::CMP_GT, 2},
        {PointwiseAttr::Mode::CMP_GE, 2},
        {PointwiseAttr::Mode::CMP_NEQ, 2},
        {PointwiseAttr::Mode::DIV, 2},
        {PointwiseAttr::Mode::MUL, 2},
        {PointwiseAttr::Mode::RELU_FWD, 1},
        {PointwiseAttr::Mode::SIGMOID_FWD, 1},
        {PointwiseAttr::Mode::SUB, 2},
        {PointwiseAttr::Mode::TANH_FWD, 1}};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
