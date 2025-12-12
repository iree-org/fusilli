// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// layer normalization nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_LAYERNORM_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_LAYERNORM_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace fusilli {

class LayernormAttr : public AttributesCRTP<LayernormAttr> {
public:
  // Names for Tensor Inputs and Outputs.
  enum class InputNames : uint8_t { X, SCALE, BIAS, EPSILON };
  enum class OutputNames : uint8_t { Y, MEAN, INV_VARIANCE };

  enum class FwdPhase : uint8_t { NOT_SET, TRAINING, INFERENCE };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(LayernormAttr, InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(LayernormAttr, InputNames, SCALE)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(LayernormAttr, InputNames, BIAS)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(LayernormAttr, InputNames, EPSILON)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(LayernormAttr, OutputNames, Y)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(LayernormAttr, OutputNames, MEAN)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(LayernormAttr, OutputNames, INV_VARIANCE)

  LayernormAttr &setForwardPhase(FwdPhase forwardPhase) {
    forwardPhase_ = forwardPhase;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, SCALE)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, BIAS)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, EPSILON)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, Y)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, MEAN)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, INV_VARIANCE)

  FwdPhase getForwardPhase() const { return forwardPhase_; }

  // Utility for forward phases.
  static const std::unordered_map<FwdPhase, std::string> kFwdPhaseToStr;

private:
  FwdPhase forwardPhase_ = FwdPhase::NOT_SET;
};

inline const std::unordered_map<LayernormAttr::FwdPhase, std::string>
    LayernormAttr::kFwdPhaseToStr = {
        {LayernormAttr::FwdPhase::NOT_SET, "NOT_SET"},
        {LayernormAttr::FwdPhase::TRAINING, "TRAINING"},
        {LayernormAttr::FwdPhase::INFERENCE, "INFERENCE"},
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_LAYERNORM_ATTRIBUTES_H
