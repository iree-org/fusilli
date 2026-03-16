// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// scaled dot-product attention (SDPA) nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_SDPA_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_SDPA_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace fusilli {

class SdpaAttr : public AttributesCRTP<SdpaAttr> {
public:
  // Names for Tensor Inputs and Outputs.
  enum class InputNames : uint8_t { Q, K, V, MASK };
  enum class OutputNames : uint8_t { O };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Tensor setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(SdpaAttr, InputNames, Q)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(SdpaAttr, InputNames, K)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(SdpaAttr, InputNames, V)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(SdpaAttr, InputNames, MASK)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(SdpaAttr, OutputNames, O)

  // Tensor getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, Q)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, K)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, V)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, MASK)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, O)

  // Scalar attribute setters:
  SdpaAttr &setDropout(float p) {
    dropout_ = p;
    return *this;
  }

  SdpaAttr &setIsCausal(bool v) {
    isCausal_ = v;
    return *this;
  }

  SdpaAttr &setScale(std::optional<float> s) {
    scale_ = s;
    return *this;
  }

  SdpaAttr &setEnableGqa(bool v) {
    enableGqa_ = v;
    return *this;
  }

  // Scalar attribute getters:
  float getDropout() const { return dropout_; }
  bool getIsCausal() const { return isCausal_; }
  std::optional<float> getScale() const { return scale_; }
  bool getEnableGqa() const { return enableGqa_; }

private:
  float dropout_ = 0.0f;
  bool isCausal_ = false;
  std::optional<float> scale_ = std::nullopt;
  bool enableGqa_ = false;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_SDPA_ATTRIBUTES_H
