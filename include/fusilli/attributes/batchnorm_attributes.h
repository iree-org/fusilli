// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// batch normalization nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_BATCHNORM_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_BATCHNORM_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/common.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace fusilli {

class BatchnormAttr : public AttributesCRTP<BatchnormAttr> {
public:
  // Names for Tensor Inputs and Outputs.
  enum class InputNames : uint8_t {
    X,
    SCALE,
    BIAS,
    MEAN,
    VAR,
    EPSILON,
    MOMENTUM
  };
  enum class OutputNames : uint8_t { Y, SAVED_MEAN, SAVED_INV_VARIANCE };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BatchnormAttr, InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BatchnormAttr, InputNames, SCALE)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BatchnormAttr, InputNames, BIAS)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BatchnormAttr, InputNames, MEAN)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(BatchnormAttr, InputNames, VAR)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(BatchnormAttr, OutputNames, Y)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(BatchnormAttr, OutputNames, SAVED_MEAN)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(BatchnormAttr, OutputNames,
                                       SAVED_INV_VARIANCE)

  BatchnormAttr &setEpsilon(const std::shared_ptr<TensorAttr> &epsilon) {
    return setInput(InputNames::EPSILON, epsilon);
  }

  BatchnormAttr &setMomentum(const std::shared_ptr<TensorAttr> &momentum) {
    return setInput(InputNames::MOMENTUM, momentum);
  }

  BatchnormAttr &setForwardPhase(NormFwdPhase forwardPhase) {
    forwardPhase_ = forwardPhase;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, SCALE)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, BIAS)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, MEAN)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, VAR)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, Y)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, SAVED_MEAN)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, SAVED_INV_VARIANCE)

  std::shared_ptr<TensorAttr> getEpsilon() const {
    return getInput(InputNames::EPSILON);
  }

  std::shared_ptr<TensorAttr> getMomentum() const {
    return getInput(InputNames::MOMENTUM);
  }

  NormFwdPhase getForwardPhase() const { return forwardPhase_; }

private:
  NormFwdPhase forwardPhase_ = NormFwdPhase::NOT_SET;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_BATCHNORM_ATTRIBUTES_H
