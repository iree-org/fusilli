// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Serialization format for the opaque_data payload in CustomOpAttributes.
// hipDNN treats this payload as opaque bytes; the fusilli plugin interprets it
// as JSON with the schema below.
//
// Both the plugin (deserialization) and integration tests (serialization)
// include this header.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_INCLUDE_CUSTOM_OP_OPAQUE_DATA_H
#define FUSILLI_PLUGIN_INCLUDE_CUSTOM_OP_OPAQUE_DATA_H

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct CustomOpOpaqueData {
  std::string mlir;
  uint32_t numOutputs = 0;
  bool isStatic = false;

  // Serialize to opaque byte vector (for hipDNN side / tests).
  static std::vector<uint8_t> serialize(const std::string &mlir,
                                        uint32_t numOutputs, bool isStatic) {
    nlohmann::json j;
    j["mlir"] = mlir;
    j["num_outputs"] = numOutputs;
    j["is_static"] = isStatic;
    auto str = j.dump();
    return {str.begin(), str.end()};
  }

  // Deserialize from opaque byte vector (for plugin side).
  static CustomOpOpaqueData deserialize(const uint8_t *data, size_t size) {
    auto j = nlohmann::json::parse(data, data + size);
    return {.mlir = j.at("mlir").get<std::string>(),
            .numOutputs = j.at("num_outputs").get<uint32_t>(),
            .isStatic = j.value("is_static", false)};
  }
};

#endif // FUSILLI_PLUGIN_INCLUDE_CUSTOM_OP_OPAQUE_DATA_H
