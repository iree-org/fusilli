// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using namespace fusilli;

TEST_CASE("SdpaAttr default constructor", "[sdpa_attr]") {
  SdpaAttr attr;
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
  REQUIRE(attr.getDropout() == 0.0f);
  REQUIRE(attr.getIsCausal() == false);
  REQUIRE(attr.getScale() == std::nullopt);
  REQUIRE(attr.getEnableGqa() == false);
}

TEST_CASE("SdpaAttr scalar setters and getters", "[sdpa_attr]") {
  SdpaAttr attr;

  attr.setDropout(0.1f).setIsCausal(true).setScale(0.125f).setEnableGqa(true);

  REQUIRE(attr.getDropout() == 0.1f);
  REQUIRE(attr.getIsCausal() == true);
  REQUIRE(attr.getScale().has_value());
  REQUIRE(*attr.getScale() == 0.125f);
  REQUIRE(attr.getEnableGqa() == true);
}

TEST_CASE("SdpaAttr tensor setters and getters", "[sdpa_attr]") {
  SdpaAttr attr;

  auto q = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("Q"));
  auto k = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("K"));
  auto v = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("V"));
  auto o = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("O"));

  attr.setQ(q).setK(k).setV(v).setO(o).setName("sdpa_test");

  REQUIRE(attr.inputs.size() == 3);
  REQUIRE(attr.outputs.size() == 1);
  REQUIRE(attr.getName() == "sdpa_test");
  REQUIRE(attr.getQ() == q);
  REQUIRE(attr.getK() == k);
  REQUIRE(attr.getV() == v);
  REQUIRE(attr.getO() == o);
  REQUIRE(attr.getMASK() == nullptr);
}

TEST_CASE("SdpaAttr with attention mask", "[sdpa_attr]") {
  SdpaAttr attr;

  auto q = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("Q"));
  auto k = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("K"));
  auto v = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 8, 64, 64}).setName("V"));
  auto mask = std::make_shared<TensorAttr>(
      TensorAttr().setDim({1, 1, 64, 64}).setName("MASK"));

  attr.setQ(q).setK(k).setV(v).setMASK(mask);

  REQUIRE(attr.inputs.size() == 4);
  REQUIRE(attr.getMASK() == mask);
  REQUIRE(attr.getMASK()->getDim() == std::vector<int64_t>{1, 1, 64, 64});
}

TEST_CASE("SdpaAttr scale can be reset to nullopt", "[sdpa_attr]") {
  SdpaAttr attr;

  attr.setScale(0.5f);
  REQUIRE(attr.getScale().has_value());
  REQUIRE(*attr.getScale() == 0.5f);

  attr.setScale(std::nullopt);
  REQUIRE(!attr.getScale().has_value());
}
