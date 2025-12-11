// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("LayernormAttr default constructor", "[layernorm_attr]") {
  LayernormAttr attr;
  REQUIRE(attr.getForwardPhase() == LayernormAttr::FwdPhase::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("LayernormAttr setters and getters", "[layernorm_attr]") {
  LayernormAttr attr;

  LayernormAttr::FwdPhase phase = LayernormAttr::FwdPhase::INFERENCE;

  attr.setForwardPhase(phase);

  REQUIRE(attr.getForwardPhase() == phase);

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto s = std::make_shared<TensorAttr>(2.0f);
  auto b = std::make_shared<TensorAttr>(3.0f);
  auto e = std::make_shared<TensorAttr>(4.0f);
  auto y = std::make_shared<TensorAttr>(5.0f);
  auto m = std::make_shared<TensorAttr>(6.0f);
  auto v = std::make_shared<TensorAttr>(7.0f);

  attr.setX(x)
      .setSCALE(s)
      .setBIAS(b)
      .setEPSILON(e)
      .setY(y)
      .setMEAN(m)
      .setINV_VARIANCE(v);

  REQUIRE(attr.inputs.size() == 4);
  REQUIRE(attr.outputs.size() == 3);

  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getSCALE() == s);
  REQUIRE(attr.getBIAS() == b);
  REQUIRE(attr.getEPSILON() == e);
  REQUIRE(attr.getY() == y);
  REQUIRE(attr.getMEAN() == m);
  REQUIRE(attr.getINV_VARIANCE() == v);

  REQUIRE(attr.getX()->getDataType() == DataType::Float);
  REQUIRE(attr.getSCALE()->getDataType() == DataType::Float);
  REQUIRE(attr.getBIAS()->getDataType() == DataType::Float);
  REQUIRE(attr.getEPSILON()->getDataType() == DataType::Float);
  REQUIRE(attr.getY()->getDataType() == DataType::Float);
  REQUIRE(attr.getMEAN()->getDataType() == DataType::Float);
  REQUIRE(attr.getINV_VARIANCE()->getDataType() == DataType::Float);

  REQUIRE(attr.getX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getSCALE()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getBIAS()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getEPSILON()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getMEAN()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getINV_VARIANCE()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getSCALE()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getBIAS()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getEPSILON()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getMEAN()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getINV_VARIANCE()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->isScalar() == true);
  REQUIRE(attr.getSCALE()->isScalar() == true);
  REQUIRE(attr.getBIAS()->isScalar() == true);
  REQUIRE(attr.getEPSILON()->isScalar() == true);
  REQUIRE(attr.getY()->isScalar() == true);
  REQUIRE(attr.getMEAN()->isScalar() == true);
  REQUIRE(attr.getINV_VARIANCE()->isScalar() == true);

  REQUIRE(attr.getX()->isVirtual() == false);
  REQUIRE(attr.getSCALE()->isVirtual() == false);
  REQUIRE(attr.getBIAS()->isVirtual() == false);
  REQUIRE(attr.getEPSILON()->isVirtual() == false);
  REQUIRE(attr.getY()->isVirtual() == false);
  REQUIRE(attr.getMEAN()->isVirtual() == false);
  REQUIRE(attr.getINV_VARIANCE()->isVirtual() == false);
}
