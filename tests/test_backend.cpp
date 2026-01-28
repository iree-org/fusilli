// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <string>
#include <vector>

using namespace fusilli;

//===----------------------------------------------------------------------===//
// Tests for parseCompilerFlags
//===----------------------------------------------------------------------===//

TEST_CASE("parseCompilerFlags", "[backend][flags]") {
  SECTION("Null input") {
    std::vector<std::string> result = parseCompilerFlags(nullptr);
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{}));
  }

  SECTION("Empty string") {
    std::vector<std::string> result = parseCompilerFlags("");
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{}));
  }

  SECTION("Whitespace-only string") {
    std::vector<std::string> result = parseCompilerFlags("   \t  \n  ");
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{}));
  }

  SECTION("Single flag") {
    std::vector<std::string> result = parseCompilerFlags("--iree-opt-level=O3");
    REQUIRE_THAT(result, Catch::Matchers::Equals(
                             std::vector<std::string>{"--iree-opt-level=O3"}));
  }

  SECTION("Multiple simple flags") {
    std::vector<std::string> result = parseCompilerFlags(
        "--iree-opt-level=O3 --iree-hal-target-backends=rocm");
    REQUIRE_THAT(
        result, Catch::Matchers::Equals(std::vector<std::string>{
                    "--iree-opt-level=O3", "--iree-hal-target-backends=rocm"}));
  }

  SECTION("Flags with equals signs") {
    std::vector<std::string> result =
        parseCompilerFlags("--iree-hip-target=gfx942 --iree-opt-level=O3 "
                           "--iree-hal-target-backends=rocm");
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{
                             "--iree-hip-target=gfx942", "--iree-opt-level=O3",
                             "--iree-hal-target-backends=rocm"}));
  }

  SECTION("Extra whitespace") {
    std::vector<std::string> result = parseCompilerFlags(
        "  --iree-opt-level=O3    --iree-hal-target-backends=rocm  ");
    REQUIRE_THAT(
        result, Catch::Matchers::Equals(std::vector<std::string>{
                    "--iree-opt-level=O3", "--iree-hal-target-backends=rocm"}));
  }

  SECTION("Tuning spec path without spaces") {
    std::vector<std::string> result = parseCompilerFlags(
        "--iree-codegen-tuning-spec-path=/home/user/tuning_specs/spec.mlir "
        "--iree-opt-level=O3");
    REQUIRE_THAT(
        result,
        Catch::Matchers::Equals(std::vector<std::string>{
            "--iree-codegen-tuning-spec-path=/home/user/tuning_specs/spec.mlir",
            "--iree-opt-level=O3"}));
  }

  SECTION("Quoted flag with spaces") {
    std::vector<std::string> result =
        parseCompilerFlags("--flag1 \"--flag2=value with spaces\"");
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{
                             "--flag1", "--flag2=value with spaces"}));
  }

  SECTION("Multiple flags with mixed quoting") {
    std::vector<std::string> result = parseCompilerFlags(
        "--iree-opt-level=O3 "
        "\"--iree-codegen-tuning-spec-path=/path/with spaces/spec.mlir\"");
    REQUIRE_THAT(
        result,
        Catch::Matchers::Equals(std::vector<std::string>{
            "--iree-opt-level=O3",
            "--iree-codegen-tuning-spec-path=/path/with spaces/spec.mlir"}));
  }

  SECTION("Complex realistic example") {
    std::vector<std::string> result =
        parseCompilerFlags("--iree-codegen-tuning-spec-path=/path/to/spec.mlir "
                           "--iree-opt-level=O3 --iree-hip-target=mi300x");
    REQUIRE_THAT(result,
                 Catch::Matchers::Equals(std::vector<std::string>{
                     "--iree-codegen-tuning-spec-path=/path/to/spec.mlir",
                     "--iree-opt-level=O3", "--iree-hip-target=mi300x"}));
  }

  SECTION("Flags with single quotes") {
    std::vector<std::string> result =
        parseCompilerFlags("'--flag1' '--flag2=value'");
    REQUIRE_THAT(result, Catch::Matchers::Equals(std::vector<std::string>{
                             "'--flag1'", "'--flag2=value'"}));
  }
}

//===----------------------------------------------------------------------===//
// Tests for getGpuSkuFromMarketingName
//===----------------------------------------------------------------------===//

TEST_CASE("getGpuSkuFromMarketingName CDNA GPUs", "[backend][sku]") {
  SECTION("MI300X variants") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI300X") == "mi300x");
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI300X OAM") == "mi300x");
    REQUIRE(getGpuSkuFromMarketingName("mi300x") == "mi300x");
  }

  SECTION("MI300A") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI300A") == "mi300a");
  }

  SECTION("MI325X") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI325X") == "mi325x");
  }

  SECTION("MI308X") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI308X") == "mi308x");
  }

  SECTION("CDNA4 - MI350X/MI355X") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI350X") == "mi350x");
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI355X") == "mi355x");
  }

  SECTION("CDNA2 - MI250 series") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI250X") == "mi250x");
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI250") == "mi250");
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI210") == "mi210");
  }

  SECTION("CDNA1 - MI100") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI100") == "mi100");
  }
}

TEST_CASE("getGpuSkuFromMarketingName RDNA3 Pro GPUs", "[backend][sku]") {
  SECTION("Radeon PRO W series") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon PRO W7900") == "w7900");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon PRO W7800") == "w7800");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon PRO W7700") == "w7700");
  }

  SECTION("Radeon PRO V710") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon PRO V710") == "v710");
  }
}

TEST_CASE("getGpuSkuFromMarketingName RDNA3 Consumer GPUs", "[backend][sku]") {
  SECTION("RX 7900 series") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 7900 XTX") ==
            "rx7900xtx");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 7900 XT") == "rx7900xt");
  }

  SECTION("RX 7800/7700 series") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 7800 XT") == "rx7800xt");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 7700 XT") == "rx7700xt");
  }
}

TEST_CASE("getGpuSkuFromMarketingName RDNA4 GPUs", "[backend][sku]") {
  SECTION("RX 9000 series") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 9070 XT") == "rx9070xt");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 9070") == "rx9070");
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon RX 9060 XT") == "rx9060xt");
  }

  SECTION("Radeon AI PRO R9700") {
    REQUIRE(getGpuSkuFromMarketingName("AMD Radeon AI PRO R9700") == "r9700");
  }
}

TEST_CASE("getGpuSkuFromMarketingName case insensitivity", "[backend][sku]") {
  SECTION("Lowercase input") {
    REQUIRE(getGpuSkuFromMarketingName("amd instinct mi300x") == "mi300x");
  }

  SECTION("Uppercase input") {
    REQUIRE(getGpuSkuFromMarketingName("AMD INSTINCT MI300X") == "mi300x");
  }

  SECTION("Mixed case input") {
    REQUIRE(getGpuSkuFromMarketingName("Amd Instinct Mi300x") == "mi300x");
  }
}

TEST_CASE("getGpuSkuFromMarketingName unrecognized inputs", "[backend][sku]") {
  SECTION("Empty string") { REQUIRE(getGpuSkuFromMarketingName("").empty()); }

  SECTION("Unknown GPU") {
    REQUIRE(getGpuSkuFromMarketingName("NVIDIA GeForce RTX 4090").empty());
    REQUIRE(getGpuSkuFromMarketingName("Unknown GPU Model").empty());
  }

  SECTION("Partial match that shouldn't match") {
    // "MI30" alone shouldn't match MI300X
    REQUIRE(getGpuSkuFromMarketingName("AMD Instinct MI30").empty());
  }
}

//===----------------------------------------------------------------------===//
// Tests for getGpuMarketingNameFromAmdSmi
//
// NOTE: These tests require amd-smi to be installed and accessible.
// They will return empty string if amd-smi is not available.
//===----------------------------------------------------------------------===//

TEST_CASE("getGpuMarketingNameFromAmdSmi returns string or empty",
          "[backend][amd-smi]") {
  // This test verifies the function doesn't crash and returns a valid result.
  // On systems without amd-smi, it should return an empty string.
  std::string result = getGpuMarketingNameFromAmdSmi();

  // Result should either be empty (no amd-smi) or contain "AMD"
  if (!result.empty()) {
    // If we got a result, it should contain AMD in the name
    REQUIRE(result.find("AMD") != std::string::npos);
  }
  // Empty result is also acceptable (amd-smi not available)
}

//===----------------------------------------------------------------------===//
// Tests for getIreeHipTargetForAmdgpu
//
// NOTE: These tests require either amd-smi or rocm_agent_enumerator
// to be installed and accessible.
//===----------------------------------------------------------------------===//

TEST_CASE("getIreeHipTargetForAmdgpu returns valid target or empty",
          "[backend][hip-target]") {
  // This test verifies the function doesn't crash and returns a valid result.
  std::string result = getIreeHipTargetForAmdgpu();

  // Result should either be empty (no tools available) or a valid target
  if (!result.empty()) {
    // Valid targets are either:
    // 1. SKU names (e.g., mi300x, w7900)
    // 2. Architecture names (e.g., gfx942, gfx1100)
    bool isSkuName = (result.find("mi") == 0 || result.find("rx") == 0 ||
                      result.find("w7") == 0 || result.find("v7") == 0 ||
                      result.find("r9") == 0);
    bool isArchName = (result.find("gfx") == 0);

    REQUIRE((isSkuName || isArchName));
  }
  // Empty result is also acceptable (no tools available)
}

TEST_CASE("getIreeHipTargetForAmdgpu prefers SKU over architecture",
          "[backend][hip-target]") {
  // Get both the marketing name and the final target
  std::string marketingName = getGpuMarketingNameFromAmdSmi();
  std::string target = getIreeHipTargetForAmdgpu();

  // If we got a marketing name that maps to a known SKU,
  // the target should be that SKU, not an architecture
  if (!marketingName.empty()) {
    std::string expectedSku = getGpuSkuFromMarketingName(marketingName);
    if (!expectedSku.empty()) {
      REQUIRE(target == expectedSku);
    }
  }
  // If marketing name is empty or unrecognized, we can't verify this
}
