// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Implementation of backend utility functions for GPU target detection,
// compiler flag management, and IREE HAL HIP device parameter configuration.
//
//===----------------------------------------------------------------------===//

#include "fusilli/backend/backend.h"

#include "fusilli/support/external_tools.h"
#include "fusilli/support/logging.h"
#include "fusilli/support/process.h"

#include <iree/base/config.h>
#include <iree/hal/drivers/hip/api.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fusilli {

// Stream operator for Backend.
std::ostream &operator<<(std::ostream &os, const Backend &backend) {
  if (kBackendToStr.contains(backend)) // C++20
    os << kBackendToStr.at(backend);
  else
    os << "UNKNOWN_BACKEND";
  return os;
}

// Maps GPU marketing name to IREE SKU target name.
// Returns empty string if not recognized.
// Supported marketing names (from IREE's KnownTargets.cpp):
//   - AMD Instinct MI355X/MI350X/MI325X/MI300X/MI300A/MI308X
//   - AMD Instinct MI250X/MI250/MI210/MI100
//   - AMD Radeon PRO W7900/W7800/W7700/V710
//   - AMD Radeon RX 7900 XTX/XT, RX 7800 XT, RX 7700 XT
// See:
// https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp
std::string getGpuSkuFromMarketingName(const std::string &marketingName) {
  // Map of known marketing name patterns to IREE SKU targets.
  // The key is a substring to match (case-insensitive), value is the SKU.
  static const std::vector<std::pair<std::string, std::string>> skuPatterns = {
      // CDNA4
      {"MI355X", "mi355x"},
      {"MI350X", "mi350x"},
      // CDNA3
      {"MI325X", "mi325x"},
      {"MI308X", "mi308x"},
      {"MI300X", "mi300x"},
      {"MI300A", "mi300a"},
      // CDNA2
      {"MI250X", "mi250x"},
      {"MI250", "mi250"},
      {"MI210", "mi210"},
      // CDNA1
      {"MI100", "mi100"},
      // RDNA3 Pro
      {"W7900", "w7900"},
      {"W7800", "w7800"},
      {"W7700", "w7700"},
      {"V710", "v710"},
      // RDNA3 Consumer
      {"RX 7900 XTX", "rx7900xtx"},
      {"RX 7900 XT", "rx7900xt"},
      {"RX 7800 XT", "rx7800xt"},
      {"RX 7700 XT", "rx7700xt"},
      // RDNA4
      {"RX 9070 XT", "rx9070xt"},
      {"RX 9070", "rx9070"},
      {"RX 9060 XT", "rx9060xt"},
      {"R9700", "r9700"},
  };

  // Convert marketing name to uppercase for case-insensitive matching.
  std::string upperName = marketingName;
  std::ranges::transform(upperName, upperName.begin(), // C++20
                         [](unsigned char c) { return std::toupper(c); });

  // Find matching SKU pattern.
  for (const auto &[pattern, sku] : skuPatterns) {
    std::string upperPattern = pattern;
    std::ranges::transform(upperPattern, upperPattern.begin(), // C++20
                           [](unsigned char c) { return std::toupper(c); });
    if (upperName.find(upperPattern) != std::string::npos)
      return sku;
  }

  return "";
}

// Queries amd-smi for GPU marketing name.
// Runs `amd-smi static --gpu 0 --json` and extracts market_name field.
// Returns empty string on failure.
std::string getGpuMarketingNameFromAmdSmi() {
  std::string cmd = getAmdSmiPath() + " static --gpu 0 --json 2>/dev/null";

  auto outputOrNone = execCommand(cmd);
  if (!outputOrNone.has_value() || outputOrNone->empty())
    return "";

  std::string output = std::move(*outputOrNone);

  // Simple JSON parsing: find "market_name": "value"
  const std::string key = "\"market_name\":";
  size_t keyPos = output.find(key);
  if (keyPos == std::string::npos)
    return "";

  // Find the opening quote of the value.
  size_t valueStart = output.find('"', keyPos + key.length());
  if (valueStart == std::string::npos)
    return "";
  valueStart++; // Move past the opening quote.

  // Find the closing quote.
  size_t valueEnd = output.find('"', valueStart);
  if (valueEnd == std::string::npos)
    return "";

  return output.substr(valueStart, valueEnd - valueStart);
}

// Parses AMDGPU arch (e.g. `gfx942`) from `rocm_agent_enumerator` CLI output.
std::string getArchFromRocmAgentEnumerator() {
  auto cmd = getRocmAgentEnumeratorPath();

  auto outputOrNone = execCommand(cmd);
  if (!outputOrNone.has_value() || outputOrNone->empty())
    return "";

  std::istringstream stream(std::move(*outputOrNone));
  std::string target;
  while (std::getline(stream, target)) {
    target.erase(target.find_last_not_of(" \n\r\t") + 1);
    if (target == "gfx000")
      continue;
    break;
  }

  return target;
}

// Returns the best available IREE ROCm target for the current AMD GPU.
// Attempts to get SKU name (e.g., `mi300x`) via amd-smi for optimal tuning,
// falls back to architecture (e.g., `gfx942`) via rocm_agent_enumerator.
// See:
// https://iree.dev/guides/deployment-configurations/gpu-rocm/#choosing-hip-targets
std::string getIreeRocmTargetForAmdgpu() {
  // Try to get SKU name first via amd-smi for better compiler tuning.
  FUSILLI_LOG_LABEL_ENDL("INFO: Detecting IREE ROCm target for AMD GPU");

  std::string marketingName = getGpuMarketingNameFromAmdSmi();
  if (!marketingName.empty()) {
    FUSILLI_LOG_LABEL_ENDL("INFO: amd-smi returned marketing name: \""
                           << marketingName << "\"");
    std::string sku = getGpuSkuFromMarketingName(marketingName);
    if (!sku.empty()) {
      FUSILLI_LOG_LABEL_ENDL("INFO: Using SKU target: \""
                             << sku << "\" (from amd-smi)");
      return sku;
    }
  }

  // Fallback to architecture from rocm_agent_enumerator.
  FUSILLI_LOG_LABEL_ENDL(
      "INFO: Marketing name / SKU not recognized from amd-smi; falling back to "
      "architecture from rocm_agent_enumerator");
  std::string arch = getArchFromRocmAgentEnumerator();
  FUSILLI_LOG_LABEL_ENDL("INFO: Using architecture target: \""
                         << arch << "\" (from rocm_agent_enumerator)");
  return arch;
}

// Parses space-separated compiler flags from a string.
// Supports double-quote quoting for flags with spaces.
// Single quotes (') are treated as literal characters, not delimiters.
// Examples:
//   "--flag1 --flag2=value".
//   "--flag1 \"--flag2=value with spaces\""  (works - double quotes).
//   "--flag1 '--flag2=value with spaces'"    (fails - single quotes are
//   literal).
// Returns an empty vector if flagsStr is null or contains no tokens.
std::vector<std::string> parseCompilerFlags(const char *flagsStr) {
  if (!flagsStr)
    return {};

  std::vector<std::string> flags;
  std::istringstream iss(flagsStr);
  std::string token;

  while (iss >> std::quoted(token)) {
    if (!token.empty()) {
      flags.push_back(token);
    }
  }

  return flags;
}

// Map from backend to IREE compile flags.
std::span<const std::string> getBackendFlags(Backend backend) {
  static std::once_flag initFlag;
  static std::unordered_map<Backend, std::vector<std::string>> kBackendFlags;
  std::call_once(initFlag, []() {
    std::vector<std::string> cpuFlags = {
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
        "--iree-torch-externalize-transients",
    };

    // Specify a ROCm target for AMD GPU. First attempts to get the SKU name
    // (e.g., `mi300x`) via `amd-smi` for optimal compiler tuning, then falls
    // back to architecture (e.g., `gfx942`) via `rocm_agent_enumerator`.
    // See:
    // https://iree.dev/guides/deployment-configurations/gpu-rocm/#choosing-hip-targets
    auto rocmTarget = getIreeRocmTargetForAmdgpu();
    std::vector<std::string> amdGpuFlags = {
        // clang-format off
                "--iree-hal-target-backends=rocm",
                std::format("--iree-rocm-target={}", rocmTarget),
                "--iree-opt-level=O3",
                "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last))",
                "--iree-flow-enable-pad-handling",
                "--iree-global-opt-propagate-transposes-through-conv",
                "--iree-global-opt-enable-sink-transpose-through-pad",
                "--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops",
                "--iree-dispatch-creation-enable-aggressive-reshape-movement",
                "--iree-dispatch-creation-enable-split-reduction",
                "--iree-torch-externalize-transients",
        // clang-format on
    };

    // Helper lambda to add extra compiler flags from environment variable.
    auto addExtraFlags = [](std::vector<std::string> &backendFlags) {
      if (const char *extraFlags =
              std::getenv("FUSILLI_EXTRA_COMPILER_FLAGS")) {
        FUSILLI_LOG_LABEL_ENDL("INFO: Adding extra compiler flags from "
                               "FUSILLI_EXTRA_COMPILER_FLAGS");
        std::vector<std::string> parsedFlags = parseCompilerFlags(extraFlags);
        backendFlags.insert(backendFlags.end(), parsedFlags.begin(),
                            parsedFlags.end());
      }
    };

    // Add extra flags to both CPU and AMDGPU backends.
    addExtraFlags(cpuFlags);
    addExtraFlags(amdGpuFlags);

    kBackendFlags[Backend::CPU] = std::move(cpuFlags);
    kBackendFlags[Backend::AMDGPU] = std::move(amdGpuFlags);
  });

  return kBackendFlags.at(backend);
}

// Set appropriate values on `iree_hal_hip_device_params_t` for fusilli hal
// hip driver creation.
void setDefaultIreeHalHipDeviceParams(iree_hal_hip_device_params_t *params) {
  constexpr iree_device_size_t kMinimalFileTransferBufferSize = 1;

  iree_hal_hip_device_params_initialize(params);
  // As buffers should be handled by users, we don't need to cache allocations.
  params->async_caching = false;
  // Fusilli use cases shouldn't require transfering files.
  params->file_transfer_buffer_size = kMinimalFileTransferBufferSize;
}

} // namespace fusilli
