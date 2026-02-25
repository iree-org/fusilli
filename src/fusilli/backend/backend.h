// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains backend specific code like the `Backend` type, code to
// map from Backend to `iree-compile` flags, IREE runtime types and deleters.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BACKEND_H
#define FUSILLI_BACKEND_BACKEND_H

#include "fusilli/attributes/types.h"

#include <iree/hal/api.h>
#include <iree/hal/drivers/hip/api.h>
#include <iree/vm/api.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

// Target backend to run the generated kernels on.
enum class Backend : uint8_t {
  CPU,
  AMDGPU,
};

static const std::unordered_map<Backend, std::string> kBackendToStr = {
    {Backend::CPU, "CPU"},
    {Backend::AMDGPU, "AMDGPU"},
};

static const std::unordered_map<Backend, bool> kBackendExecuteAsync = {
    {Backend::CPU, false},
    {Backend::AMDGPU, true},
};

// Stream operator for Backend.
std::ostream &operator<<(std::ostream &os, const Backend &backend);

// Map from backend to IREE HAL driver name.
static const std::unordered_map<Backend, const char *> kHalDriver = {
    {Backend::CPU, "local-task"},
    {Backend::AMDGPU, "hip"},
};

// Maps GPU marketing name to IREE SKU target name.
// Returns empty string if not recognized.
// Supported marketing names (from IREE's KnownTargets.cpp):
//   - AMD Instinct MI355X/MI350X/MI325X/MI300X/MI300A/MI308X
//   - AMD Instinct MI250X/MI250/MI210/MI100
//   - AMD Radeon PRO W7900/W7800/W7700/V710
//   - AMD Radeon RX 7900 XTX/XT, RX 7800 XT, RX 7700 XT
// See:
// https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp
std::string getGpuSkuFromMarketingName(const std::string &marketingName);

// Queries amd-smi for GPU marketing name.
// Runs `amd-smi static --gpu 0 --json` and extracts market_name field.
// Returns empty string on failure.
std::string getGpuMarketingNameFromAmdSmi();

// Parses AMDGPU arch (e.g. `gfx942`) from `rocm_agent_enumerator` CLI output.
std::string getArchFromRocmAgentEnumerator();

// Returns the best available IREE ROCm target for the current AMD GPU.
// Attempts to get SKU name (e.g., `mi300x`) via amd-smi for optimal tuning,
// falls back to architecture (e.g., `gfx942`) via rocm_agent_enumerator.
// See:
// https://iree.dev/guides/deployment-configurations/gpu-rocm/#choosing-hip-targets
std::string getIreeRocmTargetForAmdgpu();

// Parses space-separated compiler flags from a string.
// Supports double-quote quoting for flags with spaces.
// Single quotes (') are treated as literal characters, not delimiters.
// Examples:
//   "--flag1 --flag2=value".
//   "--flag1 \"--flag2=value with spaces\""  (works - double quotes).
//   "--flag1 '--flag2=value with spaces'"    (fails - single quotes are
//   literal).
// Returns an empty vector if flagsStr is null or contains no tokens.
std::vector<std::string> parseCompilerFlags(const char *flagsStr);

// Map from backend to IREE compile flags.
std::span<const std::string> getBackendFlags(Backend backend);

// Set appropriate values on `iree_hal_hip_device_params_t` for fusilli hal
// hip driver creation.
void setDefaultIreeHalHipDeviceParams(iree_hal_hip_device_params_t *params);

// Template specializations to map from primitive types
// to IREE HAL element type.
template <typename T> struct IreeHalElementType;
//
// float -> IREE_HAL_ELEMENT_TYPE_FLOAT_32:
template <> struct IreeHalElementType<float> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_FLOAT_32;
};
//
// half -> IREE_HAL_ELEMENT_TYPE_FLOAT_16:
template <> struct IreeHalElementType<half> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_FLOAT_16;
};
//
// bf16 -> IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
template <> struct IreeHalElementType<bf16> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
};
//
// int -> IREE_HAL_ELEMENT_TYPE_INT_32:
template <> struct IreeHalElementType<int> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_32;
};
//
// int16 -> IREE_HAL_ELEMENT_TYPE_INT_16:
template <> struct IreeHalElementType<int16_t> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_16;
};
//
// int8 -> IREE_HAL_ELEMENT_TYPE_INT_8:
template <> struct IreeHalElementType<int8_t> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_8;
};
//
// Assert for unsupported types:
template <typename T> struct IreeHalElementType {
  static_assert(sizeof(T) == 0, "Unsupported type for IREE_HAL_ELEMENT_TYPE");
};
//
// Getter:
template <typename T> iree_hal_element_type_t getIreeHalElementTypeForT() {
  return IreeHalElementType<T>::kType;
}

// Custom deleter for IREE VM instance.
struct IreeVmInstanceDeleter {
  void operator()(iree_vm_instance_t *instance) const {
    if (instance)
      iree_vm_instance_release(instance);
  }
};

// Custom deleter for IREE HAL device.
struct IreeHalDeviceDeleter {
  void operator()(iree_hal_device_t *device) const {
    if (device)
      iree_hal_device_release(device);
  }
};

// Custom deleter for IREE VM context.
struct IreeVmContextDeleter {
  void operator()(iree_vm_context_t *context) const {
    if (context)
      iree_vm_context_release(context);
  }
};

// Custom deleter for IREE VM list.
struct IreeVmListDeleter {
  void operator()(iree_vm_list_t *list) const {
    if (list)
      iree_vm_list_release(list);
  }
};

// Custom deleter for IREE HAL buffer view.
struct IreeHalBufferViewDeleter {
  void operator()(iree_hal_buffer_view_t *bufferView) const {
    if (bufferView)
      iree_hal_buffer_view_release(bufferView);
  }
};

// Aliases for IREE types with custom deleters.
using IreeVmInstanceSharedPtrType = std::shared_ptr<iree_vm_instance_t>;
using IreeHalDeviceUniquePtrType =
    std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>;
using IreeVmContextUniquePtrType =
    std::unique_ptr<iree_vm_context_t, IreeVmContextDeleter>;
using IreeVmListUniquePtrType =
    std::unique_ptr<iree_vm_list_t, IreeVmListDeleter>;
using IreeHalBufferViewUniquePtrType =
    std::unique_ptr<iree_hal_buffer_view_t, IreeHalBufferViewDeleter>;

} // namespace fusilli

#endif // FUSILLI_BACKEND_BACKEND_H
