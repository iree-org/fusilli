// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "sdpa_utils.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <optional>

// GQA with different K and V head counts: K and V may have independently
// grouped heads relative to Q.
TEST_CASE("SDPA forward: GQA Hk!=Hv f16", "[sdpa][graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  executeSdpa(handle, DataType::Half,
              /*batch=*/1, /*headsQ=*/8, /*headsK=*/4, /*headsV=*/2,
              /*seqQ=*/64, /*seqKV=*/64, /*headDim=*/64,
              /*isCausal=*/false, /*scale=*/std::nullopt,
              /*enableGqa=*/true);
}
