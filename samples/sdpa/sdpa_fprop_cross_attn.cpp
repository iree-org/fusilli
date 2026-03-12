// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "sdpa_utils.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

// Cross-attention: query sequence length differs from key/value sequence
// length. This is typical in encoder-decoder architectures where the decoder
// (query) attends to the encoder output (key/value) which has a different
// length.
TEST_CASE("SDPA forward: cross attention f16", "[sdpa][graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  executeSdpa(handle, DataType::Half,
              /*batch=*/1, /*headsQ=*/8, /*headsKV=*/8,
              /*seqQ=*/32, /*seqKV=*/128, /*headDim=*/64);
}
