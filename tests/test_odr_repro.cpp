// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal reproduction for ASAN ODR false positive (issue #180).
//
// The `half` type usage is critical: it changes the TU's content enough
// to produce a COMDAT section layout that triggers the double-registration.
// Using `float` alone does not trigger it. Upstream clang is unaffected.
//
// To reproduce with AMD clang (version 22.0.0, fails with ODR violation):
//
//   cmake -GNinja -S . -B build \
//     -DCMAKE_C_COMPILER=clang \
//     -DCMAKE_CXX_COMPILER=clang++ \
//     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
//     -DFUSILLI_ENABLE_ASAN=ON
//   cmake --build build --target fusilli_odr_repro_test_odr_repro
//   build/bin/tests/fusilli_odr_repro_test_odr_repro
//
// AMD clang failure output:
//
// ==1823160==The following global variable is not properly aligned.
// ==1823160==This may happen if another global with the same name
// ==1823160==resides in another non-instrumented module.
// ==1823160==Or the global comes from a C file built w/o -fno-common.
// ==1823160==In either case this is likely an ODR violation bug,
// ==1823160==but AddressSanitizer can not provide more details.
// =================================================================
// ==1823160==ERROR: AddressSanitizer: odr-violation (0x5bde544f1702):
//   [1] size=4 '.str' .../include/fusilli/attributes/types.h:51 in .../build/bin/tests/fusilli_odr_repro_test_odr_repro
//   [2] size=4 '.str' .../include/fusilli/attributes/types.h:51 in .../build/bin/tests/fusilli_odr_repro_test_odr_repro
// These globals were registered at these points:
//   [1]:
//     #0 0x5bde54553389 in __asan_register_globals /therock/src/compiler/amd-llvm/compiler-rt/lib/asan/asan_globals.cpp:447:3
//     #1 0x5bde545544e9 in __asan_register_elf_globals /therock/src/compiler/amd-llvm/compiler-rt/lib/asan/asan_globals.cpp:430:3
//     #2 0x7b9a2f469303 in call_init csu/../csu/libc-start.c:145:3
//     #3 0x7b9a2f469303 in __libc_start_main csu/../csu/libc-start.c:347:5
//     #4 0x5bde54539c24 in _start (../build/bin/tests/fusilli_odr_repro_test_odr_repro+0xa0c24)
// 
//   [2]:
//     #0 0x5bde54553389 in __asan_register_globals /therock/src/compiler/amd-llvm/compiler-rt/lib/asan/asan_globals.cpp:447:3
//     #1 0x5bde545544e9 in __asan_register_elf_globals /therock/src/compiler/amd-llvm/compiler-rt/lib/asan/asan_globals.cpp:430:3
//     #2 0x7b9a2f469303 in call_init csu/../csu/libc-start.c:145:3
//     #3 0x7b9a2f469303 in __libc_start_main csu/../csu/libc-start.c:347:5
//     #4 0x5bde54539c24 in _start (../build/bin/tests/fusilli_odr_repro_test_odr_repro+0xa0c24)
// 
// ==1823160==HINT: if you don't care about these errors you may set ASAN_OPTIONS=detect_odr_violation=0
// SUMMARY: AddressSanitizer: odr-violation: global '.str' at .../include/fusilli/attributes/types.h:51 in .../build/bin/tests/fusilli_odr_repro_test_odr_repro
// ==1823160==ABORTING
//
// This passes with upstream clang (tested with clang-18.1.3)

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace fusilli;

TEST_CASE("ODR repro: half type triggers ASAN false positive", "[odr]") {
  // Using the `half` type (from fusilli/support/float_types.h, transitively
  // included via fusilli.h -> types.h) produces a COMDAT section layout that
  // causes AMD clang's ASAN to register the same .str global twice.
  std::vector<half> data(6, half(1.0f));
  REQUIRE(data[0] == half(1.0f));
}
