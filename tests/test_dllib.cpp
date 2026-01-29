// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli/support/dllib.h>
#include <fusilli/support/logging.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <utility>

using namespace fusilli;

TEST_CASE("DynamicLibrary default construction", "[dllib]") {
  DynamicLibrary lib;
  REQUIRE_FALSE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary load and close libm", "[dllib]") {
  DynamicLibrary lib;

  // libm should be available on all Linux systems.
  ErrorObject loadErr = lib.load("libm.so.6");
  REQUIRE(isOk(loadErr));
  REQUIRE(lib.isLoaded());

  ErrorObject closeErr = lib.close();
  REQUIRE(isOk(closeErr));
  REQUIRE_FALSE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary getSymbol", "[dllib]") {
  DynamicLibrary lib;

  ErrorObject loadErr = lib.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  // Get the cos function from libm.
  ErrorOr<double (*)(double)> cosFunc =
      lib.getSymbol<double (*)(double)>("cos");
  REQUIRE(isOk(cosFunc));

  // Verify the function works correctly.
  double result = (*cosFunc)(0.0);
  REQUIRE(result == 1.0);

  // Get the sin function.
  ErrorOr<double (*)(double)> sinFunc =
      lib.getSymbol<double (*)(double)>("sin");
  REQUIRE(isOk(sinFunc));

  result = (*sinFunc)(0.0);
  REQUIRE(result == 0.0);
}

TEST_CASE("DynamicLibrary getSymbol not found", "[dllib]") {
  DynamicLibrary lib;

  ErrorObject loadErr = lib.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  // Try to get a symbol that doesn't exist.
  ErrorOr<void (*)()> badFunc =
      lib.getSymbol<void (*)()>("this_symbol_does_not_exist_12345");
  REQUIRE(isError(badFunc));
}

TEST_CASE("DynamicLibrary getSymbol without load", "[dllib]") {
  DynamicLibrary lib;

  // Try to get a symbol without loading a library first.
  ErrorOr<void (*)()> func = lib.getSymbol<void (*)()>("cos");
  REQUIRE(isError(func));
  ErrorObject err = func;
  REQUIRE(err.getMessage() == "Library not loaded");
}

TEST_CASE("DynamicLibrary load nonexistent library", "[dllib]") {
  DynamicLibrary lib;

  ErrorObject loadErr = lib.load("/nonexistent/path/to/library.so");
  REQUIRE(isError(loadErr));
  REQUIRE_FALSE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary close without load", "[dllib]") {
  DynamicLibrary lib;

  // close() should be safe to call on an unloaded library.
  ErrorObject closeErr = lib.close();
  REQUIRE(isOk(closeErr));
  REQUIRE_FALSE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary double close", "[dllib]") {
  DynamicLibrary lib;

  ErrorObject loadErr = lib.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  ErrorObject closeErr = lib.close();
  REQUIRE(isOk(closeErr));
  REQUIRE_FALSE(lib.isLoaded());

  // Double close should be safe.
  closeErr = lib.close();
  REQUIRE(isOk(closeErr));
  REQUIRE_FALSE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary move constructor", "[dllib]") {
  DynamicLibrary lib1;
  ErrorObject loadErr = lib1.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  // Move construct lib2 from lib1.
  DynamicLibrary lib2(std::move(lib1));

  // lib2 should have the handle.
  REQUIRE(lib2.isLoaded());

  // lib1 should be empty.
  REQUIRE_FALSE(lib1.isLoaded());

  // Verify we can still use lib2.
  ErrorOr<double (*)(double)> cosFunc =
      lib2.getSymbol<double (*)(double)>("cos");
  REQUIRE(isOk(cosFunc));
  REQUIRE((*cosFunc)(0.0) == 1.0);
}

TEST_CASE("DynamicLibrary move assignment", "[dllib]") {
  DynamicLibrary lib1;
  ErrorObject loadErr = lib1.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  // Move assign lib1 to lib2.
  DynamicLibrary lib2;
  lib2 = std::move(lib1);

  // lib2 should have the handle.
  REQUIRE(lib2.isLoaded());

  // lib1 should be empty.
  REQUIRE_FALSE(lib1.isLoaded());

  // Verify we can still use lib2.
  ErrorOr<double (*)(double)> sinFunc =
      lib2.getSymbol<double (*)(double)>("sin");
  REQUIRE(isOk(sinFunc));
  REQUIRE((*sinFunc)(0.0) == 0.0);
}

TEST_CASE("DynamicLibrary move assignment closes existing", "[dllib]") {
  DynamicLibrary lib1;
  ErrorObject loaded1 = lib1.load("libm.so.6");
  REQUIRE(isOk(loaded1));

  DynamicLibrary lib2;
  ErrorObject loaded2 = lib2.load("libm.so.6");
  REQUIRE(isOk(loaded2));

  // Move assign lib1 to lib2, which should close lib2's existing library.
  lib2 = std::move(lib1);

  REQUIRE(lib2.isLoaded());
  REQUIRE_FALSE(lib1.isLoaded());
}

TEST_CASE("DynamicLibrary self move assignment", "[dllib]") {
  DynamicLibrary lib;
  ErrorObject loadErr = lib.load("libm.so.6");
  REQUIRE(isOk(loadErr));

  // Self-move should be safe (no-op).
  lib = std::move(lib);

  // Library should still be loaded with the same handle.
  REQUIRE(lib.isLoaded());
}

TEST_CASE("DynamicLibrary destructor closes library", "[dllib]") {
  void *handle = nullptr;
  {
    DynamicLibrary lib;
    ErrorObject loadErr = lib.load("libm.so.6");
    REQUIRE(isOk(loadErr));
    // lib goes out of scope here, destructor should close the library.
  }
  // We can't directly verify the library is closed, but at least we didn't
  // crash or leak.
}

TEST_CASE("DynamicLibrary reload", "[dllib]") {
  DynamicLibrary lib;

  ErrorObject loaded1 = lib.load("libm.so.6");
  REQUIRE(isOk(loaded1));

  // Loading again should close the old library and load a new one.
  ErrorObject loaded2 = lib.load("libm.so.6");
  REQUIRE(isOk(loaded2));
  REQUIRE(lib.isLoaded());
  // Note: handle might be the same or different depending on the loader.
}
