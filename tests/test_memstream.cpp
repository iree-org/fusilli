// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli/support/memstream.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstring>
#include <string>

using namespace fusilli;

TEST_CASE("MemStream basic operations", "[memstream]") {
  SECTION("Create and validate stream") {
    MemStream ms;
    REQUIRE(ms.isValid());
    REQUIRE(ms.stream() != nullptr);
  }

  SECTION("Write and read simple string") {
    MemStream ms;
    REQUIRE(ms.isValid());

    fprintf(ms, "Hello, World!");
    std::string result = ms.str();
    REQUIRE(result == "Hello, World!");
  }

  SECTION("Write formatted output") {
    MemStream ms;
    REQUIRE(ms.isValid());

    fprintf(ms, "Value: %d, Name: %s", 42, "test");
    std::string result = ms.str();
    REQUIRE(result == "Value: 42, Name: test");
  }

  SECTION("Multiple writes accumulate") {
    MemStream ms;
    REQUIRE(ms.isValid());

    fprintf(ms, "First ");
    fprintf(ms, "Second ");
    fprintf(ms, "Third");
    std::string result = ms.str();
    REQUIRE(result == "First Second Third");
  }

  SECTION("Size tracking") {
    MemStream ms;
    REQUIRE(ms.isValid());

    REQUIRE(ms.size() == 0);
    fprintf(ms, "12345");
    REQUIRE(ms.size() == 5);
    fprintf(ms, "67890");
    REQUIRE(ms.size() == 10);
  }

  SECTION("Empty stream returns empty string") {
    MemStream ms;
    REQUIRE(ms.isValid());
    REQUIRE(ms.str().empty());
    REQUIRE(ms.size() == 0);
  }

  SECTION("Write binary data") {
    MemStream ms;
    REQUIRE(ms.isValid());

    const char data[] = "binary\x00data";
    fwrite(data, 1, sizeof(data) - 1, ms);
    std::string result = ms.str();
    REQUIRE(result.size() == sizeof(data) - 1);
    REQUIRE(memcmp(result.data(), data, sizeof(data) - 1) == 0);
  }
}

TEST_CASE("MemStream implicit FILE* conversion", "[memstream]") {
  SECTION("Pass to fprintf directly") {
    MemStream ms;
    REQUIRE(ms.isValid());

    // This tests the implicit conversion operator.
    FILE *fp = ms;
    REQUIRE(fp != nullptr);
    fprintf(fp, "test");
    REQUIRE(ms.str() == "test");
  }
}

TEST_CASE("MemStream large writes", "[memstream]") {
  SECTION("Write large amount of data") {
    MemStream ms;
    REQUIRE(ms.isValid());

    // Write a large string to test buffer growth.
    std::string largeData(10000, 'x');
    fprintf(ms, "%s", largeData.c_str());
    std::string result = ms.str();
    REQUIRE(result == largeData);
    REQUIRE(result.size() == 10000);
  }

  SECTION("Many small writes") {
    MemStream ms;
    REQUIRE(ms.isValid());

    for (int i = 0; i < 1000; ++i) {
      fprintf(ms, "%d", i % 10);
    }
    std::string result = ms.str();
    REQUIRE(result.size() == 1000);
  }
}

TEST_CASE("FprintAdapter basic operations", "[memstream]") {
  SECTION("Write to string via adapter") {
    std::string output;
    {
      FprintAdapter adapter(output);
      REQUIRE(adapter.isValid());
      fprintf(adapter, "Hello from adapter!");
    }
    REQUIRE(output == "Hello from adapter!");
  }

  SECTION("Formatted output to string") {
    std::string output;
    {
      FprintAdapter adapter(output);
      REQUIRE(adapter.isValid());
      fprintf(adapter, "Count: %d, Value: %.2f", 10, 3.14);
    }
    REQUIRE(output == "Count: 10, Value: 3.14");
  }

  SECTION("Multiple writes accumulate") {
    std::string output;
    {
      FprintAdapter adapter(output);
      REQUIRE(adapter.isValid());
      fprintf(adapter, "A");
      fprintf(adapter, "B");
      fprintf(adapter, "C");
    }
    REQUIRE(output == "ABC");
  }

  SECTION("Empty output when nothing written") {
    std::string output = "initial";
    {
      FprintAdapter adapter(output);
      REQUIRE(adapter.isValid());
    }
    REQUIRE(output.empty());
  }
}

TEST_CASE("FprintAdapter implicit FILE* conversion", "[memstream]") {
  SECTION("Use as FILE* parameter") {
    std::string output;
    {
      FprintAdapter adapter(output);
      FILE *fp = adapter;
      REQUIRE(fp != nullptr);
      fputs("test string", fp);
    }
    REQUIRE(output == "test string");
  }
}
