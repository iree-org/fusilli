// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for finding required external programs at
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
#define FUSILLI_SUPPORT_EXTERNAL_TOOLS_H

#include <string>

namespace fusilli {

std::string getIreeCompilePath();
std::string getRocmAgentEnumeratorPath();
std::string getAmdSmiPath();
std::string getIreeCompilerLibPath();

} // namespace fusilli

#endif // FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
