// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains general support methods and classes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_EXTRAS_H
#define FUSILLI_SUPPORT_EXTRAS_H

#include "fusilli/support/target_platform.h"

#include <string>

namespace fusilli {

// An STL-style algorithm similar to std::for_each that applies a second
// functor between every pair of elements.
//
// This provides the control flow logic to, for example, print a
// comma-separated list:
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return;
  eachFn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    eachFn(*begin);
  }
}

// An overload of `interleave` which additionally accepts a SkipFunctor
// to skip certain elements based on a predicate.
//
// This provides the control flow logic to, for example, print a
// comma-separated list excluding "foo":
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; },
//              [&](std::string name) { return name == "foo"; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor, typename SkipFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor eachFn, NullaryFunctor betweenFn,
                       SkipFunctor skipFn) {
  if (begin == end)
    return;
  bool first = true;
  for (; begin != end; ++begin) {
    if (!skipFn(*begin)) {
      if (!first)
        betweenFn();
      first = false;
      eachFn(*begin);
    }
  }
}

// Returns true if the argument contains characters that are special to the
// shell and require quoting (parentheses, spaces, glob characters, etc.).
// Platform-aware: on Windows, backslash is not a metacharacter (it is the
// path separator).
inline bool needsShellQuoting(const std::string &arg) {
  for (char c : arg) {
    switch (c) {
#if !defined(FUSILLI_PLATFORM_WINDOWS)
    // On POSIX shells these are metacharacters; on Windows cmd.exe they are
    // not (backslash is the path separator, and the others have no special
    // meaning outside of double quotes).
    case '\'':
    case '\\':
    case '$':
    case '`':
#endif
    case ' ':
    case '\t':
    case '"':
    case '(':
    case ')':
    case '{':
    case '}':
    case '[':
    case ']':
    case '|':
    case '&':
    case ';':
    case '<':
    case '>':
    case '~':
    case '*':
    case '?':
    case '!':
    case '#':
      return true;
    default:
      break;
    }
  }
  return false;
}

// Shell-safe argument escaping for command line serialization.
//
// On POSIX: wraps the argument in single quotes if it contains shell
// metacharacters. Single-quoting is preferred because only the single-quote
// character itself needs escaping (done via the '\'' idiom).
//
// On Windows: wraps the argument in double quotes. Internal double-quote
// characters are escaped with a backslash.
inline std::string escapeArgument(const std::string &arg) {
  if (!needsShellQuoting(arg))
    return arg;

  std::string escaped;
  // +2 for surrounding quotes. Underestimates when internal quotes need
  // escaping, but avoids repeated reallocations for long arguments.
  escaped.reserve(arg.size() + 2);
#if defined(FUSILLI_PLATFORM_WINDOWS)
  escaped += '"';
  for (char c : arg) {
    if (c == '"')
      escaped += '\\';
    escaped += c;
  }
  escaped += '"';
#else
  escaped += '\'';
  for (char c : arg) {
    if (c == '\'') {
      // End current single-quoted string, add an escaped single quote,
      // then restart single-quoting: 'foo'\''bar' -> foo'bar
      escaped += "'\\''";
    } else {
      escaped += c;
    }
  }
  escaped += '\'';
#endif
  return escaped;
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_EXTRAS_H
