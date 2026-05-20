#!/usr/bin/env python3
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
from pathlib import Path

# Add benchmarks/ to path so we can import run_tuner.
# The utility functions under test don't depend on amdsharktuner, and the
# import guard in run_tuner.py defers the sys.exit to main().
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_tuner import build_compile_args, find_cached_artifacts, load_commands


class TestFindCachedArtifacts(unittest.TestCase):
    def test_finds_mlir_and_txt(self):
        """Given a valid cache structure, returns paths to .mlir and .txt files."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            graph_dir = base / ".cache" / "fusilli" / "abc123"
            graph_dir.mkdir(parents=True)
            mlir_file = graph_dir / "iree-compile-input.mlir"
            txt_file = graph_dir / "iree-compile-command.txt"
            mlir_file.write_text("module {}")
            txt_file.write_text("iree-compile input.mlir -o out.vmfb")

            mlir_path, txt_path = find_cached_artifacts(base)
            self.assertEqual(mlir_path, mlir_file)
            self.assertEqual(txt_path, txt_file)

    def test_raises_when_no_cache_dir(self):
        """Raises FileNotFoundError when .cache/fusilli doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_cached_artifacts(Path(tmp))

    def test_raises_when_multiple_graph_dirs(self):
        """Raises FileNotFoundError when multiple graph directories exist."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cache = base / ".cache" / "fusilli"
            (cache / "hash1").mkdir(parents=True)
            (cache / "hash2").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                find_cached_artifacts(base)

    def test_raises_when_no_mlir_file(self):
        """Raises FileNotFoundError when no .mlir file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            graph_dir = base / ".cache" / "fusilli" / "abc123"
            graph_dir.mkdir(parents=True)
            (graph_dir / "cmd.txt").write_text("iree-compile ...")
            with self.assertRaises(FileNotFoundError):
                find_cached_artifacts(base)


class TestBuildCompileArgs(unittest.TestCase):
    def test_strips_output_and_stats_flags(self):
        """Filters -o, scheduling stats flags, and adds tuner flags."""
        cmd = (
            "iree-compile input.mlir "
            "--iree-hal-target-backends=rocm "
            "--iree-scheduling-dump-statistics-format=json "
            "--iree-scheduling-dump-statistics-file=stats.json "
            "-o output.vmfb"
        )
        result = build_compile_args(cmd, Path("/tmp/benchmarks"))

        self.assertEqual(result[0], "iree-compile")
        self.assertIn("--iree-hal-target-backends=rocm", result)
        # Original "-o output.vmfb" should be stripped.
        self.assertNotIn("output.vmfb", result)
        self.assertNotIn("--iree-scheduling-dump-statistics-format=json", result)
        # Tuner-specific flags should be appended.
        self.assertIn("--iree-config-add-tuner-attributes", result)
        self.assertIn("--iree-hal-dump-executable-benchmarks-to", result)
        # Output redirected to platform null device.
        idx = result.index("-o")
        self.assertEqual(result[idx + 1], os.devnull)

    def test_preserves_input_mlir(self):
        """Keeps the input MLIR path from the original command."""
        cmd = "iree-compile my_model.mlir --iree-hal-target-backends=rocm -o out.vmfb"
        result = build_compile_args(cmd, Path("/tmp/bench"))
        self.assertIn("my_model.mlir", result)


class TestLoadCommands(unittest.TestCase):
    def test_loads_from_args(self):
        """When no file given, returns fusilli_op_args as single command."""
        result = load_commands(None, ["conv", "-F", "1", "--bf16"])
        self.assertEqual(result, [["conv", "-F", "1", "--bf16"]])

    def test_loads_from_file(self):
        """Reads commands from file, skipping comments and blank lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\n")
            f.write("conv -F 1 --bf16\n")
            f.write("\n")
            f.write("matmul -M 1024 -N 1024 -K 1024\n")
            f.flush()
            tmp_path = f.name

        try:
            result = load_commands(tmp_path, [])
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], ["conv", "-F", "1", "--bf16"])
            self.assertEqual(
                result[1], ["matmul", "-M", "1024", "-N", "1024", "-K", "1024"]
            )
        finally:
            os.unlink(tmp_path)

    def test_prefers_file_when_both_given(self):
        """When a file is given, fusilli_op_args is ignored (gating is in main)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("conv -F 1 --bf16\n")
            f.flush()
            tmp_path = f.name

        try:
            result = load_commands(tmp_path, ["matmul", "-M", "16"])
            self.assertEqual(result, [["conv", "-F", "1", "--bf16"]])
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
