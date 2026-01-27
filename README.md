# Fusilli

[![CI - fusilli](https://github.com/iree-org/fusilli/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/iree-org/fusilli/actions/workflows/build-and-test.yml)

Fusilli is a C++ Graph API and JIT Frontend for IREE that leverages just-in-time compiled and code-generated kernels to accelerate training and inference workloads. Inspired by cuDNN's graph API, it exposes cuDNN-like primitives but is backed by the power of the IREE compiler and runtime stack.

We believe hand-authored GPU kernel libraries are great for highly tuned performance but they are difficult to scale to different models or target architectures and painful to package and release efficiently. This project is founded on the overarching goal to complement the ecosystem of ML frameworks and libraries with a JIT solution, while being competitive to hand-authored kernel libraries. Apart from the core benefit of having a compiler-backed JIT engine that gets progressively and pervasively better, a systemic benefit of this is it helps reduce build times and binary sizes, making it easier to ship software effectively.

> [!WARNING]
> :construction: Fusilli is in early stages of development. The operator coverage is limited but growing. APIs may change. :construction:

> [!NOTE]
> The name 'Fusilli' is inspired by the term 'fusion' - a bread-and-butter compiler optimization for improving performance.

![Fusilli](docs/fusilli.png)

## Developer Guide

### Setup

Although optional, we recommend docker as the canonical development setup for a no-fuss quick start, hermetic and reproducible builds, and consistency with CI. Follow [these steps](https://github.com/sjain-stanford/docker.git) to launch an interactive docker container with the required dependencies pre-installed (and skip to the `Build and Test` section below).

If you prefer a custom setup instead, the following dependencies need to be brought in to build/test Fusilli:

**Build Requirements:** cmake, ninja-build, clang, IREE

**Test Requirements:** catch2, lit, FileCheck, iree-opt, iree-compile

Fusilli interfaces with the IREE compiler through the CLI and C-API and with IREE runtime through its C-API. Selection between the C-API and CLI for the compiler can be controlled via an environment variable. The IREE compiler is a heavy dependency to build (due to MLIR/LLVM), so we recommend using a prebuilt release either from a python nightly package or shared library distribution. The IREE runtime on the other hand is much more lightweight and is designed to be built from source and statically linked in. IREE does not export a shared runtime library to allow for maximum flexibility with low-level and toolchain specific (LTO style) optimizations.

Easiest way to get [`lit`](https://llvm.org/docs/CommandGuide/lit.html), and the `iree-*` CLI tools is through `pip install`. [`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html) comes packaged with clang / llvm distributions. Everything else should be available via `apt` based install.

### Build and Test

Build and test Fusilli as follows:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=<Debug|Release|RelWithDebInfo> \
    -DIREE_SOURCE_DIR=</path/to/iree/source>
cmake --build build --target all
ctest --test-dir build
```

When building on an AMD GPU system, specify `-DFUSILLI_SYSTEMS_AMDGPU=ON` to enable the AMDGPU build.

To re-run failed tests verbosely:
```shell
ctest --test-dir build --rerun-failed --output-on-failure --verbose
```

To run tests in parallel (concurrently):
```shell
ctest --test-dir build --output-on-failure -j $(nproc)
```

Tests and samples are also built as standalone binary targets (in the `build/bin` directory) to make debugging isolated failures easier.

To skip building tests and samples, specify the cmake flag `-DFUSILLI_BUILD_TESTS=OFF`.

### Benchmarks

The benchmark driver is a command line tool that takes a set of args and sub-command args to run operation specific benchmarks:
```shell
build/bin/benchmarks/fusilli_benchmark_driver <ARGS> <SUB-COMMAND> <SUB-ARGS>
```

To dump compilation artifacts to disk (`${HOME}/.cache/fusilli` by default), specify the `--dump` flag on the main driver (not the subcommand). The location to dump to can be configured by setting the `FUSILLI_CACHE_DIR` environment variable.
```shell
build/bin/benchmarks/fusilli_benchmark_driver --dump <ARGS> <SUB-COMMAND> <SUB-ARGS>
```

To benchmark on a specific GPU when multiple AMD GPUs are present, specify `--device <int>` flag corresponding to the device number from `rocm-smi`. For example, this will run the benchmark on device 7 (when there are 8 GPUs):
```shell
build/bin/benchmarks/fusilli_benchmark_driver --device 7 <ARGS> <SUB-COMMAND> <SUB-ARGS>
```

An invalid device number should result in a runtime error like so:
```
RUNTIME_FAILURE: iree/runtime/src/iree/hal/drivers/hip/hip_device.c:499: FAILED_PRECONDITION; HIP driver error 'hipErrorInvalidDevice' (101): invalid device ordinal
```

The easiest way to benchmark on AMD GPU systems is using the `rocprofv3` tool (included in the docker image). Here's a sample command to dump a `*.pftrace` file that may be opened using [Perfetto](https://ui.perfetto.dev/) for further analysis.
```shell
rocprofv3 --output-format pftrace -r -- build/bin/benchmarks/fusilli_benchmark_driver --iter 10 conv -F 1 --bf16 -n 16 -c 288 --in_d 2 -H 48 -W 32 -k 288 --fil_d 2 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --in_layout "NDHWC" --out_layout "NDHWC" --fil_layout "NDHWC" --spatial_dim 3
```

To save the benchmark results as csv, specify `--output-format csv` instead.

To skip building benchmarks, specify the cmake flag `-DFUSILLI_BUILD_BENCHMARKS=OFF`.

### Code Coverage (using gcov + lcov)

This works with gcc builds (code coverage with clang instrumentation is future work).

To generate code coverage metrics:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DFUSILLI_CODE_COVERAGE=ON \
    -DIREE_SOURCE_DIR=</path/to/iree/source>
cmake --build build --target all
ctest --test-dir build -T test -T coverage
```

This generates the `*.gcda` and `*.gcno` files with coverage info. At this point one may use an IDE to visualize the coverage info inlayed with the source code. If using VSCode's gcov-viewer extension: Hit `Cmd+Shift+P` -> Gcov Viewer: Reload (Import gcda files) to load coverage info and `Cmd+Shift+P` -> Gcov Viewer: Reset (Delete gcda files) to reset it.

To generate an HTML (interactive) coverage report:
```shell
lcov --capture --directory build --output-file build/coverage.info
# Exclude external sources from being reported in code coverage
# For example:
#   /usr/include/c++/13/*
#   /usr/include/x86_64-linux-gnu/c++/*
#   /usr/local/include/catch2/*
lcov --remove build/coverage.info '/usr/*' '*/iree/*' --output-file build/coverage.info
genhtml build/coverage.info --output-directory coverage_report
```

### Lint

This project is set up to use [pre-commit](https://pre-commit.com/) hooks for lint checks (such as clang-format for C++ and black for python sources). To install it in your local clone, run `pre-commit install`. After this, hooks will automatically run when making commits locally.

To manually run pre-commit on all files:
```shell
pre-commit run --all-files
```

To run clang-format standalone:
```shell
find . -path ./build -prune -o \( -type f \( -name "*.cpp" -o -name "*.h" \) -print \) | xargs clang-format -i
```

We also use clang-tidy for static analysis. To run clang-tidy during compilation, specify the cmake flag `-DFUSILLI_ENABLE_CLANG_TIDY=ON` when building Fusilli.

### Logging

Fusilli records execution flow through the logging interface. This is disabled by default but can be enabled for debugging.

To configure logging behavior using environment variables:

|   Set output stream \ Enable logging            | `FUSILLI_LOG_INFO` = 0 | `FUSILLI_LOG_INFO` = 1
| ----------------------------------------------- | ---------------------- | ----------------------
| `FUSILLI_LOG_FILE` not set                      | no logging             | no logging
| `FUSILLI_LOG_FILE` set to `stdout` or `stderr`  | no logging             | logging to cout / cerr
| `FUSILLI_LOG_FILE` set to `/path/to/file.txt`   | no logging             | logging to file.txt

Tests and samples that are built with the cmake flag `-DFUSILLI_ENABLE_LOGGING=ON` have their environment variables automatically configured for logging to cout.

Alternatively, one may call the logging API directly as needed:
- Calling `fusilli::isLoggingEnabled() = <true|false>` has the same effect as setting `FUSILLI_LOG_INFO = 1|0`.
- Calling `fusilli::getStream() = <stream_name>` has the same effect as setting the output stream using `FUSILLI_LOG_FILE`.


### Environment Variables

| Environment Variable                     | Description
| ---------------------------------------- | -----------
| `FUSILLI_COMPILE_BACKEND_USE_CLI`        | Enables the use of the CLI tool to invoke compilation, otherwise uses CAPI
| `FUSILLI_EXTERNAL_IREE_COMPILE`          | Path to `iree-compile` binary
| `FUSILLI_EXTERNAL_IREE_COMPILER_LIB`     | Path to the IREE compiler dynamic library
| `FUSILLI_EXTERNAL_ROCM_AGENT_ENUMERATOR` | Path to `rocm_agent_enumerator` binary
| `FUSILLI_EXTERNAL_AMD_SMI`               | Path to `amd-smi` binary (used for GPU SKU detection)
