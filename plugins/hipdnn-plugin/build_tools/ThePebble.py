"""
ThePebble - A simulacrum of TheRock for fusilli's hipDNN plugin dev/CI
environment setup.

ThePebble composes build artifacts into one distribution directory at
$HOME/.cache/ThePebble/dist. The dist directory resembles what TheRock provides
when building fusilli's hipDNN plugin - namely a composition of the installed
build artifacts for the plugin's declared dependencies.

TODO: when multi-stage build lands in TheRock investigate simply running the
iree-libs phase of TheRock.

usage:
  --setup
    Installs plugin dependencies and creates a CMakeUserPresets.json for local
    development. After running --setup, configure the plugin with:
      cmake --preset thepebble

    Dependencies installed:
     - Hip. For the hip dependency ThePebble takes an approach suggested in
       TheRock's RELEASES.md and uses TheRock's CI scripts `fetch_artifacts.py`
       and `install_rocm_from_artifacts.py` to install from artifacts built by
       TheRock. We use the more granular artifacts (used primarily by tests)
       rather than the monolithic tarball. The granular artifacts allow us to
       install hip without ending up with duplicate copies of fusilli-plugin -
       ThePebble and TheRock both build one. The granular approach also allows
       us to compose final installed artifacts from multiple builds - we can use
       hip from an older build if there was a regression, and hipDNN from a
       newer build if there was an API update.

       TheRock's scripts require a github run ID. This is configured in
       thepebble_config.toml as `versions.hip_run_id`. To find a run ID go to
       https://github.com/ROCm/TheRock/actions - each action has the run id in
       the URL. A run ID can come from a PR or a nightly release build from
       `main`. Note: Filtering https://github.com/ROCm/TheRock/actions by
       "scheduled" events will display only nightly release builds, if that's
       what you're looking for.

     - HipDNN. Currently ThePebble builds + installs hipDNN from source in
       rocm-libraries. But, the source approach can and should be augmented with
       an alternative option to fetch hipDNN artifacts from TheRock like we do
       for hip (with an independent run id). Currently hipDNN has a CMake bug
       that crashes the build when using TheRock's artifacts; it contains a
       hardcoded path that won't exist if outside of build machine for TheRock.

     - IREE runtime. This is the black sheep. As fusilli and the plugin each build
       IREE from source internally this dependency isn't built and installed to
       dist, it just exists at a path where fusilli + the plugin can find it.
       ThePebble provides IREE's path to fusilli and the plugin through
       -DIREE_SOURCE_DIR CMake Cache variable (via build flags and cmake preset
       respectively).

     - Fusilli. The plugin builds independently from fusilli itself - the plugin
       may not live in the fusilli repo long term, so is designed to be easily
       relocatable. The current version of fusilli is built + installed into dist
       using IREE runtime fetched by ThePebble.

     - hipDNN integration tests. Like hip, fetched using `fetch_artifacts.py`.

       The integration test suite is plugin-agnostic. For a pre-defined set of
       graphs it generates a cartesian grid of graph X engine numerical tests
       based on engines claiming support for each graph.

  --ci-install-and-test-fusilli-plugin
    Builds and installs the fusilli plugin into the dist directory, then runs
    TheRock's test script for fusilli plugin.

    Note: This installs the plugin into the same dist directory as dependencies
    created with --setup, which may be annoying for local development - you'll
    end up with two versions of fusilli_pluggin.so running around (one in dist
    and one in your local build folder). For local development it's probably
    easiest to run the tests from your local build. TheRock doesn't run tests
    using ctest on build folder, but bugs due to test environment setup should
    be rare.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from pathlib import Path

PEBBLE_DIR = Path.home() / ".cache" / "ThePebble"
INSTALL_DIR = PEBBLE_DIR / "dist"
CACHED_CONFIG = PEBBLE_DIR / "_copy_of_thepebble_config_for_cache_invalidation.toml"
THEROCK_REPO = "git@github.com:AaronStGeorge/TheRock.git"
THEROCK_DIR = PEBBLE_DIR / "TheRock"
ROCM_LIBRARIES_REPO = "git@github.com:AaronStGeorge/rocm-libraries.git"
IREE_REPO = "https://github.com/iree-org/iree.git"
IREE_DIR = PEBBLE_DIR / "iree"
IREE_SUBMODULES = ["third_party/flatcc", "third_party/benchmark"]
HIPDNN_SRC_DIR = PEBBLE_DIR / "rocm-libraries"

# ==============================================================================
# Utils
# ==============================================================================


def load_config() -> dict:
    """Load configuration from thepebble_config.toml."""
    config_path = Path(__file__).parent / "thepebble_config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def get_fusilli_dir() -> Path:
    """Get the fusilli source directory (parent of hipdnn-plugin)."""
    # ThePebble.py is in fusilli/plugins/hipdnn-plugin/build_tools/
    return Path(__file__).parent.parent.parent.parent


def get_plugin_dir() -> Path:
    """Get the hipdnn-plugin directory."""
    # ThePebble.py is in fusilli/plugins/hipdnn-plugin/build_tools/
    return Path(__file__).parent.parent


def validate_config():
    """Check cache exists and config matches. Error if mismatch."""
    if not CACHED_CONFIG.exists():
        sys.exit("Error: No cached config. Run --setup first.")

    current_config = load_config()
    with open(CACHED_CONFIG, "rb") as f:
        cached_config = tomllib.load(f)

    if current_config != cached_config:
        sys.exit("Error: Config mismatch. Re-run --setup to update.")


# ==============================================================================
# Setup
# ==============================================================================


def setup_therock(git_ref: str):
    """Clone TheRock and set up venv (ThePebble only uses python scripts)"""
    print(f"Cloning TheRock at {git_ref}...")
    subprocess.run(
        ["git", "clone", THEROCK_REPO, str(THEROCK_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", git_ref],
        cwd=THEROCK_DIR,
        check=True,
    )

    # Set up venv
    print("Setting up TheRock venv...")
    venv_dir = THEROCK_DIR / ".venv"
    subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "-r", str(THEROCK_DIR / "requirements.txt")],
        check=True,
    )
    subprocess.run(
        [str(pip), "install", "-r", str(THEROCK_DIR / "requirements-test.txt")],
        check=True,
    )


def install_hip(run_id: str):
    """Download and install Hip artifacts using install_rocm_from_artifacts.py."""
    venv_python = THEROCK_DIR / ".venv" / "bin" / "python"

    # Use TheRock's install_rocm_from_artifacts.py
    # --run-github-repo is needed to override GITHUB_REPOSITORY env var in CI
    cmd = [
        str(venv_python),
        str(THEROCK_DIR / "build_tools" / "install_rocm_from_artifacts.py"),
        "--run-id",
        run_id,
        "--run-github-repo",
        "ROCm/TheRock",
        "--artifact-group",
        "generic",
        "--output-dir",
        str(INSTALL_DIR),
        "--base-only",
    ]
    print(f"Fetching Hip artifacts from run {run_id}...")
    subprocess.run(cmd, check=True)

    # Fetch amd-llvm_dev (not sure why this isn't included in "base")
    # NOTE: install_rocm_from_artifacts.py wipes --output-dir,
    # fetch_artifacts.py does not, here we want to compose so the latter is
    # preferable. --flatten flag merges with directory structure created in the
    # first call.
    cmd = [
        str(venv_python),
        str(THEROCK_DIR / "build_tools" / "fetch_artifacts.py"),
        "--run-id",
        run_id,
        "--run-github-repo",
        "ROCm/TheRock",
        "--artifact-group",
        "generic",
        "--output-dir",
        str(INSTALL_DIR),
        "--flatten",
        "amd-llvm_dev",
    ]
    print(f"Fetching amd-llvm_dev artifact...")
    subprocess.run(cmd, check=True)


def install_hipdnn_integration_tests(run_id: str):
    """Download and install hipDNN integration test artifacts."""
    venv_python = THEROCK_DIR / ".venv" / "bin" / "python"

    # Use fetch_artifacts.py directly to avoid wiping the directory
    # Use install_rocm_from_artifacts.py see NOTE in `install_hip`.
    cmd = [
        str(venv_python),
        str(THEROCK_DIR / "build_tools" / "fetch_artifacts.py"),
        "--run-id",
        run_id,
        "--run-github-repo",
        "ROCm/TheRock",
        "--artifact-group",
        "generic",
        "--output-dir",
        str(INSTALL_DIR),
        "--flatten",
        "hipdnn-integration-tests_test",
    ]
    print(f"Fetching hipDNN integration test artifacts from run {run_id}...")
    subprocess.run(cmd, check=True)


def build_hipdnn(git_ref: str):
    """Build and install hipDNN from rocm-libraries sparse checkout."""
    # Sparse checkout of rocm-libraries
    print(f"Sparse checkout of rocm-libraries at {git_ref}...")
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            ROCM_LIBRARIES_REPO,
            str(HIPDNN_SRC_DIR),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "init", "--cone"],
        cwd=HIPDNN_SRC_DIR,
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "projects/hipdnn"],
        cwd=HIPDNN_SRC_DIR,
        check=True,
    )
    subprocess.run(["git", "checkout", git_ref], cwd=HIPDNN_SRC_DIR, check=True)

    # Build inside projects/hipdnn so IDEs auto-discover compile_commands.json
    hipdnn_project_dir = HIPDNN_SRC_DIR / "projects" / "hipdnn"
    hipdnn_build_dir = hipdnn_project_dir / "build"
    print(f"Building hipDNN from {hipdnn_project_dir}...")

    cmake_args = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(hipdnn_project_dir),
        "-B",
        str(hipdnn_build_dir),
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
        f"-DCMAKE_PREFIX_PATH={INSTALL_DIR}",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DHIP_PLATFORM=amd",
        "-DHIP_DNN_BUILD_PLUGINS=OFF",
        # Headers are already checked into git, no need to re-generate them
        # unless you're changing the schema.
        "-DHIP_DNN_GENERATE_SDK_HEADERS=OFF",
        "-DENABLE_CLANG_TIDY=OFF",
        "-DENABLE_CLANG_FORMAT=OFF",
    ]
    subprocess.run(cmake_args, check=True)

    # Build and install
    subprocess.run(["cmake", "--build", str(hipdnn_build_dir)], check=True)
    subprocess.run(["cmake", "--install", str(hipdnn_build_dir)], check=True)


def setup_iree(tag: str):
    """Clone IREE at a tag and fetch required submodules"""
    print(f"Cloning IREE at tag {tag}...")
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch", tag, IREE_REPO, str(IREE_DIR)],
        check=True,
    )

    # Fetch only required submodules
    print(f"Fetching IREE submodules: {IREE_SUBMODULES}")
    for submodule in IREE_SUBMODULES:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--depth=1", submodule],
            cwd=IREE_DIR,
            check=True,
        )


def build_fusilli():
    """Build and install fusilli from source."""
    fusilli_src = get_fusilli_dir()

    with tempfile.TemporaryDirectory() as tmpdir:
        fusilli_build = Path(tmpdir)
        print(f"Building fusilli from {fusilli_src}...")

        # Configure fusilli - based on TheRock's CMake args
        cmake_args = [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(fusilli_src),
            "-B",
            str(fusilli_build),
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
            f"-DCMAKE_PREFIX_PATH={INSTALL_DIR}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFUSILLI_BUILD_TESTS=OFF",
            "-DFUSILLI_BUILD_BENCHMARKS=OFF",
            "-DFUSILLI_SYSTEMS_AMDGPU=ON",
            "-DFUSILLI_CODE_COVERAGE=OFF",
            "-DFUSILLI_ENABLE_LOGGING=OFF",
            "-DFUSILLI_ENABLE_CLANG_TIDY=OFF",
            f"-DIREE_SOURCE_DIR={IREE_DIR}",
            "-DHIP_PLATFORM=amd",
            "-DIREE_USE_SYSTEM_DEPS=ON",
        ]
        subprocess.run(cmake_args, check=True)

        # Build and install
        subprocess.run(["cmake", "--build", str(fusilli_build)], check=True)
        subprocess.run(["cmake", "--install", str(fusilli_build)], check=True)


def generate_cmake_user_presets():
    """Generate CMakeUserPresets.json in the hipdnn-plugin directory."""
    plugin_dir = get_plugin_dir()
    llvm_bin = INSTALL_DIR / "lib" / "llvm" / "bin"

    presets = {
        "version": 6,
        "configurePresets": [
            {
                "name": "thepebble",
                "generator": "Ninja",
                "binaryDir": "${sourceDir}/build",
                "cacheVariables": {
                    "CMAKE_C_COMPILER": str(llvm_bin / "clang"),
                    "CMAKE_CXX_COMPILER": str(llvm_bin / "clang++"),
                    "CMAKE_PREFIX_PATH": str(INSTALL_DIR),
                    "IREE_SOURCE_DIR": str(IREE_DIR),
                    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
                    "IREE_USE_SYSTEM_DEPS": "ON",
                    "HIP_PLATFORM": "amd",
                },
            }
        ],
    }

    presets_path = plugin_dir / "CMakeUserPresets.json"
    print(f"Writing {presets_path}...")
    with open(presets_path, "w") as f:
        json.dump(presets, f, indent=2)
        f.write("\n")


def generate_local_environment_setup(iree_git_tag: str):
    """Installs a local copy of the correct `iree-compile` binary and generates
    an "activate" script to setup local machine with correct $PATH etc. to use
    ThePebble installed programs."""
    # Create venv with ThePebble prompt
    venv_dir = PEBBLE_DIR / ".venv"
    print(f"Creating venv at {venv_dir}...")
    venv.EnvBuilder(with_pip=True, prompt="ThePebble").create(venv_dir)

    # Install iree-base-compiler
    pip_version = iree_git_tag.replace("iree-", "")
    pip = venv_dir / "bin" / "pip"
    print(f"Installing iree-base-compiler=={pip_version}...")
    subprocess.run(
        [
            str(pip),
            "install",
            "--find-links",
            "https://iree.dev/pip-release-links.html",
            f"iree-base-compiler=={pip_version}",
        ],
        check=True,
    )

    # Generate activate script
    bin_dir = INSTALL_DIR / "bin"
    lib_dir = INSTALL_DIR / "lib"
    venv_activate = venv_dir / "bin" / "activate"

    script_content = f"""#!/bin/bash
# ThePebble environment activation script
# Usage: source {PEBBLE_DIR}/activate

if [[ "${{BASH_SOURCE[0]}}" == "${{0}}" ]]; then
    echo "Error: This script must be sourced, not executed."
    echo "Usage: source {PEBBLE_DIR}/activate"
    exit 1
fi

source {venv_activate}

export PATH="{bin_dir}:$PATH"
export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"

echo "ThePebble environment activated."
"""

    activate_path = PEBBLE_DIR / "activate"
    print(f"Writing {activate_path}...")
    with open(activate_path, "w") as f:
        f.write(script_content)


# ==============================================================================
# CI install and test fusilli-plugin
# ==============================================================================


def build_fusilli_plugin():
    """Build and install fusilli plugin to dist."""
    plugin_src = get_plugin_dir()
    llvm_bin = INSTALL_DIR / "lib" / "llvm" / "bin"

    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_build = Path(tmpdir)
        print(f"Building fusilli plugin from {plugin_src}...")

        cmake_args = [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(plugin_src),
            "-B",
            str(plugin_build),
            f"-DCMAKE_C_COMPILER={llvm_bin / 'clang'}",
            f"-DCMAKE_CXX_COMPILER={llvm_bin / 'clang++'}",
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
            f"-DCMAKE_PREFIX_PATH={INSTALL_DIR}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DIREE_SOURCE_DIR={IREE_DIR}",
            "-DIREE_USE_SYSTEM_DEPS=ON",
            "-DHIP_PLATFORM=amd",
        ]
        subprocess.run(cmake_args, check=True)
        subprocess.run(["cmake", "--build", str(plugin_build)], check=True)
        subprocess.run(["cmake", "--install", str(plugin_build)], check=True)


def test_fusilli_plugin():
    """Run test_fusilli_plugin.py from TheRock."""
    config = load_config()

    # The test script expects THEROCK_BIN_DIR to point to the bin/ directory
    bin_dir = INSTALL_DIR / "bin"

    # Create iree_tag_for_pip.txt.
    # TheRock/iree-libs/post_hook_fusilli-plugin.cmake would create this file
    # when building in TheRock.
    iree_tag = config["versions"]["iree_git_tag"]
    # Convert tag like "iree-3.10.0rc20251210" to pip version "3.10.0rc20251210"
    pip_version = iree_tag.replace("iree-", "")
    iree_tag_file = bin_dir / "fusilli_plugin_test_infra" / "iree_tag_for_pip.txt"
    iree_tag_file.write_text(pip_version)
    print(f"Created {iree_tag_file} with version {pip_version}")

    # Run TheRock's test_fusilli_plugin.py
    therock_dir = PEBBLE_DIR / "TheRock"
    test_script = (
        therock_dir
        / "build_tools"
        / "github_actions"
        / "test_executable_scripts"
        / "test_fusilli_plugin.py"
    )

    env = os.environ.copy()
    env["THEROCK_BIN_DIR"] = str(bin_dir)

    # iree-libs/post_hook_fusilli-plugin.cmake sets up RPATHs so that a .so in
    # "lib/hipdnn_plugins/engines" will be found by tests that
    # fusilli_plugin.so, and so that
    # "lib/hipdnn_plugins/engines/fusilli_plugin.so" can find hip .so's in lib.
    # In ThePebble we just use LD_LIBRARY_PATH.
    lib_dir = INSTALL_DIR / "lib"
    plugin_lib_dir = lib_dir / "hipdnn_plugins" / "engines"
    ld_path = f"{lib_dir}:{plugin_lib_dir}"
    if "LD_LIBRARY_PATH" in env:
        ld_path = f"{ld_path}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = ld_path

    print(f"Running {test_script}...")
    subprocess.run(["python3", str(test_script)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="ThePebble a simulacrum of TheRock for fusilli plugin dev environment setup"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup deps as TheRock would, and crate CMake preset for local dev",
    )
    parser.add_argument(
        "--ci-install-and-test-fusilli-plugin",
        action="store_true",
        help="Build + install + test the plugin using TheRock's test script",
    )
    args = parser.parse_args()

    if not args.setup and not args.ci_install_and_test_fusilli_plugin:
        parser.print_help()
        sys.exit(1)

    if args.setup:
        config = load_config()
        versions = config["versions"]

        # Start fresh
        if PEBBLE_DIR.exists():
            print(f"Removing previous setup {PEBBLE_DIR}...")
            shutil.rmtree(PEBBLE_DIR)

        # Run setup
        PEBBLE_DIR.mkdir(parents=True, exist_ok=True)
        setup_therock(versions["therock_git_ref"])
        install_hip(versions["hip_run_id"])
        install_hipdnn_integration_tests(versions["hipdnn_integration_tests_run_id"])
        build_hipdnn(versions["hipdnn_git_ref"])
        setup_iree(versions["iree_git_tag"])
        build_fusilli()
        generate_cmake_user_presets()
        generate_local_environment_setup(versions["iree_git_tag"])

        # Copy config to cache for validation checks
        config_src = Path(__file__).parent / "thepebble_config.toml"
        shutil.copy(config_src, CACHED_CONFIG)

        print(f"\nSetup complete.")
        print(f"To activate the ThePebble local dev environment, run:")
        print(f"  source {PEBBLE_DIR}/activate")

    if args.ci_install_and_test_fusilli_plugin:
        validate_config()
        build_fusilli_plugin()
        test_fusilli_plugin()


if __name__ == "__main__":
    main()
