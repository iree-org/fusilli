#!/usr/bin/env python3
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discovers latest IREE and TheRock nightly versions and updates version.json.

Designed to run inside a GitHub Actions workflow but also works locally when
``gh`` is available and authenticated.

Exit codes:
    0 - success (version.json may or may not have been updated)
    1 - error (version discovery failed, wheel not available, etc.)

The workflow determines whether a PR is needed by comparing the CURRENT_*
and LATEST_* environment variables this script writes to GITHUB_ENV.

Environment variable outputs (via GITHUB_ENV when running in CI):
    CURRENT_IREE_VERSION
    CURRENT_THEROCK_VERSION
    LATEST_IREE_VERSION
    LATEST_THEROCK_VERSION
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERSION_JSON_PATH = REPO_ROOT / "version.json"
IREE_PIP_RELEASE_LINKS_URL = "https://iree.dev/pip-release-links.html"
THEROCK_CDN_BASE_URL = "https://rocm.nightlies.amd.com/tarball"


def read_version_json() -> dict:
    """Read and parse version.json."""
    try:
        return json.loads(VERSION_JSON_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to read {VERSION_JSON_PATH}: {e}", file=sys.stderr)
        sys.exit(1)


def write_version_json(data: dict) -> None:
    """Write data back to version.json with consistent formatting."""
    VERSION_JSON_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _version_sort_key(version_str: str):
    """Parse version string like '3.11.0rc20260301' into a sortable tuple."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)rc(\d+)", version_str)
    if not m:
        return (0, 0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def verify_iree_wheel(version: str) -> bool:
    """Verify that the iree-base-compiler wheel is available for the given version.

    The pip release links page uses underscores in package names
    (``iree_base_compiler-VERSION``).
    """
    print(f"Verifying iree-base-compiler wheel availability for {version}...")
    try:
        with urllib.request.urlopen(IREE_PIP_RELEASE_LINKS_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"WARNING: Failed to fetch pip release links: {e}", file=sys.stderr)
        return False

    needle = f"iree_base_compiler-{version}"
    found = needle in content
    if found:
        print(f"  OK: {needle} is available")
    else:
        print(f"  MISSING: {needle} is NOT available")
    return found


def find_latest_iree_with_wheel() -> str:
    """Find the latest IREE version that has a published pip wheel.

    Uses the GitHub git tags API (NOT releases API, which misses rc tags) to
    discover all IREE rc tags, then verifies wheel availability on the pip
    release links page. Falls back to the second-latest tag if the latest
    wheel is not yet available.
    """
    print("Querying IREE git tags via GitHub API...")
    result = subprocess.run(
        [
            "gh",
            "api",
            "repos/iree-org/iree/git/refs/tags",
            "--jq",
            ".[].ref",
            "--paginate",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: gh api failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    rc_versions = []
    for line in result.stdout.strip().splitlines():
        m = re.match(r"refs/tags/iree-(\d+\.\d+\.\d+rc\d+)$", line)
        if m:
            rc_versions.append(m.group(1))

    if not rc_versions:
        print("ERROR: No IREE rc tags found", file=sys.stderr)
        sys.exit(1)

    rc_versions.sort(key=_version_sort_key)
    print(f"Found {len(rc_versions)} rc tags, latest: {rc_versions[-1]}")

    # Try latest first, then fall back to second-latest.
    candidates = rc_versions[-2:][::-1]
    for candidate in candidates:
        if verify_iree_wheel(candidate):
            return candidate
        print(f"  Wheel not available for {candidate}, trying fallback...")

    print("ERROR: No IREE version with available wheel found", file=sys.stderr)
    sys.exit(1)


def find_latest_therock_version(current_version: str) -> str:
    """Find the latest TheRock nightly version by probing the CDN.

    Extracts the version prefix from the current version (e.g., ``7.12.0a``
    from ``7.12.0a20260228``) and checks the three most recent dates.

    Args:
        current_version: Current TheRock version string from version.json.

    Returns:
        Latest available TheRock version string, or the current version if
        nothing newer is found.
    """
    m = re.match(r"(.+?)(\d{8})$", current_version)
    if not m:
        print(
            f"WARNING: Cannot parse TheRock version '{current_version}', "
            "keeping current version",
            file=sys.stderr,
        )
        return current_version

    prefix = m.group(1)
    print(f"TheRock version prefix: {prefix}")

    today = datetime.now(timezone.utc)
    for days_ago in range(3):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime("%Y%m%d")
        candidate = f"{prefix}{date_str}"
        url = (
            f"{THEROCK_CDN_BASE_URL}/"
            f"therock-dist-linux-gfx94X-dcgpu-{candidate}.tar.gz"
        )
        print(f"Checking TheRock CDN: {candidate}...")
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    print(f"  FOUND: {candidate}")
                    return candidate
        except (urllib.error.URLError, urllib.error.HTTPError):
            continue

    print(f"WARNING: No newer TheRock version found, keeping {current_version}")
    return current_version


def set_github_env(key: str, value: str) -> None:
    """Append an environment variable to GITHUB_ENV if running in CI."""
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"  {key}={value}")


def main() -> int:
    print("=" * 72)
    print("Fusilli Dependency Bump")
    print("=" * 72)

    # 1. Read current versions.
    data = read_version_json()
    current_iree = data["iree-version"]
    current_therock = data["therock-version"]
    print(f"\nCurrent versions:")
    print(f"  IREE:    {current_iree}")
    print(f"  TheRock: {current_therock}")

    # 2. Discover latest versions.
    print(f"\n{'─' * 72}")
    latest_iree = find_latest_iree_with_wheel()

    print(f"\n{'─' * 72}")
    latest_therock = find_latest_therock_version(current_therock)

    # 3. Output to GITHUB_ENV.
    print(f"\n{'─' * 72}")
    print("Setting environment variables:")
    set_github_env("CURRENT_IREE_VERSION", current_iree)
    set_github_env("CURRENT_THEROCK_VERSION", current_therock)
    set_github_env("LATEST_IREE_VERSION", latest_iree)
    set_github_env("LATEST_THEROCK_VERSION", latest_therock)

    # 4. Compare and update.
    iree_changed = current_iree != latest_iree
    therock_changed = current_therock != latest_therock

    if not iree_changed and not therock_changed:
        print(f"\n{'─' * 72}")
        print("Already up-to-date, no changes needed.")
        return 0

    print(f"\n{'─' * 72}")
    print("Updating version.json:")
    if iree_changed:
        print(f"  IREE:    {current_iree} -> {latest_iree}")
        data["iree-version"] = latest_iree
    if therock_changed:
        print(f"  TheRock: {current_therock} -> {latest_therock}")
        data["therock-version"] = latest_therock

    write_version_json(data)
    print("version.json updated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
