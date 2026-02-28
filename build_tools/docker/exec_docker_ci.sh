#!/usr/bin/env bash
set -euo pipefail

# Read versions from version.json (single source of truth)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
read -r IREE_GIT_TAG THEROCK_GIT_TAG < <(python3 -c "
import json; d=json.load(open('${REPO_ROOT}/version.json'))
print(d['iree-version'], d['therock-version'])")

# ROCm requires accesses to the host's /dev/kfd and /dev/dri/* device nodes, typically
# owned by the `render` and `video` groups. The groups' GIDs in the container must
# match the host's to access the resources. Sometimes the device nodes may be owned by
# dynamic GIDs (that don't belong to the `render` or `video` groups). So instead of
# adding user to the GIDs of named groups (obtained from `getent group render` or
# `getent group video`), we simply check the owning GID of the device nodes on the host
# and pass it to `docker run` with `--group-add=<GID>`.
DOCKER_RUN_DEVICE_OPTS=""
for DEV in /dev/kfd /dev/dri/*; do
  # Skip if not a character device
  # /dev/dri/by-path/ symlinks are ignored
  [[ -c "${DEV}" ]] || continue
  DOCKER_RUN_DEVICE_OPTS+=" --device=${DEV} --group-add=$(stat -c '%g' ${DEV})"
done

# Bind mounts for the following:
# - current directory to /workspace in the container
docker run --rm \
           -v "${PWD}":/workspace \
           -e IREE_GIT_TAG="${IREE_GIT_TAG}" \
           -e THEROCK_GIT_TAG="${THEROCK_GIT_TAG}" \
           ${DOCKER_RUN_DEVICE_OPTS} \
           --security-opt seccomp=unconfined \
           ghcr.io/sjain-stanford/compiler-dev-ubuntu-24.04:main@sha256:0ae9798cc883c4c2b687d43be5612f9e1333d3cd9af20d47ec20efe8f7921c9b \
           "$@"
