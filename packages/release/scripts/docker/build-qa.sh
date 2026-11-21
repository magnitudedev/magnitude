#!/usr/bin/env bash
# Builds a QA image on a Linux build host from a locally built CLI-only payload
# (magnitude-cli-only-<host>.tar.gz) and the published engine of a base release, for this host's
# architecture, and prints its digest.
#
#   build-qa.sh --payload PATH --engine-version VERSION --tag IMAGE[:TAG] \
#     [--update-configuration PATH] [--push] [--smoke]
#
# Without --push the image is loaded into the local Docker engine; with --push it goes to the
# tag's registry (log in first), and the printed digest is the registry's. --smoke runs
# acceptance/test-docker.sh against the built image. Needs bun (with `bun install` done in the
# repository), docker with buildx, and git.
set -euo pipefail
usage() { echo 'Usage: build-qa.sh --payload PATH --engine-version VERSION --tag IMAGE[:TAG] [--update-configuration PATH] [--push] [--smoke]' >&2; exit 2; }
payload='' engine_version='' tag='' update_configuration='' push=0 smoke=0
while test "$#" -gt 0; do
  case "$1" in
    --payload) payload=$(realpath "$2"); shift 2 ;;
    --engine-version) engine_version=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --update-configuration) update_configuration=$(realpath "$2"); shift 2 ;;
    --push) push=1; shift ;;
    --smoke) smoke=1; shift ;;
    *) usage ;;
  esac
done
test -n "$payload" && test -n "$engine_version" && test -n "$tag" || usage

case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
repository=$(cd "$(dirname "$0")/../../../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-docker-qa.XXXXXXXX")
trap 'rm -rf "$work"' EXIT

prepare=(--version "$engine_version" --arch "$arch" --output "$work/context" --payload "$payload")
test -z "$update_configuration" || prepare+=(--update-configuration "$update_configuration")
"${BUN:-bun}" "$repository/packages/release/scripts/docker/prepare-context.ts" "${prepare[@]}"

# The payload's CLI runs on this host, which has the image's architecture.
mkdir "$work/payload"
tar -xzf "$payload" -C "$work/payload" magnitude
version=$("$work/payload/magnitude" --version 2>/dev/null || true)
build=(docker buildx build
  --platform "linux/$arch"
  --file "$repository/packaging/docker/Dockerfile"
  --build-arg "VERSION=${version:-qa}"
  --build-arg "REVISION=$(git -C "$repository" rev-parse HEAD)"
  --tag "$tag"
  --metadata-file "$work/metadata.json"
  --provenance=false --sbom=false)
# Docker media types, as the release publishes: Podman reads HEALTHCHECK only from a Docker image config.
if test "$push" = 1; then build+=(--output "type=image,name=$tag,push=true,oci-mediatypes=false"); else build+=(--load); fi
"${build[@]}" "$work/context"

digest=$(sed -n 's/.*"containerimage.digest": *"\(sha256:[0-9a-f]*\)".*/\1/p' "$work/metadata.json" | head -n 1)
if test "$smoke" = 1; then
  bash "$repository/packages/release/scripts/acceptance/test-docker.sh" "$tag" ${version:+"$version"}
fi
echo "Built $tag for linux/$arch"
echo "Digest: ${digest:-unknown}"
