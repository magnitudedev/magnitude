#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="magnitude-tbench-build"
OUTPUT_DIR="${ROOT_DIR}/evals/tbench/bin"
CONTAINER_NAME="magnitude-tbench-build-extract"

docker build -f "${ROOT_DIR}/evals/tbench/Dockerfile.build" -t "${IMAGE_TAG}" "${ROOT_DIR}"

mkdir -p "${OUTPUT_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker create --name "${CONTAINER_NAME}" "${IMAGE_TAG}" >/dev/null
docker cp "${CONTAINER_NAME}:/repo/bin/magnitude" "${OUTPUT_DIR}/magnitude"
docker rm -f "${CONTAINER_NAME}" >/dev/null

chmod +x "${OUTPUT_DIR}/magnitude"

echo "Built Linux binary at ${OUTPUT_DIR}/magnitude"