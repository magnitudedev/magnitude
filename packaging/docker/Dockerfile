# syntax=docker/dockerfile:1.7
#
# Magnitude's official image. Build one platform at a time, natively, from the context that
# packages/release/scripts/docker/prepare-context.ts writes:
#
#   magnitude-cli-only.tar.gz     the release's CLI-only payload for this architecture
#   inference/                    the release's inference engine, with installation.json
#   update-configuration.json     optional; QA only, replaces the payload's update server
#   magnitude-release.json        the verified release manifest; provenance, not installed
#
# The payload is unpacked through a bind mount, so the archive never becomes an image layer. It is
# installed the way the CLI-only installer lays it out: /opt/magnitude/cli/<version>/ with a
# `current` link, and /usr/local/bin/magnitude pointing into it.
FROM ubuntu:24.04

ARG TARGETARCH
ARG VERSION
ARG REVISION

LABEL org.opencontainers.image.title="Magnitude" \
      org.opencontainers.image.description="Run open models as fast as your hardware allows, with OpenAI and Anthropic compatible APIs." \
      org.opencontainers.image.url="https://magnitude.dev" \
      org.opencontainers.image.documentation="https://docs.magnitude.dev/installation/docker" \
      org.opencontainers.image.source="https://github.com/magnitudedev/magnitude" \
      org.opencontainers.image.vendor="Magnitude" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# ca-certificates: the engine cannot make HTTPS requests without it. libvulkan1 and Mesa give
# AMD and Intel GPUs Vulkan through /dev/dri; the NVIDIA toolkit mounts its own driver.
RUN --mount=type=bind,target=/context <<'EOF'
set -eu
test "$(dpkg --print-architecture)" = "${TARGETARCH}"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
  ca-certificates curl tini libvulkan1 mesa-vulkan-drivers
rm -rf /var/lib/apt/lists/*
cli=/opt/magnitude/cli
mkdir -p "${cli}/.staging"
tar --extract --gzip --no-same-owner --file /context/magnitude-cli-only.tar.gz --directory "${cli}/.staging"
# Running it also proves the payload matches this architecture.
version="$("${cli}/.staging/magnitude" --version)"
if [ -n "${VERSION:-}" ] && [ "${VERSION}" != qa ]; then test "${version}" = "${VERSION}"; fi
mv "${cli}/.staging" "${cli}/${version}"
ln -s "${version}" "${cli}/current"
ln -s "${cli}/current/magnitude" /usr/local/bin/magnitude
printf '{"format":"docker"}' > "${cli}/${version}/update-package.json"
if [ -f /context/update-configuration.json ]; then
  install -m 0644 /context/update-configuration.json "${cli}/${version}/update-configuration.json"
fi
chmod -R u=rwX,go=rX "${cli}"
# ubuntu:24.04 ships an `ubuntu` account with uid 1000.
if id -u ubuntu >/dev/null 2>&1; then userdel --remove ubuntu || true; fi
if getent group ubuntu >/dev/null; then groupdel ubuntu; fi
# The NVIDIA toolkit's Vulkan manifest names libGLX_nvidia, which can't create an instance without
# an X server, so a container only sees llvmpipe. Its libEGL_nvidia works headless; elsewhere the
# loader skips this manifest because the library isn't mounted.
mkdir -p /usr/share/vulkan/icd.d
printf '%s\n' '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3"}}' \
  > /usr/share/vulkan/icd.d/nvidia_egl_icd.json
useradd --uid 1000 --user-group --home-dir /data --create-home --shell /bin/bash magnitude
# The service never raises privileges, so no binary may: the base image's setuid root helpers
# (su, mount, passwd and the like) are of no use to a server.
find / -xdev -perm /6000 -type f -exec chmod a-s {} +
EOF

COPY --chmod=755 inference/ /opt/magnitude/inference/

# MAGNITUDE_DESKTOP_STATE_DIR keeps the control socket off /data, where a Docker Desktop bind
# mount refuses its chmod; under /tmp, whichever uid runs the container (`--user`) creates it.
# `graphics` lets the NVIDIA toolkit mount its Vulkan driver.
ENV HOME=/data \
    MAGNITUDE_ICN_PATH=/opt/magnitude/inference/installation.json \
    MAGNITUDE_DESKTOP_STATE_DIR=/tmp/magnitude \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

USER magnitude
WORKDIR /data
VOLUME /data
EXPOSE 10100

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
  CMD ["curl", "-fsS", "-o", "/dev/null", "http://127.0.0.1:10100/health"]

ENTRYPOINT ["tini", "--", "magnitude", "serve"]
