# syntax=docker/dockerfile:1.7
#
# Magnitude's official image. Build one platform at a time, natively, from the context that
# packages/release/scripts/docker/prepare-context.ts writes:
#
#   magnitude-desktop.deb         the release's Linux package for this architecture
#   inference/                    the release's inference engine, with installation.json
#   update-configuration.json     optional; QA only, replaces the package's update server
#   magnitude-release.json        the verified release manifest; provenance, not installed
#
# The package is installed through a bind mount, so it never becomes an image layer.
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
test "$(dpkg-deb --field /context/magnitude-desktop.deb Architecture)" = "${TARGETARCH}"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
  ca-certificates curl tini libvulkan1 mesa-vulkan-drivers /context/magnitude-desktop.deb
rm -rf /var/lib/apt/lists/*
resources=/usr/lib/magnitude-desktop/resources
printf '{"format":"docker"}' > "${resources}/update-package.json"
if [ -f /context/update-configuration.json ]; then
  install -m 0644 /context/update-configuration.json "${resources}/update-configuration.json"
fi
# ubuntu:24.04 ships an `ubuntu` account with uid 1000.
if id -u ubuntu >/dev/null 2>&1; then userdel --remove ubuntu || true; fi
if getent group ubuntu >/dev/null; then groupdel ubuntu; fi
useradd --uid 1000 --user-group --home-dir /data --create-home --shell /bin/bash magnitude
install -d -o 1000 -g 1000 -m 0700 /run/magnitude
EOF

COPY --chmod=755 inference/ /opt/magnitude/inference/

# MAGNITUDE_DESKTOP_STATE_DIR keeps the control socket off /data, where a Docker Desktop bind
# mount refuses its chmod. `graphics` lets the NVIDIA toolkit mount its Vulkan driver.
ENV HOME=/data \
    MAGNITUDE_ICN_PATH=/opt/magnitude/inference/installation.json \
    MAGNITUDE_DESKTOP_STATE_DIR=/run/magnitude \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

USER magnitude
WORKDIR /data
VOLUME /data
EXPOSE 10100

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
  CMD ["curl", "-fsS", "-o", "/dev/null", "http://127.0.0.1:10100/health"]

ENTRYPOINT ["tini", "--", "magnitude", "serve"]
