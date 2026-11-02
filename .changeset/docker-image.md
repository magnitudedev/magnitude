---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Run Magnitude in Docker with the official image, `magnitudedev/magnitude`, for x64 and ARM64 Linux. See [Docker](https://docs.magnitude.dev/installation/docker).

- NVIDIA GPUs work through the NVIDIA Container Toolkit, and AMD and Intel GPUs through Vulkan. On a Mac, a container can't use the GPU through Metal, so install the macOS app for full speed.
- Models, settings and the network key live on the `/data` volume and survive upgrades. On first start the container creates a network key and prints it to `docker logs`.
- A container's memory limit sets which models fit, so models aren't loaded into memory the container can't use.
- Update notices tell you which `docker pull` to run, since the container doesn't update itself.
- Docker Compose, Kubernetes and Podman, including rootless Podman, are supported, and a native installation's data can be mounted to keep its models.
