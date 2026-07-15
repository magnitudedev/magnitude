import type { LlamaCppReleaseManifest } from "./contracts"

const releaseUrl = (fileName: string): string =>
  `https://github.com/ggml-org/llama.cpp/releases/download/b10011/${fileName}`

export const DEFAULT_LLAMACPP_RELEASE: LlamaCppReleaseManifest = {
  build: 10011,
  tag: "b10011",
  assets: [
    {
      platform: "darwin",
      architecture: "arm64",
      accelerator: "metal",
      fileName: "llama-b10011-bin-macos-arm64.tar.gz",
      url: releaseUrl("llama-b10011-bin-macos-arm64.tar.gz"),
      sizeBytes: 10_770_249,
      sha256: "dc8a9b70737c82476662b8145fe10a491e7ef46329a7e6801816293b99534d3d",
    },
    {
      platform: "darwin",
      architecture: "x64",
      accelerator: "cpu",
      fileName: "llama-b10011-bin-macos-x64.tar.gz",
      url: releaseUrl("llama-b10011-bin-macos-x64.tar.gz"),
      sizeBytes: 11_046_813,
      sha256: "3567bb00e2422088a42ac04179f64f61025148b31a71761daf58b742edc1feeb",
    },
    {
      platform: "linux",
      architecture: "arm64",
      accelerator: "cpu",
      fileName: "llama-b10011-bin-ubuntu-arm64.tar.gz",
      url: releaseUrl("llama-b10011-bin-ubuntu-arm64.tar.gz"),
      sizeBytes: 12_810_055,
      sha256: "f120decba9e4032456ad18922c1741298b081c597fe5e20f220616db4d65d267",
    },
    {
      platform: "linux",
      architecture: "x64",
      accelerator: "cpu",
      fileName: "llama-b10011-bin-ubuntu-x64.tar.gz",
      url: releaseUrl("llama-b10011-bin-ubuntu-x64.tar.gz"),
      sizeBytes: 15_877_355,
      sha256: "3cae0a514d2e95062be5b1ca19474446080a1cc12ae5cb1a89d0534bcd013ec1",
    },
    {
      platform: "linux",
      architecture: "arm64",
      accelerator: "vulkan",
      fileName: "llama-b10011-bin-ubuntu-vulkan-arm64.tar.gz",
      url: releaseUrl("llama-b10011-bin-ubuntu-vulkan-arm64.tar.gz"),
      sizeBytes: 25_538_945,
      sha256: "ccee1bc1569e40cba5976144fad3242f665156e79398e9d26d39e4afcc8f1be9",
    },
    {
      platform: "linux",
      architecture: "x64",
      accelerator: "vulkan",
      fileName: "llama-b10011-bin-ubuntu-vulkan-x64.tar.gz",
      url: releaseUrl("llama-b10011-bin-ubuntu-vulkan-x64.tar.gz"),
      sizeBytes: 31_300_070,
      sha256: "2993a6b5d5e1852bd5d10733aebf05769334d30de064a7795f542e668436bd8a",
    },
  ],
}
