import { LlamaDistributionVariantId } from "./identity"
import type { LlamaDistributionManifest } from "./distribution"
import { Sha256Digest } from "../model-files"

const releaseUrl = (file: string): URL =>
  new URL(`https://github.com/ggml-org/llama.cpp/releases/download/b10011/${file}`)

const variant = (
  id: string,
  platform: NodeJS.Platform,
  architecture: string,
  file: string,
  sha256: string,
): LlamaDistributionManifest["variants"][number] => ({
  id: LlamaDistributionVariantId.make(id),
  platform,
  architecture,
  archiveUrl: releaseUrl(file),
  sha256: Sha256Digest.make(sha256),
  executableRelativePath: "llama-b10011/llama-server",
  archive: "tar.gz",
})

export const DEFAULT_LLAMA_DISTRIBUTION_MANIFEST: LlamaDistributionManifest = {
  version: 1,
  release: "b10011",
  variants: [
    variant("macos-arm64-metal", "darwin", "arm64", "llama-b10011-bin-macos-arm64.tar.gz", "dc8a9b70737c82476662b8145fe10a491e7ef46329a7e6801816293b99534d3d"),
    variant("macos-x64-cpu", "darwin", "x64", "llama-b10011-bin-macos-x64.tar.gz", "3567bb00e2422088a42ac04179f64f61025148b31a71761daf58b742edc1feeb"),
    variant("linux-arm64-cpu", "linux", "arm64", "llama-b10011-bin-ubuntu-arm64.tar.gz", "f120decba9e4032456ad18922c1741298b081c597fe5e20f220616db4d65d267"),
    variant("linux-x64-cpu", "linux", "x64", "llama-b10011-bin-ubuntu-x64.tar.gz", "3cae0a514d2e95062be5b1ca19474446080a1cc12ae5cb1a89d0534bcd013ec1"),
    variant("linux-arm64-vulkan", "linux", "arm64", "llama-b10011-bin-ubuntu-vulkan-arm64.tar.gz", "ccee1bc1569e40cba5976144fad3242f665156e79398e9d26d39e4afcc8f1be9"),
    variant("linux-x64-vulkan", "linux", "x64", "llama-b10011-bin-ubuntu-vulkan-x64.tar.gz", "2993a6b5d5e1852bd5d10733aebf05769334d30de064a7795f542e668436bd8a"),
  ],
}
