import { Effect } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { Schema } from "effect"
import { detectLinuxGpu } from "./gpu"
import { LLAMACPP_RELEASE_REPO } from "./version"

// ── GPU preference ──

/** GPU preference for build selection and server configuration. */
export const GpuPreference = Schema.Literal("auto", "cpu", "vulkan")
export type GpuPreference = Schema.Schema.Type<typeof GpuPreference>

// ── Platform types ──

/** Release asset identifier for a platform+arch+GPU combination. */
export const PlatformAsset = Schema.Literal(
  "macos-arm64",
  "macos-x64",
  "ubuntu-x64",
  "ubuntu-arm64",
  "ubuntu-vulkan-x64",
  "ubuntu-vulkan-arm64",
)
export type PlatformAsset = Schema.Schema.Type<typeof PlatformAsset>

/** Platform detection result: OS, architecture, and the selected release asset. */
export const PlatformInfo = Schema.Struct({
  /** Operating system: `darwin` (macOS) or `linux`. */
  platform: Schema.Literal("darwin", "linux"),
  /** CPU architecture: `arm64` (Apple Silicon, ARM) or `x64` (Intel/AMD). */
  arch: Schema.Literal("arm64", "x64"),
  /** The best-compatible release asset for this platform. */
  asset: PlatformAsset,
})
export type PlatformInfo = Schema.Schema.Type<typeof PlatformInfo>

// ── Platform detection ──

/**
 * Detect the real architecture, accounting for Rosetta on Apple Silicon.
 * `process.arch` reports `x64` under Rosetta, but `uname -m` reports `arm64`.
 */
export function realArch(): Effect.Effect<"arm64" | "x64", never, CommandExecutor.CommandExecutor> {
  if (process.platform === "darwin") {
    return Command.string(Command.make("uname", "-m")).pipe(
      Effect.map((output) => (output.trim() === "arm64" ? "arm64" : "x64") as "arm64" | "x64"),
      Effect.catchAll(() => Effect.succeed(process.arch as "arm64" | "x64")),
    )
  }
  return Effect.succeed(process.arch as "arm64" | "x64")
}

/**
 * Detect platform and select the best-compatible release asset.
 * On Linux x64, runs GPU detection to pick between CPU and Vulkan builds.
 */
export function detectPlatform(
  gpuPreference: GpuPreference = "auto",
): Effect.Effect<PlatformInfo, never, CommandExecutor.CommandExecutor | FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const platform = process.platform as "darwin" | "linux"
    const arch = yield* realArch()

    if (platform === "darwin") {
      return {
        platform,
        arch,
        asset: arch === "arm64" ? "macos-arm64" : "macos-x64",
      }
    }

    // Linux
    if (gpuPreference === "vulkan") {
      return {
        platform,
        arch,
        asset: arch === "arm64" ? "ubuntu-vulkan-arm64" : "ubuntu-vulkan-x64",
      }
    }

    if (gpuPreference === "cpu") {
      return {
        platform,
        arch,
        asset: arch === "arm64" ? "ubuntu-arm64" : "ubuntu-x64",
      }
    }

    // gpuPreference === "auto" — detect GPU on Linux x64
    if (arch === "x64") {
      const gpu = yield* detectLinuxGpu()
      if (gpu === "vulkan") {
        return { platform, arch, asset: "ubuntu-vulkan-x64" }
      }
    }

    // Default: CPU build (always works)
    return {
      platform,
      arch,
      asset: arch === "arm64" ? "ubuntu-arm64" : "ubuntu-x64",
    }
  })
}

export function assetName(tag: string, asset: PlatformAsset): string {
  return `llama-${tag}-bin-${asset}.tar.gz`
}

export function downloadUrl(tag: string, asset: PlatformAsset): string {
  return `https://github.com/${LLAMACPP_RELEASE_REPO}/releases/download/${tag}/${assetName(tag, asset)}`
}
