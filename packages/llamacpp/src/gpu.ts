import { Effect } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { LlamaCppHardwareError } from "./errors"

/**
 * Detect GPU availability on Linux using OS commands.
 * Returns the best build variant: "vulkan", "rocm", or null (CPU).
 * This runs BEFORE we have a binary — it's for asset selection only.
 * Authoritative device info comes from --list-devices after install.
 */
export function detectLinuxGpu(): Effect.Effect<
  "vulkan" | "rocm" | null,
  never,
  CommandExecutor.CommandExecutor | FileSystem.FileSystem
> {
  return Effect.gen(function* () {
    const vulkan = yield* hasVulkanGpu()
    if (vulkan) return "vulkan"
    return null
  })
}

function hasVulkanGpu(): Effect.Effect<boolean, never, CommandExecutor.CommandExecutor | FileSystem.FileSystem> {
  return Effect.gen(function* () {
    // 1. vulkaninfo --summary → parse for deviceType DISCRETE_GPU / INTEGRATED_GPU
    const vulkaninfoResult = yield* runCommand("vulkaninfo", "--summary").pipe(
      Effect.catchAll(() => Effect.succeed("")),
    )
    if (/deviceType\s*=\s*(DISCRETE_GPU|INTEGRATED_GPU)/i.test(vulkaninfoResult)) {
      return true
    }

    // 2. Check ICD files at standard locations
    const fs = yield* FileSystem.FileSystem
    const icdDirs = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"]
    for (const dir of icdDirs) {
      const exists = yield* fs.exists(dir).pipe(Effect.catchAll(() => Effect.succeed(false)))
      if (exists) {
        const files = yield* fs.readDirectory(dir).pipe(
          Effect.catchAll(() => Effect.succeed([] as readonly string[])),
        )
        if (files.some((f) => f.endsWith(".json"))) return true
      }
    }

    // 3. Check ldconfig for libvulkan
    const ldconfigResult = yield* runCommand("ldconfig", "-p").pipe(
      Effect.catchAll(() => Effect.succeed("")),
    )
    if (ldconfigResult.includes("libvulkan.so.1")) return true

    return false
  })
}

function runCommand(
  bin: string,
  ...args: readonly string[]
): Effect.Effect<string, LlamaCppHardwareError, CommandExecutor.CommandExecutor> {
  return Command.string(Command.make(bin, ...args)).pipe(
    Effect.mapError(() => new LlamaCppHardwareError({ reason: `command failed: ${bin}` })),
  )
}
