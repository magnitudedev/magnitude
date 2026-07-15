import { cpus, totalmem, freemem } from "node:os"
import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import type { CpuInfo, MemoryInfo } from "./types"

/**
 * Get CPU info. Uses node:os for core count and model (pure kernel data).
 */
export function getCpuInfo(): CpuInfo {
  const info = cpus()
  return {
    model: info[0]?.model ?? "unknown",
    cores: info.length,
  }
}

/**
 * Get memory info.
 * macOS: sysctl hw.memsize via node:os.totalmem, available via vm_stat calculation.
 * Linux: /proc/meminfo via @effect/platform/FileSystem.
 * Fallback: node:os.totalmem/freemem.
 */
export function getMemoryInfo(): Effect.Effect<
  MemoryInfo,
  never,
  FileSystem.FileSystem
> {
  if (process.platform === "linux") {
    return linuxMemoryInfo()
  }

  // macOS and others — node:os is sufficient
  const total = totalmem()
  const available = freemem()
  return Effect.succeed({ totalBytes: total, availableBytes: Math.min(available, total) })
}

function linuxMemoryInfo(): Effect.Effect<MemoryInfo, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const content = yield* fs.readFileString("/proc/meminfo").pipe(
      Effect.catchAll(() => Effect.succeed("")),
    )

    const totalKiB = parseMeminfoLine(content, "MemTotal")
    const availKiB = parseMeminfoLine(content, "MemAvailable")

    if (totalKiB && availKiB) {
      return {
        totalBytes: totalKiB * 1024,
        availableBytes: availKiB * 1024,
      }
    }

    return { totalBytes: totalmem(), availableBytes: freemem() }
  })
}

function parseMeminfoLine(text: string, key: string): number | undefined {
  const re = new RegExp(`${key}:\s*([0-9]+)\s*kB`)
  const match = text.match(re)
  if (!match) return undefined
  const kiB = Number(match[1])
  return Number.isFinite(kiB) ? kiB : undefined
}
