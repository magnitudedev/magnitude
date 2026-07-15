import { Effect } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { LlamaCppHardwareError } from "../errors"
import type {
  HardwareInfo,
  ModelFitAssessment,
  ModelFitCategory,
  DevicePlacement,
} from "./types"

const MIB = 1024 * 1024

/**
 * Compute fast limit (fully GPU-accelerated) and ceiling (will run at all)
 * for the given hardware.
 *
 * - Unified memory (Mac): fast limit = sum of GPU free bytes (recommendedMaxWorkingSetSize),
 *   ceiling = available system RAM.
 * - Discrete GPU (Linux): fast limit = sum of GPU free VRAM,
 *   ceiling = GPU free VRAM + available RAM.
 * - CPU-only: fast limit = available RAM * 0.8, ceiling = available RAM.
 */
export function computeLimits(hw: HardwareInfo): {
  fastLimitBytes: number
  ceilingBytes: number
} {
  const gpuFree = hw.gpus.reduce((sum, g) => sum + g.freeBytes, 0)

  if (hw.gpus.length === 0) {
    return {
      fastLimitBytes: Math.floor(hw.memory.availableBytes * 0.8),
      ceilingBytes: hw.memory.availableBytes,
    }
  }

  if (hw.isUnifiedMemory) {
    return {
      fastLimitBytes: gpuFree,
      ceilingBytes: hw.memory.availableBytes,
    }
  }

  return {
    fastLimitBytes: gpuFree,
    ceilingBytes: gpuFree + hw.memory.availableBytes,
  }
}

export function categorizeFit(
  modelSizeBytes: number,
  limits: { fastLimitBytes: number; ceilingBytes: number },
): ModelFitCategory {
  if (modelSizeBytes <= limits.fastLimitBytes) return "fully-accelerated"
  if (modelSizeBytes <= limits.ceilingBytes) return "partial-cpu"
  return "wont-fit"
}

/**
 * Heuristic assessment — model file size vs computed limits.
 * Used pre-download (no model file yet) or when --fit-print is unavailable.
 */
export function assessHeuristic(
  hardware: HardwareInfo,
  modelSizeBytes: number,
): ModelFitAssessment {
  const limits = computeLimits(hardware)
  return {
    category: categorizeFit(modelSizeBytes, limits),
    fastLimitBytes: limits.fastLimitBytes,
    ceilingBytes: limits.ceilingBytes,
  }
}

/**
 * Precise assessment using `llama-server --fit-print`.
 * Runs the same placement algorithm as server start without actually starting the server.
 */
export function assessWithFitPrint(
  binaryPath: string,
  modelPath: string,
  modelSizeBytes: number,
  hardware: HardwareInfo,
): Effect.Effect<
  ModelFitAssessment,
  LlamaCppHardwareError,
  CommandExecutor.CommandExecutor
> {
  return Effect.gen(function* () {
    const result = yield* Command.string(
      Command.make(binaryPath, "-m", modelPath, "--fit-print"),
    ).pipe(
      Effect.mapError((cause) =>
        new LlamaCppHardwareError({ reason: `--fit-print failed: ${cause}` }),
      ),
    )

    const placement = parseFitPrintOutput(result)
    const limits = computeLimits(hardware)

    return {
      category: categorizeFit(modelSizeBytes, limits),
      fastLimitBytes: limits.fastLimitBytes,
      ceilingBytes: limits.ceilingBytes,
      placement,
    }
  })
}

/**
 * Parse `--fit-print` output into per-device placement.
 * Format: one line per device with model/context/compute in MiB.
 */
export function parseFitPrintOutput(output: string): readonly DevicePlacement[] {
  const lines = output.split("\n").map((l) => l.trim()).filter((l) => l.length > 0)
  const parsed: DevicePlacement[] = []

  for (const line of lines) {
    const parts = line.split(/\s+/)
    if (parts.length < 4) continue

    const [device, modelMiB, contextMiB, computeMiB] = parts
    const modelBytes = Number(modelMiB) * MIB
    const contextBytes = Number(contextMiB) * MIB
    const computeBytes = Number(computeMiB) * MIB

    if ([modelBytes, contextBytes, computeBytes].some((n) => Number.isNaN(n))) continue

    parsed.push({ device: device!, modelBytes, contextBytes, computeBytes })
  }

  return parsed
}
