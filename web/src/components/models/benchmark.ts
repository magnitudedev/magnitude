import { Option } from "effect"
import type { MachineMeasurement, MachineMeasurementPhase } from "@magnitudedev/sdk"

export const HARDWARE_BENCHMARK_TITLE = "Benchmarking your hardware"

const phaseSubtitle: Record<MachineMeasurementPhase, string> = {
  Baseline: "Testing baseline compute",
  MemoryAndStorage: "Testing memory and SSD speed",
  TokenGeneration: "Testing token generation speed",
  PromptProcessing: "Testing prompt processing speed",
  LongContextAttention: "Testing long-context attention",
}

/** Shared status view for the hardware benchmark across onboarding and the Models empty state. */
export interface HardwareBenchmarkView {
  readonly title: string
  readonly subtitle: string
  readonly percent: number | null
  readonly fraction: number | null
  readonly failed: boolean
}

const retryLabel = (retryAt: number): string => {
  const formatted = new Date(retryAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
  return `Could not benchmark hardware · retrying at ${formatted}`
}

/**
 * Standardized benchmark state while recommendations are still settling. Returns `null` once recommendations are
 * complete.
 */
export const hardwareBenchmarkView = (
  measurement: MachineMeasurement | null,
  complete: boolean,
  hasGpu = true,
): HardwareBenchmarkView | null => {
  if (complete) return null
  if (measurement === null) {
    return {
      title: HARDWARE_BENCHMARK_TITLE,
      subtitle: phaseSubtitle.Baseline,
      percent: 0,
      fraction: 0,
      failed: false,
    }
  }
  switch (measurement._tag) {
    case "Unmeasured":
      return {
        title: HARDWARE_BENCHMARK_TITLE,
        subtitle: hasGpu ? "Waiting for your GPU to be free" : "Waiting for your machine to be ready",
        percent: null,
        fraction: null,
        failed: false,
      }
    case "Measuring": {
      const progress = Option.getOrNull(measurement.progress)
      const fraction = progress ? Math.min(1, Math.max(0, progress.completed / progress.total)) : 0
      return {
        title: HARDWARE_BENCHMARK_TITLE,
        subtitle: progress ? phaseSubtitle[progress.phase] : phaseSubtitle.Baseline,
        percent: Math.round(fraction * 100),
        fraction,
        failed: false,
      }
    }
    case "Measured":
      return {
        title: HARDWARE_BENCHMARK_TITLE,
        subtitle: "Ranking models for your machine",
        percent: 100,
        fraction: 1,
        failed: false,
      }
    case "Failed":
      return {
        title: HARDWARE_BENCHMARK_TITLE,
        subtitle: retryLabel(measurement.retryAt),
        percent: null,
        fraction: null,
        failed: true,
      }
  }
}
