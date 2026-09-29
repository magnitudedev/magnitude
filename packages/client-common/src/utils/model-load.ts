import type {
  LocalInferenceHardware,
  ModelInstanceAllocation,
  ModelLoadDevice,
  ModelLoadPlan,
  ModelLoadStage,
  ModelOptimizationProgress,
} from "@magnitudedev/sdk"
import { Option } from "effect"
import { formatMemorySize } from "./format-bytes"

/** One word for a load stage, where space is short (the tray, the CLI). */
export const formatModelLoadStage = (stage: ModelLoadStage): string => {
  switch (stage) {
    case "queued": return "Waiting"
    case "preparing": return "Preparing"
    case "optimizing": return "Optimizing"
    case "loading_weights": return "Loading"
    case "finalizing": return "Finalizing"
  }
}

/**
 * A load stage in full, naming the accelerator tuning is for when the hardware snapshot has it.
 * A CPU load has no accelerator.
 */
export const describeModelLoadStage = (
  stage: ModelLoadStage,
  plannedAllocation: Option.Option<ModelLoadPlan>,
  hardware: Option.Option<LocalInferenceHardware>,
): string => {
  switch (stage) {
    case "queued": return "Waiting for memory…"
    case "preparing": return "Preparing…"
    case "optimizing": return describeOptimizing(Option.map(plannedAllocation, (plan) => plan.device), hardware)
    case "loading_weights": return "Loading weights…"
    case "finalizing": return "Finalizing…"
  }
}

const describeOptimizing = (
  device: Option.Option<ModelLoadDevice>,
  hardware: Option.Option<LocalInferenceHardware>,
): string => Option.match(
  Option.flatMap(device, (selected) => Option.flatMap(hardware, (snapshot) => {
    // Accelerators and load devices are both identified by the service's hardware device id.
    const deviceId: string = selected.deviceId
    return Option.fromNullable(snapshot.accelerators.find((accelerator) => accelerator.acceleratorId === deviceId))
  })),
  {
    onNone: () => "Optimizing…",
    onSome: (accelerator) => `Optimizing for ${accelerator.name}…`,
  },
)

/** Tuning's share done, or none while preparation is still counting the work. */
export const modelOptimizationFraction = (progress: ModelOptimizationProgress): Option.Option<number> =>
  progress.stage === "tuning" && progress.total > 0
    ? Option.some(Math.min(1, progress.completed / progress.total))
    : Option.none()

/** Post-download optimization in full, naming the accelerator it tunes for when known. */
export const describeModelOptimization = (
  progress: ModelOptimizationProgress,
  hardware: Option.Option<LocalInferenceHardware>,
): string => progress.stage === "preparing"
  ? "Preparing to optimize…"
  : describeOptimizing(progress.device, hardware)

/** Post-download optimization where space is short (the CLI). */
export const formatModelOptimization = (progress: ModelOptimizationProgress): string => Option.match(
  modelOptimizationFraction(progress),
  {
    onNone: () => "Optimizing",
    onSome: (fraction) => `Optimizing ${formatModelLoadPercentage(fraction)}`,
  },
)

/** Whether a stage's fraction measures its work; the others have no progress of their own. */
export const isMeasuredModelLoadStage = (stage: ModelLoadStage): boolean =>
  stage === "optimizing" || stage === "loading_weights"

/** Whole percent done; the tolerance keeps an exact share such as 232 / 400 from reading one low. */
export const formatModelLoadPercentage = (fraction: number): string =>
  `${Math.floor(fraction * 100 + 1e-9)}%`

/** Memory a ready model holds across every memory domain. */
export const modelMemoryBytes = (allocation: ModelInstanceAllocation): number =>
  allocation.memoryDomains.reduce(
    (total, domain) => total + domain.modelBytes + domain.contextBytes + domain.computeBytes + domain.auxiliaryBytes,
    0,
  )

export const formatModelMemory = (allocation: ModelInstanceAllocation): string =>
  formatMemorySize(modelMemoryBytes(allocation))
