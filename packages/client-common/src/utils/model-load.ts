import type {
  LocalInferenceHardware,
  ModelInstanceAllocation,
  ModelLoadPlan,
  ModelLoadStage,
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
    case "optimizing": return Option.match(
      Option.flatMap(plannedAllocation, (plan) => Option.flatMap(hardware, (snapshot) => {
        // Accelerators and load devices are both identified by the service's hardware device id.
        const deviceId: string = plan.device.deviceId
        return Option.fromNullable(snapshot.accelerators.find((accelerator) => accelerator.acceleratorId === deviceId))
      })),
      {
        onNone: () => "Optimizing…",
        onSome: (accelerator) => `Optimizing for ${accelerator.name}…`,
      },
    )
    case "loading_weights": return "Loading weights…"
    case "finalizing": return "Finalizing…"
  }
}

/** Whether a stage's fraction measures its work; the others have no progress of their own. */
export const isMeasuredModelLoadStage = (stage: ModelLoadStage): boolean =>
  stage === "optimizing" || stage === "loading_weights"

export const formatModelLoadPercentage = (fraction: number): string =>
  `${Math.floor(fraction * 100)}%`

/** Memory a ready model holds across every memory domain. */
export const modelMemoryBytes = (allocation: ModelInstanceAllocation): number =>
  allocation.memoryDomains.reduce(
    (total, domain) => total + domain.modelBytes + domain.contextBytes + domain.computeBytes + domain.auxiliaryBytes,
    0,
  )

export const formatModelMemory = (allocation: ModelInstanceAllocation): string =>
  formatMemorySize(modelMemoryBytes(allocation))
