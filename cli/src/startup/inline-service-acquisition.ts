import type { BinaryAcquisitionEvent } from "@magnitudedev/sdk"
import { formatStorageSize } from "@magnitudedev/client-common"

export interface ServiceAcquisitionChildPhase {
  readonly active: string
  readonly completed: string
  readonly progress: number
}

export const serviceAcquisitionChildPhase = (
  event: BinaryAcquisitionEvent,
): ServiceAcquisitionChildPhase | null => {
  if (event._tag === "Planned" || event.event._tag !== "Downloading") return null
  const completed = event.event.progress.acceptedBytes
  const total = event.event.progress.totalBytes
  const progress = Math.max(0, Math.min(1, completed / Math.max(1, total)))
  return {
    active: `Downloading Magnitude service... ${Math.floor(progress * 100)}% (${formatStorageSize(completed)} / ${formatStorageSize(total)})`,
    completed: `Magnitude service downloaded 100% (${formatStorageSize(total)} / ${formatStorageSize(total)})`,
    progress,
  }
}
