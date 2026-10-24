import { Context, Effect, Option } from "effect"
import type { UpdateHardware } from "@magnitudedev/release/hosted-update"
import type { AcnHardwareSummary } from "@magnitudedev/acn-protocol"
import type { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"

export interface UpdateHardwareSource {
  readonly current: Effect.Effect<Option.Option<UpdateHardware>>
}
export const UpdateHardwareSource = Context.GenericTag<UpdateHardwareSource>("@magnitudedev/daemon-management/UpdateHardwareSource")

export const unavailableUpdateHardware: UpdateHardwareSource = { current: Effect.succeed(Option.none()) }

const GIGABYTE = 1024 ** 3
const label = (value: Option.Option<string>, limit: number): Option.Option<string> => Option.flatMap(value, text => {
  const clean = text.replace(/[^\x20-\x7E]+/g, " ").replace(/\s+/g, " ").trim().slice(0, limit).trim()
  return clean.length > 0 ? Option.some(clean) : Option.none()
})
const gigabytes = (bytes: number) => Math.round(bytes / GIGABYTE)

export const updateHardwareFromSummary = (summary: AcnHardwareSummary): UpdateHardware => {
  const gpuName = label(summary.gpuName, 96)
  return {
    backend: Option.some(summary.backend),
    gpu_name: gpuName,
    gpu_kind: Option.isSome(gpuName) ? summary.gpuKind : Option.none(),
    gpu_memory_gb: Option.isSome(gpuName) ? Option.map(summary.gpuMemoryBytes, gigabytes) : Option.none(),
    gpu_count: Option.some(summary.gpuCount),
    gpu_unused_reason: label(summary.gpuUnusedReason, 200),
    cpu_model: label(summary.cpuModel, 96),
    cpu_cores: summary.cpuCores,
    memory_gb: Option.some(gigabytes(summary.memoryBytes)),
  }
}

/** The service reports its hardware once it is ready; until then there is nothing to send. */
export const updateHardwareFromService = (state: OwnedServiceState): Option.Option<UpdateHardware> =>
  state._tag === "Ready" && state.health.state._tag === "Ready"
    ? Option.map(state.health.state.hardware, updateHardwareFromSummary)
    : Option.none()
