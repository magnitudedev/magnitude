import { Effect, Option, Schema } from "effect"
import { Architecture, AssertionFailure, Target } from "./domain"
import { NativeExecution } from "./execution-telemetry"

// CUDA and NVIDIA's inventory use different domain padding/case for the same PCI address.
export const PciBusId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{4,8}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]$/i),
  Schema.transform(Schema.String, { strict: true,
    decode: value => {
      const separator = value.indexOf(":")
      return value.slice(0, separator).toLowerCase().padStart(8, "0") + value.slice(separator).toLowerCase()
    },
    encode: value => value }), Schema.brand("LabPciBusId"))
export const ObservedGpu = Schema.Struct({ name: Schema.NonEmptyString, backend: Schema.Literal("cuda", "metal"),
  uuid: Schema.NonEmptyString, pciBusId: Schema.NullOr(PciBusId), driver: Schema.String, memoryBytes: Schema.NullOr(Schema.Int.pipe(Schema.nonNegative())) })
export const HostObservation = Schema.Struct({ os: Target.fields.os, version: Schema.NonEmptyString, build: Schema.String,
  arch: Architecture, cpuVendor: Schema.String, cpuName: Schema.String, machineModel: Schema.String,
  gpus: Schema.Array(ObservedGpu), memoryBytes: Schema.Int.pipe(Schema.positive()) })
export type HostObservation = typeof HostObservation.Type
const gpuMatches = (hardware: Target["hardware"], name: string) => hardware === "a10" ? /\bA10\b/i.test(name)
  : hardware === "rtx-pro-6000" ? /RTX\s*PRO\s*6000\s+Blackwell\s+Server\s+Edition/i.test(name)
  : hardware === "dgx-spark" ? /\bGB10\b/i.test(name)
  : hardware === "apple-silicon" ? /Apple|Paravirtual/i.test(name) : false
export const attestHost = (target: Target, observed: HostObservation) => Effect.gen(function* () {
  const assert = (condition: boolean, message: string) => condition ? Effect.void : Effect.fail(new AssertionFailure({ message }))
  yield* assert(target.os === observed.os && target.arch === observed.arch, "Observed OS or CPU architecture does not match the selected target")
  yield* assert(observed.version === target.version || observed.version.startsWith(`${target.version}.`), "Observed OS version does not match the selected target")
  if (target.hardware === "intel") yield* assert(/intel/i.test(observed.cpuVendor), "An Intel CPU target ran on a different CPU vendor")
  if (target.hardware === "amd") yield* assert(/amd/i.test(observed.cpuVendor), "An AMD CPU target ran on a different CPU vendor")
  if (["a10", "rtx-pro-6000", "dgx-spark"].includes(target.hardware) || target.backend === "metal") {
    yield* assert(observed.gpus.some(g => gpuMatches(target.hardware, g.name)), `Required hardware ${target.hardware} was not observed`)
  }
  if (target.backend !== "cpu") yield* assert(observed.gpus.some(g => g.backend === target.backend), `No usable ${target.backend} device was observed`)
})

/** Use actual target-model allocations; a requested layer count is not execution evidence. */
export const attestGeneration = (target: Target, host: HostObservation, expectedModel: string, observation: NativeExecution) => Effect.gen(function* () {
  yield* attestHost(target, host)
  if (observation.model !== expectedModel || observation.allocations.length === 0) {
    return yield* new AssertionFailure({ message: "Generation has no allocation evidence for the expected model" })
  }
  // GGML reports CPU buffers as device allocations too. Only non-CPU devices are accelerators.
  const devices = observation.allocations.filter(allocation => allocation.kind === "device")
    .filter(allocation => allocation.backend.toLowerCase() !== "cpu")
  if (target.backend === "cpu") {
    if (devices.length > 0) {
      return yield* new AssertionFailure({ message: "CPU-only generation used accelerator model allocations" })
    }
    return
  }
  if (devices.length === 0) return yield* new AssertionFailure({ message: "Accelerator generation has no target-model device allocation; CPU fallback cannot pass" })
  for (const allocation of devices) {
    const backend = allocation.backend.toLowerCase() === "mtl" ? "metal" : allocation.backend.toLowerCase()
    if (backend !== target.backend) return yield* new AssertionFailure({ message: "Target-model allocation used a different backend" })
    const available = host.gpus.filter(gpu => gpu.backend === backend)
    const pciBusId = Schema.decodeUnknownOption(PciBusId)(allocation.physical_id)
    // Metal has no exported physical ID today. Only one enumerated device at index zero
    // is unambiguous; CUDA and multi-device hosts require an exact physical identity.
    const matches = allocation.physical_id === null
      ? backend === "metal" && allocation.native_index === 0 && available.length === 1 ? available : []
      : available.filter(gpu => backend === "cuda" ? Option.exists(pciBusId, id => gpu.pciBusId === id) : gpu.uuid === allocation.physical_id)
    if (matches.length !== 1 || !gpuMatches(target.hardware, matches[0]!.name)) {
      return yield* new AssertionFailure({ message: "Target-model allocation cannot be uniquely matched to the requested hardware" })
    }
  }
})
