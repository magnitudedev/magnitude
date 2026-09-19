import { Effect, Schema } from "effect"
import { Architecture, AssertionFailure, Backend, Target } from "./domain"

export const ObservedGpu = Schema.Struct({ name: Schema.NonEmptyString, backend: Schema.Literal("cuda", "metal"),
  uuid: Schema.NonEmptyString, driver: Schema.String, memoryBytes: Schema.Int.pipe(Schema.nonNegative()) })
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

/** Generation receipts are observations of the executing engine, not hardware capability listings. */
export const BackendObservation = Schema.Struct({ backend: Backend, model: Schema.NonEmptyString, requestId: Schema.NonEmptyString,
  loadedModuleDigest: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/)),
  devices: Schema.Array(Schema.Struct({ uuid: Schema.NonEmptyString, name: Schema.NonEmptyString, allocatedModelBytes: Schema.Int.pipe(Schema.nonNegative()) })),
  offloadedLayers: Schema.Int.pipe(Schema.nonNegative()), totalLayers: Schema.Int.pipe(Schema.positive()),
})
export type BackendObservation = typeof BackendObservation.Type
export const attestGeneration = (target: Target, expectedModel: string, expectedRequestId: string, observation: BackendObservation) => Effect.gen(function* () {
  if (observation.backend !== target.backend || observation.model !== expectedModel || observation.requestId !== expectedRequestId) {
    return yield* new AssertionFailure({ message: "Generation evidence belongs to a different backend, model or request" })
  }
  if (target.backend === "cpu") {
    if (observation.offloadedLayers !== 0 || observation.devices.some(d => d.allocatedModelBytes > 0)) {
      return yield* new AssertionFailure({ message: "CPU-only generation used accelerator model allocations" })
    }
  } else if (observation.offloadedLayers < 1 || observation.offloadedLayers > observation.totalLayers ||
    !observation.devices.some(d => d.allocatedModelBytes > 0 && gpuMatches(target.hardware, d.name))) {
    return yield* new AssertionFailure({ message: "Accelerator generation has no matching device allocation and layer offload evidence; CPU fallback cannot pass" })
  }
})
