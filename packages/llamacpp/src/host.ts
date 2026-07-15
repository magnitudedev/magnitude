import { cpus, freemem, totalmem } from "node:os"
import { Context, Effect, Layer } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import type {
  HostDevice,
  LlamaCppHostProfile,
  LlamaCppMemoryDomain,
  ModelFitPlan,
  ModelFitRequest,
} from "./contracts"
import { LlamaCppDistribution } from "./distribution"
import { LlamaCppHostError } from "./errors"

type HostRequirements = LlamaCppDistribution | FileSystem.FileSystem | CommandExecutor.CommandExecutor

interface RuntimeDevice {
  readonly backend: string
  readonly name: string
  readonly totalBytes: number
  readonly freeBytes: number | null
}

export interface LlamaCppHostApi {
  readonly inspect: Effect.Effect<LlamaCppHostProfile, LlamaCppHostError>
  readonly plan: (request: ModelFitRequest) => Effect.Effect<ModelFitPlan, LlamaCppHostError>
}

export class LlamaCppHost extends Context.Tag("LlamaCppHost")<LlamaCppHost, LlamaCppHostApi>() {}

const MIB = 1024 * 1024
const GIB = 1024 * MIB

export const parseRuntimeDevices = (output: string): readonly RuntimeDevice[] => {
  const devices: RuntimeDevice[] = []
  const seen = new Set<string>()
  for (const line of output.split("\n")) {
    const match = line.trim().match(/^([^:]+):\s+(.+?)\s+\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s+free)?\)/i)
    if (!match) continue
    const [, backend, name, totalMiB, freeMiB] = match
    if (backend === undefined || name === undefined || totalMiB === undefined) continue
    const totalBytes = Number(totalMiB) * MIB
    if (!Number.isFinite(totalBytes) || totalBytes <= 0) continue
    const free = freeMiB === undefined ? null : Number(freeMiB) * MIB
    const identity = `${backend.trim()}\0${name.trim()}`
    if (seen.has(identity)) continue
    seen.add(identity)
    devices.push({
      backend: backend.trim(),
      name: name.trim(),
      totalBytes,
      freeBytes: free !== null && Number.isFinite(free) ? free : null,
    })
  }
  return devices
}

const systemStableCapacity = (totalBytes: number): number =>
  Math.max(0, totalBytes - Math.max(4 * GIB, Math.floor(totalBytes * 0.15)))

const toDevice = (device: RuntimeDevice): HostDevice => ({
  backend: device.backend,
  name: device.name,
})

const buildMemoryDomains = (
  devices: readonly RuntimeDevice[],
  totalBytes: number,
  currentFreeBytes: number,
): readonly LlamaCppMemoryDomain[] => {
  const stableSystem = systemStableCapacity(totalBytes)
  // Apple silicon has one physical memory pool even before a llama.cpp
  // distribution is available to enumerate the Metal device. Keep the stable
  // capacity topology independent from runtime installation state.
  const unified = process.platform === "darwin" && process.arch === "arm64"

  if (unified) {
    // Unified-memory devices describe views into the same working set. Adding
    // them would count the same physical memory more than once.
    const reportedWorkingSet = Math.max(0, ...devices.map((device) => device.totalBytes))
    return [{
      id: "unified",
      kind: "unified_working_set",
      stableCapacityBytes: Math.min(stableSystem, reportedWorkingSet || stableSystem),
      currentFreeBytes: devices.every((device) => device.freeBytes === null)
        ? Math.min(currentFreeBytes, stableSystem)
        : Math.max(0, ...devices.map((device) => device.freeBytes ?? 0)),
      sharesSystemMemory: true,
      devices: devices.map(toDevice),
      splitGroupId: null,
    }]
  }

  const domains: LlamaCppMemoryDomain[] = [{
    id: "system",
    kind: "system",
    stableCapacityBytes: stableSystem,
    currentFreeBytes: Math.min(currentFreeBytes, totalBytes),
    sharesSystemMemory: false,
    devices: [],
    splitGroupId: null,
  }]
  for (const [index, device] of devices.entries()) {
    domains.push({
      id: `device-${index}`,
      kind: "physical_device",
      stableCapacityBytes: Math.floor(device.totalBytes * 0.9),
      currentFreeBytes: device.freeBytes,
      sharesSystemMemory: false,
      devices: [toDevice(device)],
      splitGroupId: null,
    })
  }
  return domains
}

const inspectHost: Effect.Effect<LlamaCppHostProfile, LlamaCppHostError, HostRequirements> =
  Effect.gen(function* () {
    const distribution = yield* LlamaCppDistribution
    const cpuList = cpus()
    const totalMemoryBytes = totalmem()
    const currentFreeBytes = freemem()
    const distributionState = yield* distribution.inspect.pipe(
      Effect.mapError((cause) => new LlamaCppHostError({ operation: "inspect", reason: cause.reason, cause })),
    )

    if (distributionState._tag !== "Ready") {
      return {
        system: {
          totalMemoryBytes,
          cpuModel: cpuList[0]?.model ?? null,
          logicalCores: cpuList.length,
        },
        memoryDomains: buildMemoryDomains([], totalMemoryBytes, currentFreeBytes),
        runtimeProbe: "not_installed",
        warnings: distributionState._tag === "Invalid"
          ? [{ code: "distribution_invalid", message: distributionState.reason }]
          : [],
      } satisfies LlamaCppHostProfile
    }

    const probe = yield* Command.string(Command.make(
      distributionState.distribution.executablePath,
      "--list-devices",
    )).pipe(Effect.either)
    const devices = probe._tag === "Right" ? parseRuntimeDevices(probe.right) : []
    const warnings = probe._tag === "Left"
      ? [{ code: "runtime_probe_failed", message: "llama-server could not enumerate runtime devices" }]
      : probe.right.trim() && devices.length === 0
        ? [{ code: "runtime_probe_partial", message: "llama-server device output was not recognized" }]
        : []

    return {
      system: {
        totalMemoryBytes,
        cpuModel: cpuList[0]?.model ?? null,
        logicalCores: cpuList.length,
      },
      memoryDomains: buildMemoryDomains(devices, totalMemoryBytes, currentFreeBytes),
      runtimeProbe: probe._tag === "Left" ? "partial" : warnings.length > 0 ? "partial" : "complete",
      warnings,
    } satisfies LlamaCppHostProfile
  }).pipe(
    Effect.mapError((cause) => cause instanceof LlamaCppHostError
      ? cause
      : new LlamaCppHostError({ operation: "inspect", reason: "Could not inspect llama.cpp host", cause })),
  )

export const planModelForProfile = (
  request: ModelFitRequest,
  profile: LlamaCppHostProfile,
): ModelFitPlan => {
  const contextBytes = request.contextBytesPerSlot * request.parallelSlots
  const requiredBytes = request.modelBytes + contextBytes
  const systemCapacity = profile.memoryDomains
    .filter((domain) => domain.kind === "system")
    .reduce((sum, domain) => sum + domain.stableCapacityBytes, 0)
  const unifiedCapacity = Math.max(0, ...profile.memoryDomains
    .filter((domain) => domain.sharesSystemMemory)
    .map((domain) => domain.stableCapacityBytes))
  const discreteDomains = profile.memoryDomains.filter((domain) =>
    domain.kind === "physical_device" && !domain.sharesSystemMemory)
  const discreteGroups = new Map<string, number>()
  for (const domain of discreteDomains) {
    const group = domain.splitGroupId ?? domain.id
    discreteGroups.set(group, (discreteGroups.get(group) ?? 0) + domain.stableCapacityBytes)
  }
  const discreteCapacity = Math.max(0, ...discreteGroups.values())
  const stableCapacityBytes = Math.max(unifiedCapacity, systemCapacity + discreteCapacity)
  const fullAcceleratorCapacity = Math.max(unifiedCapacity, discreteCapacity)
  const fits = requiredBytes <= stableCapacityBytes
  const fullAcceleration = requiredBytes <= fullAcceleratorCapacity
  const acceleratorModelBudget = Math.max(0, fullAcceleratorCapacity - contextBytes)
  const partialGpuLayers = request.modelLayerCount === null
    ? 0
    : Math.min(
        request.modelLayerCount,
        Math.floor(request.modelLayerCount * Math.min(1, acceleratorModelBudget / request.modelBytes)),
      )
  return {
    requiredBytes,
    stableCapacityBytes,
    parallelSlots: request.parallelSlots,
    gpuLayers: fullAcceleration ? -1 : partialGpuLayers,
    splitMode: !fullAcceleration && partialGpuLayers > 0 ? "layer" : "none",
    fits: fits && (fullAcceleration || requiredBytes <= systemCapacity || partialGpuLayers > 0),
  }
}

const planModel = (
  request: ModelFitRequest,
): Effect.Effect<ModelFitPlan, LlamaCppHostError, HostRequirements> =>
  Effect.gen(function* () {
    if (
      !Number.isFinite(request.modelBytes) || request.modelBytes <= 0
      || !Number.isFinite(request.contextBytesPerSlot) || request.contextBytesPerSlot < 0
      || !Number.isInteger(request.parallelSlots) || request.parallelSlots <= 0
      || (request.modelLayerCount !== null
        && (!Number.isInteger(request.modelLayerCount) || request.modelLayerCount <= 0))
    ) {
      return yield* new LlamaCppHostError({ operation: "plan", reason: "Model fit request is invalid" })
    }
    const profile = yield* inspectHost
    return planModelForProfile(request, profile)
  })

export const LlamaCppHostLive: Layer.Layer<LlamaCppHost, never, HostRequirements> = Layer.effect(
  LlamaCppHost,
  Effect.gen(function* () {
    const context = yield* Effect.context<HostRequirements>()
    return LlamaCppHost.of({
      inspect: inspectHost.pipe(Effect.provide(context)),
      plan: (request) => planModel(request).pipe(Effect.provide(context)),
    })
  }),
)
