import { Option } from "effect"
import type {
  LocalInferenceBackend,
  LocalInferenceHardware,
  LocalInferenceMemoryDomainId,
  ModelInstanceAllocation,
} from "@magnitudedev/sdk"

export type HardwareMemoryBreakdownStatus =
  | "complete"
  | "rounding_adjusted"
  | "inconsistent"
  | "missing_free"

export interface HardwareMemoryDomainView {
  readonly id: LocalInferenceMemoryDomainId
  readonly label: string
  readonly kind: LocalInferenceHardware["memoryDomains"][number]["kind"]
  readonly totalBytes: number
  readonly usedBytes: number | null
  readonly modelBytes: number | null
  readonly overheadBytes: number | null
  readonly fixedBytes: number | null
  readonly kvCacheBytes: number | null
  readonly systemAndAppsBytes: number | null
  readonly freeBytes: number | null
  readonly status: HardwareMemoryBreakdownStatus
  readonly notice: string | null
  readonly participatesInModelServing: boolean
}

export interface HardwareMemoryView {
  readonly domains: readonly HardwareMemoryDomainView[]
  readonly compact: { readonly usedBytes: number; readonly totalBytes: number } | null
}

const MIB = 1024 ** 2

/** The memory domains an accelerator places on: every domain any of its executors places on. */
export const acceleratorMemoryDomainIds = (
  accelerator: LocalInferenceHardware["accelerators"][number],
): ReadonlySet<LocalInferenceMemoryDomainId> =>
  new Set(accelerator.executors.flatMap(({ memoryDomainIds }) => memoryDomainIds))

/** An accelerator's role: it holds the active device (in use, on that device's backend), or it is available. */
export type LocalHardwareRole =
  | { readonly _tag: "InUse"; readonly backend: LocalInferenceBackend }
  | { readonly _tag: "Available" }

export interface LocalHardwareDevice {
  readonly accelerator: LocalInferenceHardware["accelerators"][number]
  readonly role: LocalHardwareRole
  /** Its own memory's total; none for a GPU whose memory is host RAM. */
  readonly dedicatedMemoryBytes: Option.Option<number>
  /** Whether it can place memory in host RAM. */
  readonly placesOnHost: boolean
}

export interface LocalHardwareRoles {
  /** The machine's title: its product name, then its CPU; never an accelerator's name. */
  readonly machineName: string
  /** Whether host RAM is the active device's own memory (Apple Silicon, or a serving integrated GPU). */
  readonly unifiedMemory: boolean
  /** Every usable accelerator, the one holding the active device first. */
  readonly devices: readonly LocalHardwareDevice[]
}

const uniqueNames = (values: readonly string[]): string[] =>
  [...new Set(values.map((value) => value.trim()).filter(Boolean))]

/** The roles of the machine and its devices; every surface reads them from here. */
export const deriveHardwareRoles = (hardware: LocalInferenceHardware): LocalHardwareRoles => {
  const domains = new Map(hardware.memoryDomains.map((domain) => [domain.memoryDomainId, domain]))
  const devices = hardware.accelerators.map((accelerator): LocalHardwareDevice => {
    const own = [...acceleratorMemoryDomainIds(accelerator)].flatMap((id) => {
      const domain = domains.get(id)!
      return domain.kind === "PhysicalDevice" ? [domain] : []
    })
    const active = accelerator.executors.find(({ deviceId }) => deviceId === hardware.activeDeviceId)
    return {
      accelerator,
      role: active === undefined ? { _tag: "Available" } : { _tag: "InUse", backend: active.backend },
      dedicatedMemoryBytes: own.length === 0 ? Option.none() : Option.some(own.reduce((sum, domain) => sum + domain.totalBytes, 0)),
      placesOnHost: acceleratorMemoryDomainIds(accelerator).size > own.length,
    }
  })
  const isAppleSilicon = hardware.platform === "MacOS" && hardware.architecture === "Arm64"
  return {
    machineName: uniqueNames([...Option.toArray(hardware.productName), ...Option.toArray(hardware.processor)]).join(" · ")
      || (isAppleSilicon ? "Apple Silicon" : "CPU"),
    unifiedMemory: hardware.memoryDomains.some((domain) => domain.kind === "UnifiedMemory"),
    devices: [...devices.filter(({ role }) => role._tag === "InUse"), ...devices.filter(({ role }) => role._tag === "Available")],
  }
}

const domainLabel = (
  hardware: LocalInferenceHardware,
  domain: LocalInferenceHardware["memoryDomains"][number],
  physicalOrdinal: number,
): string => {
  if (domain.kind === "System") return "System memory"
  if (domain.kind === "UnifiedMemory") return "Unified memory"
  const accelerators = hardware.accelerators.filter((accelerator) =>
    acceleratorMemoryDomainIds(accelerator).has(domain.memoryDomainId))
  return `${uniqueNames(accelerators.map(({ name }) => name)).join(" + ") || "Accelerator"} · GPU ${physicalOrdinal}`
}

export interface HardwareMemoryViewOptions {
  readonly participatingDomainIds?: readonly LocalInferenceMemoryDomainId[]
  readonly fallbackToAccelerators?: boolean
}

export const deriveHardwareMemoryView = (
  hardware: LocalInferenceHardware,
  modelAllocation: Option.Option<ModelInstanceAllocation>,
  options: HardwareMemoryViewOptions = {},
): HardwareMemoryView => {
  const allocations = new Map(Option.match(modelAllocation, {
    onNone: () => [],
    onSome: ({ memoryDomains }) =>
      memoryDomains.map((domain) => [domain.memoryDomainId, domain] as const),
  }))
  const explicitFallbackIds = Option.isSome(modelAllocation)
    ? []
    : options.participatingDomainIds ?? []
  const acceleratorDomainIds = hardware.accelerators.flatMap((accelerator) =>
    [...acceleratorMemoryDomainIds(accelerator)])
  const fallbackModelServingDomainIds = new Set(explicitFallbackIds.length > 0
    ? explicitFallbackIds
    : options.fallbackToAccelerators === false
      ? []
      : acceleratorDomainIds.length > 0
        ? acceleratorDomainIds
        : hardware.memoryDomains.map(({ memoryDomainId }) => memoryDomainId))
  let physicalOrdinal = 0
  const domains = hardware.memoryDomains.map((domain): HardwareMemoryDomainView => {
    if (domain.kind === "PhysicalDevice") physicalOrdinal += 1
    const allocation = Option.fromNullable(allocations.get(domain.memoryDomainId))
    const participatesInModelServing = fallbackModelServingDomainIds.has(domain.memoryDomainId)
      || Option.exists(allocation, (resident) =>
        resident.modelBytes + resident.contextBytes + resident.computeBytes + resident.auxiliaryBytes > 0)
    const freeBytes = Option.getOrNull(Option.map(domain.availableBytes, (available) =>
      Math.min(domain.totalBytes, Math.max(0, available))))
    const usedBytes = freeBytes === null ? null : domain.totalBytes - freeBytes
    const modelBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.modelBytes,
    })
    const overheadBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.computeBytes + resident.auxiliaryBytes,
    })
    const fixedBytes = modelBytes + overheadBytes
    const kvCacheBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.contextBytes,
    })
    const base = {
      id: domain.memoryDomainId,
      label: domainLabel(hardware, domain, physicalOrdinal),
      kind: domain.kind,
      totalBytes: domain.totalBytes,
      participatesInModelServing,
    }
    if (freeBytes === null || usedBytes === null) return {
      ...base,
      usedBytes: null,
      modelBytes,
      overheadBytes,
      fixedBytes,
      kvCacheBytes,
      systemAndAppsBytes: null,
      freeBytes: null,
      status: "missing_free",
      notice: "Current memory usage is unavailable.",
    }
    const ownedBytes = fixedBytes + kvCacheBytes
    const excessBytes = ownedBytes - usedBytes
    const toleranceBytes = Math.max(64 * MIB, domain.totalBytes * 0.001)
    if (excessBytes > toleranceBytes) return {
      ...base,
      usedBytes,
      modelBytes: null,
      overheadBytes: null,
      fixedBytes: null,
      kvCacheBytes: null,
      systemAndAppsBytes: null,
      freeBytes,
      status: "inconsistent",
      notice: "Allocation and resident usage could not be reconciled.",
    }
    const adjusted = excessBytes > 0
    const displayedUsedBytes = adjusted ? ownedBytes : usedBytes
    return {
      ...base,
      usedBytes: displayedUsedBytes,
      modelBytes,
      overheadBytes,
      fixedBytes,
      kvCacheBytes,
      systemAndAppsBytes: Math.max(0, displayedUsedBytes - ownedBytes),
      freeBytes: domain.totalBytes - displayedUsedBytes,
      status: adjusted ? "rounding_adjusted" : "complete",
      notice: null,
    }
  })
  const participating = domains.filter(
    (domain) => domain.participatesInModelServing && domain.usedBytes !== null,
  )
  return {
    domains,
    compact: participating.length === 0 ? null : {
      usedBytes: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
      totalBytes: participating.reduce((sum, domain) => sum + domain.totalBytes, 0),
    },
  }
}
