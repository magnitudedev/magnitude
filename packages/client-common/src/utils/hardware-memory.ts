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
  readonly weightsBytes: number | null
  readonly overheadBytes: number | null
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

export type MemoryBarSegmentKey = "weights" | "cache" | "runtime" | "other"

export interface MemoryBarSegment {
  readonly key: MemoryBarSegmentKey
  readonly label: string
  readonly bytes: number
  /** Percentage of total domain capacity (0..100). */
  readonly widthPercent: number
}

export interface MemoryDomainBarLayout {
  readonly domain: HardwareMemoryDomainView
  /** Total bytes owned by the loaded model in this domain, or null when observations are inconsistent. */
  readonly modelBytes: number | null
  /** Left-anchored model segments (Weights, Context, Overhead), in order. */
  readonly leftSegments: readonly MemoryBarSegment[]
  /** Right-anchored segment (Other apps), present when observed. */
  readonly rightSegment: MemoryBarSegment | null
  /** Free headroom in the middle, present when observed. */
  readonly freeBytes: number | null
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

/**
 * The memory a model is held in while it serves: unified memory, a GPU's own memory beside host RAM, or system memory
 * when it runs on the CPU.
 */
export type ServingMemory = "Unified" | "Device" | "System"

export interface LocalHardwareRoles {
  /** The machine's title: its product name, then its CPU; never an accelerator's name. */
  readonly machineName: string
  /** Whether host RAM is the active device's own memory (Apple Silicon, or a serving integrated GPU). */
  readonly unifiedMemory: boolean
  readonly servingMemory: ServingMemory
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
  const unifiedMemory = hardware.memoryDomains.some((domain) => domain.kind === "UnifiedMemory")
  const servingGpu = devices.some(({ role, dedicatedMemoryBytes }) => role._tag === "InUse" && Option.isSome(dedicatedMemoryBytes))
  return {
    machineName: uniqueNames([...Option.toArray(hardware.productName), ...Option.toArray(hardware.processor)]).join(" · ")
      || (isAppleSilicon ? "Apple Silicon" : "CPU"),
    unifiedMemory,
    servingMemory: unifiedMemory ? "Unified" : servingGpu ? "Device" : "System",
    devices: [...devices.filter(({ role }) => role._tag === "InUse"), ...devices.filter(({ role }) => role._tag === "Available")],
  }
}

/** Every memory domain's name: system or unified memory, or a device's accelerators and its GPU ordinal. */
export const memoryDomainLabels = (
  hardware: LocalInferenceHardware,
): ReadonlyMap<LocalInferenceMemoryDomainId, string> => {
  let physicalOrdinal = 0
  return new Map(hardware.memoryDomains.map((domain) => {
    if (domain.kind === "System") return [domain.memoryDomainId, "System memory"]
    if (domain.kind === "UnifiedMemory") return [domain.memoryDomainId, "Unified memory"]
    physicalOrdinal += 1
    const accelerators = hardware.accelerators.filter((accelerator) =>
      acceleratorMemoryDomainIds(accelerator).has(domain.memoryDomainId))
    return [
      domain.memoryDomainId,
      `${uniqueNames(accelerators.map(({ name }) => name)).join(" + ") || "Accelerator"} · GPU ${physicalOrdinal}`,
    ]
  }))
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
  const labels = memoryDomainLabels(hardware)
  const domains = hardware.memoryDomains.map((domain): HardwareMemoryDomainView => {
    const allocation = Option.fromNullable(allocations.get(domain.memoryDomainId))
    const participatesInModelServing = fallbackModelServingDomainIds.has(domain.memoryDomainId)
      || Option.exists(allocation, (resident) =>
        resident.modelBytes + resident.contextBytes + resident.computeBytes + resident.auxiliaryBytes > 0)
    const freeBytes = Option.getOrNull(Option.map(domain.availableBytes, (available) =>
      Math.min(domain.totalBytes, Math.max(0, available))))
    const usedBytes = freeBytes === null ? null : domain.totalBytes - freeBytes
    const weightsBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.modelBytes + resident.auxiliaryBytes,
    })
    const overheadBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.computeBytes,
    })
    const kvCacheBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.contextBytes,
    })
    const base = {
      id: domain.memoryDomainId,
      label: labels.get(domain.memoryDomainId)!,
      kind: domain.kind,
      totalBytes: domain.totalBytes,
      participatesInModelServing,
    }
    if (freeBytes === null || usedBytes === null) return {
      ...base,
      usedBytes: null,
      weightsBytes,
      overheadBytes,
      kvCacheBytes,
      systemAndAppsBytes: null,
      freeBytes: null,
      status: "missing_free",
      notice: "Current memory usage is unavailable.",
    }
    const ownedBytes = weightsBytes + overheadBytes + kvCacheBytes
    const excessBytes = ownedBytes - usedBytes
    const toleranceBytes = Math.max(64 * MIB, domain.totalBytes * 0.001)
    if (excessBytes > toleranceBytes) return {
      ...base,
      usedBytes,
      weightsBytes: null,
      overheadBytes: null,
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
      weightsBytes,
      overheadBytes,
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

const capacityPercent = (bytes: number, totalBytes: number): number =>
  totalBytes > 0 ? (bytes / totalBytes) * 100 : 0

/**
 * The participating memory domains to display as full-capacity bars, device memory first.
 * Falls back to all domains when none are marked participating.
 */
export const participatingMemoryDomains = (
  view: HardwareMemoryView,
): readonly HardwareMemoryDomainView[] => {
  const active = view.domains.filter((domain) => domain.participatesInModelServing)
  const domains = active.length > 0 ? active : view.domains
  return [
    ...domains.filter((domain) => domain.kind === "PhysicalDevice"),
    ...domains.filter((domain) => domain.kind !== "PhysicalDevice"),
  ]
}

/**
 * Full-capacity bar layout for one memory domain: model segments anchored left, free headroom in
 * the middle, and other apps anchored right. Widths are percentages of `domain.totalBytes`.
 */
export const memoryDomainBarLayout = (domain: HardwareMemoryDomainView): MemoryDomainBarLayout => {
  const leftSegments: readonly MemoryBarSegment[] =
    domain.weightsBytes === null || domain.kvCacheBytes === null || domain.overheadBytes === null
      ? []
      : [
          { key: "weights", label: "Weights", bytes: domain.weightsBytes, widthPercent: capacityPercent(domain.weightsBytes, domain.totalBytes) },
          { key: "cache", label: "Context", bytes: domain.kvCacheBytes, widthPercent: capacityPercent(domain.kvCacheBytes, domain.totalBytes) },
          { key: "runtime", label: "Overhead", bytes: domain.overheadBytes, widthPercent: capacityPercent(domain.overheadBytes, domain.totalBytes) },
        ]
  const modelBytes = leftSegments.length === 0
    ? null
    : leftSegments.reduce((sum, segment) => sum + segment.bytes, 0)
  const rightSegment: MemoryBarSegment | null = domain.systemAndAppsBytes === null
    ? null
    : {
        key: "other",
        label: "Other apps",
        bytes: domain.systemAndAppsBytes,
        widthPercent: capacityPercent(domain.systemAndAppsBytes, domain.totalBytes),
      }
  return {
    domain,
    modelBytes,
    leftSegments,
    rightSegment,
    freeBytes: domain.freeBytes,
  }
}
