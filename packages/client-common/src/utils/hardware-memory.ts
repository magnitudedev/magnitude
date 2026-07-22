import { Option } from "effect"
import type { LocalInferenceHardware, LocalInferenceMemoryDomainId } from "@magnitudedev/sdk"

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
  readonly fixedBytes: number | null
  readonly kvCacheBytes: number | null
  readonly systemAndAppsBytes: number | null
  readonly freeBytes: number | null
  readonly status: HardwareMemoryBreakdownStatus
  readonly notice: string | null
  readonly participatesInRuntime: boolean
}

export interface HardwareMemoryView {
  readonly domains: readonly HardwareMemoryDomainView[]
  readonly compact: { readonly usedBytes: number; readonly totalBytes: number } | null
}

const MIB = 1024 ** 2

const domainLabel = (
  hardware: LocalInferenceHardware,
  domain: LocalInferenceHardware["memoryDomains"][number],
  physicalOrdinal: number,
): string => {
  if (domain.kind === "System") return "System memory"
  const accelerators = hardware.accelerators.filter((accelerator) =>
    accelerator.memoryDomainId === domain.memoryDomainId)
  const name = accelerators.map(({ name }) => name).join(" + ")
  return domain.kind === "UnifiedMemory"
    ? `${name || Option.getOrElse(hardware.processor, () => "System")} · Unified memory`
    : `${name || "Accelerator"} · GPU ${physicalOrdinal}`
}

export interface HardwareMemoryViewOptions {
  readonly participatingDomainIds?: readonly LocalInferenceMemoryDomainId[]
  readonly fallbackToAccelerators?: boolean
}

export const deriveHardwareMemoryView = (
  hardware: LocalInferenceHardware,
  options: HardwareMemoryViewOptions = {},
): HardwareMemoryView => {
  const allocations = new Map(Option.match(hardware.residentMemory, {
    onNone: () => [],
    onSome: ({ domains }) => domains.map((domain) => [domain.memoryDomainId, domain] as const),
  }))
  const explicitFallbackIds = Option.isSome(hardware.residentMemory)
    ? []
    : options.participatingDomainIds ?? []
  const acceleratorDomainIds = hardware.accelerators.map(({ memoryDomainId }) => memoryDomainId)
  const fallbackRuntimeDomainIds = new Set(explicitFallbackIds.length > 0
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
    const participatesInRuntime = fallbackRuntimeDomainIds.has(domain.memoryDomainId)
      || Option.exists(allocation, (resident) =>
        resident.modelBytes + resident.contextBytes + resident.computeBytes + resident.auxiliaryBytes > 0)
    const freeBytes = Option.getOrNull(Option.map(domain.availableBytes, (available) =>
      Math.min(domain.totalBytes, Math.max(0, available))))
    const usedBytes = freeBytes === null ? null : domain.totalBytes - freeBytes
    const fixedBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.modelBytes + resident.computeBytes + resident.auxiliaryBytes,
    })
    const kvCacheBytes = Option.match(allocation, {
      onNone: () => 0,
      onSome: (resident) => resident.contextBytes,
    })
    const base = {
      id: domain.memoryDomainId,
      label: domainLabel(hardware, domain, physicalOrdinal),
      kind: domain.kind,
      totalBytes: domain.totalBytes,
      participatesInRuntime,
    }
    if (freeBytes === null || usedBytes === null) return {
      ...base,
      usedBytes: null,
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
      fixedBytes,
      kvCacheBytes,
      systemAndAppsBytes: Math.max(0, displayedUsedBytes - ownedBytes),
      freeBytes: domain.totalBytes - displayedUsedBytes,
      status: adjusted ? "rounding_adjusted" : "complete",
      notice: null,
    }
  })
  const participating = domains.filter((domain) => domain.participatesInRuntime && domain.usedBytes !== null)
  return {
    domains,
    compact: participating.length === 0 ? null : {
      usedBytes: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
      totalBytes: participating.reduce((sum, domain) => sum + domain.totalBytes, 0),
    },
  }
}
