import type { LocalInferenceHostProfile } from "@magnitudedev/sdk"

const MIB = 1024 ** 2

export type HardwareMemoryBreakdownStatus =
  | "complete"
  | "rounding_adjusted"
  | "inconsistent"
  | "unavailable"
  | "missing_free"

export interface HardwareMemoryDomainView {
  readonly id: string
  readonly label: string
  readonly kind: LocalInferenceHostProfile["memoryDomains"][number]["kind"]
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
  readonly compact: {
    readonly usedBytes: number
    readonly totalBytes: number
  } | null
}

const physicalDomainLabel = (
  domain: LocalInferenceHostProfile["memoryDomains"][number],
  physicalOrdinal: number,
): string => {
  const name = domain.deviceNames[0] ?? "Accelerator"
  return `${name} · GPU ${physicalOrdinal}`
}

const domainLabel = (
  host: LocalInferenceHostProfile,
  domain: LocalInferenceHostProfile["memoryDomains"][number],
  physicalOrdinal: number,
): string => {
  switch (domain.kind) {
    case "physical_device":
      return physicalDomainLabel(domain, physicalOrdinal)
    case "unified_memory":
      return `${domain.deviceNames[0] ?? host.cpuModel ?? "System"} · Unified memory`
    case "system":
      return "System memory"
  }
}

export interface HardwareMemoryViewOptions {
  readonly participatingDomainIds?: readonly string[]
  readonly fallbackToAccelerators?: boolean
}

export const deriveHardwareMemoryView = (
  host: LocalInferenceHostProfile,
  options: HardwareMemoryViewOptions = {},
): HardwareMemoryView => {
  const allocations = new Map(
    host.residentMemory !== null
      ? host.residentMemory.domains.map((domain) => [domain.memoryDomainId, domain] as const)
      : [],
  )
  const explicitFallbackIds = host.residentMemory !== null
    ? []
    : options.participatingDomainIds ?? []
  const shouldUseAcceleratorFallback = host.residentMemory === null
    && options.fallbackToAccelerators !== false
  const fallbackRuntimeDomainIds = explicitFallbackIds.length > 0
    ? new Set(explicitFallbackIds)
    : shouldUseAcceleratorFallback
      ? new Set((() => {
        const accelerators = host.memoryDomains.filter((domain) =>
          domain.kind === "physical_device" || domain.kind === "unified_memory")
        return (accelerators.length > 0 ? accelerators : host.memoryDomains)
          .map((domain) => domain.id)
      })())
      : new Set<string>()
  let physicalOrdinal = 0
  const domains = host.memoryDomains.map((domain): HardwareMemoryDomainView => {
    if (domain.kind === "physical_device") physicalOrdinal += 1
    const allocation = allocations.get(domain.id)
    const participatesInRuntime = fallbackRuntimeDomainIds.has(domain.id) || (
      allocation !== undefined
      && allocation.modelBytes + allocation.contextBytes + allocation.computeBytes + allocation.auxiliaryBytes > 0
    )
    const freeBytes = domain.currentFreeBytes === null
      ? null
      : Math.min(domain.totalCapacityBytes, Math.max(0, domain.currentFreeBytes))
    const usedBytes = freeBytes === null ? null : domain.totalCapacityBytes - freeBytes
    const attributionUnavailable = host.residentMemory === null

    if (freeBytes === null || usedBytes === null) {
      return {
        id: domain.id,
        label: domainLabel(host, domain, physicalOrdinal),
        kind: domain.kind,
        totalBytes: domain.totalCapacityBytes,
        usedBytes: null,
        fixedBytes: allocation
          ? allocation.modelBytes + allocation.computeBytes + allocation.auxiliaryBytes
          : null,
        kvCacheBytes: allocation?.contextBytes ?? null,
        systemAndAppsBytes: null,
        freeBytes: null,
        status: "missing_free",
        notice: "Current memory usage is unavailable.",
        participatesInRuntime,
      }
    }

    if (attributionUnavailable) {
      return {
        id: domain.id,
        label: domainLabel(host, domain, physicalOrdinal),
        kind: domain.kind,
        totalBytes: domain.totalCapacityBytes,
        usedBytes,
        fixedBytes: null,
        kvCacheBytes: null,
        systemAndAppsBytes: null,
        freeBytes,
        status: "unavailable",
        notice: null,
        participatesInRuntime,
      }
    }

    const fixedBytes = allocation
      ? allocation.modelBytes + allocation.computeBytes + allocation.auxiliaryBytes
      : 0
    const kvCacheBytes = allocation?.contextBytes ?? 0
    const ownedBytes = fixedBytes + kvCacheBytes
    const excessBytes = ownedBytes - usedBytes
    const toleranceBytes = Math.max(64 * MIB, domain.totalCapacityBytes * 0.001)

    if (excessBytes > toleranceBytes) {
      return {
        id: domain.id,
        label: domainLabel(host, domain, physicalOrdinal),
        kind: domain.kind,
        totalBytes: domain.totalCapacityBytes,
        usedBytes,
        fixedBytes: null,
        kvCacheBytes: null,
        systemAndAppsBytes: null,
        freeBytes,
        status: "inconsistent",
        notice: "Allocation and resident usage could not be reconciled.",
        participatesInRuntime,
      }
    }

    const adjusted = excessBytes > 0
    const displayedUsedBytes = adjusted ? ownedBytes : usedBytes
    return {
      id: domain.id,
      label: domainLabel(host, domain, physicalOrdinal),
      kind: domain.kind,
      totalBytes: domain.totalCapacityBytes,
      usedBytes: displayedUsedBytes,
      fixedBytes,
      kvCacheBytes,
      systemAndAppsBytes: Math.max(0, displayedUsedBytes - ownedBytes),
      freeBytes: domain.totalCapacityBytes - displayedUsedBytes,
      status: adjusted ? "rounding_adjusted" : "complete",
      notice: null,
      participatesInRuntime,
    }
  })

  const participating = domains.filter((domain) =>
    domain.participatesInRuntime && domain.usedBytes !== null)
  const compact = participating.length === 0
    ? null
    : {
        usedBytes: participating.reduce((sum, domain) => sum + (domain.usedBytes ?? 0), 0),
        totalBytes: participating.reduce((sum, domain) => sum + domain.totalBytes, 0),
      }

  return { domains, compact }
}
