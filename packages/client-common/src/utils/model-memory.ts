import { Option } from "effect"
import type {
  LocalInferenceHardware,
  LocalModelCatalogCandidate,
  MemoryAssessment,
  ProviderModelCatalogEntry,
} from "@magnitudedev/sdk"

export interface ModelMemoryConditions {
  readonly exceedsCapacity: boolean
  readonly belowWarningReserve: boolean
  readonly lacksCurrentHeadroom: boolean
  readonly evidenceUnavailable: boolean
}

const lacksCurrentHeadroom = (
  requiredBytes: number,
  hardware: LocalInferenceHardware,
  systemDomains: ReadonlySet<MemoryAssessment["memoryDomainId"]>,
): boolean => {
  const reclaimableBytes = Option.match(hardware.residentMemory, {
    onNone: () => 0,
    onSome: ({ domains }) => domains
      .filter(({ memoryDomainId }) => systemDomains.has(memoryDomainId))
      .reduce(
        (total, domain) => total
          + domain.modelBytes
          + domain.contextBytes
          + domain.computeBytes
          + domain.auxiliaryBytes,
        0,
      ),
  })
  return Math.min(
    hardware.totalSystemMemoryBytes,
    hardware.availableSystemMemoryBytes + reclaimableBytes,
  ) <= requiredBytes + hardware.abortReserveBytes
}

export const requiredMemoryBytes = (
  memory: readonly MemoryAssessment[],
): number => memory.reduce((total, domain) => total + domain.requiredBytes, 0)

export const memoryCapacityBytes = (
  memory: readonly MemoryAssessment[],
): number => memory.reduce((total, domain) => total + domain.capacityBytes, 0)

const systemMemoryRequirement = (
  memory: readonly MemoryAssessment[],
  systemDomains: ReadonlySet<MemoryAssessment["memoryDomainId"]>,
): number | undefined => {
  const domains = memory.filter(({ memoryDomainId }) => systemDomains.has(memoryDomainId))
  return domains.length === 0
    ? undefined
    : domains.reduce((total, { requiredBytes }) => total + requiredBytes, 0)
}

const isWithinWarningReserve = (
  memory: readonly MemoryAssessment[],
): boolean => memory.every(
  ({ capacityBytes, requiredBytes, warningReserveBytes }) =>
    requiredBytes <= Math.max(0, capacityBytes - warningReserveBytes),
)

const exceedsCompatibleCapacity = (
  memory: readonly MemoryAssessment[],
): boolean => memory.some(
  ({ capacityBytes, requiredBytes, compatibilityReserveBytes }) =>
    requiredBytes > Math.max(0, capacityBytes - compatibilityReserveBytes),
)

const assessedMemoryConditions = (
  memory: readonly MemoryAssessment[],
  hardware: LocalInferenceHardware | undefined,
): ModelMemoryConditions => {
  const systemDomains = new Set(hardware?.memoryDomains
    .filter(({ sharesSystemMemory }) => sharesSystemMemory)
    .map(({ memoryDomainId }) => memoryDomainId))
  const systemRequiredBytes = systemMemoryRequirement(memory, systemDomains)
  return {
    exceedsCapacity: exceedsCompatibleCapacity(memory),
    belowWarningReserve: !isWithinWarningReserve(memory),
    lacksCurrentHeadroom: hardware !== undefined
      && systemRequiredBytes !== undefined
      && lacksCurrentHeadroom(systemRequiredBytes, hardware, systemDomains),
    evidenceUnavailable: hardware === undefined || systemRequiredBytes === undefined,
  }
}

export const providerModelMemoryConditions = (
  model: ProviderModelCatalogEntry,
  hardware: LocalInferenceHardware | undefined,
): ModelMemoryConditions => {
  return Option.match(model.memory, {
    onNone: () => ({
      exceedsCapacity: false,
      belowWarningReserve: false,
      lacksCurrentHeadroom: false,
      evidenceUnavailable: true,
    }),
    onSome: (memory) => assessedMemoryConditions(memory, hardware),
  })
}

export const catalogCandidateMemoryConditions = (
  candidate: LocalModelCatalogCandidate,
  hardware: LocalInferenceHardware | undefined,
): ModelMemoryConditions => assessedMemoryConditions(candidate.memory, hardware)

export const modelMemoryStatusLabel = ({
  exceedsCapacity,
  lacksCurrentHeadroom,
  belowWarningReserve,
  evidenceUnavailable,
}: ModelMemoryConditions): string =>
  evidenceUnavailable
    ? "Unavailable"
    : exceedsCapacity
      ? "Too large"
      : lacksCurrentHeadroom
        ? "Free memory"
        : belowWarningReserve
          ? "Tight fit"
          : ""

export const modelMemoryStatusDetail = ({
  exceedsCapacity,
  lacksCurrentHeadroom,
  belowWarningReserve,
  evidenceUnavailable,
}: ModelMemoryConditions): string =>
  evidenceUnavailable
    ? "Memory information is unavailable"
    : exceedsCapacity
      ? "Requires more memory than this system has"
      : lacksCurrentHeadroom
        ? "Not enough memory available - close memory-intensive apps"
        : belowWarningReserve
          ? "High memory use"
          : ""
