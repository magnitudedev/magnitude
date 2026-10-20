import { Option } from "effect"
import {
  acceleratorBackend,
  acceleratorMemoryDomainIds,
  formatLocalInferenceBackend,
  formatMemorySize,
} from "@magnitudedev/client-common"
import type { LocalInferenceHardware, LocalInferenceMemoryDomainId } from "@magnitudedev/sdk"

export const formatContext = (tokens: number): string => tokens < 1_000
  ? String(tokens)
  : tokens % 1_024 === 0
    ? `${tokens / 1_024}K`
    : `${Math.round(tokens / 1_000)}K`

export interface LocalHardwarePresentation {
  readonly system: { readonly name: string; readonly details: readonly string[] }
  readonly accelerators: readonly { readonly name: string; readonly details: string }[]
}

const unique = (values: readonly string[]): string[] =>
  [...new Set(values.map((value) => value.trim()).filter(Boolean))]

const localHardwareTopology = (hardware: LocalInferenceHardware) => {
  const unified = hardware.memoryDomains.filter((domain) => domain.kind === "UnifiedMemory")
  const discrete = hardware.memoryDomains.filter((domain) => domain.kind === "PhysicalDevice")
  // The backend the engine runs on, when its selected executor places in the domain.
  const backendFor = (memoryDomainId: LocalInferenceMemoryDomainId) => Option.firstSomeOf(hardware.accelerators
    .filter(accelerator => acceleratorMemoryDomainIds(accelerator).has(memoryDomainId))
    .map(accelerator => Option.map(acceleratorBackend(hardware, accelerator), formatLocalInferenceBackend)))
  const namesFor = (memoryDomainId: LocalInferenceMemoryDomainId) => unique(hardware.accelerators
    .filter(accelerator => acceleratorMemoryDomainIds(accelerator).has(memoryDomainId))
    .map((accelerator) => accelerator.name))
  const unifiedBackend = Option.firstSomeOf(unified.map((domain) => backendFor(domain.memoryDomainId)))
  const unifiedAcceleratorNames = unique(unified.flatMap((domain) =>
    namesFor(domain.memoryDomainId)))
  const isAppleSilicon = hardware.platform === "MacOS" && hardware.architecture === "Arm64"
  const processorName = Option.getOrElse(hardware.processor, () =>
    isAppleSilicon ? "Apple Silicon" : "CPU")
  const productName = Option.getOrElse(hardware.productName, () => "")
  const systemName = isAppleSilicon
    ? processorName
    : unique([productName, unifiedAcceleratorNames.join(" + ")]).join(" · ") || processorName
  return { unified, discrete, backendFor, namesFor, unifiedBackend, systemName }
}

export const describeLocalHardware = (
  hardware: LocalInferenceHardware,
): LocalHardwarePresentation => {
  const {
    unified,
    discrete,
    backendFor,
    namesFor,
    unifiedBackend,
    systemName,
  } = localHardwareTopology(hardware)
  return {
    system: {
      name: systemName,
      details: [
        `${hardware.platform === "MacOS" ? "macOS" : hardware.platform} · ${hardware.architecture === "Arm64" ? "ARM64" : "x86-64"} · ${hardware.logicalCores} logical CPU core${hardware.logicalCores === 1 ? "" : "s"}`,
        `${formatMemorySize(hardware.totalSystemMemoryBytes)} ${unified.length > 0 ? "unified" : "system"} memory${Option.match(unifiedBackend, { onNone: () => "", onSome: backend => ` · ${backend} GPU acceleration` })}`,
      ],
    },
    accelerators: discrete.map((domain) => {
      const names = namesFor(domain.memoryDomainId)
      const backend = backendFor(domain.memoryDomainId)
      return {
        name: names.join(" + ") || `${Option.getOrElse(backend, () => "Local")} GPU`,
        details: `${formatMemorySize(domain.totalBytes)} VRAM${Option.match(backend, { onNone: () => "", onSome: backend => ` · ${backend} acceleration` })}`,
      }
    }),
  }
}
