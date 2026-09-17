import { Option, Schema } from "effect"
import type { LocalInferenceHardware } from "@magnitudedev/sdk"
import type { MachineIdentityObservation } from "@magnitudedev/sdk/desktop-host"
import { formatMemorySize } from "@magnitudedev/client-common"
import { hardwarePresentation, normalizeGpuName, normalizeHardwareName } from "./hardware-photos"
import facts from "../../assets/hardware/facts.json"

const HardwareFact = Schema.Struct({ label: Schema.NonEmptyString, value: Schema.NonEmptyString })
const CatalogEntry = Schema.Struct({
  names: Schema.NonEmptyArray(Schema.NonEmptyString), target: Schema.Literal("Processor", "Accelerator", "Device"),
  facts: Schema.NonEmptyArray(HardwareFact), source: Schema.NonEmptyString,
  additionalSources: Schema.optionalWith(Schema.Array(Schema.NonEmptyString), { as: "Option", exact: true }),
  variants: Schema.optionalWith(Schema.Array(Schema.Struct({
    physicalCpuCores: Schema.Int.pipe(Schema.positive()), facts: Schema.NonEmptyArray(HardwareFact),
  })), { as: "Option", exact: true }),
  memoryVariants: Schema.optionalWith(Schema.Array(Schema.Struct({
    memoryGiB: Schema.Int.pipe(Schema.positive()), facts: Schema.NonEmptyArray(HardwareFact),
  })), { as: "Option", exact: true }),
})
export const hardwareFactsCatalog = Schema.decodeUnknownSync(Schema.Array(CatalogEntry))(facts)
/** Remove OS brand-string decoration while retaining the complete processor SKU. */
export const normalizeProcessorName = (name: string) => normalizeHardwareName(name)
  .replace(/\((?:r|tm)\)|[®™]/g, "")
  .replace(/^\d+(?:st|nd|rd|th) gen\s+/, "")
  .replace(/\s+@\s+\d+(?:\.\d+)?\s*ghz$/, "")
  .replace(/\s+(?:with|w\/) radeon .*$/, "")
  .replace(/\s+\d+-core processor$/, "")
  .replace(/\s+processor$/, "")
  .replace(/\s+/g, " ").trim()
const findHardwareFacts = (target: typeof CatalogEntry.Type.target, name: string) => {
  const normalize = target === "Accelerator" ? normalizeGpuName : target === "Processor" ? normalizeProcessorName : normalizeHardwareName
  return hardwareFactsCatalog.find(entry => entry.target === target && entry.names.some(candidate => normalize(candidate) === normalize(name)))
}
const publishedFacts = (target: typeof CatalogEntry.Type.target, name: string, physicalCores: Option.Option<number> = Option.none(), dedicatedMemoryBytes: Option.Option<number> = Option.none()) => {
  const entry = findHardwareFacts(target, name)
  if (!entry) return []
  const variant = Option.flatMap(physicalCores, cores => Option.flatMap(entry.variants, variants =>
    Option.fromNullable(variants.find(variant => variant.physicalCpuCores === cores))))
  const common = entry.facts.filter(fact => Option.isNone(physicalCores) || fact.label !== "CPU core options (spec)")
  // Physical device totals can exclude small driver reservations. Require a unique
  // capacity within 1%; never use free memory, allocator budgets, or shared RAM.
  const memoryVariants = Option.match(dedicatedMemoryBytes, {
    onNone: () => [],
    onSome: bytes => Option.getOrElse(entry.memoryVariants, () => []).filter(variant =>
      Math.abs(bytes - variant.memoryGiB * 1024 ** 3) <= variant.memoryGiB * 1024 ** 3 * 0.01),
  })
  return [...common, ...Option.match(variant, { onNone: () => [], onSome: value => value.facts }),
    ...(memoryVariants.length === 1 ? memoryVariants[0]!.facts : [])]
}

/** A view of existing observations and exact published specifications; no probing or benchmarking. */
export const hardwareDetails = (identity: MachineIdentityObservation | null, hardware: LocalInferenceHardware) => {
  const domainFor = (id: string) => hardware.memoryDomains.find(domain => domain.memoryDomainId === id)
  const dedicated = hardware.accelerators.filter(accelerator => {
    const domain = domainFor(accelerator.memoryDomainId)
    return domain?.kind === "PhysicalDevice" && !domain.sharesSystemMemory
  })
  const presentation = hardwarePresentation(identity, dedicated.map(accelerator => accelerator.name), hardware.processor)
  const unified = hardware.memoryDomains.some(domain => domain.kind === "UnifiedMemory" && domain.sharesSystemMemory)
  const processorFacts = Option.match(hardware.processor, { onNone: () => [], onSome: name => publishedFacts("Processor", name, hardware.physicalCores) })
  const fixedCoreCount = processorFacts.find(fact => fact.label === "CPU cores / processor (spec)")
  const cpuCores = Option.match(hardware.physicalCores, {
    onSome: cores => [{ label: "CPU cores", value: String(cores) }],
    // A per-processor catalog count cannot establish a multi-socket server's total.
    onNone: () => fixedCoreCount && presentation.category !== "Server" ? [{ label: "CPU cores", value: fixedCoreCount.value }] : [],
  })
  const summary = [
    { label: unified ? "Unified memory" : "System RAM", value: formatMemorySize(hardware.totalSystemMemoryBytes) },
    ...cpuCores,
    ...processorFacts.filter(fact => !fact.label.startsWith("CPU core")),
    ...Option.match(presentation.deviceId, { onNone: () => [], onSome: id => publishedFacts("Device", id) }),
  ]
  // One memory total per domain, even when multiple accelerator APIs expose that device.
  const displayedDomains = new Set<string>()
  const accelerators = hardware.accelerators.map(accelerator => {
    const domain = domainFor(accelerator.memoryDomainId)
    const shared = domain?.sharesSystemMemory === true
    const showMemory = !displayedDomains.has(accelerator.memoryDomainId) && !shared && domain?.kind === "PhysicalDevice" && domain.totalBytes > 0
    displayedDomains.add(accelerator.memoryDomainId)
    return {
      id: accelerator.acceleratorId, name: accelerator.name,
      detail: showMemory ? `${formatMemorySize(domain.totalBytes)} VRAM` : shared ? (unified ? "Shares unified memory" : "Shares system RAM") : "Local acceleration",
      facts: publishedFacts("Accelerator", accelerator.name, Option.none(),
        !shared && domain?.kind === "PhysicalDevice" && domain.totalBytes > 0 ? Option.some(domain.totalBytes) : Option.none()),
    }
  })
  const processorName = Option.getOrElse(hardware.processor, () => "Unknown processor")
  const integratedChip = /^(Apple M[1-9]\d*\b|NVIDIA GB10$)/i.test(processorName)
  const visibleAccelerators = accelerators.filter(accelerator => !integratedChip || !hardware.accelerators.some(observed =>
    observed.acceleratorId === accelerator.id && domainFor(observed.memoryDomainId)?.sharesSystemMemory === true &&
    normalizeHardwareName(observed.name) === normalizeHardwareName(processorName)))
  const memoryFacts = summary.filter(fact => fact.label.includes("memory") || fact.label === "System RAM" || fact.label === "Memory bandwidth (spec)")
  const processorDetails = summary.filter(fact => !memoryFacts.includes(fact) && fact.label !== "Neural Engine cores (spec)")
  const hiddenAcceleratorFactLabels = new Set(["CUDA cores (spec)", "GPU compute units (spec)", "Xe cores (spec)", "Stream processors (spec)", "GPU architecture (spec)"])
  const groups = [
    { label: integratedChip ? "Chip" : "CPU", name: processorName, truncateName: !findHardwareFacts("Processor", processorName), details: processorDetails.map(fact => `${fact.value} ${fact.label}`) },
    { label: "Memory", name: memoryFacts[0]?.value ?? "Unknown", truncateName: false, details: memoryFacts.map((fact, index) => index === 0 ? fact.label : `${fact.value} ${fact.label}`) },
    ...visibleAccelerators.map((accelerator, index) => ({
      label: visibleAccelerators.length > 1 ? `GPU ${index + 1}` : "GPU", name: accelerator.name, truncateName: !findHardwareFacts("Accelerator", accelerator.name),
      details: [accelerator.detail, ...accelerator.facts.filter(fact => !hiddenAcceleratorFactLabels.has(fact.label)).map(fact => `${fact.value} ${fact.label}`)],
    })),
  ]
  return { ...presentation, summary, accelerators, groups, cpuInference: hardware.accelerators.length === 0 }
}
