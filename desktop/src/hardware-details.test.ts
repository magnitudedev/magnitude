import { Option, Schema } from "effect"
import { expect, it } from "vitest"
import { hardwareDetails, hardwareFactsCatalog, normalizeProcessorName } from "./hardware-details"
import { hardware, identity, hardwareScenarios } from "../test/hardware/fixtures"
import inventory from "../../assets/hardware/inventory.json"
import coverageJson from "../../assets/hardware/coverage.json"
import { normalizeGpuName } from "./hardware-photos"

const coverage = Schema.decodeUnknownSync(Schema.Array(Schema.Struct({
  photoId: Schema.NonEmptyString, processors: Schema.Array(Schema.NonEmptyString),
  accelerators: Schema.Array(Schema.NonEmptyString), sources: Schema.NonEmptyArray(Schema.NonEmptyString),
  configurationNote: Schema.NonEmptyString,
})))(coverageJson)

it("audits every photo group and every researched processor configuration", () => {
  expect(coverage.map(row => row.photoId).sort()).toEqual(inventory.map(row => row.id).sort())
  for (const group of coverage) {
    expect(group.processors.length + group.accelerators.length).toBeGreaterThan(0)
    for (const name of group.processors) {
      const entry = hardwareFactsCatalog.find(entry => entry.target === "Processor" && entry.names.some(alias => normalizeProcessorName(alias) === normalizeProcessorName(name)))
      expect(entry, `${group.photoId}: ${name}`).toBeDefined()
      expect(entry?.facts.some(fact => /^CPU core/.test(fact.label)), `${name}: missing CPU core information`).toBe(true)
    }
    for (const name of group.accelerators) {
      expect(hardwareFactsCatalog.some(entry => entry.target === "Accelerator" && entry.names.some(alias => normalizeGpuName(alias) === normalizeGpuName(name))), `${group.photoId}: ${name}`).toBe(true)
    }
  }
})
it("keeps catalog aliases and physical-core variant keys unambiguous", () => {
  const identities = new Set<string>()
  for (const entry of hardwareFactsCatalog) {
    const normalize = entry.target === "Processor" ? normalizeProcessorName : entry.target === "Accelerator" ? normalizeGpuName : (value: string) => value
    for (const name of new Set(entry.names.map(normalize))) {
      const key = `${entry.target}:${name}`
      expect(identities.has(key), key).toBe(false)
      identities.add(key)
    }
    const variants = Option.getOrElse(entry.variants, () => [])
    expect(new Set(variants.map(row => row.physicalCpuCores)).size).toBe(variants.length)
    const memoryVariants = Option.getOrElse(entry.memoryVariants, () => [])
    expect(new Set(memoryVariants.map(row => row.memoryGiB)).size).toBe(memoryVariants.length)
  }
})

it.each([
  ["RTX 3060", 8, "240 GB/s"], ["RTX 3060", 12, "360 GB/s"],
  ["Arc A770", 8, "Up to 512 GB/s"], ["Arc A770", 16, "Up to 560 GB/s"],
  ["RTX A2000 Laptop GPU", 4, "Up to 192 GB/s"], ["RTX A2000 Laptop GPU", 8, "Up to 224 GB/s"],
  ["RTX 3050 Laptop GPU", 4, "Up to 192 GB/s"], ["RTX 3050 Laptop GPU", 6, "Up to 168 GB/s"],
])("selects %s bandwidth for %s GB physical VRAM", (name, memory, value) => {
  const result = hardwareDetails(null, hardware("Unknown", 64, 8, [{ name, memory }]))
  expect(result.accelerators[0]?.facts).toContainEqual({ label: "Memory bandwidth (spec)", value })
})
it("uses dedicated physical capacity, allowing small reservations but not budgets or shared memory", () => {
  const base = hardware("Unknown", 64, 8, [{ name: "RTX 3060", memory: 12 }])
  const value = { ...base, memoryDomains: base.memoryDomains.map(domain => domain.kind === "PhysicalDevice"
    ? { ...domain, totalBytes: 12 * 1024 ** 3 - 64 * 1024 ** 2, stableCapacityBytes: 8 * 1024 ** 3 } : domain) }
  expect(hardwareDetails(null, value).accelerators[0]?.facts).toContainEqual({ label: "Memory bandwidth (spec)", value: "360 GB/s" })
  for (const gpu of [{ memory: 0 }, { memory: 10 }, { memory: 11.5 }, { memory: 12, shared: true }]) {
    expect(hardwareDetails(null, hardware("Unknown", 64, 8, [{ name: "RTX 3060", ...gpu }])).accelerators[0]?.facts.some(f => f.label === "Memory bandwidth (spec)")).toBe(false)
  }
})
it.each([
  ["RTX 4090", 24, "1,008 GB/s"], ["RTX 4090 Laptop GPU", 16, "Up to 576 GB/s"],
  ["A100-SXM4-40GB", 40, "1,555 GB/s"], ["A100-PCIE-40GB", 40, "1,555 GB/s"],
  ["A100-SXM4-80GB", 80, "2,039 GB/s"], ["A100-PCIE-80GB", 80, "1,935 GB/s"],
])("keeps bandwidth specific to %s", (name, memory, value) => {
  expect(hardwareDetails(null, hardware("Unknown", 128, 8, [{ name, memory }])).accelerators[0]?.facts)
    .toContainEqual({ label: "Memory bandwidth (spec)", value })
})
it("covers bandwidth for every cataloged discrete GPU without inventing dedicated bandwidth for shared Radeon graphics", () => {
  for (const entry of hardwareFactsCatalog.filter(entry => entry.target === "Accelerator")) {
    const values = [...entry.facts, ...Option.getOrElse(entry.memoryVariants, () => []).flatMap(v => v.facts)]
    const sharedRadeon = entry.names.some(name => /^Radeon (?:80[456]0S|7[468]0M)/.test(name))
    expect(values.some(f => f.label === "Memory bandwidth (spec)"), entry.names[0]).toBe(!sharedRadeon)
  }
})

it("shows one CPU core count, preferring detection and never substituting threads", () => {
  const value = { ...hardware("Intel Core i7-13700H", 32, 4), physicalCores: Option.some(12) }
  const result = hardwareDetails(null, value)
  expect(result.summary.filter(f => /CPU|threads/.test(f.label))).toEqual([{ label: "CPU cores", value: "12" }])
  expect(hardwareDetails(null, hardware("Unknown", 32, 4)).summary.some(f => /CPU|threads/.test(f.label))).toBe(false)
  expect(hardwareDetails(null, hardware("Apple M4 Pro", 32, 12)).summary.some(f => /CPU core/.test(f.label))).toBe(false)
})
it("does not use a per-processor specification as a server total", () => {
  expect(hardwareDetails(identity("Custom", "Server", "Server"), hardware("AMD EPYC 7742", 256, 128)).summary.some(f => f.label === "CPU cores")).toBe(false)
})
it("matches decorated CPU names without collapsing different SKUs", () => {
  expect(normalizeProcessorName("13th Gen Intel(R) Core(TM) i7-13700H @ 2.40GHz")).toBe("intel core i7-13700h")
  expect(normalizeProcessorName("AMD Ryzen 7 7840U w/ Radeon 780M Graphics")).toBe("amd ryzen 7 7840u")
  const known = hardwareDetails(null, hardware("13th Gen Intel(R) Core(TM) i7-13700H", 32, 20))
  expect(known.summary).toContainEqual({ label: "CPU cores", value: "14" })
  const unknown = hardwareDetails(null, hardware("Intel Core i7-13700HK", 32, 20))
  expect(unknown.summary.some(f => f.label.endsWith("(spec)"))).toBe(false)
})

it.each(hardwareScenarios)("presents $label from observed data", scenario => {
  const result = hardwareDetails(scenario.identity, scenario.value)
  expect(result.summary.some(f => /threads/.test(f.label))).toBe(false)
  expect(result.summary.filter(f => f.label === "CPU cores").length).toBeLessThanOrEqual(1)
  expect(result.summary[0]?.label).toBe(scenario.value.memoryDomains[0]?.kind === "UnifiedMemory" ? "Unified memory" : "System RAM")
  expect(result.cpuInference).toBe(scenario.value.accelerators.length === 0)
  expect(Option.isSome(result.photo)).toBe(true)
})
it("keeps laptop identity with discrete or external graphics", () => {
  const result = hardwareDetails(identity("Dell", "XPS 15 9530", "Portable"), hardware("Intel Core i7-13700H", 32, 16, [{ name: "RTX 5090", memory: 32 }]))
  expect(Option.getOrThrow(result.photo).kind).toBe("Device")
  expect(result.accelerators[0]?.detail).toBe("32 GB VRAM")
})
it("never substitutes a graphics card for an unknown portable or all-in-one", () => {
  for (const formFactor of ["Portable", "AllInOne"] as const) {
    const result = hardwareDetails({ _tag: "Unavailable", formFactor }, hardware("Intel Core i7-13700H", 32, 16, [{ name: "RTX 5090", memory: 32 }]))
    expect(result.photo).toEqual(Option.none())
  }
})
it("does not treat shared GPU memory as VRAM or select a component photo", () => {
  const result = hardwareDetails(null, hardware("Test", 64, 16, [{ name: "RTX 5090", memory: 64, shared: true }]))
  expect(result.photo).toEqual(Option.none())
  expect(result.accelerators[0]?.detail).toBe("Shares unified memory")
})
it("keeps multi-GPU memory separate and handles duplicate names", () => {
  const result = hardwareDetails(null, hardware("AMD", 128, 32, [{ name: "RTX 5090", memory: 32 }, { name: "RTX 5090", memory: 32 }]))
  expect(result.summary[0]?.value).toBe("128 GB")
  expect(result.accelerators.map(gpu => gpu.detail)).toEqual(["32 GB VRAM", "32 GB VRAM"])
  expect(new Set(result.accelerators.map(gpu => gpu.id)).size).toBe(2)
})
it("never infers variable Apple chip specifications or a desktop SKU from a mobile suffix", () => {
  const result = hardwareDetails(null, hardware("Apple M4 Max", 64, 16, [{ name: "RTX 4090 Laptop GPU", memory: 16 }]))
  expect(result.summary.some(fact => fact.label === "GPU cores (spec)" || fact.label === "Memory bandwidth (spec)")).toBe(false)
  expect(result.accelerators[0]?.facts).toContainEqual({ label: "CUDA cores (spec)", value: "9,728" })
})
it("resolves Apple bins only from physical cores, never scheduling parallelism", () => {
  const limited = { ...hardware("Apple M4 Max", 64, 4), physicalCores: Option.some(16) }
  expect(hardwareDetails(null, limited).summary).toContainEqual({ label: "GPU cores (spec)", value: "40" })
  expect(hardwareDetails(null, limited).summary).toContainEqual({ label: "Memory bandwidth (spec)", value: "546 GB/s" })
  const ambiguous = { ...hardware("Apple M5 Max", 64, 18), physicalCores: Option.some(18) }
  expect(hardwareDetails(null, ambiguous).summary.some(fact => fact.label === "GPU cores (spec)" || fact.label === "Memory bandwidth (spec)")).toBe(false)
})
it("does not promote a chip into a particular mini PC", () => {
  const result = hardwareDetails(null, hardware("NVIDIA GB10", 128, 20, [{ name: "NVIDIA GB10", memory: 128, shared: true }]))
  expect(result.photo).toEqual(Option.none())
  expect(result.deviceId).toEqual(Option.none())
  expect(result.summary).toContainEqual({ label: "CPU cores", value: "20" })
})
it("requires exact fact identities and retains primary source URLs", () => {
  for (const entry of hardwareFactsCatalog) {
    expect(entry.source).toMatch(/^https:\/\//)
    expect(entry.facts.every(fact => fact.label.endsWith("(spec)"))).toBe(true)
  }
  const result = hardwareDetails(null, hardware("Unknown", 32, 8, [{ name: "RTX 5090 D", memory: 32 }]))
  expect(result.accelerators[0]?.facts).toEqual([])
})

it("counts dedicated memory once when two backends expose one physical domain", () => {
  const value = hardware("AMD", 64, 16, [{ name: "RTX 4090", memory: 24 }])
  const accelerator = value.accelerators[0]!
  const result = hardwareDetails(null, { ...value, accelerators: [accelerator, { ...accelerator, acceleratorId: "vulkan-alias" as typeof accelerator.acceleratorId, backend: "Vulkan" }] })
  expect(result.accelerators.map(gpu => gpu.detail)).toEqual(["24 GB VRAM", "Local acceleration"])
})
it("keeps ordinary shared system RAM distinct from unified memory in mixed systems", () => {
  const value = hardware("AMD", 64, 16, [{ name: "Integrated graphics", memory: 64, shared: true }, { name: "RTX 4090", memory: 24 }])
  const result = hardwareDetails(null, { ...value, memoryDomains: value.memoryDomains.map(domain => domain.memoryDomainId === "system" ? { ...domain, kind: "System" as const } : domain) })
  expect(result.summary[0]?.label).toBe("System RAM")
  expect(result.accelerators.map(gpu => gpu.detail)).toEqual(["Shares system RAM", "24 GB VRAM"])
  expect(Option.getOrThrow(result.photo).kind).toBe("Component")
})

it("groups an Apple chip once and retains distinct accelerators", () => {
  const result = hardwareDetails(null, hardware("Apple M4 Pro", 48, 14, [
    { name: "Apple M4 Pro", memory: 48, shared: true }, { name: "RTX 5090", memory: 32 },
  ], 14))
  expect(result.groups.map(group => group.label)).toEqual(["Chip", "Memory", "GPU"])
  expect(result.groups.filter(group => group.name === "Apple M4 Pro")).toHaveLength(1)
  expect(result.groups[0]?.details).toContain("14 CPU cores")
  expect(result.groups[0]?.details).toContain("20 GPU cores (spec)")
  expect(result.groups[1]?.details).toContain("Unified memory")
  expect(result.groups[2]?.name).toBe("RTX 5090")
})
it("labels PC components consistently and numbers separate GPUs", () => {
  const result = hardwareDetails(null, hardware("AMD Ryzen 9 9950X", 128, 32, [
    { name: "RTX 5090", memory: 32 }, { name: "RTX 5090", memory: 32 },
  ], 16))
  expect(result.groups.map(group => group.label)).toEqual(["CPU", "Memory", "GPU 1", "GPU 2"])
  expect(result.groups[0]?.name).toBe("AMD Ryzen 9 9950X")
  expect(result.groups[1]?.details).toEqual(["System RAM"])
  expect(result.groups[2]?.details[0]).toBe("32 GB VRAM")
  expect(result.groups[2]?.details).toEqual(["32 GB VRAM", "1,792 GB/s Memory bandwidth (spec)"])
})

it.each(["RTX 5090", "Radeon 8060S Graphics", "Arc B580", "RX 580"])("omits implementation-specific GPU unit counts from the %s card", name => {
  const result = hardwareDetails(null, hardware("Unknown", 64, 8, [{ name, memory: 16 }]))
  const group = result.groups.find(group => group.label === "GPU")!
  expect(group.details.some(detail => /CUDA cores|GPU compute units|Xe cores|Stream processors/.test(detail))).toBe(false)
  expect(group.details[0]).toBe("16 GB VRAM")
})
