import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema } from "effect"
import { Architecture, AssertionFailure, InfrastructureFailure, Target } from "./domain"
import { attestHost, HostObservation, ObservedGpu } from "./hardware"
import { checkedCommand, ProcessExecutor } from "./process"

const fail = (message: string) => new InfrastructureFailure({ operation: "host-inspection", message })
const decode = <A, I>(schema: Schema.Schema<A, I>, value: unknown) => Schema.decodeUnknown(schema)(value).pipe(Effect.mapError(() => fail("Malformed native hardware report")))
const optionalString = Schema.optionalWith(Schema.String, { as: "Option", exact: true })
const architecture = (value: string) => decode(Architecture, ({ x86_64: "x64", amd64: "x64", aarch64: "arm64", arm64: "arm64" } as Record<string, string>)[value.toLowerCase()])

/** Parse data, never source os-release as shell code. Only explicitly supported distributions qualify. */
const releaseValues = (wire: string) => {
  const values = new Map<string, string>()
  for (const line of wire.split(/\r?\n/)) {
    const match = /^([A-Z_]+)=(.*)$/.exec(line.trim())
    if (!match) continue
    let value = match[2]!
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1)
    values.set(match[1]!, value)
  }
  return values
}
export const linuxIdentity = (wire: string, dgx: string = "") => Effect.gen(function* () {
  const values = releaseValues(wire)
  if (dgx) {
    const metadata = releaseValues(dgx)
    const version = metadata.get("DGX_OTA_VERSION") ?? metadata.get("DGX_SWBUILD_VERSION")
    if (!version || !/^\d+(\.\d+)*$/.test(version)) return yield* fail("Unidentified DGX OS version")
    return { os: "dgx-os" as const, version }
  }
  const distributions: Readonly<Record<string, "ubuntu" | "debian" | "fedora" | "redhat">> = { ubuntu: "ubuntu", debian: "debian", fedora: "fedora", rhel: "redhat" }
  const os = distributions[values.get("ID") ?? ""]
  const version = values.get("VERSION_ID")
  if (!os || !version || !/^\d+(\.\d+)*$/.test(version)) return yield* fail("Unsupported or unidentified Linux distribution")
  return { os, version }
})

export const nvidiaDevices = (wire: string) => Effect.forEach(wire.trim() ? wire.trim().split(/\r?\n/) : [], line => Effect.gen(function* () {
  const fields = line.split(",").map(s => s.trim())
  if (fields.length !== 5 || !fields[0]!.startsWith("GPU-") || (!/^\d+(\.\d+)?$/.test(fields[3]!) && fields[3] !== "[N/A]" && fields[3] !== "N/A")) return yield* fail("Malformed NVIDIA device query")
  return yield* decode(ObservedGpu, { uuid: fields[0], name: fields[1], driver: fields[2], backend: "cuda", pciBusId: fields[4], memoryBytes: /N\/A/.test(fields[3]!) ? null : Number(fields[3]) * 1024 ** 2 })
}))

const WindowsReport = Schema.Struct({ productType: Schema.Int, version: Schema.String, build: Schema.String, architecture: Schema.Int,
  cpuVendor: Schema.NonEmptyString, cpuName: Schema.NonEmptyString, machineModel: Schema.NonEmptyString, memoryBytes: Schema.Number })
export const windowsIdentity = (report: typeof WindowsReport.Type) => Effect.gen(function* () {
  if (![1, 3].includes(report.productType) || !report.version.startsWith("10.") || !/^\d+$/.test(report.build) || Number(report.build) < 10240) {
    return yield* fail("Unsupported Windows edition or build")
  }
  const serverVersion = report.build === "20348" ? "2022" : report.build === "26100" ? "2025" : null
  if (report.productType === 3 && serverVersion === null) return yield* fail("Unsupported Windows Server build")
  const arch = yield* decode(Architecture, report.architecture === 9 ? "x64" : report.architecture === 12 ? "arm64" : "unsupported")
  return { os: report.productType === 3 ? "windows-server" as const : "windows" as const,
    version: report.productType === 3 ? serverVersion! : Number(report.build) >= 22000 ? "11" : "10", build: report.build, arch,
    cpuVendor: report.cpuVendor, cpuName: report.cpuName, machineModel: report.machineModel, memoryBytes: report.memoryBytes }
})
const windowsQuery = `$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem
$cpu = @(Get-CimInstance Win32_Processor)[0]
$machine = Get-CimInstance Win32_ComputerSystem
[ordered]@{productType=[int]$os.ProductType; version=$os.Version; build=$os.BuildNumber; architecture=[int]$cpu.Architecture;
cpuVendor=$cpu.Manufacturer; cpuName=$cpu.Name; machineModel=$machine.Model; memoryBytes=[double]$machine.TotalPhysicalMemory} | ConvertTo-Json -Compress`
// Runs in the lab tooling environment, never injected into the application. Uses native registry IDs.
const metalQuery = `import Foundation
import Metal
let devices: [[String: Any]] = MTLCopyAllDevices().map { device in
  ["name": device.name, "backend": "metal", "uuid": "metal-registry:\\(device.registryID)",
   "driver": ProcessInfo.processInfo.operatingSystemVersionString, "pciBusId": NSNull(),
   "memoryBytes": device.hasUnifiedMemory ? ProcessInfo.processInfo.physicalMemory as Any : NSNull()]
}
let data = try JSONSerialization.data(withJSONObject: devices, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)`

export interface HostInspector {
  readonly inspect: (target: Target) => Effect.Effect<HostObservation, InfrastructureFailure | AssertionFailure>
}
export const HostInspector = Context.GenericTag<HostInspector>("@magnitudedev/testing-lab/HostInspector")
export const HostInspectorLive = Layer.effect(HostInspector, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  const run = (executable: string, args: readonly string[]) => checkedCommand(executable, args, { timeoutMs: 60_000, env: { LC_ALL: "C" }, maxOutputBytes: 1024 * 1024 })
    .pipe(Effect.provideService(ProcessExecutor, executor), Effect.map(result => result.stdout.trim()))
  const exists = (path: string) => fs.exists(path).pipe(Effect.mapError(() => fail(`Cannot inspect ${path}`)))
  const read = (path: string) => fs.readFileString(path).pipe(Effect.mapError(() => fail(`Cannot read ${path}`)))
  return {
    inspect: target => Effect.gen(function* () {
      let observation: HostObservation
      if (process.platform === "darwin") {
        const hardware = yield* run("/usr/sbin/system_profiler", ["SPHardwareDataType", "-json"]).pipe(Effect.flatMap(wire => decode(Schema.parseJson(Schema.Struct({
          SPHardwareDataType: Schema.NonEmptyArray(Schema.Struct({ chip_type: optionalString, cpu_type: optionalString, machine_model: Schema.NonEmptyString })),
        })), wire)))
        const cpu = hardware.SPHardwareDataType[0]
        const build = yield* run("/usr/bin/sw_vers", ["-buildVersion"])
        const gpus = yield* run("/usr/bin/swift", ["-e", metalQuery]).pipe(Effect.flatMap(wire => decode(Schema.parseJson(Schema.Array(ObservedGpu)), wire)))
        observation = yield* decode(HostObservation, { os: "macos", version: yield* run("/usr/bin/sw_vers", ["-productVersion"]), build,
          arch: (yield* run("/usr/sbin/sysctl", ["-n", "hw.optional.arm64"])) === "1" ? "arm64" : "x64",
          cpuVendor: Option.isSome(cpu.chip_type) ? "Apple" : "Intel", cpuName: Option.getOrElse(cpu.chip_type, () => Option.getOrElse(cpu.cpu_type, () => "Unknown")),
          machineModel: cpu.machine_model, memoryBytes: Number(yield* run("/usr/sbin/sysctl", ["-n", "hw.memsize"])), gpus })
      } else {
        const requiresNvidia = target.backend === "cuda" || ["a10", "rtx-pro-6000", "dgx-spark"].includes(target.hardware)
        const queried = yield* run("nvidia-smi", ["--query-gpu=uuid,name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader,nounits"]).pipe(Effect.either)
        if (queried._tag === "Left" && requiresNvidia) return yield* fail(`NVIDIA inspection failed: ${queried.left.message}`)
        const gpus = queried._tag === "Right" ? yield* nvidiaDevices(queried.right) : []
        if (process.platform === "win32") {
          const report = yield* run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", windowsQuery]).pipe(Effect.flatMap(wire => decode(Schema.parseJson(WindowsReport), wire)))
          observation = yield* decode(HostObservation, { ...yield* windowsIdentity(report), gpus })
        } else if (process.platform === "linux") {
          const identity = yield* linuxIdentity(yield* read("/etc/os-release"), (yield* exists("/etc/dgx-release")) ? yield* read("/etc/dgx-release") : "")
          const cpu = yield* run("lscpu", ["--json"]).pipe(Effect.flatMap(wire => decode(Schema.parseJson(Schema.Struct({ lscpu: Schema.Array(Schema.Struct({ field: Schema.String, data: Schema.NullOr(Schema.String) })) })), wire)))
          const fields = new Map(cpu.lscpu.map(row => [row.field, row.data ?? ""]))
          const memory = /^MemTotal:\s+(\d+)\s+kB$/m.exec(yield* read("/proc/meminfo"))
          observation = yield* decode(HostObservation, { ...identity, build: yield* run("uname", ["-r"]), arch: yield* architecture(yield* run("uname", ["-m"])),
            cpuVendor: fields.get("Vendor ID:") ?? "", cpuName: fields.get("Model name:") ?? "", machineModel: (yield* exists("/sys/devices/virtual/dmi/id/product_name"))
              ? (yield* read("/sys/devices/virtual/dmi/id/product_name")).trim()
              : (yield* exists("/proc/device-tree/model")) ? (yield* read("/proc/device-tree/model")).replaceAll("\0", "").trim() : "Unavailable",
            memoryBytes: memory ? Number(memory[1]) * 1024 : 0, gpus })
        } else return yield* fail(`Unsupported worker platform: ${process.platform}`)
      }
      yield* attestHost(target, observation)
      return observation
    }),
  } satisfies HostInspector
}))
