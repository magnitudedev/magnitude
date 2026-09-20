import { FileSystem } from "@effect/platform"
import { DateTime, Effect, Layer, Option, Schema } from "effect"
import { join } from "node:path"
import { Digest, InfrastructureFailure, TargetId } from "../domain"
import { AzureMachine, MachineAllocator, MachineTags } from "../machines"
import { checkedCommand, ProcessExecutor } from "../process"
import { AzureInitialization, prepareAzureInitialization } from "./azure-initialization"
import { prepareWindowsMachine, windowsPreparationDiagnostics } from "./azure-windows-preparation"
import { azureInitializationWait, azureInitializationDiagnosticsScript } from "./azure-readiness"
import { ArtifactStore } from "../artifact-store"
import { retainWorkerDiagnostic } from "../worker-diagnostics"

const vmApi = "2024-11-01"
const nicApi = "2024-05-01"
const diskApi = "2024-03-02"
const marker = "magnitude-testing-lab-v1"
const fail = (message: string) => new InfrastructureFailure({ operation: "azure", message })
const ImageReference = Schema.Union(
  Schema.Struct({ id: Schema.String.pipe(Schema.pattern(/\/versions\/[^/]+$/)) }),
  Schema.Struct({ publisher: Schema.NonEmptyString, offer: Schema.NonEmptyString, sku: Schema.NonEmptyString,
    version: Schema.String.pipe(Schema.pattern(/^\d+\.\d+\.\d+$/)) }),
)
export const AzureImage = Schema.Struct({ targetId: TargetId, image: ImageReference, size: Schema.NonEmptyString,
  os: Schema.Literal("Linux", "Windows"), diskGb: Schema.Int.pipe(Schema.between(64, 2048)),
  windowsLicense: Schema.optionalWith(Schema.Literal("visual-studio-dev-test", "multitenant"), { as: "Option", exact: true }),
  initialization: Schema.optionalWith(AzureInitialization, { as: "Option", exact: true }),
  plan: Schema.optionalWith(Schema.Struct({ name: Schema.String, product: Schema.String, publisher: Schema.String }), { as: "Option", exact: true }) })
export type AzureImage = typeof AzureImage.Type
export const AzureConfig = Schema.Struct({ executable: Schema.String, subscription: Schema.UUID, resourceGroup: Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9_.-]+$/)),
  location: Schema.NonEmptyString, subnetId: Schema.NonEmptyString, adminUsername: Schema.String.pipe(Schema.pattern(/^[a-z][a-z0-9]{1,19}$/)),
  sshPublicKey: Schema.String.pipe(Schema.pattern(/^ssh-(ed25519|rsa) /)), images: Schema.Array(AzureImage) })
export type AzureConfig = typeof AzureConfig.Type
const Resource = Schema.Struct({ id: Schema.String, name: Schema.String, type: Schema.String,
  tags: Schema.NullOr(Schema.Record({ key: Schema.String, value: Schema.String })) })
type Resource = typeof Resource.Type
const Provisioning = Schema.Struct({ properties: Schema.Struct({ provisioningState: Schema.String }) })
const VmObservation = Schema.Struct({ properties: Schema.Struct({ provisioningState: Schema.String,
  storageProfile: Schema.Struct({ osDisk: Schema.Struct({ managedDisk: Schema.Struct({ id: Schema.String }) }) }) }) })

/** Provider credentials stay on the coordinator. A guest has no subscription credential or public IP. */
export const azureAllocator = (config: AzureConfig) => Layer.effect(MachineAllocator, Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const fs = yield* FileSystem.FileSystem
  const objects = yield* ArtifactStore
  const checked = (...args: Parameters<typeof checkedCommand>) => checkedCommand(...args).pipe(Effect.provideService(ProcessExecutor, executor))
  const groupId = `/subscriptions/${config.subscription}/resourceGroups/${config.resourceGroup}`
  if (!config.subnetId.toLowerCase().startsWith(`${groupId}/providers/microsoft.network/virtualnetworks/`.toLowerCase())) return yield* fail("Worker subnet must belong to the configured subscription and resource group")
  const vmId = (name: string) => `${groupId}/providers/Microsoft.Compute/virtualMachines/${name}`
  const diskId = (name: string) => `${groupId}/providers/Microsoft.Compute/disks/${name}-os`
  const nicId = (name: string) => `${groupId}/providers/Microsoft.Network/networkInterfaces/${name}-nic`
  const rest = (method: string, id: string, version: string, body?: unknown) => Effect.scoped(Effect.gen(function* () {
    if (!id.toLowerCase().startsWith(`${groupId}/`.toLowerCase())) return yield* fail("Azure operation escaped the configured resource group")
    const args = ["rest", "--subscription", config.subscription, "--method", method, "--url", `https://management.azure.com${id}?api-version=${version}`, "--only-show-errors", "--output", "json"]
    if (body !== undefined) {
      const dir = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-lab-azure-" })
      const file = join(dir, "request.json")
      yield* fs.writeFileString(file, yield* Schema.encode(Schema.parseJson(Schema.Unknown))(body), { mode: 0o600 })
      args.push("--body", `@${file}`)
    }
    return yield* checked(config.executable, args, { timeoutMs: 180_000 })
  })).pipe(Effect.mapError(() => fail(`Azure ${method} failed for ${id.split("/").at(-1)}; inspect the provider activity log`)))
  const list = () => checked(config.executable, ["resource", "list", "--subscription", config.subscription, "--resource-group", config.resourceGroup, "--output", "json", "--only-show-errors"]).pipe(
    Effect.flatMap(r => Schema.decodeUnknown(Schema.parseJson(Schema.Array(Resource)))(r.stdout)), Effect.mapError(() => fail("Cannot read Azure resource inventory")))
  const decode = (resource: Resource) => Effect.gen(function* () {
    if (resource.tags?.["lab-owner"] !== marker || !resource.tags["lab-machine"] || !resource.tags["lab-lease"]) return yield* fail("Resource lacks lab ownership metadata")
    const tags = yield* Schema.decodeUnknown(Schema.parseJson(MachineTags))(resource.tags["lab-lease"]).pipe(Effect.mapError(() => fail("Invalid Azure lease metadata")))
    const name = resource.tags["lab-machine"]
    if (!/^ml-[a-f0-9]{12}$/.test(name)) return yield* fail("Invalid Azure machine identity")
    const expected = resource.type.toLowerCase() === "microsoft.compute/virtualmachines" ? vmId(name)
      : resource.type.toLowerCase() === "microsoft.network/networkinterfaces" ? nicId(name)
      : resource.type.toLowerCase() === "microsoft.compute/disks" ? diskId(name) : undefined
    if (!expected || resource.id.toLowerCase() !== expected.toLowerCase()) return yield* fail("Azure ownership tags do not match the resource identity")
    return AzureMachine.make({ provider: "azure", id: vmId(name), name, tags })
  })
  const inventory = () => list().pipe(Effect.flatMap(rows => Effect.forEach(rows.filter(r => r.tags?.["lab-owner"] === marker), decode)),
    Effect.map(machines => [...new Map(machines.map(m => [m.id, m])).values()]))
  const matching = (machine: typeof AzureMachine.Type) => list().pipe(Effect.flatMap(rows => Effect.forEach(rows.filter(r =>
    r.id.toLowerCase() === vmId(machine.name).toLowerCase() || r.id.toLowerCase() === nicId(machine.name).toLowerCase() || (r.id.toLowerCase() === diskId(machine.name).toLowerCase() &&
      (r.tags?.["lab-owner"] === marker || !rows.some(vm => vm.id.toLowerCase() === machine.id.toLowerCase())))), r => decode(r).pipe(Effect.flatMap(actual =>
      Schema.equivalence(MachineTags)(actual.tags, machine.tags) ? Effect.succeed(r) : Effect.fail(fail("Resource identity belongs to a different lease")))))))
  const waitProvisioned = (machine: typeof AzureMachine.Type) => Effect.gen(function* () {
    for (;;) {
      const response = yield* rest("GET", machine.id, vmApi)
      const observed = yield* Schema.decodeUnknown(Schema.parseJson(Provisioning))(response.stdout).pipe(Effect.mapError(() => fail("Invalid Azure VM observation")))
      const state = observed.properties.provisioningState
      if (state === "Succeeded") return
      if (state === "Failed" || state === "Canceled" || state === "Deleting") return yield* fail(`Azure provisioning ended in ${state}`)
      yield* Effect.sleep("5 seconds")
    }
  }).pipe(Effect.timeoutFail({ duration: "15 minutes", onTimeout: () => fail("Azure VM provisioning exceeded 15 minutes; lease remains owned for cleanup") }))
  const tagDisk = (machine: typeof AzureMachine.Type, tags: Readonly<Record<string, string>>) => Effect.gen(function* () {
    const vm = yield* rest("GET", machine.id, vmApi).pipe(Effect.flatMap(r => Schema.decodeUnknown(Schema.parseJson(VmObservation))(r.stdout)), Effect.mapError(() => fail("Cannot observe the VM's managed disk before cleanup")))
    const id = vm.properties.storageProfile.osDisk.managedDisk.id
    if (id.toLowerCase() !== diskId(machine.name).toLowerCase()) return yield* fail("VM disk identity does not match its owned allocation")
    yield* rest("PATCH", id, diskApi, { tags })
  })
  const waitInitialized = (machine: typeof AzureMachine.Type, digest: Digest) => Effect.gen(function* () {
    const identity = yield* rest("GET", machine.id, vmApi).pipe(Effect.flatMap(r => Schema.decodeUnknown(Schema.parseJson(
      Schema.Struct({ tags: Schema.Record({ key: Schema.String, value: Schema.String }) })))(r.stdout)))
    if (identity.tags["lab-initialization"] !== digest) return yield* fail("Allocated VM initialization differs from configured runtime")
    const id = `${machine.id}/runCommands/lab-initialize`
    const remaining = DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now()
    if (remaining <= 0) return yield* fail("Worker expired before initialization")
    yield* rest("PUT", id, vmApi, { location: config.location, properties: {
      source: { script: azureInitializationWait() },
      asyncExecution: true, timeoutInSeconds: Math.max(1, Math.min(1200, Math.floor(remaining / 1000))),
    } })
    const View = Schema.Struct({ properties: Schema.Struct({ instanceView: Schema.optionalWith(Schema.Struct({
      executionState: Schema.String, exitCode: Schema.optionalWith(Schema.Int, { as: "Option", exact: true }),
    }), { as: "Option", exact: true }) }) })
    for (;;) {
      const response = yield* rest("GET", id, `${vmApi}&$expand=instanceView`)
      const observed = yield* Schema.decodeUnknown(Schema.parseJson(View))(response.stdout)
      if (Option.isSome(observed.properties.instanceView)) {
        const view = observed.properties.instanceView.value
        if (view.executionState === "Succeeded") {
          if (Option.isNone(view.exitCode) || view.exitCode.value !== 0) return yield* fail("Worker initialization did not exit successfully")
          return
        }
        if (["Failed", "Canceled", "TimedOut"].includes(view.executionState)) return yield* fail(`Worker initialization ended in ${view.executionState}; inspect lab-initialize and cloud-init logs`)
      }
      yield* Effect.sleep("5 seconds")
    }
  }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail("Invalid Azure initialization observation")),
    Effect.timeoutFail({ duration: Math.max(1, Math.min(20 * 60_000, DateTime.toEpochMillis(machine.tags.expiresAt) - Date.now())),
    onTimeout: () => fail("Worker initialization exceeded its deadline; allocation remains owned for cleanup") }))
  const initializationDiagnostics = (machine: typeof AzureMachine.Type, windows = false) => Effect.gen(function* () {
    if (windows) return yield* windowsPreparationDiagnostics(machine, rest).pipe(
      Effect.flatMap(output => retainWorkerDiagnostic(machine, "initialization", output)), Effect.provideService(ArtifactStore, objects))
    // Publish bounded, redacted diagnostics before deleting the failed allocation.
    // The run report grants owner access; container-local files do not survive deployment.
    const reply = yield* checked(config.executable, ["vm", "run-command", "invoke", "--subscription", config.subscription,
      "--resource-group", config.resourceGroup, "--name", machine.name, "--command-id", "RunShellScript",
      "--scripts", azureInitializationDiagnosticsScript(), "--only-show-errors", "--output", "json"],
      { timeoutMs: 180_000, maxOutputBytes: 64 * 1024 })
    const output = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ value: Schema.Array(Schema.Struct({ message: Schema.String })) })))(reply.stdout)
    return yield* retainWorkerDiagnostic(machine, "initialization", output.value.map(entry => entry.message).join("\n")).pipe(Effect.provideService(ArtifactStore, objects))
  }).pipe(Effect.mapError(() => fail("Could not retain initialization diagnostics")))
  return {
    inventory,
    ensure: (lease, target) => Effect.gen(function* () {
      if (lease.provider !== "azure" || target.provider !== "azure" || lease.targetId !== target.id) return yield* fail("Azure allocation target does not match its lease")
      if (DateTime.toEpochMillis(lease.expiresAt) <= Date.now()) return yield* fail("Cannot allocate an expired lease")
      if (!/^ml-[a-f0-9]{12}$/.test(lease.resourceName)) return yield* fail("Azure machine names must be ml- followed by twelve hexadecimal digits")
      const image = config.images.find(i => i.targetId === target.id)
      if (!image) return yield* fail(`No qualified Azure image for ${target.id}`)
      if ((target.os === "windows") !== (image.os === "Windows")) return yield* fail("Image OS does not match target")
      if (target.os === "windows" && Option.isNone(image.windowsLicense)) return yield* fail("Windows client allocation requires an operator-verified licensing basis")
      if (target.os !== "windows" && Option.isSome(image.windowsLicense)) return yield* fail("Windows client licensing cannot be applied to another OS")
      const gpu = Option.flatMap(image.initialization, setup => "kind" in setup ? setup.gpu : Option.none())
      if (Option.isSome(gpu)) {
        if (target.hardware !== gpu.value.model || !(gpu.value.model === "a10" ? /^Standard_NV\d+ads_A10_v5$/.test(image.size) : /^Standard_NC\d+.*RTX.*v6$/i.test(image.size)))
          return yield* fail("GPU driver recipe does not match the requested hardware or Azure VM family")
      } else if (target.hardware === "a10" || target.hardware === "rtx-pro-6000") return yield* fail("GPU targets require explicit native driver preparation")
      // Administrator-owned setup is separate from submitted source. Verify it before allocating anything.
      const initialization = yield* Option.match(image.initialization, { onNone: () => Effect.void, onSome: setup => Effect.gen(function* () {
        if ((image.os === "Windows") !== ("kind" in setup && setup.kind === "windows")) return yield* fail("Initialization recipe does not match the native image OS")
        return yield* prepareAzureInitialization(setup, { executable: config.executable, subscription: config.subscription,
          adminUsername: config.adminUsername, architecture: target.arch, os: target.os, version: target.version }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(ProcessExecutor, executor))
      }).pipe(Effect.mapError(error => error._tag === "InfrastructureFailure" ? error : fail("Cannot read configured worker initialization"))) })
      const tags = MachineTags.make({ schemaVersion: 1, runId: lease.runId, leaseId: lease.leaseId, expiresAt: lease.expiresAt })
      const machine = AzureMachine.make({ provider: "azure", id: vmId(lease.resourceName), name: lease.resourceName, tags })
      let existing = yield* matching(machine)
      const encodedTags = { "lab-owner": marker, "lab-machine": machine.name, "lab-lease": yield* Schema.encode(Schema.parseJson(MachineTags))(tags).pipe(Effect.orDie),
        ...(initialization ? { "lab-initialization": initialization.identity } : {}),
        "lab-expires": DateTime.formatIso(lease.expiresAt) }
      if (!existing.some(r => r.id.toLowerCase() === nicId(machine.name).toLowerCase())) {
        const attempt = yield* rest("PUT", nicId(machine.name), nicApi, { location: config.location, tags: encodedTags,
          properties: { enableIPForwarding: false, ipConfigurations: [{ name: "private", properties: { privateIPAllocationMethod: "Dynamic", subnet: { id: config.subnetId } } }] } }).pipe(Effect.either)
        existing = yield* matching(machine)
        if (attempt._tag === "Left" && !existing.some(r => r.id.toLowerCase() === nicId(machine.name).toLowerCase())) return yield* attempt.left
      }
      if (!existing.some(r => r.id.toLowerCase() === machine.id.toLowerCase())) {
        const password = `Az!${crypto.randomUUID()}a9`
        const body = { location: config.location, tags: encodedTags, ...(Option.isSome(image.plan) ? { plan: image.plan.value } : {}),
          properties: { hardwareProfile: { vmSize: image.size },
            ...(Option.contains(image.windowsLicense, "multitenant") ? { licenseType: "Windows_Client" } : {}),
            ...(Option.isSome(gpu) ? { securityProfile: { securityType: "Standard" } } : {}),
            storageProfile: { imageReference: image.image, osDisk: { name: `${machine.name}-os`, createOption: "FromImage", deleteOption: "Delete", diskSizeGB: image.diskGb,
              managedDisk: { storageAccountType: "Premium_LRS" } } },
            networkProfile: { networkInterfaces: [{ id: nicId(machine.name), properties: { primary: true, deleteOption: "Delete" } }] },
            osProfile: { computerName: machine.name, adminUsername: config.adminUsername,
              ...(initialization?.kind === "linux" ? { customData: initialization.customData } : {}),
              ...(image.os === "Windows" ? { adminPassword: password, windowsConfiguration: { provisionVMAgent: true, enableAutomaticUpdates: false } }
                : { linuxConfiguration: { disablePasswordAuthentication: true, provisionVMAgent: true,
                  ssh: { publicKeys: [{ path: `/home/${config.adminUsername}/.ssh/authorized_keys`, keyData: config.sshPublicKey }] } } }) } } }
        const attempt = yield* rest("PUT", machine.id, vmApi, body).pipe(Effect.either)
        existing = yield* matching(machine)
        if (!existing.some(r => r.id.toLowerCase() === machine.id.toLowerCase())) return yield* fail(attempt._tag === "Left" ? "Azure allocation failed; owned NIC remains discoverable for cleanup" : "Azure did not publish the allocated VM")
      }
      yield* waitProvisioned(machine)
      // Azure does not inherit VM tags onto its managed disk. Tag it before admitting work,
      // and again before deletion, so a disk left by failed asynchronous deletion is discoverable.
      yield* tagDisk(machine, encodedTags)
      if (initialization) yield* (initialization.kind === "linux" ? waitInitialized(machine, initialization.identity) : prepareWindowsMachine(machine, initialization,
        { executable: config.executable, subscription: config.subscription, location: config.location }, { rest, waitProvisioned: waitProvisioned(machine),
          restart: checked(config.executable, ["vm", "restart", "--subscription", config.subscription, "--resource-group", config.resourceGroup,
            "--name", machine.name, "--only-show-errors"], { timeoutMs: 300_000 }).pipe(Effect.mapError(() => fail("Windows desktop restart failed or its observation expired; allocation remains owned for cleanup"))),
        }).pipe(Effect.provideService(ProcessExecutor, executor))).pipe(
        Effect.catchAll(error => initializationDiagnostics(machine, initialization.kind === "windows").pipe(Effect.either, Effect.flatMap(diagnostics =>
          Effect.fail(new InfrastructureFailure({ operation: "azure", message: `${error.message}; ${diagnostics._tag === "Right" ? "initialization diagnostics retained in run evidence" : diagnostics.left.message}`,
            evidence: diagnostics._tag === "Right" ? Option.some([diagnostics.right]) : Option.none() }))))))
      return machine
    }),
    release: machine => Effect.gen(function* () {
      if (machine.provider !== "azure" || machine.id.toLowerCase() !== vmId(machine.name).toLowerCase()) return yield* fail("Machine belongs to another Azure scope")
      // NIC-only inventory is retained after a failed VM creation, so the janitor can remove it.
      const resources = yield* matching(machine)
      const vm = resources.find(r => r.type.toLowerCase() === "microsoft.compute/virtualmachines")
      if (vm) {
        yield* tagDisk(machine, vm.tags!)
        yield* rest("DELETE", vm.id, vmApi)
        yield* Effect.gen(function* () {
          while ((yield* matching(machine)).some(r => r.id.toLowerCase() === vm.id.toLowerCase())) yield* Effect.sleep("5 seconds")
        }).pipe(Effect.timeoutFail({ duration: "10 minutes", onTimeout: () => fail("Azure VM deletion is still pending") }))
      }
      for (const resource of yield* matching(machine)) yield* rest("DELETE", resource.id,
        resource.type.toLowerCase() === "microsoft.compute/disks" ? diskApi : nicApi)
      yield* Effect.gen(function* () {
        while ((yield* matching(machine)).length) yield* Effect.sleep("5 seconds")
      }).pipe(Effect.timeoutFail({ duration: "5 minutes", onTimeout: () => fail("Azure network cleanup is still pending") }))
    }),
  } satisfies MachineAllocator
}))
