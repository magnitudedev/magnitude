import { FileSystem } from "@effect/platform"
import { DateTime, Effect, Layer, Option, Schema } from "effect"
import { join } from "node:path"
import { InfrastructureFailure, TargetId } from "../domain"
import { AzureMachine, MachineAllocator, MachineTags } from "../machines"
import { checkedCommand, ProcessExecutor } from "../process"

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
  return {
    inventory,
    ensure: (lease, target) => Effect.gen(function* () {
      if (lease.provider !== "azure" || target.provider !== "azure" || lease.targetId !== target.id) return yield* fail("Azure allocation target does not match its lease")
      if (DateTime.toEpochMillis(lease.expiresAt) <= Date.now()) return yield* fail("Cannot allocate an expired lease")
      if (!/^ml-[a-f0-9]{12}$/.test(lease.resourceName)) return yield* fail("Azure machine names must be ml- followed by twelve hexadecimal digits")
      const image = config.images.find(i => i.targetId === target.id)
      if (!image) return yield* fail(`No qualified Azure image for ${target.id}`)
      if ((target.os === "windows") !== (image.os === "Windows")) return yield* fail("Image OS does not match target")
      const tags = MachineTags.make({ schemaVersion: 1, runId: lease.runId, leaseId: lease.leaseId, expiresAt: lease.expiresAt })
      const machine = AzureMachine.make({ provider: "azure", id: vmId(lease.resourceName), name: lease.resourceName, tags })
      let existing = yield* matching(machine)
      const encodedTags = { "lab-owner": marker, "lab-machine": machine.name, "lab-lease": yield* Schema.encode(Schema.parseJson(MachineTags))(tags).pipe(Effect.orDie),
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
            storageProfile: { imageReference: image.image, osDisk: { name: `${machine.name}-os`, createOption: "FromImage", deleteOption: "Delete", diskSizeGB: image.diskGb,
              managedDisk: { storageAccountType: "Premium_LRS" } } },
            networkProfile: { networkInterfaces: [{ id: nicId(machine.name), properties: { primary: true, deleteOption: "Delete" } }] },
            osProfile: { computerName: machine.name, adminUsername: config.adminUsername,
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
