import { expect, test } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Option, Schema } from "effect"
import { readFileSync } from "node:fs"
import { azureAllocator, type AzureConfig } from "../src/providers/azure"
import { ProcessExecutor, type CommandSpec } from "../src/process"
import { MachineAllocator, MachineTags } from "../src/machines"
import { Allocating } from "../src/lease"
import { targets } from "../src/catalog"
import { LeaseId, RunId } from "../src/domain"
const target = targets.find(t => t.os === "ubuntu" && t.hardware === "intel")!
const subscription = "5304c4b3-d605-4193-b0cb-766c065acfa6"
const group = `/subscriptions/${subscription}/resourceGroups/magnitude-ci`
const name = "ml-123456789abc"
const vmId = `${group}/providers/Microsoft.Compute/virtualMachines/${name}`
const nicId = `${group}/providers/Microsoft.Network/networkInterfaces/${name}-nic`
const config: AzureConfig = { executable: "az", subscription, resourceGroup: "magnitude-ci", location: "westus2",
  subnetId: `${group}/providers/Microsoft.Network/virtualNetworks/lab/subnets/workers`, adminUsername: "labworker", sshPublicKey: "ssh-ed25519 fixture",
  images: [{ targetId: target.id, image: { publisher: "Canonical", offer: "ubuntu-24_04-lts", sku: "server", version: "24.04.202609040" },
    size: "Standard_D4s_v6", os: "Linux", diskGb: 128, plan: Option.none() }] }
const lease = () => new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id,
  provider: "azure", resourceName: name, expiresAt: DateTime.unsafeMake(Date.now() + 3600_000) })
const tags = (l: Allocating) => ({ "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name, "lab-lease": Schema.encodeSync(Schema.parseJson(MachineTags))({ schemaVersion: 1, runId: l.runId, leaseId: l.leaseId, expiresAt: l.expiresAt }) })
const resource = (l: Allocating, nic = false) => ({ id: nic ? nicId : vmId, name: nic ? `${name}-nic` : name,
  type: nic ? "Microsoft.Network/networkInterfaces" : "Microsoft.Compute/virtualMachines", tags: tags(l) })
const output = (value: unknown) => ({ stdout: JSON.stringify(value), stderr: "", exitCode: 0 })
const run = <A, E>(effect: Effect.Effect<A, E, MachineAllocator>, handler: (spec: CommandSpec) => ReturnType<typeof output>) => {
  const processes = Layer.succeed(ProcessExecutor, { run: (c: CommandSpec) => Effect.sync(() => handler(c)) })
  const allocator = azureAllocator(config).pipe(Layer.provide(Layer.merge(BunContext.layer, processes)))
  return Effect.runPromise(effect.pipe(Effect.provide(allocator)))
}
test("creates private resources in the explicit subscription and reconciles ambiguous VM creation", async () => {
  const l = lease(), rows: ReturnType<typeof resource>[] = []
  let puts = 0
  const machine = await run(Effect.flatMap(MachineAllocator, a => a.ensure(l, target)), spec => {
    expect(spec.args[spec.args.indexOf("--subscription") + 1]).toBe(subscription)
    if (spec.args[0] === "resource") return output(rows)
    const method = spec.args[spec.args.indexOf("--method") + 1]
    if (method === "GET") return output({ properties: { provisioningState: "Succeeded", storageProfile: { osDisk: { managedDisk: { id: `${group}/providers/Microsoft.Compute/disks/${name}-os` } } } } })
    if (method === "PATCH") return output({})
    expect(method).toBe("PUT"); puts++
    const body = JSON.parse(readFileSync(spec.args[spec.args.indexOf("--body") + 1]!.slice(1), "utf8"))
    expect(body.tags["lab-lease"]).toBe(tags(l)["lab-lease"])
    if (!rows.length) {
      expect(body.properties.ipConfigurations[0].properties.publicIPAddress).toBeUndefined()
      rows.push(resource(l, true)); return output({})
    }
    expect(body.properties.storageProfile.imageReference.version).not.toBe("latest")
    expect(body.properties.storageProfile.osDisk.deleteOption).toBe("Delete")
    expect(body.properties.osProfile.linuxConfiguration.disablePasswordAuthentication).toBe(true)
    rows.push(resource(l)); return { stdout: "", stderr: "Connection lost", exitCode: 1 }
  })
  expect(machine.provider).toBe("azure"); if (machine.provider !== "azure") throw new Error("Wrong provider"); expect(machine.id).toBe(vmId); expect(puts).toBe(2)
})
test("refuses to adopt another lease", async () => {
  const result = await run(Effect.flatMap(MachineAllocator, a => a.ensure(lease(), target)).pipe(Effect.either), spec => {
    expect(spec.args[0]).toBe("resource"); return output([resource(lease())])
  })
  expect(result._tag).toBe("Left")
})
test("discovers and removes NIC-only orphan allocations", async () => {
  const l = lease(); let exists = true
  await run(Effect.gen(function* () {
    const a = yield* MachineAllocator
    const inventory = yield* a.inventory()
    expect(inventory).toHaveLength(1)
    yield* a.release(inventory[0]!); expect(yield* a.inventory()).toEqual([])
  }), spec => {
    if (spec.args[0] === "resource") return output(exists ? [resource(l, true)] : [])
    expect(spec.args[spec.args.indexOf("--method") + 1]).toBe("DELETE")
    expect(spec.args[spec.args.indexOf("--url") + 1]).toContain(nicId)
    exists = false; return output({})
  })
})
test("rechecks deletion ownership after inventory", async () => {
  const l = lease(); let reads = 0
  const result = await run(Effect.gen(function* () {
    const a = yield* MachineAllocator
    yield* a.release((yield* a.inventory())[0]!)
  }).pipe(Effect.either), spec => {
    expect(spec.args[0]).toBe("resource"); return output([resource(reads++ === 0 ? l : lease())])
  })
  expect(result._tag).toBe("Left")
})
