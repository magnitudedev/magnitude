import { Fence } from "../src/lease"
import { WorkId } from "../src/work-identity"
import { expect, test } from "vitest"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer, Option, Schema, Stream } from "effect"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { sha256 } from "../src/snapshot"
import { azureAllocator, type AzureConfig } from "../src/providers/azure"
import { ProcessExecutor, type CommandSpec } from "../src/process"
import { MachineAllocator, MachineTags } from "../src/machines"
import { Allocating } from "../src/lease"
import { targets } from "../src/catalog"
import { LeaseId, RunId } from "../src/domain"
import { ArtifactStore } from "../src/artifact-store"
import { InitializationDiagnostic } from "../src/providers/initialization-diagnostics"
const target = targets.find(t => t.os === "ubuntu" && t.hardware === "intel")!
const subscription = "5304c4b3-d605-4193-b0cb-766c065acfa6"
const group = `/subscriptions/${subscription}/resourceGroups/magnitude-ci`
const name = "ml-123456789abc"
const vmId = `${group}/providers/Microsoft.Compute/virtualMachines/${name}`
const nicId = `${group}/providers/Microsoft.Network/networkInterfaces/${name}-nic`
const config: AzureConfig = { executable: "az", subscription, resourceGroup: "magnitude-ci", location: "westus2",
  subnetId: `${group}/providers/Microsoft.Network/virtualNetworks/lab/subnets/workers`, adminUsername: "labworker", sshPublicKey: "ssh-ed25519 fixture",
  images: [{ targetId: target.id, image: { publisher: "Canonical", offer: "ubuntu-24_04-lts", sku: "server", version: "24.04.202609040" },
    size: "Standard_D4s_v6", os: "Linux", diskGb: 128, plan: Option.none(), initialization: Option.none() }] }
const lease = () => new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id, workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
  provider: "azure", resourceName: name, expiresAt: DateTime.unsafeMake(Date.now() + 3600_000) })
const tags = (l: Allocating) => ({ "lab-owner": "magnitude-testing-lab-v1", "lab-machine": name, "lab-lease": Schema.encodeSync(Schema.parseJson(MachineTags))({ schemaVersion: 1, runId: l.runId, leaseId: l.leaseId, expiresAt: l.expiresAt }) })
const resource = (l: Allocating, nic = false) => ({ id: nic ? nicId : vmId, name: nic ? `${name}-nic` : name,
  type: nic ? "Microsoft.Network/networkInterfaces" : "Microsoft.Compute/virtualMachines", tags: tags(l) })
const output = (value: unknown) => ({ stdout: JSON.stringify(value), stderr: "", exitCode: 0 })
const run = <A, E>(effect: Effect.Effect<A, E, MachineAllocator>, handler: (spec: CommandSpec) => ReturnType<typeof output>, settings = config, artifacts = new Map<string, Uint8Array>()) => {
  const processes = Layer.succeed(ProcessExecutor, { run: (c: CommandSpec) => Effect.sync(() => handler(c)) })
  const objects = Layer.succeed(ArtifactStore, { put: (digest, stream) => Effect.gen(function* () {
    const bytes = Buffer.concat(Array.from(yield* Stream.runCollect(stream)))
    expect(sha256(bytes)).toBe(digest)
    artifacts.set(digest, bytes)
  }), get: digest => Stream.make(artifacts.get(digest)!), exists: digest => Effect.succeed(artifacts.has(digest)) })
  const allocator = azureAllocator(settings).pipe(Layer.provide(Layer.mergeAll(BunContext.layer, processes, objects)))
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

for (const mode of ["ready", "failed", "missing-exit", "changed-file", "oversized", "wrong-identity"] as const) {
  test(`Azure runtime initialization: ${mode}`, async () => {
    const directory = mkdtempSync(join(tmpdir(), "lab-init-"))
    try {
      const file = join(directory, "cloud-init.yml"), script = "#cloud-config\nruncmd:\n  - [true]\n"
      const digest = sha256(new TextEncoder().encode(script))
      writeFileSync(file, mode === "changed-file" ? `${script}# changed` : mode === "oversized" ? "x".repeat(65537) : script)
      const settings = { ...config, images: [{ ...config.images[0]!, initialization: Option.some({ file, sha256: digest }) }] }
      const l = lease(), rows: ReturnType<typeof resource>[] = []
      let requests = 0, launched = false, observed = false, diagnostics = false
      const artifacts = new Map<string, Uint8Array>()
      const result = await run(Effect.flatMap(MachineAllocator, a => a.ensure(l, target)).pipe(Effect.either), spec => {
        requests++
        if (spec.args[0] === "resource") return output(rows)
        if (spec.args[0] === "vm") {
          diagnostics = true
          expect(spec.args).toContain("tail -c 2200 /var/log/cloud-init-output.log; cloud-init status --long --format json")
          return output({ value: [{ message: 'setup diagnostic: https://user:private-password@account.blob.core.windows.net/archive?sig=private-signature Bearer private-token\n{"password":"private-json-password with spaces"}\nLAB_WORKER_TOKEN=private-env-token' }] })
        }
        const method = spec.args[spec.args.indexOf("--method") + 1], url = spec.args[spec.args.indexOf("--url") + 1]!
        if (url.includes("/runCommands/")) {
          if (method === "PUT") {
            const body = JSON.parse(readFileSync(spec.args[spec.args.indexOf("--body") + 1]!.slice(1), "utf8"))
            expect(body.properties.source.script).toContain("cloud-init status --wait")
            expect(body.properties.protectedParameters).toBeUndefined()
            expect(body.properties.timeoutInSeconds).toBeLessThanOrEqual(1200)
            launched = true
            return output({})
          }
          expect(launched).toBe(true)
          expect(url).toContain("$expand=instanceView")
          observed = true
          return output({ properties: { instanceView: { executionState: mode === "failed" ? "Failed" : "Succeeded",
            ...(mode === "missing-exit" ? {} : { exitCode: mode === "failed" ? 1 : 0 }) } } })
        }
        if (method === "GET") return output({ tags: { "lab-initialization": mode === "wrong-identity" ? "another-digest" : digest },
          properties: { provisioningState: "Succeeded", storageProfile: { osDisk: { managedDisk: { id: `${group}/providers/Microsoft.Compute/disks/${name}-os` } } } } })
        if (method === "PATCH") return output({})
        const body = JSON.parse(readFileSync(spec.args[spec.args.indexOf("--body") + 1]!.slice(1), "utf8"))
        if (!rows.length) rows.push(resource(l, true))
        else {
          expect(Buffer.from(body.properties.osProfile.customData, "base64").toString()).toBe(script)
          expect(body.tags["lab-initialization"]).toBe(digest)
          rows.push(resource(l))
        }
        return output({})
      }, settings, artifacts)
      expect(result._tag).toBe(mode === "ready" ? "Right" : "Left")
      if (mode === "changed-file" || mode === "oversized") expect(requests).toBe(0)
      else if (mode === "wrong-identity") expect(launched).toBe(false)
      else expect(observed).toBe(true)
      if (diagnostics && result._tag === "Left") {
        expect(result.left.message).toContain("retained in run evidence")
        const evidence = Option.getOrThrow(result.left.evidence)[0]!
        const bytes = artifacts.get(evidence.sha256)!
        expect(bytes.byteLength).toBe(evidence.bytes)
        const retained = Schema.decodeUnknownSync(Schema.parseJson(InitializationDiagnostic))(Buffer.from(bytes).toString())
        expect(retained.leaseId).toBe(l.leaseId)
        expect(retained.runId).toBe(l.runId)
        expect(retained.output).toContain("setup diagnostic:")
        expect(retained.output).not.toContain("private-")
      }
    } finally { rmSync(directory, { recursive: true, force: true }) }
  })
}
