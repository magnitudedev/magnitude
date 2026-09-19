import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, DateTime, Effect, Layer, Schema } from "effect"
import { join } from "node:path"
import { findTarget } from "../src/catalog"
import { InfrastructureFailure, LeaseId, RunId, TargetId } from "../src/domain"
import { Allocating } from "../src/lease"
import { MachineAllocator } from "../src/machines"
import { azureAllocator, AzureConfig } from "../src/providers/azure"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

const fail = (message: string) => new InfrastructureFailure({ operation: "azure-probe", message })
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("LAB_AZURE_PROBE_ROOT")
  const config = yield* fs.readFileString(yield* Config.string("LAB_AZURE_CONFIG")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(AzureConfig))))
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_AZURE_TARGET")))
  if (target.os !== "ubuntu" || target.arch !== "x64" || target.hardware !== "intel") return yield* fail("This diagnostic qualifies Ubuntu x64 Intel provisioning only")
  yield* fs.makeDirectory(root, { recursive: true })
  const allocator = Context.get(yield* Layer.build(azureAllocator(config)), MachineAllocator)
  const lease = new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id,
    provider: "azure", resourceName: `ml-${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`, expiresAt: DateTime.unsafeMake(Date.now() + 60 * 60_000) })
  yield* fs.writeFileString(join(root, "lease.json"), yield* Schema.encode(Schema.parseJson(Allocating))(lease))
  const cleanupErrors: string[] = []
  const result = yield* Effect.gen(function* () {
    const machine = yield* allocator.ensure(lease, target)
    const repeated = yield* allocator.ensure(lease, target)
    if (machine.provider !== "azure" || repeated.provider !== "azure" || repeated.id !== machine.id) return yield* fail("Repeated allocation did not resolve to the same VM")
    const script = join(root, "inspect.sh")
    yield* fs.writeFileString(script, `set -eu\npython3 - <<'PY'\nimport json, platform\nr={}\nfor line in open('/etc/os-release'):\n k,sep,v=line.strip().partition('=')\n if sep:r[k]=v.strip(chr(34))\ncpu=open('/proc/cpuinfo').read()\nprint(json.dumps({'os':r['ID'],'version':r['VERSION_ID'],'arch':platform.machine(),'intel':'GenuineIntel' in cpu}))\nPY\n`)
    const response = yield* checkedCommand(config.executable, ["vm", "run-command", "invoke", "--subscription", config.subscription,
      "--resource-group", config.resourceGroup, "--name", machine.name, "--command-id", "RunShellScript", "--scripts", `@${script}`, "--output", "json", "--only-show-errors"], { timeoutMs: 600_000 })
    yield* fs.writeFileString(join(root, "run-command.json"), response.stdout)
    const body = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ value: Schema.Array(Schema.Struct({ code: Schema.String, message: Schema.String })) })))(response.stdout)
    const output = body.value.find(value => value.code.endsWith("/succeeded"))?.message.match(/\[stdout\]\s*([\s\S]*?)\s*\[stderr\]/)?.[1]
    if (!output) return yield* fail("Guest command did not return a successful bounded host observation")
    const host = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ os: Schema.Literal("ubuntu"), version: Schema.String, arch: Schema.Literal("x86_64"), intel: Schema.Literal(true) })))(output)
    if (host.version !== target.version) return yield* fail("Actual guest version differs from target")
  }).pipe(Effect.ensuring(allocator.inventory().pipe(Effect.flatMap(machines => Effect.forEach(machines.filter(machine => machine.tags.leaseId === lease.leaseId), machine => allocator.release(machine))),
    Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(error.message) })))), Effect.either)
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ passed: result._tag === "Right" && cleanupErrors.length === 0,
    detail: result._tag === "Right" ? "Allocated idempotently, observed Ubuntu Intel guest and released the lease" : String(result.left), cleanupErrors }))
  if (result._tag === "Left") return yield* result.left
  if (cleanupErrors.length) return yield* fail("Probe cleanup failed; inspect owned resources before continuing")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
