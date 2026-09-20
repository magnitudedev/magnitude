import { Fence } from "../src/lease"
import { WorkId } from "../src/work-identity"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Context, DateTime, Effect, Layer, Redacted, Schema } from "effect"
import { join } from "node:path"
import { findTarget } from "../src/catalog"
import { InfrastructureFailure, LeaseId, RunId, TargetId } from "../src/domain"
import { Allocating } from "../src/lease"
import { MachineAllocator } from "../src/machines"
import { azureAllocator, AzureConfig } from "../src/providers/azure"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { azureLinuxBootstrap } from "../src/providers/azure-bootstrap"
import { WorkerLaunch } from "../src/outward-runner"
import { sha256 } from "../src/snapshot"
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
  const lease = new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id, workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
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
    const bootstrap = yield* azureLinuxBootstrap(config)
    const token = Redacted.make(crypto.randomUUID())
    const guestRoot = `/home/${config.adminUsername}/bootstrap-probe`
    const guestScript = "import os,json,hashlib,pwd; print(json.dumps(dict(user=pwd.getpwuid(os.getuid()).pw_name,root=os.environ['LAB_WORKER_ROOT'],origin=os.environ['LAB_URL'],credentialDigest=hashlib.sha256(os.environ['LAB_WORKER_TOKEN'].encode()).hexdigest())))"
    yield* bootstrap.start(machine, WorkerLaunch.make({ executable: "/usr/bin/python3", args: ["-c", guestScript], root: guestRoot,
      origin: "https://lab.example.com", token, deadline: DateTime.unsafeMake(Date.now() + 10 * 60_000) }))
    const observation = yield* Effect.gen(function* () {
      for (;;) {
        const response = yield* checkedCommand(config.executable, ["rest", "--subscription", config.subscription, "--method", "GET", "--url",
          `https://management.azure.com${machine.id}/runCommands/lab-worker?api-version=2024-11-01&$expand=instanceView`, "--only-show-errors", "--output", "json"])
        const body = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ properties: Schema.Struct({
          instanceView: Schema.optionalWith(Schema.Struct({ executionState: Schema.String,
            output: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
            error: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
          }), { as: "Option", exact: true }),
        }) })))(response.stdout)
        if (body.properties.instanceView._tag === "Some") {
          const view = body.properties.instanceView.value
          yield* fs.writeFileString(join(root, "bootstrap-observation.json"), (yield* Schema.encode(Schema.parseJson(Schema.Unknown))(view)).replaceAll(Redacted.value(token), "[REDACTED]"))
          if (view.executionState === "Failed" || view.executionState === "Canceled" || view.executionState === "TimedOut") return yield* fail(`Bootstrap guest ended in ${view.executionState}`)
          if (view.executionState === "Succeeded" && view.output._tag === "Some") return view.output.value
        }
        yield* Effect.sleep("5 seconds")
      }
    }).pipe(Effect.timeoutFail({ duration: "10 minutes", onTimeout: () => fail("Managed guest probe did not complete") }))
    const guest = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ user: Schema.String, root: Schema.String, origin: Schema.String, credentialDigest: Schema.String })))(observation.trim())
    if (guest.user !== config.adminUsername || guest.root !== guestRoot || guest.origin !== "https://lab.example.com" || guest.credentialDigest !== sha256(Redacted.value(token))) return yield* fail("Protected credential or guest execution identity differed from the launch")
    yield* fs.writeFileString(join(root, "bootstrap.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ passed: true, user: guest.user, root: guest.root, credentialMatched: true }))
  }).pipe(Effect.ensuring(allocator.inventory().pipe(Effect.flatMap(machines => Effect.forEach(machines.filter(machine => machine.tags.leaseId === lease.leaseId), machine => allocator.release(machine))),
    Effect.catchAll(error => Effect.sync(() => { cleanupErrors.push(error.message) })))), Effect.either)
  yield* fs.writeFileString(join(root, "report.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))({ passed: result._tag === "Right" && cleanupErrors.length === 0,
    detail: result._tag === "Right" ? "Allocated idempotently, observed Ubuntu Intel guest, verified protected bootstrap credential and guest user, and released the lease" : String(result.left), cleanupErrors }))
  if (result._tag === "Left") return yield* result.left
  if (cleanupErrors.length) return yield* fail("Probe cleanup failed; inspect owned resources before continuing")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
