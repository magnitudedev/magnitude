import { DateTime, Effect, Layer, Schema } from "effect"
import { InfrastructureFailure } from "../domain"
import { Machine, MachineAllocator, MachineTags, SshMachine, WorkerTransport } from "../machines"
import { posix } from "node:path"
import { command, ProcessExecutor } from "../process"

export const SparkConfig = Schema.Struct({
  executable: Schema.NonEmptyString,
  host: Schema.String.pipe(Schema.pattern(/^ssh:\/\/[a-z][a-z0-9_-]*@[a-zA-Z0-9.-]+$/)),
  image: Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9./_-]+@sha256:[a-f0-9]{64}$/)),
})
const name = "magnitude-lab-spark"
const label = "dev.magnitude.lab.lease"
const failed = (message: string) => new InfrastructureFailure({ operation: "spark", message })

/** One named container is the exclusive lab lease. Other containers and host jobs are never inspected. */
const sparkAccess = (config: typeof SparkConfig.Type) => Effect.gen(function* () {
  const executor = yield* ProcessExecutor
  const run = (args: readonly string[], timeoutMs = 120_000) => command(config.executable, ["--host", config.host, ...args], { timeoutMs }).pipe(
    Effect.provideService(ProcessExecutor, executor))
  const inspect = () => Effect.gen(function* () {
    const result = yield* run(["container", "inspect", "--format", '{"id":{{json .Id}},"labels":{{json .Config.Labels}}}', name])
    if (result.exitCode !== 0) {
      if (result.exitCode === 1 && /No such (object|container)/i.test(result.stderr)) return []
      return yield* failed("Cannot inspect the owned Spark container")
    }
    const record = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ id: SshMachine.fields.containerId, labels: Schema.Record({ key: Schema.String, value: Schema.String }) })))(result.stdout)
    const labels = record.labels
    if (!labels[label]) return yield* failed("Spark container name is occupied without lab ownership")
    const tags = yield* Schema.decodeUnknown(Schema.parseJson(MachineTags))(Buffer.from(labels[label], "base64").toString("utf8"))
    return [SshMachine.make({ provider: "spark", host: config.host, containerId: record.id, tags })]
  }).pipe(Effect.mapError(() => failed("Cannot verify Spark container ownership")))
  return { run, inspect }
})

export const sparkAllocator = (config: typeof SparkConfig.Type) => Layer.effect(MachineAllocator, Effect.gen(function* () {
  const { run, inspect } = yield* sparkAccess(config)
  return MachineAllocator.of({
    inventory: inspect,
    ensure: (lease, target) => Effect.gen(function* () {
      if (lease.provider !== "spark" || target.provider !== "spark" || target.arch !== "arm64" || target.hardware !== "dgx-spark") {
        return yield* failed("Spark requires its explicit ARM64 hardware target")
      }
      if (!lease.workId.startsWith("test:")) return yield* failed("Spark does not build source; use the Azure ARM64 producer")
      if (DateTime.toEpochMillis(lease.expiresAt) <= Date.now()) return yield* failed("Spark lease has expired")
      const tags = MachineTags.make({ schemaVersion: 1, leaseId: lease.leaseId, runId: lease.runId, expiresAt: lease.expiresAt })
      const current = yield* inspect()
      if (current.length) {
        if (!Schema.equivalence(MachineTags)(current[0]!.tags, tags)) return yield* failed("Spark already has an active lab lease")
        return current[0]!
      }
      const encoded = Buffer.from(yield* Schema.encode(Schema.parseJson(MachineTags))(tags)).toString("base64")
      const result = yield* run(["run", "--detach", "--name", name, "--label", `${label}=${encoded}`,
        "--cpus", "2", "--memory", "8g", "--pids-limit", "512", "--device", "nvidia.com/gpu=0",
        "--cap-add", "NET_ADMIN",
        "--init", config.image, "sleep", String(Math.max(1, Math.ceil((DateTime.toEpochMillis(lease.expiresAt) - Date.now()) / 1000)))])
      if (result.exitCode !== 0) return yield* failed("Could not create the isolated Spark lease")
      const created = yield* inspect()
      if (created.length !== 1 || !Schema.equivalence(MachineTags)(created[0]!.tags, tags)) return yield* failed("Spark allocation ownership mismatch")
      return created[0]!
    }).pipe(Effect.mapError(error => error instanceof InfrastructureFailure ? error : failed("Spark allocation failed"))),
    release: machine => Effect.gen(function* () {
      if (machine.provider !== "spark" || machine.host !== config.host) return yield* failed("Wrong Spark allocation endpoint")
      const current = yield* inspect()
      if (!current.length) return
      if (!Schema.equivalence(MachineTags)(current[0]!.tags, machine.tags)) return yield* failed("Spark lease changed; refusing cleanup")
      if (current[0]!.containerId !== machine.containerId) return yield* failed("Spark container identity changed; refusing cleanup")
      const result = yield* run(["rm", "--force", machine.containerId])
      if (result.exitCode !== 0 || (yield* inspect()).length) return yield* failed("Spark container cleanup did not finish")
    }),
  })
}))

export const sparkTransport = (config: typeof SparkConfig.Type) => Layer.effect(WorkerTransport, Effect.gen(function* () {
  const { run, inspect } = yield* sparkAccess(config)
  const verify = (machine: Machine) => Effect.gen(function* () {
    if (machine.provider !== "spark" || machine.host !== config.host) return yield* failed("Wrong Spark transport endpoint")
    const current = yield* inspect()
    if (current.length !== 1 || current[0]!.containerId !== machine.containerId || !Schema.equivalence(MachineTags)(current[0]!.tags, machine.tags)) {
      return yield* failed("Spark transport lease no longer owns its container")
    }
    return machine
  })
  const checked = (args: readonly string[]) => run(args).pipe(Effect.flatMap(result => result.exitCode === 0 ? Effect.void
    : Effect.fail(failed("Spark file transfer failed"))))
  const transferPath = (machine: typeof SshMachine.Type, path: string) => Effect.gen(function* () {
    const root = `/lab/${machine.tags.leaseId}/`
    if (!path.startsWith(root) || path.includes("\0") || posix.normalize(path) !== path || path.endsWith("/")) {
      return yield* failed("Spark transfer path escapes its lease workspace")
    }
    // Check inside the container, never resolve a container path on the office host.
    yield* checked(["exec", "--user", "labworker", machine.containerId, "python3", "-c", String.raw`
import os,sys
p=sys.argv[1]
if os.path.realpath(p)!=p: raise SystemExit('Redirected transfer path')
os.makedirs(os.path.dirname(p),exist_ok=True)
`, path])
    return path
  })
  return WorkerTransport.of({
    execute: (machine, executable, args, timeoutMs) => Effect.gen(function* () {
      const owned = yield* verify(machine)
      return yield* run(["exec", "--user", "labworker", "--workdir", "/lab", owned.containerId, executable, ...args], timeoutMs)
    }),
    upload: (machine, local, remote) => Effect.gen(function* () {
      const owned = yield* verify(machine)
      const path = yield* transferPath(owned, remote)
      yield* checked(["cp", local, `${owned.containerId}:${path}`])
      yield* checked(["exec", "--user", "0", owned.containerId, "chown", "labworker:labworker", path])
    }),
    download: (machine, remote, local) => Effect.gen(function* () {
      const owned = yield* verify(machine)
      const path = yield* transferPath(owned, remote)
      yield* checked(["cp", `${owned.containerId}:${path}`, local])
    }),
  })
}))
