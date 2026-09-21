import { DateTime, Effect, Layer, Schema } from "effect"
import { expect, test } from "vitest"
import { targets } from "../src/catalog"
import { LeaseId, RunId } from "../src/domain"
import { Allocating, Fence } from "../src/lease"
import { MachineAllocator, MachineTags, SshMachine, WorkerTransport } from "../src/machines"
import { ProcessExecutor } from "../src/process"
import { sparkAllocator, sparkTransport } from "../src/providers/spark"
import { WorkId } from "../src/work-identity"

test("Spark leases are exclusive, bounded, idempotent and remove only their exact container", async () => {
  const target = targets.find(t => t.provider === "spark" && t.backend === "cuda")!
  const lease = new Allocating({ provider: "spark", leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`),
    runId: RunId.make(`run-${crypto.randomUUID()}`), workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
    targetId: target.id, resourceName: "ml-123456789abc", expiresAt: DateTime.unsafeMake(Date.now() + 60_000) })
  let labels: Record<string, string> | undefined
  const recorded: string[][] = []
  const executor = Layer.succeed(ProcessExecutor, ProcessExecutor.of({ run: spec => Effect.sync(() => {
    expect(spec.args.slice(0, 2)).toEqual(["--host", "ssh://tom@sparky"])
    const args = spec.args.slice(2); recorded.push([...args])
    if (args[0] === "container") return labels
      ? { exitCode: 0, stdout: Schema.encodeSync(Schema.parseJson(Schema.Unknown))({ id: "a".repeat(64), labels }), stderr: "" }
      : { exitCode: 1, stdout: "", stderr: "Error: No such container: magnitude-lab-spark" }
    if (args[0] === "run") {
      expect(args).toContain("--cpus"); expect(args).toContain("--memory")
      expect(args).not.toContain("--privileged"); expect(args).not.toContain("--volume")
      expect(args[args.indexOf("--cap-add") + 1]).toBe("NET_ADMIN")
      expect(args).not.toContain("--network=host")
      const value = args[args.indexOf("--label") + 1]!, split = value.indexOf("=")
      labels = { [value.slice(0, split)]: value.slice(split + 1) }
      return { exitCode: 0, stdout: "a".repeat(64), stderr: "" }
    }
    expect(args).toEqual(["rm", "--force", "a".repeat(64)])
    labels = undefined
    return { exitCode: 0, stdout: "", stderr: "" }
  }) }))
  await Effect.runPromise(Effect.gen(function* () {
    const allocator = yield* MachineAllocator
    const machine = yield* allocator.ensure(lease, target)
    expect(yield* allocator.ensure(lease, target)).toEqual(machine)
    const conflict = yield* allocator.ensure(new Allocating({ ...lease, leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`) }), target).pipe(Effect.either)
    expect(conflict._tag).toBe("Left")
    const build = yield* allocator.ensure(new Allocating({ ...lease, workId: WorkId.make("build:linux-arm64-gnu:cuda") }), target).pipe(Effect.either)
    expect(build._tag).toBe("Left")
    yield* allocator.release(machine)
    yield* allocator.release(machine)
  }).pipe(Effect.provide(sparkAllocator({ executable: "docker", host: "ssh://tom@sparky", image: `ubuntu@sha256:${"b".repeat(64)}` }).pipe(Layer.provide(executor)))))
  expect(recorded.filter(args => args[0] === "run")).toHaveLength(1)
  expect(recorded.filter(args => args[0] === "rm")).toHaveLength(1)
})

test("Spark transfers use container IDs and reject escaped paths and changed ownership", async () => {
  const tags = MachineTags.make({ schemaVersion: 1, leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`),
    runId: RunId.make(`run-${crypto.randomUUID()}`), expiresAt: DateTime.unsafeMake(Date.now() + 60_000) })
  const machine = SshMachine.make({ provider: "spark", host: "ssh://tom@sparky", containerId: "a".repeat(64), tags })
  const commands: string[][] = []
  let changed = false
  const executor = Layer.succeed(ProcessExecutor, ProcessExecutor.of({ run: spec => Effect.sync(() => {
    const args = spec.args.slice(2); commands.push([...args])
    return { exitCode: 0, stderr: "", stdout: args[0] === "container" ? Schema.encodeSync(Schema.parseJson(Schema.Unknown))({
      id: changed ? "b".repeat(64) : machine.containerId,
      labels: { "dev.magnitude.lab.lease": Buffer.from(Schema.encodeSync(Schema.parseJson(MachineTags))(tags)).toString("base64") },
    }) : "" }
  }) }))
  await Effect.runPromise(Effect.gen(function* () {
    const transport = yield* WorkerTransport
    const path = `/lab/${tags.leaseId}/attempt-1/input.tar.gz`
    yield* transport.upload(machine, "/local/input.tar.gz", path)
    yield* transport.download(machine, path, "/local/output.tar.gz")
    expect(commands.filter(args => args[0] === "cp")).toEqual([
      ["cp", "/local/input.tar.gz", `${machine.containerId}:${path}`],
      ["cp", `${machine.containerId}:${path}`, "/local/output.tar.gz"],
    ])
    for (const escaped of ["/etc/shadow", `/lab/${tags.leaseId}/../other/file`, "/lab/other/file"]) {
      const result = yield* transport.upload(machine, "/local/file", escaped).pipe(Effect.either)
      expect(result._tag).toBe("Left")
    }
    expect(commands.filter(args => args[0] === "cp")).toHaveLength(2)
    changed = true
    const result = yield* transport.execute(machine, "true", [], 1000).pipe(Effect.either)
    expect(result._tag).toBe("Left")
    expect(commands.some(args => args.includes("true"))).toBe(false)
  }).pipe(Effect.provide(sparkTransport({ executable: "docker", host: machine.host, image: `ubuntu@sha256:${"b".repeat(64)}` }).pipe(Layer.provide(executor)))))
})
