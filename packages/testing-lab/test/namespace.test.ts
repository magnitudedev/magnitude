import { Fence } from "../src/lease"
import { WorkId } from "../src/work-identity"
import { expect, test } from "vitest"
import { DateTime, Effect, Layer, Schema } from "effect"
import { namespaceAllocator, type NamespaceImage } from "../src/providers/namespace"
import { ProcessExecutor, type CommandSpec } from "../src/process"
import { MachineAllocator, MachineTags } from "../src/machines"
import { Allocating } from "../src/lease"
import { targets } from "../src/catalog"
import { LeaseId, RunId } from "../src/domain"

const target = targets.find(t => t.os === "macos" && t.version === "26")!
const image: NamespaceImage = { version: "26", selector: "tahoe-slim", catalogCreatedAt: "2026-09-10T08:02:37.522Z", productVersion: "26.6.2", buildVersion: "25G83" }
const lease = () => new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id, workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
  provider: "namespace", resourceName: "magnitude-lab-fixture", expiresAt: DateTime.unsafeMake(Date.now() + 3600_000) })
const box = (allocation: Allocating) => ({ id: "owned-devbox", name: allocation.resourceName, documented_purpose: "magnitude-lab/v1 " + Schema.encodeSync(Schema.parseJson(MachineTags))({
  schemaVersion: 1, runId: allocation.runId, leaseId: allocation.leaseId, expiresAt: allocation.expiresAt,
}) })
const run = <A, E>(effect: Effect.Effect<A, E, MachineAllocator>, callback: (command: CommandSpec) => { exitCode: number; stdout: string; stderr: string }) => Effect.runPromise(effect.pipe(
  Effect.provide(namespaceAllocator("devbox", [image]).pipe(Layer.provide(Layer.succeed(ProcessExecutor, { run: command => Effect.sync(() => callback(command)) })))),
))
const output = (stdout: string) => ({ exitCode: 0, stdout, stderr: "" })

test("ambiguous create is reconciled through deterministic provider inventory", async () => {
  const allocation = lease()
  let created = false, creates = 0
  const machine = await run(Effect.flatMap(MachineAllocator, a => a.ensure(allocation, target)), command => {
    if (command.args[0] === "list") return output(JSON.stringify(created ? [box(allocation)] : []))
    if (command.args[0] === "image") return output(JSON.stringify([{ name: image.selector, created_at: image.catalogCreatedAt }]))
    if (command.args[0] === "create") { created = true; creates++; return { exitCode: 1, stdout: "", stderr: "Connection lost after provider allocation" } }
    if (command.args[0] === "exec") return output("ProductVersion:\t\t26.6.2\nBuildVersion:\t\t25G83\n")
    throw new Error("Unexpected command")
  })
  expect(machine.provider).toBe("namespace")
  expect(creates).toBe(1)
})
test("catalog drift is blocked before provisioning", async () => {
  const commands: string[] = []
  const result = await run(Effect.flatMap(MachineAllocator, a => a.ensure(lease(), target)).pipe(Effect.either), command => {
    commands.push(command.args[0]!)
    return output(command.args[0] === "list" ? "[]" : JSON.stringify([{ name: image.selector, created_at: "changed" }]))
  })
  expect(result._tag).toBe("Left")
  expect(commands).not.toContain("create")
})
test("a coincident resource name never authorizes adopting someone else's machine", async () => {
  const allocation = lease()
  const result = await run(Effect.flatMap(MachineAllocator, a => a.ensure(allocation, target)).pipe(Effect.either), command => {
    expect(command.args[0]).toBe("list")
    return output(JSON.stringify([{ ...box(lease()), name: allocation.resourceName }]))
  })
  expect(result._tag).toBe("Left")
})
test("inventory discovers tagged allocations without database acknowledgment", async () => {
  const allocation = lease()
  const result = await run(Effect.flatMap(MachineAllocator, a => a.inventory()), () => output(JSON.stringify([box(allocation), { id: "unrelated", name: "office" }])))
  expect(result).toHaveLength(1)
  expect(result[0]?.tags.leaseId).toBe(allocation.leaseId)
})

const emptyInventory = "No devbox available yet. Try running `devbox create`.\n[\n]\n"
test("empty Namespace inventory accepts only its exact documented CLI notice", async () => {
  expect(await run(Effect.flatMap(MachineAllocator, a => a.inventory()), () => output(emptyInventory))).toEqual([])
  for (const wire of ["unexpected diagnostic\n[]", emptyInventory.replace("[\n]", "{}"), emptyInventory.replace("[\n]", Schema.encodeSync(Schema.parseJson(Schema.Unknown))([box(lease())]))]) {
    expect((await run(Effect.flatMap(MachineAllocator, a => a.inventory()).pipe(Effect.either), () => output(wire)))._tag).toBe("Left")
  }
})
test("Namespace release verifies the provider's banner-prefixed empty inventory", async () => {
  const allocation = lease()
  let present = true
  const commands: string[] = []
  await run(Effect.gen(function* () {
    const allocator = yield* MachineAllocator
    const machine = (yield* allocator.inventory())[0]!
    yield* allocator.release(machine)
    expect(yield* allocator.inventory()).toEqual([])
  }), command => {
    commands.push(command.args[0]!)
    if (command.args[0] === "list") return output(present ? Schema.encodeSync(Schema.parseJson(Schema.Unknown))([box(allocation)]) : emptyInventory)
    if (command.args[0] === "shutdown") return output("")
    if (command.args[0] === "expire") { present = false; return output("") }
    throw new Error("Unexpected command")
  })
  expect(commands).toContain("shutdown")
  expect(commands).toContain("expire")
})

test("ephemeral shutdown can complete deletion without an expire request", async () => {
  const allocation = lease()
  let present = true
  const commands: string[] = []
  await run(Effect.gen(function* () {
    const allocator = yield* MachineAllocator
    const machine = (yield* allocator.inventory())[0]!
    yield* allocator.release(machine)
    yield* allocator.release(machine)
  }), command => {
    commands.push(command.args[0]!)
    if (command.args[0] === "list") return output(present ? JSON.stringify([box(allocation)]) : emptyInventory)
    if (command.args[0] === "shutdown") { present = false; return output("") }
    throw new Error("Deleted ephemeral machine must not receive further mutations")
  })
  expect(commands.filter(c => c === "shutdown")).toHaveLength(1)
  expect(commands).not.toContain("expire")
})

test("release refuses expiration when ownership changes during shutdown", async () => {
  const allocation = lease()
  let stopped = false
  const commands: string[] = []
  const result = await run(Effect.gen(function* () {
    const allocator = yield* MachineAllocator
    yield* allocator.release((yield* allocator.inventory())[0]!)
  }).pipe(Effect.either), command => {
    commands.push(command.args[0]!)
    if (command.args[0] === "list") return output(JSON.stringify([box(stopped ? lease() : allocation)]))
    if (command.args[0] === "shutdown") { stopped = true; return output("") }
    throw new Error("Ownership changes must prevent expiration")
  })
  expect(result._tag).toBe("Left")
  expect(commands).not.toContain("expire")
})
