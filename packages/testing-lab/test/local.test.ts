import { Fence } from "../src/lease"
import { WorkId } from "../src/work-identity"
import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { DateTime, Effect, Layer } from "effect"
import { join } from "node:path"
import { LeaseId, RunId } from "../src/domain"
import { Allocating } from "../src/lease"
import { MachineAllocator, WorkerTransport } from "../src/machines"
import { localAllocator, localTransport } from "../src/providers/local"
import { ProcessExecutorLive } from "../src/process"
import { targets } from "../src/catalog"

test("local leases isolate files and reject replacement, traversal and foreign ownership", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-local-test-" })
  const target = { ...targets[0]!, arch: process.arch === "arm64" ? "arm64" as const : "x64" as const,
    os: process.platform === "darwin" ? "macos" as const : process.platform === "win32" ? "windows" as const : "ubuntu" as const, provider: "local" as const }
  const lease = new Allocating({ leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`), runId: RunId.make(`run-${crypto.randomUUID()}`), targetId: target.id, workId: WorkId.make(`test:${target.id}`), workFence: Fence.make(1),
    provider: "local", resourceName: "ml-123456789abc", expiresAt: DateTime.unsafeMake(Date.now() + 60_000) })
  yield* Effect.gen(function* () {
    const allocator = yield* MachineAllocator
    const transport = yield* WorkerTransport
    const machine = yield* allocator.ensure(lease, target)
    if (machine.provider !== "local") return yield* Effect.dieMessage("Wrong local provider")
    const original = join(root, "input.txt")
    yield* fs.writeFileString(original, "source bytes")
    yield* transport.upload(machine, original, "source/input.txt")
    const output = yield* transport.execute(machine, process.execPath, ["-e", "console.log(await Bun.file('source/input.txt').text())"], 10_000)
    expect(output.stdout.trim()).toBe("source bytes")
    const downloaded = join(root, "downloaded.txt")
    yield* transport.download(machine, "source/input.txt", downloaded)
    expect(yield* fs.readFileString(downloaded)).toBe("source bytes")
    expect((yield* transport.upload(machine, original, "../escape").pipe(Effect.either))._tag).toBe("Left")
    const foreign = new Allocating({ ...lease, leaseId: LeaseId.make(`lease-${crypto.randomUUID()}`) })
    expect((yield* allocator.ensure(foreign, target).pipe(Effect.either))._tag).toBe("Left")
    yield* fs.symlink(root, join(machine.root, "outside"))
    expect((yield* transport.download(machine, "outside/input.txt", downloaded).pipe(Effect.either))._tag).toBe("Left")
    yield* allocator.release(machine)
    yield* allocator.release(machine)
    expect(yield* allocator.inventory()).toEqual([])
    expect(yield* fs.readFileString(original)).toBe("source bytes")
  }).pipe(Effect.provide(Layer.merge(localAllocator(join(root, "workers")), localTransport).pipe(Layer.provide(Layer.merge(BunContext.layer, ProcessExecutorLive)))))
})).pipe(Effect.provide(BunContext.layer))))
