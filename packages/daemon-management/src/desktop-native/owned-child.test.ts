import { fileURLToPath } from "node:url"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { makeUnixOwnedChildSpawner, type OwnedChild } from "./owned-child"

const addon = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/desktop-host.node`, import.meta.url))
const fixture = fileURLToPath(new URL("./fixtures/process.cjs", import.meta.url))
const run = <A, E>(work: Effect.Effect<A, E, ProcessGroupController>) => Effect.runPromise(work.pipe(
  Effect.provideService(ProcessGroupController, ProcessGroupControllerLive),
))
const command = (mode: string) => ({ executable: process.execPath, arguments: [fixture, addon, mode], environment: process.env })
const workerPid = (child: OwnedChild) => Effect.gen(function* () {
  for (;;) {
    const tail = yield* child.diagnosticTail
    if (tail.includes("\n")) return (yield* Schema.decode(Schema.parseJson(Schema.Struct({ worker: Schema.Number })))(tail.trim())).worker
    yield* Effect.sleep("10 millis")
  }
}).pipe(Effect.timeout("3 seconds"))
const alive = (pid: number) => { try { process.kill(pid, 0); return true } catch { return false } }

describe.skipIf(process.platform === "win32")("owned service child", () => {
  it("scope close retires a real child and its worker", async () => {
    let leader = 0, worker = 0
    await run(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeUnixOwnedChildSpawner
      const child = yield* spawner.spawn(command("owned"))
      leader = child.identity.pid
      worker = yield* workerPid(child)
      expect(alive(worker)).toBe(true)
    })))
    expect(alive(leader)).toBe(false)
    expect(alive(worker)).toBe(false)
  })

  it("cleans survivors when the leader has already died", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeUnixOwnedChildSpawner
      const child = yield* spawner.spawn(command("owned"))
      const worker = yield* workerPid(child)
      process.kill(child.identity.pid, "SIGKILL")
      yield* child.exit
      expect(alive(worker)).toBe(true)
      yield* child.stop
      expect(alive(worker)).toBe(false)
    })))
  })

  it("escalates a stubborn child and shares repeated stop calls", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeUnixOwnedChildSpawner
      const child = yield* spawner.spawn(command("owned-stubborn"))
      const worker = yield* workerPid(child)
      yield* Effect.all([child.stop, child.stop, child.stop], { concurrency: "unbounded" })
      expect(alive(child.identity.pid)).toBe(false)
      expect(alive(worker)).toBe(false)
    })))
  }, 10000)

  it("reports spawn failure as a typed outcome", async () => {
    await run(Effect.scoped(Effect.gen(function* () {
      const spawner = yield* makeUnixOwnedChildSpawner
      const result = yield* spawner.spawn({ executable: "/missing/magnitude-service", arguments: [], environment: {} }).pipe(Effect.either)
      expect(result._tag === "Left" && result.left._tag).toBe("OwnedChildSpawnFailed")
    })))
  })
})
