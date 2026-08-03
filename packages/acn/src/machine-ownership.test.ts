import { describe, expect, it } from "vitest"
import * as FileSystem from "@effect/platform/FileSystem"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import { BunCommandExecutor, BunFileSystem } from "@effect/platform-bun"
import { Deferred, Effect, Exit, Fiber, Layer, Option, Scope } from "effect"
import { mkdtemp, mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { acquireAcnMachineOwnership } from "./machine-ownership"
import { AcnOwnerIdSchema } from "@magnitudedev/acn-protocol"

const ownerId = AcnOwnerIdSchema.make

const run = <A, E>(
  effect: Effect.Effect<A, E, FileSystem.FileSystem | CommandExecutor.CommandExecutor>
) => Effect.runPromise(effect.pipe(
  Effect.provide(Layer.merge(
    BunFileSystem.layer,
    BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
  )),
))

describe("ACN machine ownership", () => {
  it("waits for the exact live owner and acquires after its release", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-owner-"))
    await run(
      Effect.gen(function* () {
        const firstScope = yield* Scope.make()
        const secondScope = yield* Scope.make()
        yield* acquireAcnMachineOwnership({ dataDir, id: ownerId("first"), version: "1" }).pipe(
          Effect.provideService(Scope.Scope, firstScope),
        )
        const waiting = yield* Effect.fork(
          acquireAcnMachineOwnership({ dataDir, id: ownerId("second"), version: "2" }).pipe(
            Effect.provideService(Scope.Scope, secondScope),
          ),
        )
        yield* Effect.sleep("150 millis")
        expect(Option.isNone(yield* Fiber.poll(waiting))).toBe(true)
        yield* Scope.close(firstScope, Exit.void)
        yield* Fiber.join(waiting)
        yield* Scope.close(secondScope, Exit.void)
      }),
    )
  })

  it("recovers a dead owner's exact record", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-stale-owner-"))
    const directory = join(dataDir, "acn")
    await mkdir(directory, { recursive: true })
    await writeFile(
      join(directory, "owner"),
      JSON.stringify({ id: "dead", pid: 2_147_483_647, version: "old", startedAt: 1 }),
    )
    await run(
      Effect.scoped(
        acquireAcnMachineOwnership({ dataDir, id: ownerId("replacement"), version: "new" }),
      ),
    )
    expect(await Bun.file(join(directory, "owner")).exists()).toBe(false)
  })

  it("does not acquire after an interrupted ownership wait", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-owner-interrupt-"))
    const ownerExists = await run(
      Effect.gen(function* () {
        const firstScope = yield* Scope.make()
        yield* acquireAcnMachineOwnership({
          dataDir,
          id: ownerId("active"),
          version: "1",
        }).pipe(Effect.provideService(Scope.Scope, firstScope))
        const waiting = yield* acquireAcnMachineOwnership({
          dataDir,
          id: ownerId("superseded"),
          version: "2",
        }).pipe(
          Effect.scoped,
          Effect.fork,
        )
        yield* Effect.sleep("150 millis")
        yield* Fiber.interrupt(waiting)
        yield* Scope.close(firstScope, Exit.void)
        yield* Effect.sleep("150 millis")
        const exists = yield* Effect.promise(() =>
          Bun.file(join(dataDir, "acn", "owner")).exists()
        )
        return exists
      }),
    )
    expect(ownerExists).toBe(false)
  })

  it("serializes simultaneous candidates without overlapping ownership", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-owner-race-"))
    const order = await run(
      Effect.gen(function* () {
        const scopes = yield* Effect.all([Scope.make(), Scope.make()])
        const firstAcquired = yield* Deferred.make<string>()
        const acquired = yield* Effect.all(
          ["left", "right"].map((id, index) =>
            acquireAcnMachineOwnership({ dataDir, id: ownerId(id), version: "1.0.0" }).pipe(
              Effect.provideService(Scope.Scope, scopes[index]!),
              Effect.tap(() => Deferred.succeed(firstAcquired, id)),
              Effect.as(id),
              Effect.fork,
            ),
          ),
          { concurrency: "unbounded" },
        )
        const first = yield* Deferred.await(firstAcquired)
        const firstIndex = first === "left" ? 0 : 1
        const secondIndex = firstIndex === 0 ? 1 : 0
        expect(Option.isNone(yield* Fiber.poll(acquired[secondIndex]!))).toBe(true)
        yield* Scope.close(scopes[firstIndex]!, Exit.void)
        const second = yield* Fiber.join(acquired[secondIndex]!)
        yield* Scope.close(scopes[secondIndex]!, Exit.void)
        return [first, second]
      }),
    )
    expect(new Set(order)).toEqual(new Set(["left", "right"]))
  })
})
