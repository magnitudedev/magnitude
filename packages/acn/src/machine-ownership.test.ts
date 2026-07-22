import { describe, expect, it } from "vitest"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Either, Exit, Scope } from "effect"
import { mkdtemp, mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { AcnMachineAlreadyOwned, acquireAcnMachineOwnership } from "./machine-ownership"

const run = <A, E, R>(effect: Effect.Effect<A, E, R>) =>
  Effect.runPromise(effect.pipe(Effect.provide(BunFileSystem.layer)) as Effect.Effect<A, E, never>)

describe("ACN machine ownership", () => {
  it("admits one owner and releases only its exact record", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-owner-"))
    const result = await run(
      Effect.gen(function* () {
        const firstScope = yield* Scope.make()
        const secondScope = yield* Scope.make()
        yield* acquireAcnMachineOwnership({ dataDir, id: "first", version: "1" }).pipe(
          Effect.provideService(Scope.Scope, firstScope),
        )
        const collision = yield* Effect.either(
          acquireAcnMachineOwnership({ dataDir, id: "second", version: "2" }).pipe(
            Effect.provideService(Scope.Scope, secondScope),
          ),
        )
        yield* Scope.close(firstScope, Exit.void)
        yield* acquireAcnMachineOwnership({ dataDir, id: "second", version: "2" }).pipe(
          Effect.provideService(Scope.Scope, secondScope),
        )
        yield* Scope.close(secondScope, Exit.void)
        return collision
      }),
    )
    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) expect(result.left).toBeInstanceOf(AcnMachineAlreadyOwned)
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
        acquireAcnMachineOwnership({ dataDir, id: "replacement", version: "new" }),
      ),
    )
    expect(await Bun.file(join(directory, "owner")).exists()).toBe(false)
  })

  it("linearizes simultaneous candidates to exactly one owner", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-owner-race-"))
    const outcomes = await run(
      Effect.gen(function* () {
        const scopes = yield* Effect.all([Scope.make(), Scope.make()])
        const attempts = yield* Effect.all(
          ["left", "right"].map((id, index) =>
            acquireAcnMachineOwnership({ dataDir, id, version: "1.0.0" }).pipe(
              Effect.provideService(Scope.Scope, scopes[index]!),
              Effect.either,
            ),
          ),
          { concurrency: "unbounded" },
        )
        yield* Effect.forEach(scopes, (scope) => Scope.close(scope, Exit.void), {
          discard: true,
        })
        return attempts
      }),
    )
    expect(outcomes.filter(Either.isRight)).toHaveLength(1)
    expect(outcomes.filter(Either.isLeft)).toHaveLength(1)
  })
})
