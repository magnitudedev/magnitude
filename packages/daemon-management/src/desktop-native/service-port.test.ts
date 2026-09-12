import { createServer } from "node:http"
import { Effect, Either } from "effect"
import { expect, it } from "vitest"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"
import { requireServicePort } from "./service-port"

const command = { executable: "test-service", arguments: [], environment: {} }
const reachedChild = new OwnedChildSpawnFailed({ executable: command.executable, message: "Reached child creation" })
const listener = Effect.acquireRelease(
  Effect.async<ReturnType<typeof createServer>>(resume => {
    const server = createServer((_request, response) => response.end("incumbent"))
    server.listen(0, "127.0.0.1", () => resume(Effect.succeed(server)))
  }),
  server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }),
)

it("leaves an occupied port untouched and releases its probe before retrying child creation", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const incumbent = yield* listener
    const address = incumbent.address()
    if (!address || typeof address === "string") throw new Error("Expected TCP address")
    let spawns = 0
    const spawner = yield* requireServicePort(address.port).pipe(Effect.provideService(OwnedChildSpawner, {
      spawn: () => Effect.sync(() => { spawns++ }).pipe(Effect.zipRight(Effect.fail(reachedChild))),
    }))
    for (let attempt = 0; attempt < 2; attempt++) {
      const result = yield* Effect.either(spawner.spawn(command))
      expect(Either.isLeft(result) && result.left.message).toContain(`Port ${address.port} is already in use`)
      expect(spawns).toBe(0)
      expect(yield* Effect.promise(() => fetch(`http://127.0.0.1:${address.port}`).then(response => response.text()))).toBe("incumbent")
    }
    yield* Effect.async<void>(resume => { incumbent.close(() => resume(Effect.void)) })
    const result = yield* Effect.either(spawner.spawn(command))
    expect(Either.isLeft(result) && result.left).toBe(reachedChild)
    expect(spawns).toBe(1)
    // The preflight is not a persistent listener; a child can bind the exact port.
    const replacement = yield* listener
    yield* Effect.async<void>(resume => { replacement.close(() => resume(Effect.void)) })
    yield* Effect.async<void>(resume => { replacement.listen(address.port, "127.0.0.1", () => resume(Effect.void)) })
  })))
})

it("reports invalid port admission without starting a child", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const spawner = yield* requireServicePort(-1).pipe(Effect.provideService(OwnedChildSpawner, {
      spawn: () => Effect.die("Invalid port must not reach child creation"),
    }))
    const result = yield* Effect.either(spawner.spawn(command))
    expect(Either.isLeft(result) && result.left.message).toContain("could not open its local service port")
  })))
})
