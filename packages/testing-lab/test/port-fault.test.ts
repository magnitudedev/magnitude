import { Effect } from "effect"
import { createServer } from "node:net"
import { expect, test } from "vitest"
import { occupyServicePort } from "../src/port-fault"

test("reserves a loopback port and releases it when the owning scope ends", async () => {
  const port = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const port = yield* occupyServicePort(0)
    expect(port).toBeGreaterThan(0)
    const collision = yield* Effect.scoped(occupyServicePort(port)).pipe(Effect.either)
    expect(collision._tag).toBe("Left")
    return port
  })))
  const server = createServer()
  try {
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject)
      server.listen({ port, host: "127.0.0.1", exclusive: true }, resolve)
    })
    expect(server.listening).toBe(true)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
})
