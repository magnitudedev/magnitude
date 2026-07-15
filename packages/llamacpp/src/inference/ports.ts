import { Effect } from "effect"
import * as net from "node:net"

/**
 * Find a free TCP port on localhost.
 * Tries the preferred port first; if taken, lets the OS assign one.
 */
export function findFreePort(
  preferred: number,
): Effect.Effect<number, never, never> {
  return Effect.gen(function* () {
    const preferredFree = yield* checkPortFree(preferred)
    if (preferredFree) return preferred
    return yield* getOsPort()
  })
}

function checkPortFree(port: number): Effect.Effect<boolean, never, never> {
  return Effect.async<boolean>((resume) => {
    const server = net.createServer()
    server.once("error", () => resume(Effect.succeed(false)))
    server.listen(port, "127.0.0.1", () => {
      server.close(() => resume(Effect.succeed(true)))
    })
  })
}

function getOsPort(): Effect.Effect<number, never, never> {
  return Effect.async<number>((resume) => {
    const server = net.createServer()
    server.once("error", () => resume(Effect.succeed(0)))
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address()
      const port = addr && typeof addr === "object" ? addr.port : 0
      server.close(() => resume(Effect.succeed(port)))
    })
  })
}
