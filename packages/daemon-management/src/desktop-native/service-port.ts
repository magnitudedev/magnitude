import { createServer } from "node:net"
import { Effect } from "effect"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"

/** A diagnostic preflight, not ownership: the service's own bind remains authoritative. */
export const checkServicePort = (port: number, executable: string) => Effect.scoped(Effect.gen(function* () {
  const server = yield* Effect.acquireRelease(Effect.sync(() => createServer(socket => socket.destroy())),
    server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }))
  yield* Effect.async<void, OwnedChildSpawnFailed>(resume => {
    const code = (error: unknown) => typeof error === "object" && error !== null && "code" in error ? error.code : undefined
    // Windows refuses a port another account holds exclusively with EACCES rather than EADDRINUSE.
    const inUse = (error: unknown) => code(error) === "EADDRINUSE" || (code(error) === "EACCES" && port >= 1024)
    const failed = (error: unknown) => resume(Effect.fail(new OwnedChildSpawnFailed({
      executable,
      message: inUse(error)
        ? `Port ${port} is already in use, by Magnitude running for another account on this computer or by another program. `
          + "Stop it, then retry. If upgrading Magnitude, quit the old app and disable its service startup first."
        : `Magnitude could not open its local service port (${port}). Check local network permissions, then retry.`,
    })))
    server.once("error", failed)
    server.once("listening", () => resume(Effect.void))
    try { server.listen({ host: "127.0.0.1", port, exclusive: true }) } catch (error) { failed(error) }
  })
}))

export const requireServicePort = (port: number) => Effect.map(OwnedChildSpawner, spawner => OwnedChildSpawner.of({
  spawn: command => checkServicePort(port, command.executable).pipe(Effect.zipRight(spawner.spawn(command))),
}))
