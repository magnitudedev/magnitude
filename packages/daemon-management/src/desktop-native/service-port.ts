import { createServer } from "node:net"
import { Effect } from "effect"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"

/** A diagnostic preflight, not ownership: the service's own bind remains authoritative. */
export const checkServicePort = (port: number, executable: string) => Effect.scoped(Effect.gen(function* () {
  const server = yield* Effect.acquireRelease(Effect.sync(() => createServer(socket => socket.destroy())),
    server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }))
  yield* Effect.async<void, OwnedChildSpawnFailed>(resume => {
    const failed = (error: unknown) => resume(Effect.fail(new OwnedChildSpawnFailed({
      executable,
      message: typeof error === "object" && error !== null && "code" in error && error.code === "EADDRINUSE"
        ? `Port ${port} is already in use. Stop the other service, then retry. If upgrading Magnitude, quit the old app and disable its service startup first.`
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
