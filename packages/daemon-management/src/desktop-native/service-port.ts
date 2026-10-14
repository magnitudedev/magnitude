import { createServer } from "node:net"
import { Effect, Option } from "effect"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"

const errorCode = (error: unknown) =>
  typeof error === "object" && error !== null && "code" in error && typeof error.code === "string" ? Option.some(error.code) : Option.none()

/** Windows reports a port bound exclusively by another account as access denied rather than in use. */
const occupied = (error: unknown) => Option.exists(errorCode(error), code => code === "EADDRINUSE" || (code === "EACCES" && process.platform === "win32"))

const servedByMagnitude = (port: number) => Effect.tryPromise(signal =>
  fetch(`http://127.0.0.1:${port}/health`, { signal }).then(response => response.ok ? response.json() : null)).pipe(
  Effect.map(body => typeof body === "object" && body !== null && "service" in body && body.service === "magnitude-acn"),
  Effect.timeout("2 seconds"), Effect.orElseSucceed(() => false))

const occupiedMessage = (port: number, magnitude: boolean) => magnitude
  ? `Another Magnitude server is already running on this computer on port ${port}, for example \`magnitude serve\` under a service such as NSSM or another account. `
    + "Stop it (for an NSSM service, run `nssm stop Magnitude` as an administrator), then retry."
  : `Port ${port} is already in use. Stop the other service, then retry. If upgrading Magnitude, quit the old app and disable its service startup first.`

/** A diagnostic preflight, not ownership: the service's own bind remains authoritative. */
export const checkServicePort = (port: number, executable: string) => Effect.scoped(Effect.gen(function* () {
  const server = yield* Effect.acquireRelease(Effect.sync(() => createServer(socket => socket.destroy())),
    server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }))
  const failure = yield* Effect.async<Option.Option<unknown>>(resume => {
    const failed = (error: unknown) => resume(Effect.succeed(Option.some(error)))
    server.once("error", failed)
    server.once("listening", () => resume(Effect.succeed(Option.none())))
    try { server.listen({ host: "127.0.0.1", port, exclusive: true }) } catch (error) { failed(error) }
  })
  if (Option.isNone(failure)) return
  const message = occupied(failure.value)
    ? occupiedMessage(port, yield* servedByMagnitude(port))
    : `Magnitude could not open its local service port (${port}). Check local network permissions, then retry.`
  return yield* new OwnedChildSpawnFailed({ executable, message })
}))

export const requireServicePort = (port: number) => Effect.map(OwnedChildSpawner, spawner => OwnedChildSpawner.of({
  spawn: command => checkServicePort(port, command.executable).pipe(Effect.zipRight(spawner.spawn(command))),
}))
