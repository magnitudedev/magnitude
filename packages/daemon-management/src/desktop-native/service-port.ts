import { connect, createServer } from "node:net"
import { FetchHttpClient } from "@effect/platform"
import { Duration, Effect, Layer, Option } from "effect"
import { MagnitudeClient } from "@magnitudedev/sdk"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"

const errorCode = (error: unknown) =>
  typeof error === "object" && error !== null && "code" in error && typeof error.code === "string" ? Option.some(error.code) : Option.none()

/** Windows reports a port bound exclusively by another account as access denied rather than in use. */
const occupied = (error: unknown) => Option.exists(errorCode(error), code => code === "EADDRINUSE" || (code === "EACCES" && process.platform === "win32"))

const servedByMagnitude = (port: number) => Effect.tryPromise(signal =>
  fetch(`http://127.0.0.1:${port}/health`, { signal }).then(response => response.ok ? response.json() : null)).pipe(
  Effect.map(body => typeof body === "object" && body !== null && "service" in body && body.service === "magnitude-acn"),
  Effect.timeout("2 seconds"), Effect.orElseSucceed(() => false))

/** Where another Magnitude server on this computer usually comes from, and how its owner stops it, per platform. */
const otherMagnitude = (platform: NodeJS.Platform) => platform === "win32"
  ? "for example for another account or under a service such as NSSM. Stop it (for an NSSM service, run `nssm stop Magnitude` as an administrator), then retry."
  : platform === "darwin"
    ? "run by another account or as a background service. Ask the account that runs it to quit Magnitude, or stop its service, then retry."
    : "run by another account or as a system service. Stop it (for a service, `sudo systemctl stop magnitude`), or ask the account that runs Magnitude to quit it, then retry."

export const occupiedMessage = (port: number, magnitude: boolean, platform: NodeJS.Platform = process.platform) => magnitude
  ? `Another Magnitude server is already running on this computer on port ${port}, ${otherMagnitude(platform)}`
  : `Port ${port} is already in use by another program. Stop it, then retry. If upgrading Magnitude, quit the old app and disable its service startup first.`

const probePort = (port: number) => Effect.scoped(Effect.gen(function* () {
  const server = yield* Effect.acquireRelease(Effect.sync(() => createServer(socket => socket.destroy())),
    server => Effect.async<void>(resume => { server.close(() => resume(Effect.void)) }))
  return yield* Effect.async<Option.Option<unknown>>(resume => {
    const failed = (error: unknown) => resume(Effect.succeed(Option.some(error)))
    server.once("error", failed)
    server.once("listening", () => resume(Effect.succeed(Option.none())))
    try { server.listen({ host: "127.0.0.1", port, exclusive: true }) } catch (error) { failed(error) }
  })
}))

/** Whether another Magnitude, one this process's ownership lock cannot see, serves `port`. */
export const portServedByOtherMagnitude = (port: number) => probePort(port).pipe(
  Effect.flatMap(failure => Option.exists(failure, occupied) ? servedByMagnitude(port) : Effect.succeed(false)))

/** Whether anything holds `port` on loopback, including a server that is still shutting down. */
export const servicePortBound = (port: number) => probePort(port).pipe(Effect.map(failure => Option.exists(failure, occupied)))

/** Whether something accepts connections on `port` on loopback. */
const servicePortAccepts = (port: number) => Effect.async<boolean>(resume => {
  const socket = connect({ host: "127.0.0.1", port })
  const settle = (accepted: boolean) => { socket.destroy(); resume(Effect.succeed(accepted)) }
  socket.setTimeout(1000, () => settle(false))
  socket.once("connect", () => settle(true))
  socket.once("error", () => settle(false))
  return Effect.sync(() => socket.destroy())
})

/**
 * Whether `port` is still being released: it cannot be bound, yet nothing accepts connections on it. Connections a
 * stopped server closed linger (TCP TIME_WAIT: 30 s on macOS, read with `net.inet.tcp.msl` 15000), and until they
 * expire another account cannot bind the port, even with SO_REUSEADDR.
 */
export const servicePortLingering = (port: number) => servicePortBound(port).pipe(
  Effect.flatMap(bound => bound ? servicePortAccepts(port).pipe(Effect.map(accepts => !accepts)) : Effect.succeed(false)))

/**
 * The desktop always serves. A Magnitude it cannot reach through its own lock (another account's, or a
 * service with another data folder) is asked to quit with the `QuitApplication` request, which every
 * `serve` since 0.2.10 honours, and the port is awaited. Anything else leaves the port check to report.
 */
export const takeOverServicePort = (port: number) => Effect.gen(function* () {
  if (!(yield* portServedByOtherMagnitude(port))) return
  yield* Effect.logInfo(`Asking the Magnitude serving port ${port} to stop so the desktop app can serve`)
  yield* Effect.scoped(Effect.flatMap(MagnitudeClient, client => client.application.quitApplication({}))).pipe(
    Effect.provide(MagnitudeClient.layer({ origin: `http://127.0.0.1:${port}`, autoStart: false }).pipe(Layer.provide(FetchHttpClient.layer))),
    Effect.timeout("10 seconds"), Effect.catchAll(error => Effect.logWarning(`The other Magnitude did not accept the request to stop: ${String(error)}`)))
  const deadline = Date.now() + Duration.toMillis("30 seconds")
  while (Date.now() < deadline && Option.isSome(yield* probePort(port))) yield* Effect.sleep("250 millis")
})

/** A diagnostic preflight, not ownership: the service's own bind remains authoritative. */
export const checkServicePort = (port: number, executable: string) => Effect.gen(function* () {
  const failure = yield* probePort(port)
  if (Option.isNone(failure)) return
  const message = occupied(failure.value)
    ? occupiedMessage(port, yield* servedByMagnitude(port))
    : `Magnitude could not open its local service port (${port}). Check local network permissions, then retry.`
  return yield* new OwnedChildSpawnFailed({ executable, message })
})

export const requireServicePort = (port: number) => Effect.map(OwnedChildSpawner, spawner => OwnedChildSpawner.of({
  spawn: command => checkServicePort(port, command.executable).pipe(Effect.zipRight(spawner.spawn(command))),
}))
