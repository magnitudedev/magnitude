import { createRequire } from "node:module"
import { dirname, join } from "node:path"
import { createReadStream, createWriteStream, closeSync } from "node:fs"
import { Duplex } from "node:stream"
import { Socket } from "node:net"
import { DesktopChildEvent, DesktopOwnerCommand } from "@magnitudedev/acn-protocol/desktop-control"
import { receiveJsonLines, sendJsonLine, JsonLineChannelFailed } from "@magnitudedev/utils/json-line-channel"
import { Deferred, Effect, Schema, Stream, type Scope } from "effect"
import type { MagnitudeHealthResponse } from "@magnitudedev/acn-protocol"

export class AcnOwnerUnavailable extends Schema.TaggedError<AcnOwnerUnavailable>()("AcnOwnerUnavailable", {
  message: Schema.String,
}) {}

export interface AcnOwnerControl {
  readonly awaitStart: Effect.Effect<void, JsonLineChannelFailed>
  readonly awaitShutdown: Effect.Effect<void, JsonLineChannelFailed>
  readonly reportHealth: (health: MagnitudeHealthResponse) => Effect.Effect<void, JsonLineChannelFailed>
}

/** Installed before any application/engine scope. fd 0 is exclusively native-owned. */
export const installOwnerGuard = Effect.try({
  try: () => {
    const addon = process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node")
    const native = createRequire(import.meta.url)(addon) as { guardParent: (fd: number) => void }
    native.guardParent(0)
  },
  catch: error => new AcnOwnerUnavailable({ message: `Magnitude must be launched by its desktop owner: ${String(error)}` }),
})

const openOwnerChannel = process.platform === "win32" ? Effect.gen(function* () {
  const name = yield* Schema.decodeUnknown(Schema.String.pipe(Schema.maxLength(240), Schema.pattern(/^\\\\\.\\pipe\\magnitude-child-[0-9a-f-]+$/)))(process.env.MAGNITUDE_OWNER_PIPE).pipe(
    Effect.mapError(() => new AcnOwnerUnavailable({ message: "Missing private desktop control pipe." })),
  )
  return yield* Effect.acquireRelease(Effect.async<Socket, AcnOwnerUnavailable>(resume => {
    const client = new Socket()
    client.on("error", () => resume(Effect.fail(new AcnOwnerUnavailable({ message: "Could not connect to the desktop owner." }))))
    client.once("connect", () => resume(Effect.succeed(client)))
    client.connect(name)
    return Effect.sync(() => client.destroy())
  }).pipe(Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => new AcnOwnerUnavailable({ message: "Desktop control connection timed out." }) })),
  client => Effect.sync(() => client.destroy()))
}) : Effect.acquireRelease(Effect.try({
    // Bun cannot wrap an inherited descriptor with net.Socket; its fs streams support it.
    try: () => Duplex.from({
      readable: createReadStream("", { fd: 3, autoClose: false }),
      writable: createWriteStream("", { fd: 3, autoClose: false }),
    }),
    catch: error => new AcnOwnerUnavailable({ message: `Missing inherited desktop control channel: ${String(error)}` }),
  }), socket => Effect.sync(() => { socket.destroy(); closeSync(3) }))

export const makeAcnOwnerControl: Effect.Effect<AcnOwnerControl, AcnOwnerUnavailable | JsonLineChannelFailed, Scope.Scope> = Effect.gen(function* () {
  const socket = yield* openOwnerChannel
  const started = yield* Deferred.make<void, JsonLineChannelFailed>()
  const shutdown = yield* Deferred.make<void, JsonLineChannelFailed>()
  const stoppingObserved = yield* Deferred.make<void, JsonLineChannelFailed>()
  yield* receiveJsonLines(socket, DesktopOwnerCommand).pipe(
    Stream.runForEach(command => command._tag === "StoppingObserved"
      ? Deferred.succeed(stoppingObserved, undefined).pipe(Effect.asVoid)
      : command._tag === "Shutdown"
      ? Effect.all([Deferred.succeed(shutdown, undefined), Deferred.fail(started, new JsonLineChannelFailed({ message: "Desktop stopped before startup authorization" }))]).pipe(Effect.asVoid)
      : Effect.gen(function* () {
          if (yield* Deferred.isDone(started)) return yield* new JsonLineChannelFailed({ message: "Duplicate owner Start" })
          yield* Deferred.succeed(started, undefined)
        })),
    Effect.zipRight(Effect.fail(new JsonLineChannelFailed({ message: "Desktop control channel closed" }))),
    Effect.catchAll(error => Effect.all([Deferred.fail(started, error), Deferred.fail(shutdown, error), Deferred.fail(stoppingObserved, error)])),
    Effect.forkScoped,
  )
  yield* sendJsonLine(socket, DesktopChildEvent, { _tag: "Booted", pid: process.pid })
  return {
    awaitStart: Deferred.await(started).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new JsonLineChannelFailed({ message: "Desktop did not authorize service startup" }) })),
    awaitShutdown: Deferred.await(shutdown),
    reportHealth: health => sendJsonLine(socket, DesktopChildEvent, { _tag: "Health", health }).pipe(
      Effect.zipRight(health.state._tag === "Stopping" ? Deferred.await(stoppingObserved).pipe(
        Effect.timeoutFail({ duration: "2 seconds", onTimeout: () => new JsonLineChannelFailed({ message: "Desktop did not acknowledge the final service status." }) }),
      ) : Effect.void),
    ),
  }
})
