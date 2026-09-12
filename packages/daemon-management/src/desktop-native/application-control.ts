import type { Duplex } from "node:stream"
import { createServer, Socket } from "node:net"
import { lstat, unlink, chmod } from "node:fs/promises"
import { Deferred, Effect, Option, Schema, Stream } from "effect"
import { receiveJsonLines, sendJsonLine } from "@magnitudedev/utils/json-line-channel"
import { ApplicationIntent, ApplicationRequest, ApplicationSnapshot, ApplicationLoginRequest, ApplicationLoginReply, LoginStartupFailed, type LoginStartupAction, type LoginStartupState } from "@magnitudedev/sdk/desktop-host"
export { ApplicationIntent, ApplicationRequest, ApplicationSnapshot } from "@magnitudedev/sdk/desktop-host"

export class ApplicationControlFailed extends Schema.TaggedError<ApplicationControlFailed>()("ApplicationControlFailed", { message: Schema.String }) {}
export class ApplicationControlUnavailable extends Schema.TaggedError<ApplicationControlUnavailable>()("ApplicationControlUnavailable", { message: Schema.String }) {}
const failure = (error: unknown) => new ApplicationControlFailed({ message: String(error) })

const validateUnixControlPath = (path: string) => Effect.gen(function* () {
  const maximumBytes = process.platform === "darwin" ? 103 : 107
  if (path.includes("\0") || Buffer.byteLength(path, "utf8") > maximumBytes) return yield* new ApplicationControlFailed({
    message: `The application control path is invalid or exceeds ${maximumBytes} UTF-8 bytes. Use a shorter application state directory.`,
  })
})

/** Call only while holding the native lifetime lock, in a private current-user directory.
 * The lock must outlive this server. Unlinking a stale socket never changes the lock inode.
 */
export interface ApplicationControlOptions {
  readonly snapshot: Effect.Effect<ApplicationSnapshot>
  readonly dispatch: (intent: ApplicationIntent) => Effect.Effect<void>
  readonly login: (action: LoginStartupAction) => Effect.Effect<LoginStartupState, LoginStartupFailed>
}

export const serveApplicationControl = (path: string, options: ApplicationControlOptions) => Effect.gen(function* () {
  if (process.platform === "win32") return yield* new ApplicationControlFailed({ message: "Windows application control requires the named-pipe ACL adapter" })
  yield* validateUnixControlPath(path)
  yield* Effect.tryPromise({ try: async () => {
    const existing = await lstat(path).catch(error => { if (error.code === "ENOENT") return null; throw error })
    if (existing !== null) {
      if (!existing.isSocket() || existing.uid !== process.getuid!()) throw new Error("Unsafe application control endpoint")
      await unlink(path)
    }
  }, catch: failure })
  const listening = yield* Deferred.make<void, ApplicationControlFailed>()
  const connections = Stream.asyncScoped<Socket, ApplicationControlFailed>(emit => Effect.acquireRelease(
    Effect.async<{ server: ReturnType<typeof createServer>; sockets: Set<Socket> }, ApplicationControlFailed>(resume => {
      const sockets = new Set<Socket>()
      const server = createServer(socket => {
        // Slow or malicious local clients cannot create an unbounded queue of open handles.
        if (sockets.size >= 16) { socket.destroy(); return }
        sockets.add(socket)
        socket.on("error", () => {})
        socket.once("close", () => sockets.delete(socket))
        socket.setTimeout(5000, () => socket.destroy())
        void emit.single(socket)
      })
      server.on("error", error => { resume(Effect.fail(failure(error))); void emit.fail(failure(error)) })
      server.listen(path, () => resume(Effect.succeed({ server, sockets })))
      return Effect.sync(() => { for (const socket of sockets) socket.destroy(); server.close() })
    }),
    ({ server, sockets }) => Effect.promise(async () => {
      for (const socket of sockets) socket.destroy()
      await new Promise<void>(resolve => server.close(() => resolve()))
      await unlink(path).catch(error => { if (error.code !== "ENOENT") throw error })
    }),
  ).pipe(Effect.tap(() => Effect.tryPromise({ try: () => chmod(path, 0o600), catch: failure })), Effect.tap(() => Deferred.succeed(listening, undefined))))
  const worker = yield* serveApplicationRequests(connections.pipe(Stream.tapError(error => Deferred.fail(listening, error))), options)
  yield* Deferred.await(listening)
  return worker
})

/** Shared framing, dispatch, and per-request cleanup for both native transport adapters. */
export const serveApplicationRequests = <E, R>(connections: Stream.Stream<Duplex, E, R>, options: ApplicationControlOptions) =>
  connections.pipe(Stream.mapEffect(socket => Effect.scoped(Effect.gen(function* () {
    yield* Effect.addFinalizer(() => Effect.sync(() => socket.destroy()))
    const request = yield* receiveJsonLines(socket, Schema.Union(ApplicationRequest, ApplicationLoginRequest)).pipe(Stream.take(1), Stream.runHead)
    if (Option.isNone(request)) return
    if ("login" in request.value) {
      const reply = yield* options.login(request.value.login).pipe(Effect.map(state => ({ _tag: "LoginStartup" as const, state })), Effect.catchAll(Effect.succeed))
      yield* sendJsonLine(socket, ApplicationLoginReply, reply)
      return
    }
    // Reply before Quit tears down the server and before ShowWindow does any UI work.
    yield* sendJsonLine(socket, ApplicationSnapshot, yield* options.snapshot)
    yield* options.dispatch(request.value.intent)
  })).pipe(Effect.timeout("5 seconds"), Effect.catchAll(() => Effect.void)), { concurrency: 16 }), Stream.runDrain, Effect.forkScoped)

const exchange = <Q, QI, A, AI>(path: string, requestSchema: Schema.Schema<Q, QI>, request: Q, replySchema: Schema.Schema<A, AI>) => Effect.scoped(Effect.gen(function* () {
  if (process.platform !== "win32") yield* validateUnixControlPath(path)
  const socket = yield* Effect.acquireRelease(
    Effect.async<Socket, ApplicationControlFailed | ApplicationControlUnavailable>(resume => {
      const client = new Socket()
      client.on("error", error => resume(Effect.fail("code" in error && (error.code === "ENOENT" || error.code === "ECONNREFUSED")
        ? new ApplicationControlUnavailable({ message: "Magnitude desktop is not running" }) : failure(error))))
      client.once("connect", () => resume(Effect.succeed(client)))
      client.connect(path)
      return Effect.sync(() => client.destroy())
    }),
    client => Effect.sync(() => client.destroy()),
  )
  // Start reading before sending so immediate replies are retained by the stream.
  const response = yield* receiveJsonLines(socket, replySchema).pipe(Stream.take(1), Stream.runHead, Effect.forkScoped)
  yield* sendJsonLine(socket, requestSchema, request)
  const value = yield* response.await.pipe(Effect.flatten)
  return yield* Option.match(value, { onNone: () => Effect.fail(new ApplicationControlFailed({ message: "Application closed without a response" })), onSome: Effect.succeed })
})).pipe(Effect.catchTag("JsonLineChannelFailed", error => Effect.fail(new ApplicationControlFailed({ message: error.message }))), Effect.timeoutFail({ duration: "5 seconds", onTimeout: () => new ApplicationControlFailed({ message: "Application did not respond within five seconds" }) }))

export const requestApplication = (path: string, intent: ApplicationIntent) => exchange(path, ApplicationRequest, { version: 1 as const, intent }, ApplicationSnapshot)
export const requestLoginStartup = (path: string, login: LoginStartupAction) => exchange(path, ApplicationLoginRequest, { version: 1 as const, login }, ApplicationLoginReply).pipe(
  Effect.flatMap(reply => reply._tag === "LoginStartupFailed" ? Effect.fail(reply) : Effect.succeed(reply.state)),
)
