import { Duplex } from "node:stream"
import { Cause, Deferred, Effect, ExecutionStrategy, Exit, Option, Runtime, Scope, Stream } from "effect"
import { serveApplicationRequests, type ApplicationControlOptions } from "./application-control"
import { WindowsPrivatePipes, type WindowsPrivatePipe, type WindowsPipeName, WindowsPipeFailed } from "@magnitudedev/utils/windows-native"

/** The Node stream boundary shares schema framing with Unix without using Node's pipe server. */
export const windowsPipeDuplex = (pipe: WindowsPrivatePipe) => Effect.gen(function* () {
  const run = Runtime.runCallback(yield* Effect.runtime<never>())
  const socket = new Duplex({
    read() {
      run(pipe.read, { onExit: result => {
        if (socket.destroyed) return
        if (Exit.isFailure(result)) socket.destroy(new Error(Cause.pretty(result.cause)))
        else socket.push(result.value.length === 0 ? null : Buffer.from(result.value))
      } })
    },
    write(chunk: Buffer, _encoding, callback) {
      run(pipe.write(chunk), { onExit: result => callback(Exit.isFailure(result) ? new Error(Cause.pretty(result.cause)) : null) })
    },
    destroy(error, callback) {
      run(pipe.close, { onExit: result => callback(Exit.isFailure(result) ? new Error(Cause.pretty(result.cause)) : error) })
    },
  })
  // Errors also reach framing subscribers. A failure before subscription must not crash the host.
  socket.on("error", () => {})
  yield* Effect.addFinalizer(() => Effect.async<void>(resume => {
    if (socket.closed) return resume(Effect.void)
    socket.once("close", () => resume(Effect.void))
    socket.destroy()
  }))
  return socket
})

/** A pending native instance always retains the endpoint, with at most sixteen admitted clients. */
export const serveWindowsApplicationControl = (name: WindowsPipeName, options: ApplicationControlOptions) => Effect.gen(function* () {
  const ready = yield* Deferred.make<void, WindowsPipeFailed>()
  const connections = Stream.unwrapScoped(Effect.gen(function* () {
    const driver = yield* WindowsPrivatePipes
    const parent = yield* Scope.Scope
    const run = Runtime.runFork(yield* Effect.runtime<never>())
    const permits = yield* Effect.makeSemaphore(16)
    const allocate = (first: boolean) => Effect.gen(function* () {
      const scope = yield* Scope.fork(parent, ExecutionStrategy.sequential)
      const pipe = yield* driver.bind(name, first).pipe(Scope.extend(scope))
      return { scope, pipe }
    })
    const initial = yield* allocate(true)
    yield* Deferred.succeed(ready, undefined)
    return Stream.unfoldEffect(initial, current => Effect.gen(function* () {
      yield* Effect.acquireRelease(permits.take(1), () => permits.release(1)).pipe(Scope.extend(current.scope))
      yield* current.pipe.accept
      // Replenish before publishing: closing a completed request never releases the pipe name.
      const next = yield* allocate(false)
      const socket = yield* windowsPipeDuplex(current.pipe).pipe(Scope.extend(current.scope))
      socket.once("close", () => { run(Scope.close(current.scope, Exit.void)) })
      return [socket, next] as const
    }).pipe(Effect.map(Option.some)))
  }))
  const worker = yield* serveApplicationRequests(connections.pipe(Stream.tapError(error => Deferred.fail(ready, error))), options)
  yield* Deferred.await(ready)
  return worker
})
