import { NodeRuntime } from "@effect/platform-node"
import { Deferred, Effect, Option, Queue, Runtime, Schema, Stream } from "effect"
import { createRequire } from "node:module"
import { createInterface } from "node:readline"
import { TerminalCommand, TerminalEvent, TerminalStart, TerminalExit, TerminalBridgeFailure } from "./terminal-protocol.ts"

const error = () => new TerminalBridgeFailure({ message: "Native terminal bridge failed" })
process.stderr.write("Terminal bridge loaded\n")
const program = Effect.scoped(Effect.gen(function* () {
  if (process.versions.bun || Number(process.versions.node.split(".")[0]) < 24) return yield* new TerminalBridgeFailure({ message: "The native terminal bridge requires Node.js 24 or newer" })
  // Use readline's line event directly for the Windows named pipe. The async iterator
  // can leave that pipe paused while Stream.peel waits for its first frame.
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity })
  const frames = yield* Queue.unbounded<string>()
  const onLine = (line: string) => { Runtime.runSync(Runtime.defaultRuntime)(Queue.offer(frames, line)) }
  const onClose = () => { Runtime.runSync(Runtime.defaultRuntime)(Queue.shutdown(frames)) }
  lines.on("line", onLine)
  lines.on("close", onClose)
  yield* Effect.addFinalizer(() => Effect.sync(() => { lines.off("line", onLine); lines.off("close", onClose); lines.close() }))
  process.stdin.resume()
  const decode = (line: string) => line.length <= 256 * 1024
    ? Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(line).pipe(Effect.mapError(error))
    : Effect.fail(error())
  const first = yield* Queue.take(frames).pipe(Effect.mapError(error), Effect.flatMap(decode), Effect.flatMap(Schema.decodeUnknown(TerminalStart)))
  process.stderr.write("Terminal bridge received launch\n")
  const rest = Stream.fromQueue(frames).pipe(Stream.mapEffect(decode))
  const events = yield* Queue.unbounded<typeof TerminalEvent.Type>()
  const sentExit = yield* Deferred.make<void>()
  yield* Queue.take(events).pipe(Effect.flatMap(event => Effect.gen(function* () {
    const encoded = yield* Schema.encode(Schema.parseJson(TerminalEvent))(event)
    yield* Effect.async<void, TerminalBridgeFailure>(resume => { process.stdout.write(encoded + "\n", failure => resume(failure ? Effect.fail(error()) : Effect.void)) })
    if (event._tag === "Exited" || event._tag === "Failed") yield* Deferred.succeed(sentExit, undefined)
  })), Effect.forever, Effect.forkScoped)
  const runtime = yield* Effect.runtime<never>()
  const emit = (event: typeof TerminalEvent.Type) => Runtime.runSync(runtime)(Queue.offer(events, event))
  const pty: typeof import("node-pty") = yield* Effect.try({ try: () => createRequire(import.meta.url)("node-pty"), catch: error })
  process.stderr.write("Terminal bridge loaded native PTY\n")
  const config = first.launch
  const nativeExit = yield* Deferred.make<void>()
  let ended = false
  let detach = () => {}
  const child = yield* Effect.acquireRelease(Effect.try({ try: () => pty.spawn(config.executable, [...config.args], {
    cwd: config.cwd, env: { ...config.environment, TERM: "xterm-256color", COLORTERM: "truecolor" },
    cols: config.columns, rows: config.rows, name: "xterm-256color",
  }), catch: error }), child => Effect.gen(function* () {
    if (ended) return
    yield* Effect.try({ try: () => child.kill(), catch: error })
    const graceful = yield* Deferred.await(nativeExit).pipe(Effect.interruptible, Effect.timeoutOption("500 millis"))
    if (Option.isNone(graceful)) {
      yield* Effect.try({ try: () => child.kill("SIGKILL"), catch: error })
      yield* Deferred.await(nativeExit).pipe(Effect.interruptible, Effect.timeoutFail({ duration: "1 second", onTimeout: error }))
    }
  }).pipe(Effect.ensuring(Effect.sync(() => detach())), Effect.catchAll(() => Effect.sync(() => { process.exitCode = 1 }))))
  emit({ _tag: "Started", pid: child.pid })
  let bytes = 0, overflow = false
  const data = child.onData(text => {
    if (overflow) return
    bytes += Buffer.byteLength(text)
    if (bytes > 16 * 1024 * 1024) { overflow = true; emit({ _tag: "Failed", message: "Terminal output exceeded 16 MiB" }); return }
    emit({ _tag: "Output", text })
  })
  const stopped = child.onExit(event => {
    ended = true
    Runtime.runSync(runtime)(Deferred.succeed(nativeExit, undefined))
    emit(TerminalExit.make({ _tag: "Exited", code: event.exitCode, signal: event.signal ? Option.some(String(event.signal)) : Option.none() }))
  })
  detach = () => { data.dispose(); stopped.dispose() }
  const commands = rest.pipe(Stream.mapEffect(Schema.decodeUnknown(TerminalCommand)), Stream.runForEach(command => Effect.try({ try: () => {
    if (ended) throw error()
    if (command._tag === "Write") child.write(command.text)
    else if (command._tag === "Resize") child.resize(command.columns, command.rows)
    else child.kill(command.force ? "SIGKILL" : "SIGTERM")
  }, catch: error })))
  yield* Effect.raceFirst(commands, Deferred.await(sentExit))
})).pipe(Effect.catchAll(failure => Effect.async<void>(resume => {
  process.exitCode = 1
  process.stdout.write(JSON.stringify({ _tag: "Failed", message: failure instanceof TerminalBridgeFailure ? failure.message : "Invalid terminal bridge input" }) + "\n", () => resume(Effect.void))
})))
NodeRuntime.runMain(program)
