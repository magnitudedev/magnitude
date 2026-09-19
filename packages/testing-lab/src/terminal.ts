import { FileSystem } from "@effect/platform"
import { Terminal as Screen } from "@xterm/headless"
import { Context, Deferred, Effect, FiberSet, Layer, Option, Schedule, Schema, Scope } from "effect"
import { isAbsolute, join } from "node:path"
import { fileURLToPath } from "node:url"
import { AssertionFailure, InfrastructureFailure } from "./domain"
import { LabProcessId } from "./application-identity"
import { jsonProcess } from "./json-process"
import { TerminalColumns, TerminalRows, TerminalLaunch, TerminalExit, TerminalEvent } from "./terminal-protocol"

export { TerminalExit } from "./terminal-protocol"
export const TerminalConfig = Schema.Struct({ ...TerminalLaunch.fields,
  runtime: Schema.NonEmptyString.pipe(Schema.filter(isAbsolute)), evidence: Schema.NonEmptyString.pipe(Schema.filter(isAbsolute)),
})
export const TerminalScreen = Schema.Struct({ columns: TerminalColumns, rows: TerminalRows, alternate: Schema.Boolean, lines: Schema.Array(Schema.String) })
export interface TerminalSession {
  readonly pid: typeof LabProcessId.Type
  readonly write: (text: string) => Effect.Effect<void, InfrastructureFailure>
  readonly resize: (columns: number, rows: number) => Effect.Effect<void, InfrastructureFailure>
  readonly screen: Effect.Effect<typeof TerminalScreen.Type, InfrastructureFailure>
  readonly exited: Effect.Effect<typeof TerminalExit.Type, InfrastructureFailure>
}
export interface TerminalDriver {
  readonly start: (config: typeof TerminalConfig.Type, onCleanupError: (message: string) => void) => Effect.Effect<TerminalSession, InfrastructureFailure, Scope.Scope>
}
export const TerminalDriver = Context.GenericTag<TerminalDriver>("@magnitudedev/testing-lab/TerminalDriver")
const failure = (message: string) => new InfrastructureFailure({ operation: "terminal", message })
const boundary = <A>(run: () => A) => Effect.try({ try: run, catch: () => failure("Terminal operation failed") })

/** Native bindings live in a Node process; the worker owns its lifetime and terminal interpretation. */
export const NativeTerminalDriver = Layer.effect(TerminalDriver, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return { start: (config, onCleanupError) => Effect.gen(function* () {
    yield* fs.stat(config.executable).pipe(Effect.flatMap(stat => stat.type === "File" ? Effect.void : Effect.fail(failure("Terminal executable is not a regular file"))),
      Effect.mapError(() => failure("Terminal executable is missing or unreadable")))
    yield* fs.makeDirectory(config.evidence, { recursive: true }).pipe(Effect.mapError(() => failure("Cannot create terminal evidence directory")))
    const display = yield* Effect.acquireRelease(boundary(() => new Screen({ cols: config.columns, rows: config.rows, scrollback: 2000, allowProposedApi: true })), display => Effect.sync(() => display.dispose()))
    let transcript = "", bytes = 0
    let fault = Option.none<InfrastructureFailure>()
    const healthy = Effect.suspend(() => Option.match(fault, { onNone: () => Effect.void, onSome: Effect.fail }))
    const snapshot = Effect.async<void>(resume => { display.write("", () => resume(Effect.void)) }).pipe(Effect.zipRight(boundary(() => {
      const buffer = display.buffer.active
      return TerminalScreen.make({ columns: display.cols, rows: display.rows, alternate: buffer.type === "alternate",
        lines: Array.from({ length: buffer.length }, (_, index) => buffer.getLine(index)?.translateToString(true) ?? "") })
    })))
    const screen = healthy.pipe(Effect.zipRight(snapshot))
    yield* Effect.addFinalizer(() => Effect.gen(function* () {
      yield* fs.writeFileString(join(config.evidence, "terminal-output.txt"), transcript, { mode: 0o600 })
      yield* fs.writeFileString(join(config.evidence, "terminal-screen.json"), yield* Schema.encode(Schema.parseJson(TerminalScreen))(yield* snapshot), { mode: 0o600 })
    }).pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(`Terminal evidence: ${error.message}`) }))))
    const child = yield* jsonProcess({ executable: config.runtime, args: [fileURLToPath(new URL("./terminal-entry.ts", import.meta.url))],
      cwd: config.cwd, environment: config.environment, stdoutLog: join(config.evidence, "terminal-events.jsonl"), stderrLog: join(config.evidence, "terminal-bridge.stderr.log") }).pipe(Effect.provideService(FileSystem.FileSystem, fs))
    yield* child.send({ _tag: "Start", launch: config })
    const receive = child.receive.pipe(Effect.flatMap(Schema.decodeUnknown(TerminalEvent)), Effect.mapError(() => failure("Invalid or incomplete terminal bridge event")))
    const started = yield* receive.pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => failure("Native terminal did not start") }))
    if (started._tag !== "Started") return yield* failure(started._tag === "Failed" ? started.message : "Terminal bridge did not establish its child")
    const exited = yield* Deferred.make<typeof TerminalExit.Type, InfrastructureFailure>()
    const sendReply = yield* FiberSet.makeRuntime<never>()
    const replies = display.onData(text => { sendReply(child.send({ _tag: "Write", text }).pipe(Effect.catchAll(error => Effect.sync(() => { fault = Option.some(error) })))) })
    yield* Effect.addFinalizer(() => Effect.sync(() => replies.dispose()))
    yield* receive.pipe(Effect.flatMap(event => Effect.gen(function* () {
      if (event._tag === "Output") {
        bytes += Buffer.byteLength(event.text)
        if (bytes > 16 * 1024 * 1024) return yield* failure("Terminal output exceeded 16 MiB")
        transcript += event.text
        yield* Effect.async<void>(resume => { display.write(event.text, () => resume(Effect.void)) })
      } else if (event._tag === "Exited") { yield* Deferred.succeed(exited, event); return true }
      else return yield* failure(event._tag === "Failed" ? event.message : "Terminal bridge started more than one child")
      return false
    })), Effect.repeat({ until: done => done }), Effect.catchAll(error => Effect.gen(function* () {
      fault = Option.some(error)
      yield* Deferred.fail(exited, error)
    })), Effect.forkScoped)
    yield* Effect.addFinalizer(() => Effect.gen(function* () {
      if (Option.isSome(yield* Deferred.poll(exited))) return
      yield* child.send({ _tag: "Stop", force: false })
      const stopped = yield* Deferred.await(exited).pipe(Effect.interruptible, Effect.timeoutOption("1 second"))
      if (Option.isNone(stopped)) {
        yield* child.send({ _tag: "Stop", force: true })
        yield* Deferred.await(exited).pipe(Effect.interruptible, Effect.timeoutFail({ duration: "3 seconds", onTimeout: () => failure("Terminal child survived cleanup") }))
      }
    }).pipe(Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
    return { pid: LabProcessId.make(started.pid), screen,
      write: text => healthy.pipe(Effect.zipRight(child.send({ _tag: "Write", text }))),
      resize: (columns, rows) => Schema.decodeUnknown(Schema.Struct({ columns: TerminalColumns, rows: TerminalRows }))({ columns, rows }).pipe(
        Effect.mapError(() => failure("Invalid terminal dimensions")), Effect.flatMap(size => child.send({ _tag: "Resize", ...size }).pipe(
          Effect.zipRight(boundary(() => display.resize(size.columns, size.rows)))))),
      exited: Deferred.await(exited).pipe(Effect.tap(() => healthy)),
    } satisfies TerminalSession
  }) } satisfies TerminalDriver
}))

export const waitForTerminal = (session: TerminalSession, predicate: (screen: typeof TerminalScreen.Type) => boolean, description: string) =>
  session.screen.pipe(Effect.repeat({ until: predicate, schedule: Schedule.identity<typeof TerminalScreen.Type>().pipe(Schedule.addDelay(() => "100 millis")) }),
    Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new AssertionFailure({ message: `Terminal did not ${description}` }) }))
