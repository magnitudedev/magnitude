import { FileSystem } from "@effect/platform"
import { Effect, Option, Schedule, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { command } from "../process"
import { HarnessTerminalReceipt, TerminalInput, TerminalTurn } from "./terminal"
import { TerminalConfig, TerminalDriver, TerminalScreen, waitForTerminal } from "../terminal"

const optional = <A, I>(schema: Schema.Schema<A, I>) => Schema.optionalWith(schema, { as: "Option", exact: true })
export const OpenCodeTerminalConfig = Schema.Struct({ ...TerminalConfig.omit("args", "columns", "rows").fields,
  model: TerminalInput, modelName: TerminalInput, initialModel: TerminalInput, initialModelName: TerminalInput, interrupt: TerminalTurn, recovery: TerminalTurn })
const Sessions = Schema.Array(Schema.Struct({ id: Schema.NonEmptyString, directory: Schema.String }))
const Export = Schema.Struct({ info: Schema.Struct({ id: Schema.NonEmptyString }),
  messages: Schema.Array(Schema.Struct({ info: Schema.Unknown, parts: Schema.Array(Schema.Unknown) })) })
const Assistant = Schema.Struct({ role: Schema.Literal("assistant"), id: Schema.NonEmptyString, sessionID: Schema.NonEmptyString,
  modelID: Schema.String, providerID: Schema.String, finish: optional(Schema.String), error: optional(Schema.Struct({ name: Schema.String })),
  time: Schema.Struct({ completed: optional(Schema.Number) }) })
const Text = Schema.Struct({ type: Schema.Literal("text"), text: Schema.String })
const fail = (message: string) => new AssertionFailure({ message })

/** Read the UI label from the app-created connection; persisted IDs remain the assertion. */
export const openCodeModelName = (home: string, model: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const file = join(home, ".config", "opencode", "opencode.json")
  if (Number((yield* fs.stat(file)).size) > 4 * 1024 * 1024) return yield* fail("OpenCode configuration exceeds 4 MiB")
  const config = yield* fs.readFileString(file).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({
    provider: Schema.Struct({ magnitude: Schema.Struct({ models: Schema.Record({ key: Schema.String, value: Schema.Struct({ name: optional(Schema.String) }) }) }) }),
  })))))
  const selected = config.provider.magnitude.models[model]
  if (!selected) return yield* fail("The application connection does not declare the requested OpenCode model")
  return Option.getOrElse(selected.name, () => model)
})

/** OpenCode 1.18.31 TUI. Exported native messages establish interruption and completion. */
export const openCodeTerminal = (config: typeof OpenCodeTerminalConfig.Type, onCleanupError: (message: string) => void) => Effect.scoped(Effect.gen(function* () {
  yield* Schema.decodeUnknown(OpenCodeTerminalConfig)(config)
  const fs = yield* FileSystem.FileSystem
  for (const turn of [config.interrupt, config.recovery]) if (turn.prompt.includes(turn.expected)) return yield* fail("Terminal response marker must not appear in echoed input")
  yield* fs.makeDirectory(config.evidence, { recursive: true })
  const run = (args: readonly string[]) => command(config.executable, args, { env: config.environment, inheritEnv: false,
    cwd: Option.some(config.cwd), timeoutMs: 30_000, maxOutputBytes: 16 * 1024 * 1024 }).pipe(Effect.flatMap(output =>
      output.exitCode === 0 ? Effect.succeed(output.stdout) : Effect.fail(fail("OpenCode terminal inspection command failed"))))
  if ((yield* run(["--version"])).trim() !== "1.18.31") return yield* fail("OpenCode terminal qualification requires version 1.18.31")
  const list = run(["session", "list", "--format", "json"]).pipe(Effect.flatMap(text => Schema.decodeUnknown(Schema.parseJson(Sessions))(text.trim() || "[]")))
  const previous = new Set((yield* list).map(session => session.id))
  const terminal = yield* (yield* TerminalDriver).start(TerminalConfig.make({ ...config,
    args: ["--pure", "--model", `magnitude/${config.initialModel}`], columns: 160, rows: 50,
  }), onCleanupError)
  const wait = (predicate: (lines: readonly string[]) => boolean, description: string, timeoutMs = 30_000) =>
    waitForTerminal(terminal, screen => predicate(screen.lines), description, timeoutMs)
  const saveScreen = (name: string, screen: typeof TerminalScreen.Type) => Schema.encode(Schema.parseJson(TerminalScreen))(screen).pipe(
    Effect.flatMap(json => fs.writeFileString(join(config.evidence, `${name}.json`), json)))
  const submit = (prompt: string) => terminal.write(prompt).pipe(Effect.zipRight(wait(lines => lines.some(line => line.includes(prompt.slice(0, 40))), "render prompt input")),
    Effect.zipRight(terminal.write("\r")))
  yield* wait(lines => lines.some(line => line.includes(config.initialModelName)), "show the initial model")
  yield* terminal.write("\u0018m")
  yield* wait(lines => lines.some(line => line.includes("Select model")), "open model selection")
  yield* terminal.write(config.modelName)
  yield* wait(lines => lines.some(line => line.includes(config.modelName)), "filter the desired model")
  yield* terminal.write("\r")
  const selected = yield* wait(lines => !lines.some(line => line.includes("Select model")) && lines.some(line => line.includes(config.modelName)), "select the model")
  if (selected.lines.some(line => line.includes("Select variant"))) {
    yield* terminal.write("off")
    yield* wait(lines => lines.some(line => line.trim() === "off"), "filter the off variant")
    yield* terminal.write("\r")
    yield* wait(lines => !lines.some(line => line.includes("Select variant")), "select the off variant")
  }
  yield* submit(config.interrupt.prompt)
  yield* saveScreen("streaming", yield* wait(lines => lines.some(line => line.includes(config.interrupt.expected)), "render streamed assistant output", 300_000))
  const sessions = (yield* list).filter(session => !previous.has(session.id))
  if (sessions.length !== 1) return yield* fail("OpenCode did not create exactly one identifiable terminal session")
  const id = sessions[0]!.id
  const read = run(["export", id]).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Export))), Effect.flatMap(saved => Effect.gen(function* () {
    if (saved.info.id !== id) return yield* fail("OpenCode exported a different session")
    const assistants = []
    for (const entry of saved.messages) {
      if (!Schema.is(Schema.Struct({ role: Schema.Literal("assistant") }))(entry.info)) continue
      assistants.push({ info: yield* Schema.decodeUnknown(Assistant)(entry.info), parts: entry.parts })
    }
    if (assistants.some(entry => entry.info.sessionID !== id || entry.info.providerID !== "magnitude" || entry.info.modelID !== config.model)) {
      return yield* fail("OpenCode generated with another session, provider or model")
    }
    return { saved, assistants }
  })))
  const waitForTurn = (count: number) => read.pipe(Effect.repeat({ until: state => state.assistants.length >= count && Option.isSome(state.assistants[count - 1]!.info.time.completed),
    schedule: Schedule.identity<Effect.Effect.Success<typeof read>>().pipe(Schedule.addDelay(() => "100 millis")) }),
    Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("OpenCode did not persist terminal turn completion") }))
  const active = yield* read
  if (active.assistants.length !== 1 || Option.isSome(active.assistants[0]!.info.time.completed)) {
    return yield* fail("OpenCode generation completed before keyboard interruption")
  }
  yield* terminal.write("\u001b")
  yield* wait(lines => lines.some(line => line.includes("again to interrupt")), "arm interruption")
  yield* terminal.write("\u001b")
  const aborted = yield* waitForTurn(1)
  yield* fs.writeFileString(join(config.evidence, "aborted.session.json"), yield* Schema.encode(Schema.parseJson(Export))(aborted.saved))
  if (aborted.assistants.length !== 1 || aborted.assistants[0]!.info.id !== active.assistants[0]!.info.id || !Option.exists(aborted.assistants[0]!.info.error, error => error.name === "MessageAbortedError")) {
    return yield* fail("OpenCode did not persist exactly one interrupted assistant turn")
  }
  yield* submit(config.recovery.prompt)
  yield* saveScreen("recovered", yield* wait(lines => lines.some(line => line.includes(config.recovery.expected)), "render the follow-up", 300_000))
  const completed = yield* waitForTurn(2)
  yield* fs.writeFileString(join(config.evidence, "recovered.session.json"), yield* Schema.encode(Schema.parseJson(Export))(completed.saved))
  const first = completed.assistants[0]!, last = completed.assistants[1]!
  const text = last.parts.filter(Schema.is(Text)).map(part => part.text).join("")
  if (completed.assistants.length !== 2 || first.info.id !== aborted.assistants[0]!.info.id || last.info.id === first.info.id ||
    Option.isSome(last.info.error) || !Option.contains(last.info.finish, "stop") || !text.includes(config.recovery.expected)) {
    return yield* fail("OpenCode did not persist a completed recovery answer in the same session")
  }
  yield* terminal.write("\u0004")
  const exited = yield* terminal.exited.pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => fail("OpenCode did not exit through keyboard input") }))
  if (exited.code !== 0 || Option.isSome(exited.signal)) return yield* fail("OpenCode terminal did not exit normally")
  return HarnessTerminalReceipt.make({ sessionId: id, pid: terminal.pid, model: config.model,
    interruptedMessageId: first.info.id, recoveredMessageId: last.info.id, text })
}))
