import { FileSystem } from "@effect/platform"
import { Effect, Option, Schedule, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { command } from "../process"
import { LabProcessId } from "../application-identity"
import { TerminalConfig, TerminalDriver, TerminalScreen, waitForTerminal } from "../terminal"

const safeInput = Schema.NonEmptyString.pipe(Schema.pattern(/^[^\x00-\x1f\x7f]+$/))
const Turn = Schema.Struct({ prompt: safeInput, expected: safeInput })
export const PiTerminalConfig = Schema.Struct({ ...TerminalConfig.omit("args", "columns", "rows").fields,
  model: safeInput, initialModel: safeInput, interrupt: Turn, recovery: Turn })
export const PiTerminalReceipt = Schema.Struct({ sessionId: Schema.NonEmptyString, pid: LabProcessId, model: Schema.NonEmptyString,
  interruptedMessageId: Schema.NonEmptyString, recoveredMessageId: Schema.NonEmptyString, text: Schema.NonEmptyString })
const Header = Schema.Struct({ type: Schema.Literal("session"), id: Schema.NonEmptyString })
const Assistant = Schema.Struct({ type: Schema.Literal("message"), id: Schema.NonEmptyString,
  message: Schema.Struct({ role: Schema.Literal("assistant"), model: Schema.String, provider: Schema.String,
    stopReason: Schema.String, content: Schema.Array(Schema.Unknown) }) })
const Text = Schema.Struct({ type: Schema.Literal("text"), text: Schema.String })
const fail = (message: string) => new AssertionFailure({ message })

/** Pi 0.85.1 keyboard journey. Native session records, not screen text, prove turn completion. */
export const piTerminal = (config: typeof PiTerminalConfig.Type, onCleanupError: (message: string) => void) => Effect.scoped(Effect.gen(function* () {
  yield* Schema.decodeUnknown(PiTerminalConfig)(config)
  const fs = yield* FileSystem.FileSystem
  for (const turn of [config.interrupt, config.recovery]) {
    if (turn.prompt.includes(turn.expected)) return yield* fail("Terminal response marker must not appear in echoed input")
  }
  const version = yield* command(config.executable, ["--version"], { env: config.environment, inheritEnv: false, timeoutMs: 30_000 })
  if (version.exitCode !== 0 || version.stdout.trim() !== "0.85.1") return yield* fail("Pi terminal qualification requires version 0.85.1")
  yield* fs.makeDirectory(config.evidence, { recursive: true })
  const session = join(config.evidence, "session.jsonl")
  if (yield* fs.exists(session)) return yield* fail("Pi terminal session must start with a fresh evidence directory")
  const read = Effect.gen(function* () {
    if (!(yield* fs.exists(session))) return { headers: [], messages: [] }
    if (Number((yield* fs.stat(session)).size) > 16 * 1024 * 1024) return yield* fail("Pi session exceeded 16 MiB")
    const contents = yield* fs.readFileString(session)
    const complete = contents.slice(0, contents.lastIndexOf("\n") + 1)
    const entries = yield* Effect.forEach(complete.split("\n").filter(Boolean), line => Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(line))
    return { headers: entries.filter(Schema.is(Header)), messages: entries.filter(Schema.is(Assistant)) }
  })
  const waitForMessages = (count: number) => read.pipe(Effect.repeat({ until: state => state.messages.length >= count,
    schedule: Schedule.identity<Effect.Effect.Success<typeof read>>().pipe(Schedule.addDelay(() => "100 millis")) }),
    Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("Pi did not persist the expected terminal turn") }))
  const terminal = yield* (yield* TerminalDriver).start(TerminalConfig.make({ ...config,
    args: ["--provider", "magnitude", "--model", config.initialModel, "--thinking", "off", "--no-tools", "--offline", "--session", session],
    columns: 160, rows: 50,
  }), onCleanupError)
  const saveScreen = (name: string, screen: typeof TerminalScreen.Type) =>
    Schema.encode(Schema.parseJson(TerminalScreen))(screen).pipe(Effect.flatMap(json => fs.writeFileString(join(config.evidence, `${name}.json`), json)))
  yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes(config.initialModel)), "show the initial model")
  yield* terminal.write(`/model magnitude/${config.model}`)
  // The footer appears before Pi enables submission. Retry only this idempotent selection;
  // after it is consumed, Enter on the empty editor is a no-op. Never retry generation.
  yield* terminal.write("\r").pipe(Effect.zipRight(terminal.screen), Effect.repeat({
    until: screen => screen.lines.some(line => line.includes(`Model: ${config.model}`)), schedule: Schedule.spaced("100 millis"),
  }), Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("Pi did not confirm keyboard model selection") }))
  yield* terminal.write(`${config.interrupt.prompt}\r`)
  yield* saveScreen("streaming", yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes(config.interrupt.expected)), "render streamed assistant output", 300_000))
  yield* terminal.write("\u001b")
  const aborted = yield* waitForMessages(1)
  if (aborted.headers.length !== 1 || aborted.messages.length !== 1 || aborted.messages[0]!.message.stopReason !== "aborted") {
    return yield* fail("Pi did not persist exactly one interrupted assistant turn")
  }
  yield* terminal.write(`${config.recovery.prompt}\r`)
  yield* saveScreen("recovered", yield* waitForTerminal(terminal, screen => screen.lines.some(line => line.includes(config.recovery.expected)), "render the follow-up answer", 300_000))
  const completed = yield* waitForMessages(2)
  if (completed.headers.length !== 1 || completed.headers[0]!.id !== aborted.headers[0]!.id || completed.messages.length !== 2 ||
    completed.messages[0]!.id !== aborted.messages[0]!.id || completed.messages[1]!.id === aborted.messages[0]!.id ||
    completed.messages[1]!.message.stopReason !== "stop" ||
    completed.messages.some(entry => entry.message.model !== config.model || entry.message.provider !== "magnitude")) {
    return yield* fail("Pi did not recover in the same session with the selected provider and model")
  }
  const text = completed.messages[1]!.message.content.filter(Schema.is(Text)).map(part => part.text).join("")
  if (!text.includes(config.recovery.expected)) return yield* fail("Pi's persisted answer differs from the rendered recovery marker")
  yield* terminal.write("\u0004")
  const exited = yield* terminal.exited.pipe(Effect.timeoutFail({ duration: "15 seconds", onTimeout: () => fail("Pi did not exit through keyboard input") }))
  if (exited.code !== 0 || Option.isSome(exited.signal)) return yield* fail("Pi terminal did not exit normally")
  return PiTerminalReceipt.make({ sessionId: completed.headers[0]!.id, pid: terminal.pid, model: config.model,
    interruptedMessageId: completed.messages[0]!.id, recoveredMessageId: completed.messages[1]!.id, text })
}))
