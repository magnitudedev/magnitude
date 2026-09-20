import { FileSystem } from "@effect/platform"
import { Effect, Option, Schedule, Schema } from "effect"
import { join } from "node:path"
import { parseDocument } from "yaml"
import { AssertionFailure } from "../domain"
import { command } from "../process"
import { TerminalConfig, TerminalDriver, TerminalScreen, waitForTerminal } from "../terminal"
import { HarnessTerminalReceipt, TerminalInput, TerminalTurn } from "./terminal"
import { decodeHermesTerminalEvents, verifyHermesTerminalLifecycle } from "./hermes-terminal-events"
import { hermesInstallation } from "./installation"

export const HermesTerminalConfig = Schema.Struct({ ...TerminalConfig.omit("args", "columns", "rows").fields,
  model: TerminalInput, initialModel: TerminalInput, endpoint: Schema.NonEmptyString, interrupt: TerminalTurn, recovery: TerminalTurn })
const Export = Schema.Struct({ id: Schema.NonEmptyString, model: Schema.String, billing_provider: Schema.String, billing_base_url: Schema.String,
  messages: Schema.Array(Schema.Struct({ id: Schema.Int, session_id: Schema.String, role: Schema.String,
    content: Schema.NullOr(Schema.String), finish_reason: Schema.NullOr(Schema.String) })) })
const fail = (message: string) => new AssertionFailure({ message })

/** Pinned Hermes keyboard journey; its own observer and persisted transcript prove the two turns. */
export const hermesTerminal = (config: typeof HermesTerminalConfig.Type, onCleanupError: (message: string) => void) => Effect.scoped(Effect.gen(function* () {
  yield* Schema.decodeUnknown(HermesTerminalConfig)(config)
  const fs = yield* FileSystem.FileSystem
  for (const turn of [config.interrupt, config.recovery]) if (turn.prompt.includes(turn.expected)) return yield* fail("Terminal response marker must not appear in echoed input")
  yield* fs.makeDirectory(config.evidence, { recursive: true })
  const run = (args: readonly string[]) => command(config.executable, args, { env: config.environment, inheritEnv: false,
    cwd: Option.some(config.cwd), timeoutMs: 30_000, maxOutputBytes: 16 * 1024 * 1024 }).pipe(Effect.flatMap(output =>
      output.exitCode === 0 ? Effect.succeed(output.stdout) : Effect.fail(fail("Hermes terminal inspection command failed"))))
  if (!(yield* run(["--version"])).includes(hermesInstallation.version)) return yield* fail("Hermes terminal executable differs from its pinned installation")
  const home = config.environment.HERMES_HOME
  if (!home) return yield* fail("Hermes terminal requires an explicit owned HERMES_HOME")
  const configuration = join(home, "config.yaml"), eventsFile = join(config.evidence, "lifecycle.jsonl")
  const observer = join(config.evidence, "observer.cjs")
  yield* fs.writeFileString(eventsFile, "", { flag: "wx", mode: 0o600 })
  yield* fs.writeFileString(observer, `const fs=require('node:fs');const bytes=fs.readFileSync(0);if(bytes.length>65536)process.exit(1);const event=JSON.parse(bytes);const path=process.env.LAB_HERMES_LIFECYCLE;if(fs.statSync(path).size+bytes.length>1048576)process.exit(1);fs.appendFileSync(path,JSON.stringify(event)+'\\n');`, { flag: "wx", mode: 0o600 })
  const original = yield* fs.readFileString(configuration)
  const patched = yield* Effect.try({ try: () => {
    const document = parseDocument(original)
    if (document.errors.length || document.hasIn(["hooks", "on_session_end"])) throw new Error("Existing or invalid observer configuration")
    document.setIn(["display", "streaming"], true)
    document.setIn(["hooks", "on_session_end"], [{ command: [config.runtime, observer].map(value => '"' + value.replaceAll('\\', '\\\\').replaceAll('"', '\\"') + '"').join(" "), timeout: 5 }])
    return String(document)
  }, catch: () => fail("Cannot add an observation-only Hermes hook to this owned configuration") })
  // This fixture exclusively owns its private harness home; restore the exact input,
  // including display preferences, after the terminal process has been reaped.
  yield* fs.writeFileString(join(config.evidence, "configuration.before.yaml"), original, { flag: "wx", mode: 0o600 })
  yield* Effect.acquireRelease(fs.writeFileString(configuration, patched), () => fs.writeFileString(configuration, original).pipe(
    Effect.catchAll(error => Effect.sync(() => { onCleanupError(error.message) }))))
  const readEvents = Effect.gen(function* () {
    if (Number((yield* fs.stat(eventsFile)).size) > 1024 * 1024) return yield* fail("Hermes lifecycle evidence exceeds 1 MiB")
    const text = yield* fs.readFileString(eventsFile)
    return yield* decodeHermesTerminalEvents(text.slice(0, text.lastIndexOf("\n") + 1))
  })
  const waitEvents = (count: number) => readEvents.pipe(Effect.repeat({ until: events => events.length >= count,
    schedule: Schedule.identity<Effect.Effect.Success<typeof readEvents>>().pipe(Schedule.addDelay(() => "100 millis")) }), Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("Hermes did not persist the expected native turn lifecycle") }))
  const terminal = yield* (yield* TerminalDriver).start(TerminalConfig.make({ ...config,
    environment: { ...config.environment, LAB_HERMES_LIFECYCLE: eventsFile },
    args: ["chat", "--cli", "--accept-hooks", "--provider", "custom:magnitude", "--model", config.initialModel, "--reasoning", "none", "--toolsets", "file", "--max-turns", "8", "--run-budget", "240"], columns: 160, rows: 50,
  }), onCleanupError)
  const wait = (predicate: (lines: readonly string[]) => boolean, label: string, timeout = 30_000) => waitForTerminal(terminal, screen => predicate(screen.lines), label, timeout)
  const save = (name: string, screen: typeof TerminalScreen.Type) => Schema.encode(Schema.parseJson(TerminalScreen))(screen).pipe(Effect.flatMap(value => fs.writeFileString(join(config.evidence, `${name}.json`), value)))
  yield* wait(lines => lines.some(line => line.includes(config.initialModel)), "show the initial Hermes model")
  yield* terminal.write(`/model ${config.model} --provider custom:magnitude --session`)
  // The banner precedes the input loop. Only retry Enter for this idempotent selection;
  // an empty input is a no-op. Generation is submitted exactly once.
  yield* terminal.write("\r").pipe(Effect.zipRight(terminal.screen), Effect.repeat({
    until: screen => screen.lines.some(line => line.includes("Model switched:") && line.includes(config.model)),
    schedule: Schedule.identity<typeof TerminalScreen.Type>().pipe(Schedule.addDelay(() => "100 millis")),
  }), Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("Hermes did not confirm keyboard model selection") }))
  yield* save("selected-model", yield* terminal.screen)
  const submit = (prompt: string) => terminal.write(prompt).pipe(Effect.zipRight(wait(lines => lines.some(line => line.includes(prompt)), "render the entered prompt")), Effect.zipRight(terminal.write("\r")))
  yield* submit(config.interrupt.prompt)
  yield* save("streaming", yield* wait(lines => lines.some(line => line.includes(config.interrupt.expected)), "render streamed Hermes output", 300_000))
  if ((yield* readEvents).length !== 0) return yield* fail("Hermes completed before the interruption input")
  yield* terminal.write("\u0003")
  const interrupted = yield* waitEvents(1)
  if (interrupted.length !== 1 || !interrupted[0]!.extra.interrupted) return yield* fail("Hermes did not acknowledge keyboard interruption")
  yield* submit(config.recovery.prompt)
  yield* save("recovered", yield* wait(lines => lines.some(line => line.includes(config.recovery.expected)), "render the Hermes recovery answer", 300_000))
  const completed = yield* waitEvents(2)
  const lifecycle = yield* verifyHermesTerminalLifecycle(completed, interrupted[0]!.session_id, config.model)
  yield* submit("/exit")
  const exited = yield* terminal.exited.pipe(Effect.timeoutFail({ duration: "20 seconds", onTimeout: () => fail("Hermes did not exit through keyboard input") }))
  if (exited.code !== 0 || Option.isSome(exited.signal)) return yield* fail("Hermes terminal did not exit normally")
  yield* verifyHermesTerminalLifecycle(yield* decodeHermesTerminalEvents(yield* fs.readFileString(eventsFile)), lifecycle.sessionId, config.model)
  const transcript = yield* run(["sessions", "export", "--session-id", lifecycle.sessionId, "--format", "jsonl", "-", "--yes"])
  yield* fs.writeFileString(join(config.evidence, "session.jsonl"), transcript)
  const saved = yield* Schema.decodeUnknown(Schema.parseJson(Export))(transcript)
  // Hermes persists an interruption notice instead of partial generated text. Its native
  // lifecycle plus the rendered nonce prove interruption; the final persisted answer proves recovery.
  const messages = saved.messages.filter(message => message.role === "assistant" && message.content?.trim())
  if (saved.id !== lifecycle.sessionId || saved.model !== config.model || saved.billing_provider !== "custom" || saved.billing_base_url !== config.endpoint || messages.length !== 2 || messages.some(message => message.session_id !== saved.id)
    || messages[0]!.finish_reason !== null || messages[1]!.finish_reason !== "stop" || !messages[1]!.content!.includes(config.recovery.expected)) return yield* fail("Hermes persisted transcript does not match interrupted and recovered terminal output")
  return HarnessTerminalReceipt.make({ sessionId: lifecycle.sessionId, pid: terminal.pid, model: config.model,
    interruptedMessageId: String(messages[0]!.id), recoveredMessageId: String(messages[1]!.id), text: messages[1]!.content! })
}))
