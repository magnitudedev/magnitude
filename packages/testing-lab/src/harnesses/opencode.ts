import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { command } from "../process"

const optional = <A, I>(s: Schema.Schema<A, I>) => Schema.optionalWith(s, { as: "Option", exact: true })
const Event = Schema.Struct({ type: Schema.String, sessionID: Schema.NonEmptyString, part: optional(Schema.Unknown), error: optional(Schema.Unknown) })
const Text = Schema.Struct({ type: Schema.Literal("text"), text: Schema.String })
const Finish = Schema.Struct({ type: Schema.Literal("step-finish"), reason: Schema.String })
const Tool = Schema.Struct({ type: Schema.Literal("tool"), tool: Schema.NonEmptyString, state: Schema.Struct({ status: Schema.String }) })
const Export = Schema.Struct({ info: Schema.Struct({ id: Schema.NonEmptyString }), messages: Schema.Array(Schema.Struct({ info: Schema.Unknown })) })
const Assistant = Schema.Struct({ role: Schema.Literal("assistant"), providerID: Schema.String, modelID: Schema.String })
export const OpenCodeTurn = Schema.Struct({ sessionId: Schema.NonEmptyString, text: Schema.String, tools: Schema.Array(Schema.String) })
export type OpenCodeTurn = typeof OpenCodeTurn.Type
export const OpenCodeConfig = Schema.Struct({ executable: Schema.String, cwd: Schema.String, environment: Schema.Record({ key: Schema.String, value: Schema.String }),
  evidence: Schema.String, model: Schema.String })
const fail = (message: string) => new AssertionFailure({ message })

/** OpenCode's JSON mode emits completed text parts, not token deltas. Streaming is observed separately. */
export const openCode = (config: typeof OpenCodeConfig.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(config.evidence, { recursive: true })
  const run = (args: readonly string[]) => command(config.executable, args, { cwd: Option.some(config.cwd), env: config.environment,
    inheritEnv: false, timeoutMs: 300_000, maxOutputBytes: 16 * 1024 * 1024 })
  const models = yield* run(["models", "magnitude"])
  if (models.exitCode !== 0 || !models.stdout.split(/\r?\n/).includes(`magnitude/${config.model}`)) return yield* fail("OpenCode did not discover the model from the application's connection")
  yield* fs.writeFileString(join(config.evidence, "models.txt"), models.stdout)
  let index = 0
  const prompt = (message: string, session: Option.Option<string> = Option.none()) => Effect.gen(function* () {
    const label = `turn-${++index}`
    const output = yield* run(["run", "--format", "json", "--model", `magnitude/${config.model}`, "--variant", "off",
      ...Option.match(session, { onNone: () => [], onSome: id => ["--session", id] }), message])
    yield* fs.writeFileString(join(config.evidence, `${label}.jsonl`), output.stdout)
    yield* fs.writeFileString(join(config.evidence, `${label}.stderr.log`), output.stderr)
    if (output.exitCode !== 0) return yield* fail(`OpenCode exited ${output.exitCode}: ${output.stderr.slice(-1200)}`)
    const events = yield* Effect.forEach(output.stdout.split("\n").filter(line => line.trim()), line => Schema.decodeUnknown(Schema.parseJson(Event))(line).pipe(
      Effect.mapError(() => fail("OpenCode emitted an invalid JSON event"))))
    const id = events[0]?.sessionID
    if (!id || events.some(e => e.sessionID !== id) || Option.exists(session, expected => id !== expected)) return yield* fail("OpenCode did not preserve the requested session")
    let text = "", reason = ""
    const tools: string[] = []
    for (const event of events) {
      if (event.type === "error" || Option.isSome(event.error)) return yield* fail("OpenCode reported a generation error; inspect the retained event stream")
      const part = Option.getOrUndefined(event.part)
      if (event.type === "text") {
        const decoded = yield* Schema.decodeUnknown(Text)(part).pipe(Effect.mapError(() => fail("OpenCode text part is invalid")))
        text += decoded.text
      }
      if (event.type === "step_finish") {
        const decoded = yield* Schema.decodeUnknown(Finish)(part).pipe(Effect.mapError(() => fail("OpenCode completion part is invalid")))
        reason = decoded.reason
      }
      if (event.type === "tool_use") {
        const decoded = yield* Schema.decodeUnknown(Tool)(part).pipe(Effect.mapError(() => fail("OpenCode tool part is invalid")))
        if (decoded.state.status !== "completed") return yield* fail(`OpenCode tool ${decoded.tool} did not complete successfully`)
        tools.push(decoded.tool)
      }
    }
    if (!text.trim() || reason !== "stop") return yield* fail(`OpenCode produced no completed answer (finish reason ${reason})`)
    const transcript = yield* run(["export", id])
    yield* fs.writeFileString(join(config.evidence, `${label}.session.json`), transcript.stdout)
    if (transcript.exitCode !== 0) return yield* fail("OpenCode could not export the persisted session")
    const saved = yield* Schema.decodeUnknown(Schema.parseJson(Export))(transcript.stdout).pipe(Effect.mapError(() => fail("OpenCode exported an invalid transcript")))
    const assistants = saved.messages.flatMap(m => Schema.is(Assistant)(m.info) ? [m.info] : [])
    if (saved.info.id !== id || assistants.length === 0 || assistants.some(a => a.providerID !== "magnitude" || a.modelID !== config.model)) return yield* fail("OpenCode's persisted assistant messages used another provider or model")
    return OpenCodeTurn.make({ sessionId: id, text, tools })
  })
  return { prompt }
})
