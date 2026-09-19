import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { command } from "../process"

const optional = <A, I>(s: Schema.Schema<A, I>) => Schema.optionalWith(s, { as: "Option", exact: true })
const Event = Schema.Union(
  Schema.Struct({ type: Schema.Literal("system"), subtype: Schema.Literal("init"), model: Schema.String, session_id: Schema.String }),
  Schema.Struct({ type: Schema.Literal("text"), text: Schema.String }),
  Schema.Struct({ type: Schema.Literal("tool_use"), name: Schema.NonEmptyString, tool_call_id: optional(Schema.String) }),
  Schema.Struct({ type: Schema.Literal("tool_result"), name: Schema.NonEmptyString, tool_call_id: optional(Schema.String), is_error: Schema.Boolean }),
  Schema.Struct({ type: Schema.Literal("result"), session_id: Schema.NonEmptyString, exit_code: Schema.Int, text: Schema.String, error: optional(Schema.String) }),
)
export const HermesTurn = Schema.Struct({ sessionId: Schema.NonEmptyString, text: Schema.String, streamed: Schema.Boolean, tools: Schema.Array(Schema.String) })
export type HermesTurn = typeof HermesTurn.Type
export const HermesConfig = Schema.Struct({ executable: Schema.String, cwd: Schema.String, environment: Schema.Record({ key: Schema.String, value: Schema.String }),
  evidence: Schema.String, model: Schema.String })
const fail = (message: string) => new AssertionFailure({ message })

export const hermes = (config: typeof HermesConfig.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(config.evidence, { recursive: true })
  let index = 0
  const prompt = (message: string, session: Option.Option<string> = Option.none()) => Effect.gen(function* () {
    const label = `turn-${++index}`
    const query = join(config.evidence, `${label}.prompt.txt`)
    yield* fs.writeFileString(query, message)
    const output = yield* command(config.executable, ["chat", "--query-file", query, "--format", "stream-json", "--oneshot", "--provider", "custom:magnitude",
      "--model", config.model, "--reasoning", "none", "--toolsets", "file", "--max-turns", "8", "--run-budget", "240",
      ...Option.match(session, { onNone: () => [], onSome: id => ["--resume", id] })],
      { cwd: Option.some(config.cwd), env: config.environment, inheritEnv: false, timeoutMs: 300_000, maxOutputBytes: 16 * 1024 * 1024 })
    yield* fs.writeFileString(join(config.evidence, `${label}.jsonl`), output.stdout)
    yield* fs.writeFileString(join(config.evidence, `${label}.stderr.log`), output.stderr)
    if (output.exitCode !== 0) return yield* fail(`Hermes exited ${output.exitCode}: ${(output.stderr || output.stdout).slice(-1200)}`)
    const events = yield* Effect.forEach(output.stdout.split("\n").filter(line => line.trim()), line => Schema.decodeUnknown(Schema.parseJson(Event))(line).pipe(
      Effect.mapError(() => fail("Hermes emitted an invalid JSON event"))))
    const first = events[0], last = events.at(-1)
    if (first?.type !== "system" || first.model !== config.model) return yield* fail("Hermes did not initialize the requested model")
    if (last?.type !== "result" || last.exit_code !== 0 || Option.isSome(last.error) || !last.text.trim()) return yield* fail("Hermes did not finish with a successful result")
    if (events.filter(e => e.type === "result").length !== 1 || events.filter(e => e.type === "system").length !== 1) return yield* fail("Hermes emitted duplicate lifecycle events")
    if ((first.session_id && first.session_id !== last.session_id) || Option.exists(session, id => last.session_id !== id)) return yield* fail("Hermes did not preserve the selected session")
    const pending = new Map<string, string>(), tools: string[] = []
    let streamed = ""
    for (const event of events) {
      if (event.type === "text") streamed += event.text
      if (event.type === "tool_use") {
        const id = Option.getOrElse(event.tool_call_id, () => event.name)
        if (pending.has(id)) return yield* fail("Hermes emitted a duplicate active tool call")
        pending.set(id, event.name)
      }
      if (event.type === "tool_result") {
        const id = Option.getOrElse(event.tool_call_id, () => event.name)
        if (event.is_error || pending.get(id) !== event.name) return yield* fail(`Hermes tool ${event.name} failed or had no matching invocation`)
        pending.delete(id); tools.push(event.name)
      }
    }
    if (pending.size || !streamed.trim() || !streamed.endsWith(last.text)) return yield* fail("Hermes result did not match completed streamed generation and tools")
    return HermesTurn.make({ sessionId: last.session_id, text: last.text, streamed: true, tools })
  })
  return { prompt }
})
