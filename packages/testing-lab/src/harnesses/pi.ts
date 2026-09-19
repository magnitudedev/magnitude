import { Effect, Option, Schema } from "effect"
import { AssertionFailure } from "../domain"
import { jsonProcess, JsonProcessConfig } from "../json-process"

const opt = <A, I>(s: Schema.Schema<A, I>) => Schema.optionalWith(s, { as: "Option", exact: true })
const Event = Schema.Struct({ type: Schema.String, id: opt(Schema.String), success: opt(Schema.Boolean), error: opt(Schema.String), data: opt(Schema.Unknown),
  message: opt(Schema.Unknown), toolName: opt(Schema.String), isError: opt(Schema.Boolean), result: opt(Schema.Unknown),
  assistantMessageEvent: opt(Schema.Struct({ type: Schema.String, delta: opt(Schema.String) })) })
const Assistant = Schema.Struct({ role: Schema.Literal("assistant"), provider: Schema.String, model: Schema.String,
  stopReason: Schema.String, content: Schema.Array(Schema.Unknown) })
export const PiTurn = Schema.Struct({ text: Schema.String, streamed: Schema.Boolean, tools: Schema.Array(Schema.Struct({ name: Schema.String, failed: Schema.Boolean })),
  sessionId: Schema.NonEmptyString })
export type PiTurn = typeof PiTurn.Type
const assertion = (message: string) => new AssertionFailure({ message })

/** Pi 0.85.1 RPC adapter. Acceptance is agent_settled, never prompt admission or agent_end. */
export const piSession = (config: typeof JsonProcessConfig.Type, model: string) => Effect.gen(function* () {
  const process = yield* jsonProcess(config)
  const next = process.receive.pipe(Effect.flatMap(Schema.decodeUnknown(Event)), Effect.mapError(error => assertion(`Pi RPC failed: ${String(error).slice(-800)}`)))
  const command = (type: string, fields: Readonly<Record<string, unknown>> = {}) => Effect.gen(function* () {
    const id = crypto.randomUUID()
    yield* process.send({ id, type, ...fields })
    for (;;) {
      const event = yield* next
      if (event.type !== "response" || !Option.contains(event.id, id)) continue
      if (!Option.contains(event.success, true)) return yield* assertion(Option.getOrElse(event.error, () => `Pi rejected ${type}`))
      return event.data
    }
  }).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => assertion(`Pi command ${type} did not respond`) }))
  const state = () => command("get_state").pipe(Effect.flatMap(data => Schema.decodeUnknown(Schema.Struct({ sessionId: Schema.NonEmptyString,
    model: Schema.Struct({ id: Schema.String, provider: Schema.String }) }))(Option.getOrUndefined(data))),
    Effect.mapError(() => assertion("Pi did not report its session and selected model")))
  const selected = yield* state()
  if (selected.model.id !== model || selected.model.provider !== "magnitude") return yield* assertion("Pi did not consume the Magnitude provider/model selection")
  const prompt = (message: string) => Effect.gen(function* () {
    const id = crypto.randomUUID()
    yield* process.send({ type: "prompt", id, message })
    let accepted = false, text = "", streamed = false, completed = false
    const tools: { name: string; failed: boolean }[] = []
    for (;;) {
      const event = yield* next
      if (event.type === "response" && Option.contains(event.id, id)) {
        if (!Option.contains(event.success, true)) return yield* assertion(Option.getOrElse(event.error, () => "Pi rejected the prompt"))
        accepted = true
      }
      if (event.type === "auto_retry_start" || event.type === "extension_error") return yield* assertion(`Pi reported ${event.type}; the lab does not retry model assertions`)
      if (event.type === "message_update" && Option.isSome(event.assistantMessageEvent) && event.assistantMessageEvent.value.type === "text_delta") {
        text += Option.getOrElse(event.assistantMessageEvent.value.delta, () => ""); streamed = true
      }
      if (event.type === "tool_execution_end") {
        if (Option.isNone(event.toolName) || Option.isNone(event.isError)) return yield* assertion("Pi omitted tool completion fields")
        tools.push({ name: event.toolName.value, failed: event.isError.value })
      }
      if (event.type === "message_end" && Option.isSome(event.message) && Schema.is(Assistant)(event.message.value)) {
        const message = event.message.value
        if (message.provider !== "magnitude" || message.model !== model) return yield* assertion("Pi generated with another provider or model")
        if (!["stop", "toolUse"].includes(message.stopReason)) return yield* assertion(`Pi generation ended with ${message.stopReason}`)
        completed = message.stopReason === "stop"
      }
      if (event.type === "agent_settled") break
    }
    if (!accepted || !completed || !streamed || !text.trim()) return yield* assertion("Pi settled without an accepted prompt and streamed assistant output")
    if (tools.some(t => t.failed)) return yield* assertion("Pi reported a failed tool execution")
    const current = yield* state()
    if (current.sessionId !== selected.sessionId) return yield* assertion("Pi switched sessions during the turn")
    return PiTurn.make({ text, streamed, tools, sessionId: current.sessionId })
  }).pipe(Effect.timeoutFail({ duration: "5 minutes", onTimeout: () => assertion("Pi turn exceeded five minutes") }))
  return { prompt, state, abort: command("abort").pipe(Effect.asVoid) }
})
