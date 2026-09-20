import { Effect, Schema } from "effect"
import { AssertionFailure } from "../domain"

export const HermesTerminalEvent = Schema.Struct({
  hook_event_name: Schema.Literal("on_session_end"), session_id: Schema.NonEmptyString,
  extra: Schema.Struct({ task_id: Schema.NonEmptyString, turn_id: Schema.NonEmptyString,
    model: Schema.NonEmptyString, platform: Schema.Literal("cli"), completed: Schema.Boolean,
    failed: Schema.Boolean, interrupted: Schema.Boolean, turn_exit_reason: Schema.String }),
})
export const HermesTerminalLifecycle = Schema.Struct({ sessionId: Schema.NonEmptyString, model: Schema.NonEmptyString,
  interruptedTurnId: Schema.NonEmptyString, recoveredTurnId: Schema.NonEmptyString,
  interruptedTaskId: Schema.NonEmptyString, recoveredTaskId: Schema.NonEmptyString })
const fail = (message: string) => new AssertionFailure({ message })

/** Decode the pinned client's observer envelope; legacy exit callbacks cannot prove a turn. */
export const decodeHermesTerminalEvents = (jsonl: string) => Effect.gen(function* () {
  if (Buffer.byteLength(jsonl) > 1024 * 1024) return yield* fail("Hermes terminal lifecycle evidence exceeds 1 MiB")
  if (jsonl && !jsonl.endsWith("\n")) return yield* fail("Hermes terminal lifecycle evidence has an incomplete frame")
  const events: typeof HermesTerminalEvent.Type[] = []
  const ids = new Set<string>()
  for (const line of jsonl.split("\n").filter(Boolean)) {
    const event = yield* Schema.decodeUnknown(Schema.parseJson(HermesTerminalEvent))(line).pipe(
      Effect.mapError(() => fail("Hermes terminal observer omitted canonical turn lifecycle fields")))
    if (ids.has(event.extra.turn_id)) return yield* fail("Hermes terminal observer emitted a duplicate turn")
    ids.add(event.extra.turn_id)
    events.push(event)
  }
  return events
})

/** Lifecycle proof is combined with actual rendered output and the native transcript by H7. */
export const verifyHermesTerminalLifecycle = (events: ReadonlyArray<typeof HermesTerminalEvent.Type>, sessionId: string, model: string) => Effect.gen(function* () {
  if (events.length !== 2) return yield* fail("Hermes terminal journey requires exactly two completed lifecycle records")
  if (events.some(event => event.session_id !== sessionId || event.extra.model !== model || event.extra.platform !== "cli")) {
    return yield* fail("Hermes terminal lifecycle belongs to another session, model or surface")
  }
  const first = events[0]!.extra, second = events[1]!.extra
  if (!first.interrupted || first.completed || first.failed) return yield* fail("Hermes did not acknowledge an interrupted first turn")
  if (second.interrupted || !second.completed || second.failed) return yield* fail("Hermes did not complete the recovery turn")
  if (first.turn_id === second.turn_id) return yield* fail("Hermes reused a turn identity during recovery")
  return HermesTerminalLifecycle.make({ sessionId, model, interruptedTurnId: first.turn_id, recoveredTurnId: second.turn_id,
    interruptedTaskId: first.task_id, recoveredTaskId: second.task_id })
})
