import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { decodeHermesTerminalEvents, HermesTerminalEvent, verifyHermesTerminalLifecycle } from "../src/harnesses/hermes-terminal-events"

const event = (interrupted: boolean, suffix: string) => HermesTerminalEvent.make({ hook_event_name: "on_session_end", session_id: "native-session",
  extra: { task_id: `task-${suffix}`, turn_id: `turn-${suffix}`, model: "served-model", platform: "cli", completed: !interrupted,
    failed: false, interrupted, turn_exit_reason: interrupted ? "interrupted" : "text_response(finish_reason=stop)" } })
const first = event(true, "1"), second = event(false, "2")
const encode = (value: unknown) => Schema.encodeSync(Schema.parseJson(Schema.Unknown))(value) + "\n"

test("Hermes canonical observer records distinguish interruption from successful recovery", async () => {
  const proof = await Effect.runPromise(decodeHermesTerminalEvents(encode(first) + encode(second)).pipe(
    Effect.flatMap(events => verifyHermesTerminalLifecycle(events, "native-session", "served-model"))))
  expect(proof.interruptedTurnId).toBe("turn-1")
  expect(proof.recoveredTurnId).toBe("turn-2")
})

test.each([
  ["normal completion before interruption", [event(false, "1"), second]],
  ["failed recovery", [first, { ...second, extra: { ...second.extra, failed: true } }]],
  ["another model", [first, { ...second, extra: { ...second.extra, model: "other" } }]],
  ["another session", [first, { ...second, session_id: "other" }]],
  ["reused turn", [first, { ...second, extra: { ...second.extra, turn_id: first.extra.turn_id } }]],
  ["missing recovery", [first]],
] as const)("rejects %s", async (_, events) => {
  expect((await Effect.runPromise(verifyHermesTerminalLifecycle(events, "native-session", "served-model").pipe(Effect.either)))._tag).toBe("Left")
})

test.each([
  encode({ hook_event_name: "on_session_end", session_id: "native-session", extra: {} }),
  encode(first) + encode(first), encode(first).trimEnd(), "not JSON\n", "x".repeat(1024 * 1024 + 1),
])("rejects incomplete, duplicate, legacy or oversized lifecycle evidence", async wire => {
  expect((await Effect.runPromise(decodeHermesTerminalEvents(wire).pipe(Effect.either)))._tag).toBe("Left")
})


test("native CLI keeps its session task while starting a distinct recovery turn", async () => {
  const result = await Effect.runPromise(verifyHermesTerminalLifecycle([first, { ...second, extra: { ...second.extra, task_id: first.extra.task_id } }], "native-session", "served-model"))
  expect(result.interruptedTaskId).toBe(result.recoveredTaskId)
  expect(result.interruptedTurnId).not.toBe(result.recoveredTurnId)
})
