import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { expect, test } from "vitest"
import { command, ProcessExecutorLive } from "../src/process"
import { decodeHermesTerminalEvents, verifyHermesTerminalLifecycle } from "../src/harnesses/hermes-terminal-events"

const python = process.env.LAB_HERMES_PYTHON_EXECUTABLE
// Uses Hermes's installed observer serializer, not a lab-authored wire fixture.
// It establishes envelope compatibility only; no generation or interruption is claimed.
test.skipIf(!python)("decodes the installed Hermes observer's canonical native envelopes", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-hermes-observer-" })
  const output = yield* command(python!, ["-c", `from agent.shell_hooks import _serialize_payload
for index, interrupted in [(1, True), (2, False)]:
    print(_serialize_payload("on_session_end", dict(session_id="native-session", task_id=f"task-{index}",
        turn_id=f"turn-{index}", model="served-model", platform="cli", completed=not interrupted,
        failed=False, interrupted=interrupted, turn_exit_reason="interrupted" if interrupted else "text_response(finish_reason=stop)")))
`], { cwd: Option.some(root), env: { HOME: root, USERPROFILE: root, PATH: process.env.PATH ?? "" }, inheritEnv: false, timeoutMs: 15_000 })
  expect(output.exitCode).toBe(0)
  const events = yield* decodeHermesTerminalEvents(output.stdout)
  const result = yield* verifyHermesTerminalLifecycle(events, "native-session", "served-model")
  expect(result.interruptedTurnId).toBe("turn-1")
  expect(result.recoveredTurnId).toBe("turn-2")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 20000)
