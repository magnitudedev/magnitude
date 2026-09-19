import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import { openCode } from "../src/harnesses/opencode"
import { hermes } from "../src/harnesses/hermes"
import { ProcessExecutor } from "../src/process"

const json = Schema.encodeSync(Schema.parseJson(Schema.Unknown))
const check = (kind: "opencode" | "hermes", variant: string) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-cli-protocol-" })
  const session = variant === "session" ? "other-session" : "fixture-session"
  const executor: ProcessExecutor = { run: spec => Effect.sync(() => {
    expect(spec.inheritEnv).toBe(false)
    expect(spec.env).toEqual({ HOME: root })
    if (spec.args[0] === "models") return { exitCode: 0, stdout: "magnitude/fixture-model\n", stderr: "" }
    if (spec.args[0] === "export") return { exitCode: 0, stderr: "", stdout: json({ info: { id: session }, messages: [{ info: {
      role: "assistant", providerID: variant === "provider" ? "other" : "magnitude", modelID: "fixture-model",
    } }] }) }
    if (variant === "setup") return { exitCode: 1, stdout: "Hermes is not configured; run hermes setup", stderr: "" }
    const events = kind === "opencode" ? [
      { type: "step_start", sessionID: session, part: { type: "step-start" } },
      { type: "tool_use", sessionID: session, part: { type: "tool", tool: "read", state: { status: variant === "tool" ? "error" : "completed" } } },
      { type: "text", sessionID: session, part: { type: "text", text: "HELLO" } },
      { type: "step_finish", sessionID: session, part: { type: "step-finish", reason: variant === "truncated" ? "length" : "stop" } },
    ] : [
      { type: "system", subtype: "init", model: variant === "provider" ? "other-model" : "fixture-model", session_id: session },
      { type: "tool_use", name: "read_file", tool_call_id: "call-1" },
      { type: "tool_result", name: "read_file", tool_call_id: "call-1", is_error: variant === "tool" },
      { type: "text", text: "HELLO" },
      ...(variant === "truncated" ? [] : [{ type: "result", session_id: session, text: "HELLO", exit_code: 0 }]),
    ]
    return { exitCode: variant === "exit" ? 1 : 0, stdout: events.map(json).join("\n") + "\n", stderr: "" }
  }) }
  return yield* Effect.gen(function* () {
    const config = { executable: kind, cwd: root, environment: { HOME: root }, evidence: root, model: "fixture-model" }
    const client = yield* kind === "opencode" ? openCode(config) : hermes(config)
    const result = yield* client.prompt("Read the fixture then say HELLO", Option.some("fixture-session")).pipe(Effect.either)
    expect(result._tag).toBe(variant === "pass" ? "Right" : "Left")
    if (result._tag === "Right") expect(result.right.text).toBe("HELLO")
    if (kind === "hermes" && variant === "setup" && result._tag === "Left") expect(result.left.message).toContain("run hermes setup")
  }).pipe(Effect.provideService(ProcessExecutor, executor))
})).pipe(Effect.provide(BunContext.layer)))

test.each(["opencode", "hermes"] as const)("%s validates terminal output, selected model, tools and resumed session", async kind => {
  for (const variant of ["pass", "session", "provider", "tool", "truncated", "exit", "setup"]) await check(kind, variant)
})
