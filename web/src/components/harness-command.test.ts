import { expect, it } from "vitest"
import { Effect, Schema } from "effect"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Command } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { harnessCommand, shellArgument } from "./harness-command"

const model = Schema.decodeUnknownSync(ProviderModelIdSchema)("Qwen3.6-35B-A3B-Q4_K_M")
it.each([
  ["pi", `pi --provider magnitude --model ${model}`],
  ["opencode", `opencode --model magnitude/${model}`],
  ["hermes", `hermes --provider custom:magnitude --model ${model}`],
  ["codex", `codex -c model_provider=magnitude --model magnitude-local/${model}`],
  ["claude-code", `claude --model anthropic-local/${model}`],
  ["oh-my-pi", `omp --model magnitude/${model}`],
  ["cline", `cline --tui --provider openai-compatible --model ${model}`],
  ["openclaw", "openclaw tui"],
])("formats %s using its actual provider/model interface", (id, expected) => {
  const harness = Schema.decodeUnknownSync(HarnessIdSchema)(id)
  expect(harnessCommand(harness, model, "darwin")).toBe(expected)
  expect(harnessCommand(harness, model, "win32")).toBe(expected)
})
it.skipIf(process.platform === "win32")("quotes model arguments literally without shell evaluation", async () => {
  const value = "org/model ' $(echo injected) `echo injected`; more"
  const output = await Effect.runPromise(Command.string(Command.make("/bin/sh", "-c", `printf '%s' ${shellArgument(value, "darwin")}`)).pipe(Effect.provide(NodeContext.layer)))
  expect(output).toBe(value)
})
it("quotes apostrophes and dollar expressions literally for PowerShell", () => {
  expect(shellArgument("model's $(unsafe)", "win32")).toBe("'model''s $(unsafe)'")
})
it("reflects changed model IDs while retaining the harness's provider", () => {
  const id = Schema.decodeUnknownSync(HarnessIdSchema)("opencode")
  const next = Schema.decodeUnknownSync(ProviderModelIdSchema)("another-model-Q8")
  expect(harnessCommand(id, next, "linux")).toBe("opencode --model magnitude/another-model-Q8")
  expect(harnessCommand(id, next, "linux")).not.toContain(model)
})
