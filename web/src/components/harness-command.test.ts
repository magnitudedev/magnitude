import { expect, it } from "vitest"
import { Effect, Option, Schema } from "effect"
import { HarnessIdSchema } from "@magnitudedev/sdk"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Command } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { harnessCommand, shellArgument } from "@magnitudedev/sdk"

const model = Schema.decodeUnknownSync(ProviderModelIdSchema)("Qwen3.6-35B-A3B-Q4_K_M")
it.each([
  ["pi", `pi --provider magnitude --model ${model}`],
  ["opencode", `OPENCODE_CONFIG_CONTENT='{"model":"magnitude/${model}"}' opencode --standalone`],
  ["hermes", `hermes --provider custom:magnitude --model ${model}`],
  ["codex", `codex -c model_provider=magnitude --model magnitude-local/${model}`],
  ["claude-code", `claude --model anthropic-local/${model}`],
  ["oh-my-pi", `omp --model magnitude/${model}`],
  ["cline", `cline --tui --provider openai-compatible --model ${model}`],
  ["openclaw", "openclaw tui"],
])("formats %s using its actual provider/model interface", (id, expected) => {
  const harness = Schema.decodeUnknownSync(HarnessIdSchema)(id)
  expect(harnessCommand(harness, model, "darwin", Option.none())).toBe(expected)
  if (id !== "opencode") expect(harnessCommand(harness, model, "win32", Option.none())).toBe(expected)
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
  expect(harnessCommand(id, next, "linux", Option.some("1.18.23"))).toBe("opencode --model magnitude/another-model-Q8")
  expect(harnessCommand(id, next, "linux", Option.some("1.18.23"))).not.toContain(model)
})
it.each([
  ["1.18.23", `opencode --model magnitude/${model}`],
  ["2.2.0", `OPENCODE_CONFIG_CONTENT='{"model":"magnitude/${model}"}' opencode --standalone`],
])("launches OpenCode %s with the interface that version accepts", (version, expected) => {
  const id = Schema.decodeUnknownSync(HarnessIdSchema)("opencode")
  expect(harnessCommand(id, model, "darwin", Option.some(version))).toBe(expected)
})
it("pins OpenCode 2's model for PowerShell", () => {
  const id = Schema.decodeUnknownSync(HarnessIdSchema)("opencode")
  expect(harnessCommand(id, model, "win32", Option.some("2.2.0"))).toBe(`$env:OPENCODE_CONFIG_CONTENT='{"model":"magnitude/${model}"}'; opencode --standalone`)
})
it.skipIf(process.platform === "win32")("pins OpenCode 2's model as configuration a POSIX shell passes through unchanged", async () => {
  const id = Schema.decodeUnknownSync(HarnessIdSchema)("opencode")
  const odd = Schema.decodeUnknownSync(ProviderModelIdSchema)(`hf:org/it's "quoted" $(echo injected).gguf`)
  const command = harnessCommand(id, odd, "darwin", Option.some("2.2.0")).replace(" opencode --standalone", " /usr/bin/printenv OPENCODE_CONFIG_CONTENT")
  const output = await Effect.runPromise(Command.string(Command.make("/bin/sh", "-c", command)).pipe(Effect.provide(NodeContext.layer)))
  expect(Schema.decodeUnknownSync(Schema.parseJson(Schema.Struct({ model: Schema.String })))(output.trim())).toEqual({ model: `magnitude/${odd}` })
})
