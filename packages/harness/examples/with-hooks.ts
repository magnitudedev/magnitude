import { Effect, Schema, Stream, Ref } from "effect"
import { FetchHttpClient } from "@effect/platform"
import {
  Auth,
  defineTool,
  NativeChatCompletions,
  PromptBuilder,
} from "@magnitudedev/ai"
import {
  defineHarnessTool,
  defineToolkit,
  createHarness,
  type HarnessHooks,
} from "../src/index.js"

// ── Define tools ─────────────────────────────────────────────────────

const shellDef = defineTool({
  name: "run_shell",
  description: "Run a shell command.",
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: Schema.String,
})

const shellTool = defineHarnessTool({
  definition: shellDef,
  execute: (input) =>
    Effect.succeed(`$ ${input.command}\nCommand executed successfully.`),
})

const readDef = defineTool({
  name: "read_file",
  description: "Read a file.",
  inputSchema: Schema.Struct({ path: Schema.String }),
  outputSchema: Schema.String,
})

const readTool = defineHarnessTool({
  definition: readDef,
  execute: (input) =>
    Effect.succeed(`Contents of ${input.path}: ...`),
})

// ── Build toolkit ────────────────────────────────────────────────────

const toolkit = defineToolkit({
  shell: { tool: shellTool },
  read: { tool: readTool },
})

// ── Define hooks ─────────────────────────────────────────────────────

const hooks: HarnessHooks = {
  beforeExecute: (ctx) =>
    Effect.gen(function* () {
      console.log(`[hook] beforeExecute: ${ctx.toolName}`)
      if (ctx.toolName === "run_shell") {
        const input = ctx.input as { command: string }
        if (input.command.includes("rm -rf")) {
          return { _tag: "Reject" as const, rejection: "Dangerous command blocked" }
        }
      }
      return { _tag: "Proceed" as const }
    }),
  afterExecute: (ctx) =>
    Effect.sync(() => {
      console.log(`[hook] afterExecute: ${ctx.toolName} → ${ctx.result._tag}`)
    }),
  onEvent: (event) =>
    Effect.sync(() => {
      if (event._tag === "ThoughtDelta") {
        process.stderr.write(event.text)
      }
    }),
}

// ── Bind model ───────────────────────────────────────────────────────

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) throw new Error("Set OPENAI_API_KEY before running this example")

const openaiGpt4o = NativeChatCompletions.model({
  id: "openai/gpt-4o",
  modelId: "gpt-4o",
  endpoint: "https://api.openai.com/v1",
  contextWindow: 128_000,
  maxOutputTokens: 16_384,
  options: NativeChatCompletions.options,
})

const model = openaiGpt4o.bind({ auth: Auth.bearer(apiKey) })

// ── Create harness and run ───────────────────────────────────────────

const harness = createHarness({ model, toolkit, hooks })

const prompt = PromptBuilder.empty()
  .system("You have shell and file reading tools.")
  .user("List the files in the current directory.")
  .build()

const program = Effect.gen(function* () {
  const turn = yield* harness.runTurn(prompt)

  yield* Stream.runForEach(turn.events, (event) =>
    Effect.sync(() => {
      switch (event._tag) {
        case "MessageDelta":
          process.stdout.write(event.text)
          break
        case "TurnEnd":
          console.log(`\n[done] ${event.outcome._tag}`)
          break
      }
    }),
  )

  const canonical = yield* Ref.get(turn.canonicalTurn)
  console.log(`\nTool results: ${canonical.toolResults.length}`)
  console.log(`Outcome: ${canonical.outcome?._tag}`)
})

program.pipe(Effect.provide(FetchHttpClient.layer), Effect.runPromise)
