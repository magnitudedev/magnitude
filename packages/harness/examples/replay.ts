import { Effect, Ref, Schema } from "effect"
import {
  defineTool,
  Auth,
  NativeChatCompletions,
  type ToolCallId,
} from "@magnitudedev/ai"
import {
  defineHarnessTool,
  defineToolkit,
  createHarness,
  type HarnessEvent,
} from "../src/index.js"

// ── Replay example ───────────────────────────────────────────────────
//
// Shows how to replay a sequence of events through a ReplayTurn
// to reconstruct canonical state without running the model.

const echoDef = defineTool({
  name: "echo",
  description: "Echo input back.",
  inputSchema: Schema.Struct({ text: Schema.String }),
  outputSchema: Schema.String,
})

const echoTool = defineHarnessTool({
  definition: echoDef,
  execute: (input) => Effect.succeed(input.text),
})

const toolkit = defineToolkit({
  echo: { tool: echoTool },
})

// Dummy model — not called during replay
const dummyModel = NativeChatCompletions.model({
  id: "dummy/model",
  modelId: "dummy",
  endpoint: "http://localhost",
  contextWindow: 1000,
  maxOutputTokens: 1000,
  options: NativeChatCompletions.options,
}).bind({ auth: Auth.bearer("dummy") })

const harness = createHarness({ model: dummyModel, toolkit })

// Simulate a recorded event sequence
const recordedEvents: HarnessEvent[] = [
  { _tag: "MessageStart" },
  { _tag: "MessageDelta", text: "I'll echo " },
  { _tag: "MessageDelta", text: "that for you." },
  { _tag: "MessageEnd" },
  {
    _tag: "ToolInputStarted",
    toolCallId: "tc_1" as ToolCallId,
    toolName: "echo",
    toolKey: "echo",
    group: "",
  },
  {
    _tag: "ToolInputReady",
    toolCallId: "tc_1" as ToolCallId,
    input: { text: "hello" },
  },
  {
    _tag: "ToolExecutionStarted",
    toolCallId: "tc_1" as ToolCallId,
    toolName: "echo",
    toolKey: "echo",
    group: "",
    input: { text: "hello" },
    cached: false,
  },
  {
    _tag: "ToolExecutionEnded",
    toolCallId: "tc_1" as ToolCallId,
    toolName: "echo",
    toolKey: "echo",
    group: "",
    result: { _tag: "Success", output: "hello" },
  },
  {
    _tag: "ToolResultFormatted",
    toolCallId: "tc_1" as ToolCallId,
    toolName: "echo",
    toolKey: "echo",
    parts: [{ _tag: "TextPart", text: "hello" }],
  },
  {
    _tag: "TurnEnd",
    outcome: { _tag: "Completed", toolCallsCount: 1 },
    usage: null,
  },
]

const program = Effect.gen(function* () {
  const replay = yield* harness.createReplayTurn()

  for (const event of recordedEvents) {
    yield* replay.feed(event)
  }

  const canonical = yield* Ref.get(replay.canonicalTurn)
  console.log("Reconstructed turn:")
  console.log(`  Message: ${canonical.assistantMessage.text}`)
  console.log(`  Tool calls: ${canonical.assistantMessage.toolCalls?.length ?? 0}`)
  console.log(`  Tool results: ${canonical.toolResults.length}`)
  console.log(`  Outcome: ${canonical.outcome?._tag}`)
})

Effect.runPromise(program)
