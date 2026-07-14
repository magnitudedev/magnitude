import { FetchHttpClient } from "@effect/platform"
import { Chunk, Effect, Option, Schema, Stream } from "effect"
import {
  Prompt,
  mergeReasoningDetails,
  type BaseCallOptions,
  type JsonValue,
  type Provider,
  type ProviderModel,
  type ResponseStreamEvent,
  type ToolCallId,
  type ProviderToolCallId,
  type ToolDefinition,
} from "@magnitudedev/ai"
import {
  createDeepSeekProvider,
  createKimiApiProvider,
  createKimiForCodingProvider,
  createLlamaCppProvider,
  createOpenRouterProvider,
  createVercelProvider,
  createZaiCodingPlanProvider,
  createZaiProvider,
} from "../src"

interface LiveProvider {
  readonly provider: Provider
  readonly keyName: string | null
}

function requiredKey(name: string): string | null {
  return process.env[name]?.trim() || null
}

const providers = new Map<string, LiveProvider>()

function register(
  id: string,
  keyName: string | null,
  create: (key: string | undefined) => { readonly provider: Provider },
): void {
  const key = keyName ? requiredKey(keyName) : undefined
  if (keyName && !key) return
  providers.set(id, { provider: create(key).provider, keyName })
}

register("llamacpp", null, () => createLlamaCppProvider())
register("openrouter", "OPENROUTER_API_KEY", (apiKey) => createOpenRouterProvider({ apiKey }))
register("vercel", "AI_GATEWAY_API_KEY", (apiKey) => createVercelProvider({ apiKey }))
register("deepseek", "DEEPSEEK_API_KEY", (apiKey) => createDeepSeekProvider({ apiKey }))
register("zai", "ZAI_API_KEY", (apiKey) => createZaiProvider({ apiKey }))
register("zai-coding-plan", "ZAI_API_KEY", (apiKey) => createZaiCodingPlanProvider({ apiKey }))
register("kimi-api", "MOONSHOT_API_KEY", (apiKey) => createKimiApiProvider({ apiKey }))
register("kimi-for-coding", "KIMI_API_KEY", (apiKey) => createKimiForCodingProvider({ apiKey }))

function runHttp<A, E>(effect: Effect.Effect<A, E, never | import("@effect/platform/HttpClient").HttpClient>) {
  return Effect.runPromise(effect.pipe(Effect.provide(FetchHttpClient.layer)))
}

function errorText(cause: unknown): string {
  const credentialNames = [
    "OPENROUTER_API_KEY",
    "AI_GATEWAY_API_KEY",
    "DEEPSEEK_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
  ]
  let message = String(cause).replace(/\s+/g, " ")
  for (const name of credentialNames) {
    const value = requiredKey(name)
    if (value) message = message.replaceAll(value, "[REDACTED]")
  }
  return message.slice(0, 1_000)
}

function maxTokens(fallback: number): number {
  const configured = Number.parseInt(process.env.LIVE_MAX_TOKENS ?? "", 10)
  return Number.isFinite(configured) && configured > 0 ? configured : fallback
}

async function catalog(): Promise<void> {
  const required = [
    "OPENROUTER_API_KEY",
    "AI_GATEWAY_API_KEY",
    "DEEPSEEK_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
  ]
  console.log(JSON.stringify({
    configured: Object.fromEntries(required.map((name) => [name, Boolean(requiredKey(name))])),
  }))

  for (const [id, entry] of providers) {
    try {
      const models = await runHttp(entry.provider.catalog.refresh)
      console.log(JSON.stringify({
        provider: id,
        status: "ok",
        count: models.length,
        models: models.map((model) => ({
          id: model.providerModelId,
          displayName: model.displayName,
          family: model.modelFamilyId,
          reasoning: model.reasoningEfforts,
          openWeights: model.openWeightStatus ?? null,
          tools: model.capabilities.toolCalls,
        })),
      }))
    } catch (cause) {
      console.log(JSON.stringify({ provider: id, status: "error", error: errorText(cause) }))
    }
  }
}

const echoTool: ToolDefinition = {
  name: "echo_value",
  description: "Return a supplied short value.",
  inputSchema: Schema.Struct({ value: Schema.String }),
  outputSchema: Schema.String,
}

function userMessage(text: string) {
  return {
    _tag: "UserMessage" as const,
    parts: [{ _tag: "TextPart" as const, text }],
  }
}

interface CallResult {
  readonly text: string
  readonly reasoning: string
  readonly reasoningDetails: readonly JsonValue[]
  readonly toolCall: {
    readonly id: ToolCallId
    readonly providerId: ProviderToolCallId
    readonly name: string
    readonly input: unknown
  } | null
  readonly terminal: string
  readonly finishReason: string | null
  readonly usage: unknown
}

function summarizeEvents(events: readonly ResponseStreamEvent[]): CallResult {
  let text = ""
  let reasoning = ""
  let reasoningDetails: readonly JsonValue[] = []
  let toolCall: CallResult["toolCall"] = null
  let terminal = "missing"
  let finishReason: string | null = null
  let usage: unknown = null

  for (const event of events) {
    if (event._tag === "message_delta") text += event.text
    if (event._tag === "thought_delta") reasoning += event.text
    if (event._tag === "reasoning_details") {
      reasoningDetails = mergeReasoningDetails(reasoningDetails, event.details)
    }
    if (event._tag === "tool_call_start") {
      toolCall = {
        id: event.toolCallId,
        providerId: event.providerToolCallId,
        name: event.toolName,
        input: {},
      }
    }
    if (event._tag === "tool_call_field_end" && event.path.length === 0 && toolCall) {
      toolCall = { ...toolCall, input: event.value }
    }
    if (event._tag === "stream_end") {
      terminal = event.terminal._tag
      if (event.terminal._tag === "StreamCompleted") {
        finishReason = event.terminal.finishReason
        usage = event.terminal.usage
      } else {
        usage = event.terminal.cause._tag
      }
    }
  }
  return { text, reasoning, reasoningDetails, toolCall, terminal, finishReason, usage }
}

async function collectCall(
  provider: Provider,
  modelId: string,
  prompt: Prompt,
  tools: readonly ToolDefinition[],
  options: BaseCallOptions,
): Promise<CallResult> {
  const bound = await Effect.runPromise(provider.bindModel(modelId))
  const events = await runHttp(Effect.gen(function* () {
    const result = yield* bound.stream(prompt, tools, options)
    return Chunk.toReadonlyArray(yield* Stream.runCollect(result.events))
  }))
  return summarizeEvents(events)
}

async function call(
  providerId: string,
  modelId: string,
  effort: string,
  scenario: string,
): Promise<void> {
  const entry = providers.get(providerId)
  if (!entry) throw new Error(`Provider is not configured: ${providerId}`)

  if (scenario === "text") {
    const result = await collectCall(
      entry.provider,
      modelId,
      Prompt.from({
        system: "Follow the user's output-format instruction exactly.",
        messages: [userMessage("Reply with exactly: LIVE_OK")],
      }),
      [],
      { maxTokens: maxTokens(96), reasoningEffort: effort },
    )
    console.log(JSON.stringify({
      provider: providerId,
      model: modelId,
      effort,
      scenario,
      ...result,
      text: result.text.slice(0, 300),
      reasoningChars: result.reasoning.length,
      reasoningDetailCount: result.reasoningDetails.length,
      reasoning: undefined,
      reasoningDetails: undefined,
    }))
    return
  }

  if (scenario !== "tool") throw new Error(`Unknown scenario: ${scenario}`)
  const firstUser = userMessage("Call echo_value once with value LIVE_TOOL. Do not answer directly.")
  const first = await collectCall(
    entry.provider,
    modelId,
    Prompt.from({ system: "Use the supplied tool when requested.", messages: [firstUser] }),
    [echoTool],
    { maxTokens: maxTokens(192), reasoningEffort: effort, toolChoice: "auto" },
  )
  if (!first.toolCall) {
    console.log(JSON.stringify({
      provider: providerId,
      model: modelId,
      effort,
      scenario,
      status: "no_tool_call",
      first: {
        terminal: first.terminal,
        finishReason: first.finishReason,
        text: first.text.slice(0, 300),
        reasoningChars: first.reasoning.length,
        reasoningDetailCount: first.reasoningDetails.length,
        usage: first.usage,
      },
    }))
    return
  }

  const second = await collectCall(
    entry.provider,
    modelId,
    Prompt.from({
      system: "Use the supplied tool when requested.",
      messages: [
        firstUser,
        {
          _tag: "AssistantMessage",
          reasoning: first.reasoning ? Option.some(first.reasoning) : Option.none(),
          reasoningDetails: first.reasoningDetails,
          text: first.text ? Option.some(first.text) : Option.none(),
          toolCalls: Option.some([{
            _tag: "ToolCallPart",
            id: first.toolCall.id,
            providerToolCallId: first.toolCall.providerId,
            name: first.toolCall.name,
            input: first.toolCall.input as never,
          }]),
        },
        {
          _tag: "ToolResultMessage",
          toolCallId: first.toolCall.id,
          providerToolCallId: first.toolCall.providerId,
          toolName: first.toolCall.name,
          parts: [{ _tag: "TextPart", text: "LIVE_TOOL" }],
        },
      ],
    }),
    [echoTool],
    { maxTokens: maxTokens(192), reasoningEffort: effort, toolChoice: "none" },
  )
  console.log(JSON.stringify({
    provider: providerId,
    model: modelId,
    effort,
    scenario,
    status: second.terminal === "StreamCompleted" ? "ok" : "failed",
    first: {
      terminal: first.terminal,
      finishReason: first.finishReason,
      toolCall: first.toolCall,
      reasoningChars: first.reasoning.length,
      reasoningDetailCount: first.reasoningDetails.length,
    },
    second: {
      terminal: second.terminal,
      finishReason: second.finishReason,
      text: second.text.slice(0, 300),
      reasoningChars: second.reasoning.length,
      usage: second.usage,
    },
  }))
}

const [mode = "catalog", providerId, modelId, effort = "default", scenario = "text"] = process.argv.slice(2)

if (mode === "catalog") {
  await catalog()
} else if (mode === "call" && providerId && modelId) {
  await call(providerId, modelId, effort, scenario)
} else {
  throw new Error("Usage: live-provider-check.ts catalog | call <provider> <model> <effort> <text|tool>")
}
