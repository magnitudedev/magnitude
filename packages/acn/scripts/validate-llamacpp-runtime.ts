import { Schema } from "effect"

const RawModelSchema = Schema.Struct({
  id: Schema.optional(Schema.Unknown),
  meta: Schema.optional(Schema.Struct({
    n_ctx: Schema.optional(Schema.Unknown),
    n_ctx_train: Schema.optional(Schema.Unknown),
    n_params: Schema.optional(Schema.Unknown),
    size: Schema.optional(Schema.Unknown),
    ftype: Schema.optional(Schema.Unknown),
  })),
})

const ModelListSchema = Schema.Struct({
  data: Schema.optional(Schema.Array(RawModelSchema)),
})

const ServerPropsSchema = Schema.Struct({
  model_alias: Schema.optional(Schema.Unknown),
  model_ftype: Schema.optional(Schema.Unknown),
  model_path: Schema.optional(Schema.Unknown),
  build_info: Schema.optional(Schema.Unknown),
  default_generation_settings: Schema.optional(Schema.Struct({
    n_ctx: Schema.optional(Schema.Unknown),
  })),
  chat_template_caps: Schema.optional(Schema.Struct({
    supports_tool_calls: Schema.optional(Schema.Unknown),
    supports_tools: Schema.optional(Schema.Unknown),
  })),
})

const ChatCompletionSchema = Schema.Struct({
  choices: Schema.optional(Schema.Array(Schema.Struct({
    finish_reason: Schema.optional(Schema.Unknown),
    message: Schema.optional(Schema.Struct({
      content: Schema.optional(Schema.Unknown),
      reasoning_content: Schema.optional(Schema.Unknown),
      tool_calls: Schema.optional(Schema.Array(Schema.Struct({
        function: Schema.optional(Schema.Struct({
          name: Schema.optional(Schema.Unknown),
          arguments: Schema.optional(Schema.Unknown),
        })),
      }))),
    })),
  }))),
  timings: Schema.optional(Schema.Struct({
    prompt_per_second: Schema.optional(Schema.Unknown),
    predicted_per_second: Schema.optional(Schema.Unknown),
  })),
})

const endpointArgument = process.argv.find((argument) => argument.startsWith("--endpoint="))
const endpoint = (endpointArgument?.slice("--endpoint=".length)
  ?? process.env.LLAMACPP_ENDPOINT
  ?? "http://127.0.0.1:8080").replace(/\/$/, "")
const apiKey = process.env.LLAMACPP_API_KEY?.trim()

const headers = (json = false): Headers => {
  const result = new Headers()
  if (json) result.set("content-type", "application/json")
  if (apiKey) result.set("authorization", `Bearer ${apiKey}`)
  return result
}

const requestJson = async <A, I>(
  path: string,
  schema: Schema.Schema<A, I>,
  init?: RequestInit,
): Promise<{ readonly body: A; readonly elapsedMs: number }> => {
  const started = performance.now()
  const response = await fetch(`${endpoint}${path}`, {
    ...init,
    signal: AbortSignal.timeout(120_000),
  })
  const text = await response.text()
  if (!response.ok) {
    throw new Error(`${path} returned HTTP ${response.status}: ${text.slice(0, 500)}`)
  }
  try {
    return {
      body: Schema.decodeUnknownSync(schema)(JSON.parse(text)),
      elapsedMs: Math.round(performance.now() - started),
    }
  } catch {
    throw new Error(`${path} did not return JSON: ${text.slice(0, 500)}`)
  }
}

const requireString = (value: unknown, description: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`llama.cpp did not report ${description}`)
  }
  return value
}

const chat = async (model: string, body: Record<string, unknown>) =>
  requestJson("/v1/chat/completions", ChatCompletionSchema, {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify({
      model,
      temperature: 0,
      chat_template_kwargs: { enable_thinking: false },
      ...body,
    }),
  })

const health = await requestJson("/health", Schema.Struct({
  status: Schema.optional(Schema.Unknown),
}), { headers: headers() })
if (health.body.status !== "ok") throw new Error(`llama.cpp health is ${String(health.body.status)}`)

const [{ body: models }, { body: props }] = await Promise.all([
  requestJson("/v1/models", ModelListSchema, { headers: headers() }),
  requestJson("/props", ServerPropsSchema, { headers: headers() }),
])
if (!Array.isArray(models.data) || models.data.length === 0) {
  throw new Error("llama.cpp returned no loaded models from /v1/models")
}

const results: Record<string, unknown>[] = []
for (const rawModel of models.data) {
  const model = requireString(rawModel.id, "a model ID")

  const simple = await chat(model, {
    messages: [{ role: "user", content: "Reply with exactly: runtime-ok" }],
    max_tokens: 128,
  })
  const simpleChoice = simple.body.choices?.[0]
  const content = simpleChoice?.message?.content
  if (typeof content !== "string" || !content.toLowerCase().includes("runtime-ok")) {
    throw new Error(`${model}: chat returned no expected visible answer (finish_reason=${String(simpleChoice?.finish_reason)})`)
  }

  const tool = await chat(model, {
    messages: [{ role: "user", content: "Use add_numbers to add 2 and 3." }],
    tools: [{
      type: "function",
      function: {
        name: "add_numbers",
        description: "Add two numbers",
        parameters: {
          type: "object",
          properties: { a: { type: "number" }, b: { type: "number" } },
          required: ["a", "b"],
          additionalProperties: false,
        },
      },
    }],
    tool_choice: "required",
    max_tokens: 256,
  })
  const toolChoice = tool.body.choices?.[0]
  const toolCall = toolChoice?.message?.tool_calls?.[0]?.function
  if (toolChoice?.finish_reason !== "tool_calls" || toolCall?.name !== "add_numbers") {
    throw new Error(`${model}: required tool call was not returned (finish_reason=${String(toolChoice?.finish_reason)})`)
  }
  try {
    Schema.decodeUnknownSync(Schema.Struct({
      a: Schema.Literal(2),
      b: Schema.Literal(3),
    }))(JSON.parse(requireString(toolCall.arguments, "tool arguments")))
  } catch (cause) {
    throw new Error(`${model}: tool arguments were not {"a":2,"b":3}`, { cause })
  }

  results.push({
    model,
    quant: rawModel.meta?.ftype ?? props.model_ftype,
    modelBytes: rawModel.meta?.size,
    parameters: rawModel.meta?.n_params,
    configuredContextTokens: props.default_generation_settings?.n_ctx ?? rawModel.meta?.n_ctx,
    trainedContextTokens: rawModel.meta?.n_ctx_train,
    build: props.build_info,
    reportsToolTemplateSupport: props.chat_template_caps?.supports_tools === true
      && props.chat_template_caps?.supports_tool_calls === true,
    healthLatencyMs: health.elapsedMs,
    chatLatencyMs: simple.elapsedMs,
    chatTokensPerSecond: simple.body.timings?.predicted_per_second,
    toolLatencyMs: tool.elapsedMs,
    toolTokensPerSecond: tool.body.timings?.predicted_per_second,
    chat: "passed",
    toolCalling: "passed",
  })
}

// This manual validator observes an already active endpoint and is deliberately
// read-only. Managed process lifecycle and activation identity are covered by
// the LlamaCppRuntime integration tests, which own their scoped test processes.
console.log(JSON.stringify({ endpoint, models: results }, null, 2))
