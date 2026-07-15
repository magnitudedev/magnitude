interface RawModel {
  readonly id?: unknown
  readonly meta?: {
    readonly n_ctx?: unknown
    readonly n_ctx_train?: unknown
    readonly n_params?: unknown
    readonly size?: unknown
    readonly ftype?: unknown
  }
}

interface ModelList {
  readonly data?: readonly RawModel[]
}

interface ServerProps {
  readonly model_alias?: unknown
  readonly model_ftype?: unknown
  readonly model_path?: unknown
  readonly build_info?: unknown
  readonly default_generation_settings?: {
    readonly n_ctx?: unknown
  }
  readonly chat_template_caps?: {
    readonly supports_tool_calls?: unknown
    readonly supports_tools?: unknown
  }
}

interface ChatCompletion {
  readonly choices?: readonly {
    readonly finish_reason?: unknown
    readonly message?: {
      readonly content?: unknown
      readonly reasoning_content?: unknown
      readonly tool_calls?: readonly {
        readonly function?: {
          readonly name?: unknown
          readonly arguments?: unknown
        }
      }[]
    }
  }[]
  readonly timings?: {
    readonly prompt_per_second?: unknown
    readonly predicted_per_second?: unknown
  }
}

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

const requestJson = async <T>(
  path: string,
  init?: RequestInit,
): Promise<{ readonly body: T; readonly elapsedMs: number }> => {
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
      body: JSON.parse(text) as T,
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
  requestJson<ChatCompletion>("/v1/chat/completions", {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify({
      model,
      temperature: 0,
      chat_template_kwargs: { enable_thinking: false },
      ...body,
    }),
  })

const health = await requestJson<{ readonly status?: unknown }>("/health", { headers: headers() })
if (health.body.status !== "ok") throw new Error(`llama.cpp health is ${String(health.body.status)}`)

const [{ body: models }, { body: props }] = await Promise.all([
  requestJson<ModelList>("/v1/models", { headers: headers() }),
  requestJson<ServerProps>("/props", { headers: headers() }),
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
  let toolArguments: unknown
  try {
    toolArguments = JSON.parse(requireString(toolCall.arguments, "tool arguments"))
  } catch (cause) {
    throw new Error(`${model}: tool arguments were not valid JSON`, { cause })
  }
  if (
    typeof toolArguments !== "object"
    || toolArguments === null
    || (toolArguments as Record<string, unknown>).a !== 2
    || (toolArguments as Record<string, unknown>).b !== 3
  ) {
    throw new Error(`${model}: tool arguments were not {"a":2,"b":3}`)
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

// TODO(llamacpp-lifecycle-integration): The CTO-owned lifecycle layer should
// load each selected catalog artifact on its target backend and invoke this
// endpoint validator. This script intentionally does not start, stop, or
// replace servers, so validation cannot race the daemon's server manager.
console.log(JSON.stringify({ endpoint, models: results }, null, 2))
