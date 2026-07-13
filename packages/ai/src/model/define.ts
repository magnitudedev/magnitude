import { Clock, Effect, Option, Stream } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Prompt } from "../prompt/prompt"
import type { ToolDefinition } from "../tools/tool-definition"
import type { AuthApplicator } from "../auth/auth"
import type { Codec } from "../codec/codec"
import type { ChatCompletionsRequest } from "../wire/chat-completions"
import type {
  ProviderCall,
  RejectedHttpResponse,
  StreamStartFailure,
  StreamFailure,
  StreamFailureContext,
} from "../errors/failure"
import { causeInfoText, StreamStartClientCorrectnessViolation, toCauseInfo } from "../errors/failure"
import type { StreamStartProviderCorrectnessViolation, StreamStartProviderRejection } from "../errors/failure"
import { snapshotModelAttemptFailure } from "../errors/classify"
import { executeHttpStream, type HttpStreamResult } from "../transport/stream"
import type { ProviderModelCapabilities, ImagePlaceholderConfig } from "./capabilities"
import type { ModelSpec, ModelStreamResult } from "./model-spec"
import type { BoundModel } from "./bound-model"
import { normalizeVision } from "../prompt/normalize-vision"
import { TraceListener, type AssembledToolCall, type ModelCallTrace } from "../trace"
import type { FinishReason } from "../response/events"
import type { ResponseUsage } from "../response/usage"
import type { ProviderToolCallId, ToolCallId } from "../prompt/ids"
import type { RawInputToken, RawOutputToken } from "../response/events"

// ---------------------------------------------------------------------------
// Model.define — internal factory used by protocol namespaces
// ---------------------------------------------------------------------------

export interface ModelDefineConfig<
  TCallOptions,
  TWireReq,
  TWireChunk,
> {
  readonly modelId: string
  readonly endpoint: string
  readonly path: string
  readonly codec: Codec<TWireReq, TWireChunk>
  readonly buildWireRequest: (
    prompt: Prompt,
    tools: readonly ToolDefinition[],
    options: TCallOptions,
  ) => TWireReq
  readonly decodePayload: (raw: string) => Effect.Effect<TWireChunk, unknown>
  readonly classifyRejectedResponse?: (
    call: ProviderCall,
    response: RejectedHttpResponse,
  ) => StreamStartProviderRejection | StreamStartProviderCorrectnessViolation
  readonly doneSignal?: string
  readonly capabilities?: ProviderModelCapabilities
}

function joinUrl(endpoint: string, path: string): string {
  return endpoint.replace(/\/+$/, "") + path
}

function makeDecodeOptions(
  httpResult: HttpStreamResult<unknown>,
  tools: readonly ToolDefinition[],
  generateToolCallId: (() => ToolCallId) | undefined,
) {
  return {
    tools,
    streamContext: {
      responseHeaders: httpResult.responseHeaders,
      call: httpResult.call,
      response: httpResult.response,
    } as StreamFailureContext,
    generateToolCallId,
    toStreamFailure: (err: StreamFailure) => err,
  }
}

export function modelDefine<
  TCallOptions,
  TWireReq extends ChatCompletionsRequest,
  TWireChunk,
>(
  config: ModelDefineConfig<TCallOptions, TWireReq, TWireChunk>,
): ModelSpec<TCallOptions> {
  const url = joinUrl(config.endpoint, config.path)
  const call: ProviderCall = {
    provider: config.endpoint,
    model: config.modelId,
    method: "POST",
    url,
  }

  const spec: ModelSpec<TCallOptions> = {
    modelId: config.modelId,
    endpoint: config.endpoint,
    capabilities: config.capabilities,

    bind: (args) => modelBind(spec, args.auth, args.defaults, { imagePlaceholders: args.imagePlaceholders }),

    _execute: (
      auth: AuthApplicator,
      prompt: Prompt,
      tools: readonly ToolDefinition[],
      options: TCallOptions,
    ): Effect.Effect<
      ModelStreamResult,
      StreamStartFailure,
      HttpClient.HttpClient
    > => {
      return Effect.gen(function* () {
        const listenerOption = yield* Effect.serviceOption(TraceListener)
        const runtimeOptions = options as TCallOptions & { readonly generateToolCallId?: () => ToolCallId }
        const wireRequest = yield* Effect.try({
          try: () => config.buildWireRequest(prompt, tools, options),
          catch: (cause) => {
            const causeInfo = toCauseInfo(cause)
            return new StreamStartClientCorrectnessViolation({
              call,
              component: "request_builder",
              message: `Could not build model request: ${causeInfoText(causeInfo)}`,
              evidence: { _tag: "UnexpectedDefectCaught", cause: causeInfo },
            })
          },
        })

        const httpEffect = executeHttpStream({
          call,
          body: wireRequest,
          auth,
          decodePayload: config.decodePayload,
          doneSignal: config.doneSignal,
          classifyRejectedResponse: config.classifyRejectedResponse,
        })

        if (Option.isNone(listenerOption)) {
          // No listener — zero overhead path
          return yield* httpEffect.pipe(
            Effect.map((httpResult) => ({
              ...config.codec.decode(httpResult.stream, makeDecodeOptions(httpResult, tools, runtimeOptions.generateToolCallId)),
              requestId: httpResult.response.requestId,
            })),
          )
        }

        const listener = listenerOption.value
        const startedAt = yield* Clock.currentTimeMillis
        const startTime = performance.now()

        // Mutable accumulators for trace assembly
        let reasoning = ""
        let text = ""
        const toolCallMap = new Map<ToolCallId, { id: ToolCallId; providerToolCallId: ProviderToolCallId; name: string; args: Record<string, unknown> }>()
        let finishReason: FinishReason | null = null
        let usage: ResponseUsage | null = null
        let rawInput: ReadonlyArray<RawInputToken> | null = null
        let rawOutput: ReadonlyArray<RawOutputToken> | null = null

        const result = yield* httpEffect.pipe(
          Effect.map((httpResult) => ({
            ...config.codec.decode(httpResult.stream, makeDecodeOptions(httpResult, tools, runtimeOptions.generateToolCallId)),
            requestId: httpResult.response.requestId,
          })),
          Effect.mapError((failure) => {
            // Emit trace for stream-start failures.
            const trace: ModelCallTrace = {
              modelId: config.modelId,
              url,
              startedAt,
              durationMs: performance.now() - startTime,
              request: wireRequest,
              response: {
                reasoning: null,
                text: null,
                toolCalls: [],
                finishReason: null,
                usage: null,
                logprobs: null,
              },
              ...(rawInput ? { rawInput } : {}),
              ...(rawOutput ? { rawOutput } : {}),
              modelAttemptFailure: snapshotModelAttemptFailure(failure),
            }
            listener.onTrace(trace)
            return failure
          }),
        )

        // Wrap the event stream to accumulate trace data
        const tracedEvents = result.events.pipe(
          Stream.tap((event) =>
            Effect.sync(() => {
              switch (event._tag) {
                case "thought_delta":
                  reasoning += event.text
                  break
                case "message_delta":
                  text += event.text
                  break
                case "tool_call_start":
                  toolCallMap.set(event.toolCallId, { id: event.toolCallId, providerToolCallId: event.providerToolCallId, name: event.toolName, args: {} })
                  break
                case "tool_call_field_end": {
                  const tc = toolCallMap.get(event.toolCallId)
                  if (tc) {
                    if (event.path.length === 0) {
                      tc.args = event.value as Record<string, unknown>
                    } else {
                      let target: any = tc.args
                      for (let i = 0; i < event.path.length - 1; i++) {
                        if (!(event.path[i] in target)) {
                          target[event.path[i]] = {}
                        }
                        target = target[event.path[i]]
                      }
                      target[event.path[event.path.length - 1]] = event.value
                    }
                  }
                  break
                }
                case "stream_end": {
                  const terminal = event.terminal
                  if (terminal._tag === "StreamCompleted") {
                    finishReason = terminal.finishReason
                    if (terminal.usage._tag === "UsageReported") {
                      usage = terminal.usage.usage
                    }
                  }
                  rawInput = event.rawInput ? [...event.rawInput] : null
                  rawOutput = event.rawOutput ? [...event.rawOutput] : null
                  break
                }
              }
            }),
          ),
          Stream.ensuring(
            Effect.sync(() => {
              const assembledToolCalls: AssembledToolCall[] = Array.from(toolCallMap.values()).map(
                (tc) => ({ id: tc.id, providerToolCallId: tc.providerToolCallId, name: tc.name, arguments: tc.args }),
              )
              const trace: ModelCallTrace = {
                modelId: config.modelId,
                url,
                startedAt,
                durationMs: performance.now() - startTime,
                request: wireRequest,
                response: {
                  reasoning: reasoning.length > 0 ? reasoning : null,
                  text: text.length > 0 ? text : null,
                  toolCalls: assembledToolCalls,
                  finishReason,
                  usage,
                  logprobs: result.logprobs.length > 0 ? result.logprobs : null,
                },
                ...(rawInput ? { rawInput } : {}),
                ...(rawOutput ? { rawOutput } : {}),
              }
              listener.onTrace(trace)
            }),
          ),
        )

        return {
          events: tracedEvents,
          parsers: result.parsers,
          logprobs: result.logprobs,
          requestId: result.requestId,
        }
      })
    },
  }

  return spec
}

// ---------------------------------------------------------------------------
// Model.bind — public binding API
// ---------------------------------------------------------------------------

export function modelBind<
  TCallOptions,
>(
  spec: ModelSpec<TCallOptions>,
  auth: AuthApplicator,
  defaults?: Partial<TCallOptions>,
  options?: { imagePlaceholders?: ImagePlaceholderConfig },
): BoundModel<TCallOptions> {
  return {
    stream: (prompt, tools, callOptions?) => {
      const merged = { ...defaults, ...callOptions } as TCallOptions
      const normalizedPrompt = (options?.imagePlaceholders?.enabled && spec.capabilities?.vision === false)
        ? normalizeVision(prompt, options.imagePlaceholders.format)
        : prompt
      return spec._execute(auth, normalizedPrompt, tools, merged)
    },
  }
}

// ---------------------------------------------------------------------------
// Model namespace — public API
// ---------------------------------------------------------------------------

export const Model = {
  /** @internal — used by protocol namespaces */
  define: modelDefine,
  /** Bind a ModelSpec with auth and optional defaults to create a BoundModel */
  bind: modelBind,
} as const
