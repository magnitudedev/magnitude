/**
 * Model Proxy — unified abstraction for all LLM calls.
 *
 * Clean API with usage tracking and cost calculation:
 *
 *   import { primary, secondary } from '@magnitudedev/providers'
 *
 *   const { stream, getUsage } = primary.chat(messages)
 *   const { result, usage } = await primary.compact(messages)
 *   const { result, usage } = await primary.autopilot(systemPrompt, messages)
 */

import { b, type ChatMessage, Collector } from '@magnitudedev/llm-core'
import { logger } from '@magnitudedev/logger'
import {
  accumulateUsage,
  type ResolvedModel, type ModelSlot, type CallUsage,
} from './provider-state'
import { buildUsage } from './usage'
import { getCodexReasoningEffort } from './reasoning-effort'
import { createProviderClient, type ProviderClient } from './provider-client'
import { buildClientRegistry } from './client-registry-builder'
import { toIncrementalStream } from './incremental-stream'
import { normalizeModelOutput, normalizeQuotesInString } from './output-normalization'
import {
  type CollectorData, type TraceData, type AgentTraceMeta,
  isTracing, emitTrace, extractCollectorData, wrapStreamForTrace,
  onTrace,
} from '@magnitudedev/tracing'

export type { CollectorData, TraceData }
export { onTrace }

export interface ChatStream {
  readonly stream: AsyncIterable<string>
  getUsage(): CallUsage
  getCollectorData(): CollectorData
}

export interface ChatOptions {
  stopSequences?: string[]
}

export interface ModelProxy {
  chat(systemPrompt: string, messages: ChatMessage[], meta: AgentTraceMeta | undefined, options: ChatOptions | undefined, ackTurn: string): ChatStream
  compact(systemPrompt: string, messages: ChatMessage[], meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }>
  autopilot(systemPrompt: string, driverMessages: ChatMessage[], meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }>
  generateChatTitle(conversation: string, defaultChatName: string, meta?: AgentTraceMeta): Promise<{ title: string } | null>
  gatherReport(query: string, context: string, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }>
  gatherSplit(query: string, fileTree: string, tokenBudget: number, meta?: AgentTraceMeta): Promise<{ result: { path: string; query: string }[]; usage: CallUsage }>
  extractMemoryDiff(
    transcript: string,
    currentMemory: string,
    meta?: AgentTraceMeta
  ): Promise<{
    result: {
      reasoning: string
      additions: Array<{ category: 'code_style' | 'codebase' | 'workflow' | 'tools'; content: string; evidence: string }>
      updates: Array<{ existing: string; replacement: string; evidence: string }>
      deletions: Array<{ existing: string; evidence: string }>
    }
    usage: CallUsage
  }>
  patchFile(instructions: string, fileContent: string, previousAttempts?: Array<{response: string, error: string}>, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }>
  createFile(instructions: string, filePath: string, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }>
  baml: typeof b
  readonly slot: ModelSlot
}

// buildUsage and calculateCosts imported from ./usage
// Auth, endpoints, and headers provided by ProviderClient from ./provider-client

// =============================================================================
// Token Extraction from Collector
// =============================================================================

/** Basic token count validation — reject clearly invalid values */
function validateTokenCount(tokens: number): number | null {
  if (tokens <= 0) return null
  return tokens
}

function extractUsageFromCollector(collector: Collector, resolved: ResolvedModel | null): CallUsage {
  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let cacheReadTokens: number | null = null
  let cacheWriteTokens: number | null = null

  const lastCall = collector.last?.calls.at(-1)

  if (lastCall) {
    // Strategy 1: Raw HTTP response body
    try {
      const rawUsage = lastCall.httpResponse?.body.json()?.usage
      if (rawUsage) {
        if (typeof rawUsage.input_tokens === 'number') {
          const total = rawUsage.input_tokens
            + (rawUsage.cache_creation_input_tokens ?? 0)
            + (rawUsage.cache_read_input_tokens ?? 0)
          inputTokens = validateTokenCount(total)
          cacheReadTokens = rawUsage.cache_read_input_tokens ?? null
          cacheWriteTokens = rawUsage.cache_creation_input_tokens ?? null
        }
        if (typeof rawUsage.output_tokens === 'number') {
          outputTokens = rawUsage.output_tokens
        }
      }
    } catch {}

    // Strategy 2: SSE responses (streaming)
    if (inputTokens === null) {
      try {
        const sseResponses = 'sseResponses' in lastCall ? (lastCall as any).sseResponses() : null
        if (Array.isArray(sseResponses)) {
          for (const sse of sseResponses) {
            const data = sse.json?.() ?? null
            if (data?.type === 'message_start' && data?.message?.usage) {
              const usage = data.message.usage
              if (typeof usage.input_tokens === 'number') {
                const total = usage.input_tokens
                  + (usage.cache_creation_input_tokens ?? 0)
                  + (usage.cache_read_input_tokens ?? 0)
                inputTokens = validateTokenCount(total)
                cacheReadTokens = usage.cache_read_input_tokens ?? null
                cacheWriteTokens = usage.cache_creation_input_tokens ?? null
              }
            }
            if (data?.type === 'message_delta' && data?.usage) {
              if (typeof data.usage.output_tokens === 'number') {
                outputTokens = data.usage.output_tokens
              }
            }
          }
        }
      } catch {}
    }
  }

  // Strategy 3: Collector usage fields (fallback)
  if (inputTokens === null) {
    const usage = collector.usage
    if (usage) {
      const total = (usage.inputTokens ?? 0) + (usage.cachedInputTokens ?? 0)
      inputTokens = validateTokenCount(total)
    }
  }
  if (outputTokens === null) {
    const usage = collector.usage
    if (usage && typeof usage.outputTokens === 'number') {
      outputTokens = usage.outputTokens
    }
  }

  return buildUsage(resolved, inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens)
}

// =============================================================================
// Responses API (Codex/Copilot)
// =============================================================================

function transformForResponsesApi(body: any, stream: boolean, options?: ChatOptions, maxOutputTokens?: number): Record<string, any> {
  const messages = body.input ?? body.messages ?? []
  const systemParts: string[] = []
  const nonSystemMessages: any[] = []

  for (const msg of messages) {
    if (msg.role === 'system') {
      const text = typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((p: any) => p.text ?? '').join('')
          : ''
      systemParts.push(text)
    } else {
      nonSystemMessages.push(msg)
    }
  }

  const instructions = systemParts.join('\n\n')
  if (!instructions) logger.warn('[ModelProxy] No system messages found in BAML request')

  const result: Record<string, any> = {
    model: body.model, instructions, input: nonSystemMessages, stream, store: false,
    ...(maxOutputTokens ? { max_output_tokens: maxOutputTokens } : {}),
    text: { verbosity: 'low' },
  }

  const reasoningEffort = getCodexReasoningEffort(body.model)
  if (reasoningEffort) {
    result.reasoning = { effort: reasoningEffort }
    logger.info(`[ModelProxy] Applied reasoning.effort=${reasoningEffort} for ${body.model}`)
  }

  // NOTE: Responses API does not support stop sequences — omitted intentionally.
  // Stop sequences are also excluded in the BAML registry path in client-registry-builder.ts#buildOpenAIResponsesOptions.

  return result
}

function extractTextFromResponsesApi(data: any): string {
  if (Array.isArray(data.output)) {
    const texts: string[] = []
    for (const item of data.output) {
      if (item.type === 'message' && Array.isArray(item.content)) {
        for (const content of item.content) {
          if (content.type === 'output_text') texts.push(content.text ?? '')
        }
      }
    }
    if (texts.length > 0) return texts.join('')
  }
  if (typeof data.text === 'string') return data.text
  if (typeof data.content === 'string') return data.content
  if (data.choices?.[0]?.message?.content) return data.choices[0].message.content
  throw new Error('[ModelProxy] Could not extract text from Responses API response')
}

async function callViaResponsesApi(
  fnName: string, args: any[], client: ProviderClient,
): Promise<{ result: any; usage: CallUsage }> {
  const { stream, getUsage } = streamViaResponsesApi(fnName, args, client)

  let outputText = ''
  for await (const chunk of stream) {
    outputText += chunk
  }

  const result = await (b.parse as any)[fnName].call(b.parse, outputText)
  const usage = getUsage()

  return { result, usage }
}

function streamViaResponsesApi(
  fnName: string, args: any[], client: ProviderClient, options?: ChatOptions, maxOutputTokens?: number,
): ChatStream {
  let inputTokens: number | null = null
  let outputTokens: number | null = null
  let capturedResponseBody: unknown | null = null

  async function* generate(): AsyncGenerator<string> {
    const resolved = client.resolve()
    const req = await (b.streamRequest as any)[fnName].call(b.streamRequest, ...args, { clientRegistry: resolved?.registry })
    const body = req.body.json()
    const codexBody = transformForResponsesApi(body, true, options, maxOutputTokens)

    const response = await fetch(client.getResponsesEndpoint(), {
      method: 'POST',
      headers: client.getHeaders(),
      body: JSON.stringify(codexBody),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Responses API streaming error ${response.status}: ${errorText}`)
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)
        if (data === '[DONE]') return

        try {
          const event = JSON.parse(data)
          if (event.type === 'response.output_text.delta') {
            yield normalizeQuotesInString(event.delta ?? '')
          } else if (event.type === 'response.completed') {
            capturedResponseBody = event.response ?? null
            const u = event.response?.usage
            if (u) {
              if (typeof u.input_tokens === 'number') inputTokens = u.input_tokens
              if (typeof u.output_tokens === 'number') outputTokens = u.output_tokens
            }
          }
        } catch {}
      }
    }
  }

  return {
    stream: generate(),
    getUsage(): CallUsage {
      const resolved = client.resolve()
      return buildUsage(resolved, inputTokens, outputTokens, null, null)
    },
    getCollectorData(): CollectorData {
      return { rawResponseBody: capturedResponseBody, sseEvents: null, rawRequestBody: null }
    },
  }
}

// =============================================================================
// BAML Streaming (standard providers)
// =============================================================================

function streamViaBaml(
  fnName: string, args: any[], resolved: ResolvedModel | null, slot: ModelSlot, options?: ChatOptions,
): ChatStream {
  const collector = new Collector('cortex-turn')
  const callRegistry = resolved && options?.stopSequences && options.stopSequences.length > 0
    ? buildClientRegistry(resolved.providerId, resolved.modelId, resolved.auth, options.stopSequences)
    : resolved?.registry ?? undefined
  const opts = { clientRegistry: callRegistry, collector }

  // Call method on b.stream with proper this binding
  const bamlStream = (b.stream as any)[fnName].call(b.stream, ...args, opts)

  return {
    stream: toNormalizedAsyncStream(toIncrementalStream(bamlStream)),
    getUsage(): CallUsage {
      const usage = extractUsageFromCollector(collector, resolved)
      accumulateUsage(slot, usage)
      return usage
    },
    getCollectorData(): CollectorData {
      return extractCollectorData(collector)
    },
  }
}

// =============================================================================
// BAML Non-streaming with Collector (standard providers)
// =============================================================================

async function callViaBaml(
  fnName: string, args: any[], resolved: ResolvedModel | null, slot: ModelSlot,
): Promise<{ result: any; usage: CallUsage; collectorData: CollectorData }> {
  const collector = new Collector(`${fnName}-call`)
  const opts = { clientRegistry: resolved?.registry ?? undefined, collector }

  // Call method on b with proper this binding
  const result = await (b as any)[fnName].call(b, ...args, opts)
  const usage = extractUsageFromCollector(collector, resolved)
  accumulateUsage(slot, usage)
  const collectorData = extractCollectorData(collector)
  const normalizedResult = normalizeModelOutput(result)
  return { result: normalizedResult, usage, collectorData }
}

// =============================================================================
// Option Injection (for raw baml proxy)
// =============================================================================

function injectOptions(args: any[], resolved: ResolvedModel | null): any[] {
  if (!resolved) return args
  const opts = { clientRegistry: resolved.registry }
  const lastArg = args[args.length - 1]
  if (lastArg && typeof lastArg === 'object' && !Array.isArray(lastArg) &&
      ('clientRegistry' in lastArg || 'tb' in lastArg || 'collector' in lastArg || 'signal' in lastArg)) {
    return [...args.slice(0, -1), { ...lastArg, ...opts }]
  }
  return [...args, opts]
}

// =============================================================================
// Raw BAML Proxy (for .baml accessor)
// =============================================================================

function toNormalizedAsyncStream(stream: AsyncIterable<string>): AsyncIterable<string> {
  return {
    async *[Symbol.asyncIterator]() {
      for await (const chunk of stream) {
        yield normalizeQuotesInString(chunk)
      }
    },
  }
}

function normalizeMaybeAsyncOutput<T>(value: T): T
function normalizeMaybeAsyncOutput<T>(value: Promise<T>): Promise<T>
function normalizeMaybeAsyncOutput<T>(value: T | Promise<T>): T | Promise<T> {
  if (value && typeof (value as any).then === 'function') {
    return (value as Promise<T>).then(v => normalizeModelOutput(v) as T)
  }
  return normalizeModelOutput(value as T)
}

function createBamlProxy(slot: ModelSlot, client: ProviderClient): typeof b {
  return new Proxy(b, {
    get(target, prop: string | symbol) {
      if (prop === 'stream') {
        return new Proxy(target.stream, {
          get(st, sp: string | symbol) {
            if (typeof sp === 'symbol') return (st as any)[sp]
            const fn = (st as any)[sp]
            if (typeof fn !== 'function') return fn
            return (...args: any[]) => {
              const resolved = client.resolve()
              if (resolved && (resolved.isCodex || resolved.isCopilotCodex)) {
                return streamViaResponsesApi(sp as string, args, client).stream
              }
              return toNormalizedAsyncStream(fn.call(st, ...injectOptions(args, resolved)))
            }
          }
        })
      }

      if (prop === 'request' || prop === 'streamRequest') {
        const sub = (target as any)[prop]
        return new Proxy(sub, {
          get(st: any, sp: string | symbol) {
            if (typeof sp === 'symbol') return st[sp]
            const fn = st[sp]
            if (typeof fn !== 'function') return fn
            return (...args: any[]) => {
              const resolved = client.resolve()
              return fn.call(st, ...injectOptions(args, resolved))
            }
          }
        })
      }

      if (prop === 'parse' || prop === 'parseStream') return (target as any)[prop]
      if (typeof prop === 'symbol') return (target as any)[prop]

      const fn = (target as any)[prop]
      if (typeof fn !== 'function') return fn

      return (...args: any[]) => {
        const resolved = client.resolve()
        if (resolved && (resolved.isCodex || resolved.isCopilotCodex)) {
          return callViaResponsesApi(prop as string, args, client).then(r => r.result)
        }
        return normalizeMaybeAsyncOutput(fn.call(target, ...injectOptions(args, resolved)))
      }
    }
  }) as typeof b
}

// =============================================================================
// Factory
// =============================================================================

export function createModelProxy(slot: ModelSlot): ModelProxy {
  const client = createProviderClient(slot)
  const baml = createBamlProxy(slot, client)

  /** Make a streaming call via responses API or BAML, with automatic tracing */
  function tracedStream(
    fnName: string, args: any[], callType: string,
    meta: AgentTraceMeta | undefined, request: TraceData['request'], options?: ChatOptions,
  ): ChatStream {
    const startTime = Date.now()
    // ensureAuth is called lazily inside the stream generator (see streamViaBaml/streamViaResponsesApi)
    // since tracedStream is synchronous. Callers that need eager refresh should call ensureAuth() first.
    const resolved = client.resolve()
    const includeClaudeSpoof = resolved?.isAnthropicOAuth ?? false
    const fullArgs = [...args, includeClaudeSpoof]

    let cs: ChatStream
    if (resolved && (resolved.isCodex || resolved.isCopilotCodex)) {
      // max_output_tokens omitted — Codex/Copilot Responses API endpoints reject it with 400
      const inner = streamViaResponsesApi(fnName, fullArgs, client, options)
      cs = {
        stream: inner.stream,
        getUsage(): CallUsage {
          const usage = inner.getUsage()
          accumulateUsage(slot, usage)
          return usage
        },
        getCollectorData() { return inner.getCollectorData() },
      }
    } else {
      cs = streamViaBaml(fnName, fullArgs, resolved, slot, options)
    }

    if (!isTracing() || callType === 'chat') return cs

    const traceCtx = { startTime, model: resolved?.modelId ?? null, provider: resolved?.providerId ?? null, slot, defaultCallType: callType, meta }
    const traced = wrapStreamForTrace(cs.stream, () => {
      const usage = cs.getUsage()
      const cd = cs.getCollectorData()
      const traceRequest = (cd?.rawRequestBody as any)?.messages
        ? { messages: (cd.rawRequestBody as any).messages }
        : request
      emitTrace(traceCtx, traceRequest, { rawBody: cd.rawResponseBody, sseEvents: cd.sseEvents, rawOutput: traced.getChunks().join('') }, usage)
    })
    return {
      stream: traced.stream,
      getUsage(): CallUsage { return cs.getUsage() },
      getCollectorData(): CollectorData { return cs.getCollectorData() },
    }
  }

  /** Make a non-streaming call with automatic tracing */
  async function tracedCall(
    fnName: string, args: any[], callType: string,
    meta: AgentTraceMeta | undefined, request: TraceData['request'],
  ): Promise<{ result: any; usage: CallUsage }> {
    await client.ensureAuth()
    const startTime = Date.now()
    const resolved = client.resolve()
    const includeClaudeSpoof = resolved?.isAnthropicOAuth ?? false
    const fullArgs = [...args, includeClaudeSpoof]

    let result: any, usage: CallUsage, collectorData: CollectorData | null = null
    if (resolved && (resolved.isCodex || resolved.isCopilotCodex)) {
      const r = await callViaResponsesApi(fnName, fullArgs, client)
      result = r.result; usage = r.usage
      accumulateUsage(slot, usage)
    } else {
      const r = await callViaBaml(fnName, fullArgs, resolved, slot)
      result = r.result; usage = r.usage; collectorData = r.collectorData
    }

    // Use collector's request body if available, otherwise fall back to manually passed request
    const traceRequest = collectorData
      ? (collectorData.rawRequestBody as any)?.messages
        ? { messages: (collectorData.rawRequestBody as any).messages }
        : request
      : request

    emitTrace(
      { startTime, model: resolved?.modelId ?? null, provider: resolved?.providerId ?? null, slot, defaultCallType: callType, meta },
      traceRequest,
      { rawBody: collectorData?.rawResponseBody ?? null, sseEvents: collectorData?.sseEvents ?? null, rawOutput: typeof result === 'string' ? result : undefined },
      usage,
    )
    return { result, usage }
  }


  return {
    slot,
    baml,

    chat(systemPrompt: string, messages: ChatMessage[], meta: AgentTraceMeta | undefined, options: ChatOptions | undefined, ackTurn: string): ChatStream {
      return tracedStream('CodingAgentChat', [systemPrompt, messages, ackTurn], 'chat', meta, { messages }, options)
    },

    async compact(systemPrompt: string, messages: ChatMessage[], meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }> {
      return tracedCall('CodingAgentCompact', [systemPrompt, messages], 'compact', meta, { messages })
    },

    async autopilot(systemPrompt: string, driverMessages: ChatMessage[], meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }> {
      return tracedCall('AutopilotContinuation', [systemPrompt, driverMessages], 'autopilot', meta, { messages: driverMessages })
    },

    async generateChatTitle(conversation: string, defaultChatName: string, meta?: AgentTraceMeta): Promise<{ title: string } | null> {
      const { result, usage } = await tracedCall('GenerateChatTitle', [conversation, defaultChatName], 'title', meta, { input: conversation })
      return result
    },

    async gatherReport(query: string, context: string, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }> {
      return tracedCall('GatherReport', [query, context], 'gather-report', meta, { input: query })
    },

    async gatherSplit(query: string, fileTree: string, tokenBudget: number, meta?: AgentTraceMeta): Promise<{ result: { path: string; query: string }[]; usage: CallUsage }> {
      return tracedCall('GatherSplit', [query, fileTree, tokenBudget], 'gather-split', meta, { input: query })
    },

    async extractMemoryDiff(
      transcript: string,
      currentMemory: string,
      meta?: AgentTraceMeta
    ): Promise<{
      result: {
        reasoning: string
        additions: Array<{ category: 'code_style' | 'codebase' | 'workflow' | 'tools'; content: string; evidence: string }>
        updates: Array<{ existing: string; replacement: string; evidence: string }>
        deletions: Array<{ existing: string; evidence: string }>
      }
      usage: CallUsage
    }> {
      return tracedCall(
        'ExtractMemoryDiff',
        [transcript, currentMemory],
        'extract-memory-diff',
        meta,
        { input: transcript }
      )
    },
    
    async patchFile(instructions: string, fileContent: string, previousAttempts?: Array<{response: string, error: string}>, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }> {
      return tracedCall('PatchFile', [instructions, fileContent, previousAttempts ?? []], 'patch-file', meta, { input: instructions })
    },

    async createFile(instructions: string, filePath: string, meta?: AgentTraceMeta): Promise<{ result: string; usage: CallUsage }> {
      return tracedCall('CreateFile', [instructions, filePath], 'create-file', meta, { input: instructions })
    },
  }
}

// =============================================================================
// Pre-built Instances
// =============================================================================

export const primary: ModelProxy = createModelProxy('primary')
export const secondary: ModelProxy = createModelProxy('secondary')
export const browser: ModelProxy = createModelProxy('browser')
