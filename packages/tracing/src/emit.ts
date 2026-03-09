/**
 * Trace emission — callback registration and helpers for emitting traces.
 */

import { logger } from '@magnitudedev/logger'
import type { Collector } from '@magnitudedev/llm-core'
import type { TraceData, CollectorData, ModelSlot, CallUsage } from './types'

// =============================================================================
// Callback Registration
// =============================================================================

type TraceCallback = (trace: TraceData<any>) => void
let traceCallback: TraceCallback | null = null

/** Register a callback to receive trace data for every LLM call */
export function onTrace(cb: TraceCallback): void {
  traceCallback = cb
}

/** Check if tracing is active */
export function isTracing(): boolean {
  return traceCallback !== null
}

// =============================================================================
// Collector Data Extraction
// =============================================================================

export function extractCollectorData(collector: Collector): CollectorData {
  const lastCall = collector.last?.calls.at(-1)
  let rawRequestBody: unknown | null = null
  let rawResponseBody: unknown | null = null
  let sseEvents: unknown[] | null = null
  try { rawRequestBody = lastCall?.httpRequest?.body.json() ?? null } catch {}
  try { rawResponseBody = lastCall?.httpResponse?.body.json() ?? null } catch {}
  try {
    sseEvents = 'sseResponses' in (lastCall ?? {})
      ? (lastCall as any).sseResponses()?.map((s: any) => { try { return s.json?.() ?? null } catch { return null } }) ?? null
      : null
  } catch {}
  return { rawRequestBody, rawResponseBody, sseEvents }
}

// =============================================================================
// Trace Emission
// =============================================================================

export interface TraceContext {
  startTime: number
  model: string | null
  provider: string | null
  slot: ModelSlot
  defaultCallType: string
  meta?: Record<string, unknown>
  strategyId?: string | null
  systemPrompt?: string | null
}

/**
 * Emit a trace to the registered callback.
 */
export function emitTrace(
  ctx: TraceContext,
  request: TraceData['request'],
  response: { rawBody: unknown | null; sseEvents: unknown[] | null; rawOutput?: string },
  usage: CallUsage,
): void {
  if (!traceCallback) return
  try {
    traceCallback({
      timestamp: new Date().toISOString(),
      model: ctx.model,
      provider: ctx.provider,
      slot: ctx.slot,
      callType: (ctx.meta?.callType as string) ?? ctx.defaultCallType,
      request,
      response,
      usage,
      durationMs: Date.now() - ctx.startTime,
      metadata: ctx.meta ?? {},
      strategyId: ctx.strategyId ?? null,
      systemPrompt: ctx.systemPrompt ?? null,
    })
  } catch (e) {
    logger.error({ error: e }, '[Tracing] Failed to emit trace')
  }
}

export interface TracedStream {
  stream: AsyncIterable<string>
  getChunks(): string[]
}

/**
 * Wrap a stream to capture chunks for tracing.
 * onComplete is called when the stream finishes being consumed.
 */
export function wrapStreamForTrace(innerStream: AsyncIterable<string>, onComplete?: (chunks: string[]) => void): TracedStream {
  const chunks: string[] = []
  async function* wrapped() {
    try {
      for await (const chunk of innerStream) {
        chunks.push(chunk)
        yield chunk
      }
    } finally {
      onComplete?.(chunks)
    }
  }
  return {
    stream: wrapped(),
    getChunks() { return chunks },
  }
}
