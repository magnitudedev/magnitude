import { describe, it, expect } from 'vitest'
import { Stream, Effect } from 'effect'
import { processChunk, decode, tryParseJson, mapReason, initialDecoderState, type DecoderState } from '../decode'
import type { ResponseStreamEvent } from '../../../events/turn-part-event'
import { ChatCompletionsStreamChunk } from '@magnitudedev/drivers'

// =============================================================================
// Fixtures helpers
// =============================================================================

function makeChunk(opts: {
  reasoningContent?: string
  content?: string
  toolCalls?: Array<{
    index: number
    id?: string
    functionName?: string
    functionArguments?: string
  }>
  finishReason?: string | null
  usage?: { prompt_tokens: number; completion_tokens: number; prompt_tokens_details?: { cached_tokens?: number } }
}): ChatCompletionsStreamChunk {
  return new ChatCompletionsStreamChunk({
    id:      'chatcmpl-test',
    object:  'chat.completion.chunk',
    created: 1234567890,
    model:   'test-model',
    choices: [{
      index:         0,
      delta:         {
        role:              undefined,
        reasoning_content: opts.reasoningContent ?? null,
        content:           opts.content ?? null,
        tool_calls:        opts.toolCalls?.map(tc => ({
          index:    tc.index,
          id:       tc.id,
          type:     'function' as const,
          function: {
            name:      tc.functionName,
            arguments: tc.functionArguments,
          },
        })),
      },
      finish_reason: opts.finishReason ?? null,
    }],
    usage: opts.usage,
  })
}

function collectEvents(chunks: ChatCompletionsStreamChunk[]): Promise<ResponseStreamEvent[]> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const stream = decode(Stream.fromIterable(chunks) as any)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return Effect.runPromise(Stream.runCollect(stream).pipe(
    Effect.map(chunk => Array.from(chunk)),
  ) as any)
}

// =============================================================================
// tryParseJson
// =============================================================================

describe('tryParseJson', () => {
  it('valid JSON → parsed', () => {
    expect(tryParseJson('{"a":1}')).toEqual({ a: 1 })
  })
  it('invalid JSON → { _parseError }', () => {
    const result = tryParseJson('{bad') as Record<string, unknown>
    expect(result._parseError).toBe('{bad')
  })
  it('empty string → { _parseError }', () => {
    const result = tryParseJson('') as Record<string, unknown>
    expect(result._parseError).toBe('')
  })
})

// =============================================================================
// mapReason
// =============================================================================

describe('mapReason', () => {
  it('maps known reasons', () => {
    expect(mapReason('stop')).toBe('stop')
    expect(mapReason('tool_calls')).toBe('tool_calls')
    expect(mapReason('length')).toBe('length')
    expect(mapReason('content_filter')).toBe('content_filter')
  })
  it('maps unknown to other', () => {
    expect(mapReason('something_else')).toBe('other')
    expect(mapReason(null)).toBe('other')
    expect(mapReason(undefined)).toBe('other')
  })
})

// =============================================================================
// processChunk — unit tests
// =============================================================================

describe('processChunk', () => {
  it('empty chunk (no delta fields) → no events', () => {
    const chunk = makeChunk({})
    const { events } = processChunk(chunk, initialDecoderState)
    expect(events).toHaveLength(0)
  })

  it('reasoning_content opens thought', () => {
    const chunk = makeChunk({ reasoningContent: 'Let me think' })
    const { events, state } = processChunk(chunk, initialDecoderState)
    expect(events[0]).toMatchObject({ type: 'thought_start', level: 'medium' })
    expect(events[1]).toMatchObject({ type: 'thought_delta', text: 'Let me think' })
    expect(state.thoughtOpen).toBe(true)
  })

  it('reasoning_content across two chunks keeps thought open', () => {
    const c1 = makeChunk({ reasoningContent: 'Part 1' })
    const { state: s1, events: e1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ reasoningContent: ' Part 2' })
    const { state: s2, events: e2 } = processChunk(c2, s1)
    expect(e1.map(e => e.type)).toEqual(['thought_start', 'thought_delta'])
    expect(e2).toEqual([{ type: 'thought_delta', text: ' Part 2' }])
    expect(s2.thoughtOpen).toBe(true)
  })

  it('content after reasoning → closes thought, opens message', () => {
    const c1 = makeChunk({ reasoningContent: 'Thinking' })
    const { state: s1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ content: 'Response' })
    const { events } = processChunk(c2, s1)
    expect(events.map(e => e.type)).toContain('thought_end')
    expect(events.map(e => e.type)).toContain('message_start')
    expect(events.map(e => e.type)).toContain('message_delta')
  })

  it('finish_reason closes open thought', () => {
    const c1 = makeChunk({ reasoningContent: 'Thinking' })
    const { state: s1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ finishReason: 'stop' })
    const { events } = processChunk(c2, s1)
    expect(events.some(e => e.type === 'thought_end')).toBe(true)
    expect(events.some(e => e.type === 'response_done')).toBe(true)
  })

  it('finish_reason closes open message', () => {
    const c1 = makeChunk({ content: 'Hello' })
    const { state: s1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ finishReason: 'stop' })
    const { events } = processChunk(c2, s1)
    expect(events.some(e => e.type === 'message_end')).toBe(true)
    expect(events.some(e => e.type === 'response_done')).toBe(true)
  })

  it('tool_call first chunk → tool_call_start', () => {
    const chunk = makeChunk({ toolCalls: [{ index: 0, id: 'srv-id', functionName: 'read_file', functionArguments: '' }] })
    const { events } = processChunk(chunk, initialDecoderState)
    expect(events[0].type).toBe('tool_call_start')
    const startEvt = events[0] as { type: string; toolName: string }
    expect(startEvt.toolName).toBe('read_file')
  })

  it('server tool_call id is IGNORED — codec generates its own', () => {
    const chunk = makeChunk({ toolCalls: [{ index: 0, id: 'server-id-123', functionName: 'read_file' }] })
    const { events } = processChunk(chunk, initialDecoderState)
    const startEvt = events[0] as { type: string; toolCallId: string }
    expect(startEvt.toolCallId).not.toBe('server-id-123')
    expect(startEvt.toolCallId).toMatch(/^call-/)
  })

  it('tool_call field events accumulate across chunks', () => {
    const c1 = makeChunk({ toolCalls: [{ index: 0, functionName: 'run', functionArguments: '{"a"' }] })
    const { state: s1, events: e1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ toolCalls: [{ index: 0, functionArguments: ':1}' }] })
    const { events: e2 } = processChunk(c2, s1)
    expect(e1.some(e => e.type === 'tool_call_field_start')).toBe(true)
    expect(e2.some(e => e.type === 'tool_call_field_end')).toBe(true)
  })

  it('finish_reason=tool_calls → emits tool_call_end and response_done', () => {
    const c1 = makeChunk({ toolCalls: [{ index: 0, functionName: 'run', functionArguments: '{"cmd":"ls"}' }] })
    const { state: s1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ finishReason: 'tool_calls', usage: { prompt_tokens: 7, completion_tokens: 3 } })
    const { events } = processChunk(c2, s1)
    const endEvt = events.find(e => e.type === 'tool_call_end') as { type: string; toolCallId: string } | undefined
    expect(endEvt).toBeDefined()
    const done = events.find(e => e.type === 'response_done') as { reason: string; usage: { inputTokens: number; outputTokens: number } } | undefined
    expect(done).toBeDefined()
    expect(done!.reason).toBe('tool_calls')
    expect(done!.usage).toMatchObject({ inputTokens: 7, outputTokens: 3 })
  })

  it('malformed tool args still emits tool_call_end and response_done', () => {
    const c1 = makeChunk({ toolCalls: [{ index: 0, functionName: 'run', functionArguments: '{bad' }] })
    const { state: s1 } = processChunk(c1, initialDecoderState)
    const c2 = makeChunk({ finishReason: 'tool_calls', usage: { prompt_tokens: 1, completion_tokens: 1 } })
    const { events } = processChunk(c2, s1)
    expect(events.some(e => e.type === 'tool_call_end')).toBe(true)
    expect(events.some(e => e.type === 'response_done')).toBe(true)
  })

  it('usage in final chunk is included on response_done', () => {
    const c = makeChunk({
      finishReason: 'stop',
      usage: { prompt_tokens: 100, completion_tokens: 50, prompt_tokens_details: { cached_tokens: 12 } },
    })
    const { events } = processChunk(c, initialDecoderState)
    const done = events.find(e => e.type === 'response_done') as {
      usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number | null; cacheWriteTokens: number | null }
    } | undefined
    expect(done).toBeDefined()
    expect(done!.usage).toEqual({
      inputTokens: 100,
      outputTokens: 50,
      cacheReadTokens: 12,
      cacheWriteTokens: null,
    })
  })
})

// =============================================================================
// decode stream integration
// =============================================================================

describe('decode stream', () => {
  it('thought-only stream → thought_start/delta/end + response_done(stop)', async () => {
    const chunks = [
      makeChunk({ reasoningContent: 'Thinking...' }),
      makeChunk({ finishReason: 'stop', usage: { prompt_tokens: 2, completion_tokens: 1 } }),
    ]
    const events = await collectEvents(chunks)
    expect(events.map(e => e.type)).toEqual([
      'thought_start', 'thought_delta', 'thought_end', 'response_done',
    ])
    const done = events.find(e => e.type === 'response_done') as { reason: string }
    expect(done.reason).toBe('stop')
  })

  it('content-only stream → message_start/delta/end + response_done', async () => {
    const chunks = [
      makeChunk({ content: 'Hello' }),
      makeChunk({ finishReason: 'stop', usage: { prompt_tokens: 2, completion_tokens: 1 } }),
    ]
    const events = await collectEvents(chunks)
    expect(events.map(e => e.type)).toEqual([
      'message_start', 'message_delta', 'message_end', 'response_done',
    ])
  })

  it('tool_call stream → tool_call_start + field events + tool_call_end + response_done(tool_calls)', async () => {
    const chunks = [
      makeChunk({ toolCalls: [{ index: 0, functionName: 'read', functionArguments: '{"path":' }] }),
      makeChunk({ toolCalls: [{ index: 0, functionArguments: '"/foo"}' }] }),
      makeChunk({ finishReason: 'tool_calls', usage: { prompt_tokens: 4, completion_tokens: 2 } }),
    ]
    const events = await collectEvents(chunks)
    const types = events.map(e => e.type)
    expect(types).toContain('tool_call_start')
    expect(types).toContain('tool_call_field_delta')
    expect(types).toContain('tool_call_field_end')
    expect(types).toContain('tool_call_end')
    expect(types).toContain('response_done')
  })

  it('parallel tool calls (different indices) → independent sequences', async () => {
    const chunks = [
      makeChunk({ toolCalls: [
        { index: 0, functionName: 'tool_a', functionArguments: '{"x":1}' },
        { index: 1, functionName: 'tool_b', functionArguments: '{"y":2}' },
      ]}),
      makeChunk({ finishReason: 'tool_calls' }),
    ]
    const events = await collectEvents(chunks)
    const starts = events.filter(e => e.type === 'tool_call_start')
    const ends   = events.filter(e => e.type === 'tool_call_end')
    expect(starts).toHaveLength(2)
    expect(ends).toHaveLength(2)
    const toolNames = starts.map(e => (e as { toolName: string }).toolName)
    expect(toolNames).toContain('tool_a')
    expect(toolNames).toContain('tool_b')
    // Each end has correct parsed input
    const endA = ends.find(e => {
      const toolCallId = (e as { toolCallId: string }).toolCallId
      return (starts.find(s => (s as { toolCallId: string }).toolCallId === toolCallId && (s as { toolName: string }).toolName === 'tool_a')) != null
    })
    expect(endA).toBeDefined()
  })

  it('server tool_call IDs are ignored — codec generates its own call- IDs', async () => {
    const chunks = [
      makeChunk({ toolCalls: [{ index: 0, id: 'server-xyz', functionName: 'foo', functionArguments: '{}' }] }),
      makeChunk({ finishReason: 'tool_calls' }),
    ]
    const events = await collectEvents(chunks)
    const start = events.find(e => e.type === 'tool_call_start') as { toolCallId: string } | undefined
    expect(start!.toolCallId).toMatch(/^call-/)
    expect(start!.toolCallId).not.toContain('server-xyz')
  })

  it('finish_reason absent on non-final chunk → no response_done', async () => {
    const chunks = [
      makeChunk({ content: 'Part 1' }),
      makeChunk({ content: ' Part 2' }),
    ]
    const events = await collectEvents(chunks)
    expect(events.some(e => e.type === 'response_done')).toBe(false)
  })

  it('thought + content + tool_call ordering → correct nesting', async () => {
    const chunks = [
      makeChunk({ reasoningContent: 'Thinking' }),
      makeChunk({ content: 'Response' }),
      makeChunk({ toolCalls: [{ index: 0, functionName: 'do_it', functionArguments: '{}' }] }),
      makeChunk({ finishReason: 'tool_calls', usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    ]
    const events = await collectEvents(chunks)
    const types = events.map(e => e.type)
    const thoughtStartIdx = types.indexOf('thought_start')
    const thoughtEndIdx   = types.indexOf('thought_end')
    const msgStartIdx     = types.indexOf('message_start')
    const msgEndIdx       = types.indexOf('message_end')
    const toolStartIdx    = types.indexOf('tool_call_start')
    // thought closes before message opens
    expect(thoughtEndIdx).toBeLessThan(msgStartIdx)
    // message closes before tool call opens
    expect(msgEndIdx).toBeLessThan(toolStartIdx)
    // response_done at the end
    expect(types[types.length - 1]).toBe('response_done')
  })
})
