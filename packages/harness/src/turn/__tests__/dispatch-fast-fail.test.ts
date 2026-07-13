import { describe, it } from '@effect/vitest'
import { Data, Effect, Stream, Queue, Schema } from 'effect'
import { expect } from 'vitest'
import { dispatch, TurnAbort } from '../dispatcher'
import { defineHarnessTool } from '../../tool/tool'
import { defineToolkit } from '../../tool/toolkit'
import type { HarnessEvent } from '../../events'
import type { ProviderToolCallId, ToolCallId, StreamingFieldParser } from '@magnitudedev/ai'
import { createStreamingFieldParser, ModelStreamTerminal, StreamProviderCorrectnessViolation } from '@magnitudedev/ai'

// ── Test tools ────────────────────────────────────────────────────────

const inputSchema = Schema.Struct({ value: Schema.String })
const testToolErrorSchema = Schema.Struct({ message: Schema.String })

class TestToolError extends Data.TaggedError('TestToolError')<{
  readonly message: string
}> {}

const succeedEntry = {
  tool: defineHarnessTool({
    definition: { name: 'succeed', description: 'Succeeds', inputSchema, outputSchema: Schema.String },
    execute: () => Effect.succeed('ok'),
  }),
}

const failEntry = {
  tool: defineHarnessTool({
    definition: { name: 'fail', description: 'Fails', inputSchema, outputSchema: Schema.String },
    errorSchema: testToolErrorSchema,
    execute: () => Effect.fail(new TestToolError({ message: 'tool failed' })),
  }),
}

const toolkit = defineToolkit({ succeed: succeedEntry, fail: failEntry })

// ── Stream builder ───────────────────────────────────────────────────

type StreamEvent = import('@magnitudedev/ai').ResponseStreamEvent

function toolCallEvents(id: string, name: string): StreamEvent[] {
  const toolCallId = id as ToolCallId
  const providerToolCallId = id as ProviderToolCallId
  return [
    { _tag: 'tool_call_start', toolCallId, providerToolCallId, toolName: name },
    { _tag: 'tool_call_field_start', toolCallId, providerToolCallId, path: ['value'] },
    { _tag: 'tool_call_field_delta', toolCallId, providerToolCallId, path: ['value'], delta: '"x"' },
    { _tag: 'tool_call_field_end', toolCallId, providerToolCallId, path: ['value'], value: 'x' },
    { _tag: 'tool_call_ready', toolCallId, providerToolCallId },
  ]
}

function streamEndCompleted(finishReason: string = 'tool_calls'): StreamEvent {
  return {
    _tag: 'stream_end',
    terminal: ModelStreamTerminal.StreamCompleted({
      call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
      response: { status: 200, headers: [], requestId: null },
      finishReason: finishReason as any,
      progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
      usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
    }),
  }
}

function streamEndValidationFailure(
  toolCallId: string,
  providerToolCallId: string,
  toolName: string,
): StreamEvent {
  return {
    _tag: 'stream_end',
    terminal: ModelStreamTerminal.StreamFailed({
      cause: new StreamProviderCorrectnessViolation({
        call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
        response: { status: 200, headers: [], requestId: null },
        violation: {
          _tag: 'InvalidConstrainedOutput',
          output: {
            _tag: 'InvalidToolInput',
            toolCallId: toolCallId as ToolCallId,
            providerToolCallId: providerToolCallId as ProviderToolCallId,
            toolName,
            issue: { message: 'bad input' },
          },
        },
        progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
      }),
      usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
    }),
  }
}

function messageEvents(text: string): StreamEvent[] {
  return [
    { _tag: 'message_start' },
    { _tag: 'message_delta', text },
    { _tag: 'message_end' },
  ]
}

/** Build parsers for the given tool call IDs. */
function makeParsers(ids: string[]): Map<ToolCallId, StreamingFieldParser> {
  const parsers = new Map<ToolCallId, StreamingFieldParser>()
  for (const id of ids) {
    const parser = createStreamingFieldParser(inputSchema)
    parser.push('{"value":"x"}')
    parser.end()
    parsers.set(id as ToolCallId, parser)
  }
  return parsers
}

// ── Event collector ───────────────────────────────────────────────────

function runDispatch(
  streamEvents: StreamEvent[],
  opts?: {
    toolkit?: ReturnType<typeof defineToolkit>
    hooks?: Parameters<typeof dispatch>[0]['hooks']
  },
): Effect.Effect<readonly HarnessEvent[]> {
  const tk = opts?.toolkit ?? toolkit
  const allIds = streamEvents
    .filter((e): e is Extract<StreamEvent, { _tag: 'tool_call_start' }> => e._tag === 'tool_call_start')
    .map((e) => e.toolCallId as string)

  return Effect.gen(function* () {
    const queue = yield* Queue.unbounded<HarnessEvent>()
    const events: HarnessEvent[] = []

    const emit = (event: HarnessEvent) =>
      Effect.gen(function* () {
        events.push(event)
        yield* Queue.offer(queue, event)
      })

    yield* Effect.fork(
      dispatch({
        events: Stream.fromIterable(streamEvents),
        parsers: makeParsers(allIds),
        toolkit: tk,
        hooks: opts?.hooks,
        emit,
        requestId: null,
      }).pipe(Effect.ensuring(Queue.shutdown(queue))),
    )

    yield* Stream.fromQueue(queue).pipe(
      Stream.takeUntil((e) => e._tag === 'TurnEnd'),
      Stream.runDrain,
    )

    return events
  })
}

// ── Assertion helpers ──────────────────────────────────────────────────

function getTurnEnd(events: readonly HarnessEvent[]) {
  const end = events.find((e) => e._tag === 'TurnEnd')
  expect(end).toBeDefined()
  return end as Extract<HarnessEvent, { _tag: 'TurnEnd' }>
}

function eventTags(events: readonly HarnessEvent[]): string[] {
  return events.map((e) => e._tag)
}

function tagsForToolCall(events: readonly HarnessEvent[], toolCallId: string): string[] {
  return events
    .filter((e) => '_tag' in e && 'toolCallId' in (e as any) && (e as any).toolCallId === toolCallId)
    .map((e) => e._tag)
}

// ── Tests ──────────────────────────────────────────────────────────────

describe('dispatch fast-fail', () => {
  // ── Property: Normal completion is unaffected ─────────────────────

  it('completes normally when all tools succeed', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch([
        ...toolCallEvents('call-1', 'succeed'),
        streamEndCompleted(),
      ])

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('Completed')
      expect(turnEnd.outcome).toEqual({ _tag: 'Completed', toolCallsCount: 1, requestId: null })
    }).pipe(Effect.runPromise)
  )

  // ── Property: Tool execution error fast-fails ──────────────────────

  it('emits full lifecycle then TurnEnd with ToolExecutionError when tool throws', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch([
        ...toolCallEvents('call-1', 'fail'),
        // Model would produce more after the error — must be ignored
        ...messageEvents('should be ignored'),
        streamEndCompleted(),
      ])

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('ToolExecutionError')
      if (turnEnd.outcome._tag === 'ToolExecutionError') {
        expect(turnEnd.outcome.toolCallId).toBe('call-1')
        expect(turnEnd.outcome.toolName).toBe('fail')
        expect(turnEnd.outcome.error.message).toBe('tool failed')
      }

      // Full lifecycle was emitted for the failing tool
      const toolTags = tagsForToolCall(events, 'call-1')
      expect(toolTags).toContain('ToolInputStarted')
      expect(toolTags).toContain('ToolExecutionStarted')
      expect(toolTags).toContain('ToolExecutionEnded')

      // TurnEnd is the last event
      const tags = eventTags(events)
      expect(tags[tags.length - 1]).toBe('TurnEnd')

      // Events after the error were NOT processed
      expect(tags).not.toContain('MessageStart')
      expect(tags).not.toContain('MessageDelta')
      expect(tags).not.toContain('MessageEnd')
    }).pipe(Effect.runPromise)
  )

  // ── Property: Successful tool before error is preserved ─────────────

  it('preserves successful tool A results when tool B errors', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch([
        ...toolCallEvents('call-A', 'succeed'),
        ...toolCallEvents('call-B', 'fail'),
        // More model tokens after B — must be ignored
        ...messageEvents('ignored'),
        streamEndCompleted(),
      ])

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('ToolExecutionError')

      // Tool A lifecycle complete with success
      const toolATags = tagsForToolCall(events, 'call-A')
      expect(toolATags).toContain('ToolExecutionStarted')
      expect(toolATags).toContain('ToolExecutionEnded')

      const toolAEnded = events.find(
        (e) => e._tag === 'ToolExecutionEnded' && (e as any).toolCallId === 'call-A',
      ) as Extract<HarnessEvent, { _tag: 'ToolExecutionEnded' }>
      expect(toolAEnded.result._tag).toBe('Success')

      // Tool B lifecycle complete with error
      const toolBTags = tagsForToolCall(events, 'call-B')
      expect(toolBTags).toContain('ToolExecutionEnded')

      const toolBEnded = events.find(
        (e) => e._tag === 'ToolExecutionEnded' && (e as any).toolCallId === 'call-B',
      ) as Extract<HarnessEvent, { _tag: 'ToolExecutionEnded' }>
      expect(toolBEnded.result._tag).toBe('Error')

      // Outcome references tool B
      if (turnEnd.outcome._tag === 'ToolExecutionError') {
        expect(turnEnd.outcome.toolCallId).toBe('call-B')
      }

      // No message events after the error
      expect(eventTags(events)).not.toContain('MessageStart')
    }).pipe(Effect.runPromise)
  )

  // ── Property: Gate rejection fast-fails ─────────────────────────────

  it('fast-fails with GateRejected when beforeExecute hook rejects', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch(
        [
          ...toolCallEvents('call-1', 'succeed'),
          ...messageEvents('ignored'),
          streamEndCompleted(),
        ],
        {
          hooks: {
            beforeExecute: () =>
              Effect.succeed({ _tag: 'Deny' as const, denial: 'forbidden' }),
          },
        },
      )

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('GateRejected')
      if (turnEnd.outcome._tag === 'GateRejected') {
        expect(turnEnd.outcome.toolCallId).toBe('call-1')
      }

      // Events after rejection were NOT processed
      expect(eventTags(events)).not.toContain('MessageDelta')
    }).pipe(Effect.runPromise)
  )

  // ── Property: Engine defect fast-fails ──────────────────────────────

  it('fast-fails with EngineDefect for unknown tool name', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch(
        [
          { _tag: 'tool_call_start', toolCallId: 'call-X' as ToolCallId, providerToolCallId: 'call-X' as ProviderToolCallId, toolName: 'nonexistent' },
          ...messageEvents('ignored'),
          streamEndCompleted(),
        ],
        { toolkit: defineToolkit({ succeed: succeedEntry }) },
      )

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('EngineDefect')
      expect(eventTags(events)).not.toContain('MessageDelta')
    }).pipe(Effect.runPromise)
  )

  // ── Property: provider correctness violation from codec terminates clearly ───────

  it('handles tool input schema violation from codec as provider correctness', () =>
    Effect.gen(function* () {
      const events = yield* runDispatch([
        { _tag: 'tool_call_start', toolCallId: 'call-1' as ToolCallId, providerToolCallId: 'call-1' as ProviderToolCallId, toolName: 'succeed' },
        { _tag: 'tool_call_field_start', toolCallId: 'call-1' as ToolCallId, providerToolCallId: 'call-1' as ProviderToolCallId, path: ['value'] },
        { _tag: 'tool_call_field_delta', toolCallId: 'call-1' as ToolCallId, providerToolCallId: 'call-1' as ProviderToolCallId, path: ['value'], delta: 'bad' },
        streamEndValidationFailure('call-1', 'call-1', 'succeed'),
      ])

      const turnEnd = getTurnEnd(events)
      expect(turnEnd.outcome._tag).toBe('StreamFailed')
      if (turnEnd.outcome._tag === 'StreamFailed') {
        expect(turnEnd.outcome.terminal.cause._tag).toBe('StreamProviderCorrectnessViolation')
        if (turnEnd.outcome.terminal.cause._tag === 'StreamProviderCorrectnessViolation') {
          expect(turnEnd.outcome.terminal.cause.violation._tag).toBe('InvalidConstrainedOutput')
        }
      }
    }).pipe(Effect.runPromise)
  )
})
