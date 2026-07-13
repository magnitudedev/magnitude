import { describe, it } from '@effect/vitest'
import { Data, Effect, Stream, Schema } from 'effect'
import { expect } from 'vitest'
import { createHarness } from '../harness'
import { defineHarnessTool } from '../../tool/tool'
import { defineToolkit } from '../../tool/toolkit'
import type { BoundModel, ProviderToolCallId, ToolCallId, ResponseStreamEvent, StreamingFieldParser } from '@magnitudedev/ai'
import { createStreamingFieldParser, ModelStreamTerminal, Prompt, createToolCallId } from '@magnitudedev/ai'

// ── Test tool that always errors ─────────────────────────────────────

const inputSchema = Schema.Struct({ value: Schema.String })
const testToolErrorSchema = Schema.Struct({ message: Schema.String })

class TestToolError extends Data.TaggedError('TestToolError')<{
  readonly message: string
}> {}

const failTool = defineHarnessTool({
  definition: { name: 'fail', description: 'Always fails', inputSchema, outputSchema: Schema.String },
  errorSchema: testToolErrorSchema,
  execute: () => Effect.fail(new TestToolError({ message: 'tool failed' })),
})

const toolkit = defineToolkit({ fail: { tool: failTool } })

// ── Mock model ───────────────────────────────────────────────────────

type StreamEvent = ResponseStreamEvent

function createMockModel(events: StreamEvent[], parsers: Map<ToolCallId, StreamingFieldParser>): BoundModel<any> {
  return {
    stream: () =>
      Effect.succeed({
        events: Stream.fromIterable(events),
        parsers,
        logprobs: [],
        requestId: null,
      }),
  }
}

// ── Test ─────────────────────────────────────────────────────────────

describe('queue race', () => {
  it('completes cleanly when tool errors occur during forked dispatch', () =>
    Effect.gen(function* () {
      const providerToolCallId = 'ptc-1' as ProviderToolCallId
      const callId = createToolCallId()
      const parser = createStreamingFieldParser(Schema.Struct({ value: Schema.String }))
      const parsers = new Map([[callId, parser]])

      const streamEvents: StreamEvent[] = [
        { _tag: 'tool_call_start', toolCallId: callId, providerToolCallId, toolName: 'fail' },
        { _tag: 'tool_call_field_start', toolCallId: callId, providerToolCallId, path: ['value'] },
        { _tag: 'tool_call_field_delta', toolCallId: callId, providerToolCallId, path: ['value'], delta: '"x"' },
        { _tag: 'tool_call_field_end', toolCallId: callId, providerToolCallId, path: ['value'], value: 'x' },
        { _tag: 'tool_call_ready', toolCallId: callId, providerToolCallId },
        {
          _tag: 'stream_end',
          terminal: ModelStreamTerminal.StreamCompleted({
            call: { provider: 'test', model: 'test', method: 'POST', url: 'http://test' },
            response: { status: 200, headers: [], requestId: null },
            finishReason: 'tool_calls',
            progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
            usage: { _tag: 'UsageNotReported', reason: 'provider_does_not_report_usage' },
          }),
        },
      ]

      const model = createMockModel(streamEvents, parsers)

      const harness = createHarness({
        model,
        toolkit,
      })

      // Run a turn and consume events — exactly as cortex does
      const turn = yield* harness.runTurn(Prompt.from({
        messages: [{ _tag: 'UserMessage', parts: [{ _tag: 'TextPart', text: 'run' }] }],
      }))

      // Allow the forked dispatch fiber to complete and shut down the
      // queue before we start consuming. This reliably reproduces the
      // race that occurs in production when the cortex does async work
      // between obtaining the LiveTurn and draining its event stream.
      yield* Effect.sleep("50 millis")

      const events: Array<{ _tag: string }> = []
      yield* Stream.runForEach(turn.events, (event) =>
        Effect.sync(() => {
          events.push(event)
        }),
      )

      // The consumer must see a TurnEnd event
      const turnEnd = events.find((e) => e._tag === 'TurnEnd') as any
      expect(turnEnd, 'TurnEnd event must be delivered to the consumer').toBeDefined()

      // The outcome must be ToolExecutionError, not Completed
      expect(turnEnd.outcome._tag).toBe('ToolExecutionError')
    }))
})
