/**
 * Display projection streaming tool input test.
 *
 * Verifies that assistant_tool_call_input_delta events accumulate jsonChunk
 * into the ToolStep's streamingInput, and that assistant_tool_call_end
 * finalizes it.
 */

import { describe, it, expect } from 'vitest'
import { Effect, Layer } from 'effect'
import {
  ProjectionBusTag,
  makeProjectionBusLayer,
  makeAmbientServiceLayer,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporterLive,
} from '@magnitudedev/event-core'
import type { AppEvent } from '../src/events'
import { TurnProjection } from '../src/projections/turn'
import { AgentRoutingProjection } from '../src/projections/agent-routing'
import { AgentStatusProjection } from '../src/projections/agent-status'
import { DisplayProjection } from '../src/projections/display'
import type { DisplayState, ToolStep } from '../src/projections/display'

const ts = (n: number) => 1_700_100_000_000 + n

const makeDisplay = async (events: AppEvent[]): Promise<DisplayState> => {
  const baseBusLayer = Layer.provideMerge(
    makeProjectionBusLayer<AppEvent>(),
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
  )
  const baseLayer = Layer.provideMerge(
    makeAmbientServiceLayer<AppEvent>(),
    baseBusLayer,
  )

  const runtimeLayer = Layer.mergeAll(
    FrameworkErrorPubSubLive,
    Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive),
    baseLayer,
    Layer.provide(TurnProjection.Layer, baseLayer),
    Layer.provide(AgentRoutingProjection.Layer, baseLayer),
    Layer.provide(AgentStatusProjection.Layer, baseLayer),
    Layer.provide(DisplayProjection.Layer, baseLayer),
  )

  const program = Effect.gen(function* () {
    const bus = yield* (ProjectionBusTag<AppEvent>())
    const projection = yield* DisplayProjection.Tag

    for (const event of events) {
      yield* bus.processEvent(event as any)
    }

    return yield* projection.getFork(null)
  })

  return Effect.runPromise(
    program.pipe(Effect.provide(runtimeLayer)) as Effect.Effect<DisplayState>,
  )
}

const forkId = null
const turnId = 'turn-1'

describe('Display projection — streaming tool input', () => {
  it('accumulates jsonChunks in ToolStep.streamingInput during streaming', async () => {
    const toolCallId = 'tc-1'

    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      { type: 'assistant_tool_call_start', timestamp: ts(2), forkId, turnId, toolCallId, toolName: 'shell', toolKey: 'shell' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(3), forkId, turnId, toolCallId, path: ['command'] as readonly string[], delta: '{"command"' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(4), forkId, turnId, toolCallId, path: ['command'] as readonly string[], delta: ':"ls"}' } as AppEvent,
    ])

    const thinkBlock = state.messages.find(m => m.type === 'think_block')
    expect(thinkBlock).toBeDefined()
    if (!thinkBlock || thinkBlock.type !== 'think_block') return

    const toolStep = thinkBlock.steps.find(s => s.id === toolCallId && s.type === 'tool') as ToolStep | undefined
    expect(toolStep).toBeDefined()
    expect(toolStep?.streamingInput).toBe('{"command":"ls"}')
  })

  it('finalizes streamingInput on assistant_tool_call_end', async () => {
    const toolCallId = 'tc-1'

    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      { type: 'assistant_tool_call_start', timestamp: ts(2), forkId, turnId, toolCallId, toolName: 'shell', toolKey: 'shell' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(3), forkId, turnId, toolCallId, path: ['command'] as readonly string[], delta: '{"command"' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(4), forkId, turnId, toolCallId, path: ['command'] as readonly string[], delta: ':"ls"}' } as AppEvent,
      { type: 'assistant_tool_call_end', timestamp: ts(5), forkId, turnId, toolCallId, input: { command: 'ls' } } as AppEvent,
    ])

    const thinkBlock = state.messages.find(m => m.type === 'think_block')
    if (!thinkBlock || thinkBlock.type !== 'think_block') {
      throw new Error('Expected think block')
    }

    const toolStep = thinkBlock.steps.find(s => s.id === toolCallId && s.type === 'tool') as ToolStep | undefined
    expect(toolStep).toBeDefined()
    // After tool_call_end, streamingInput should contain the JSON-serialized input
    expect(toolStep?.streamingInput).toBeDefined()
    expect(typeof toolStep?.streamingInput).toBe('string')
    expect(toolStep!.streamingInput!).toContain('"command"')
  })

  it('multiple tool calls tracked independently via streamingInput', async () => {
    const state = await makeDisplay([
      { type: 'turn_started', timestamp: ts(1), forkId, turnId, chainId: 'chain-1' } as AppEvent,
      { type: 'assistant_tool_call_start', timestamp: ts(2), forkId, turnId, toolCallId: 'tc-1', toolName: 'shell', toolKey: 'shell' } as AppEvent,
      { type: 'assistant_tool_call_start', timestamp: ts(3), forkId, turnId, toolCallId: 'tc-2', toolName: 'read', toolKey: 'fileRead' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(4), forkId, turnId, toolCallId: 'tc-1', path: ['cmd'] as readonly string[], delta: '"cmd"' } as AppEvent,
      { type: 'assistant_tool_call_field_delta', timestamp: ts(5), forkId, turnId, toolCallId: 'tc-2', path: ['path'] as readonly string[], delta: '"path"' } as AppEvent,
    ])

    const thinkBlock = state.messages.find(m => m.type === 'think_block')
    if (!thinkBlock || thinkBlock.type !== 'think_block') throw new Error('Expected think block')

    const step1 = thinkBlock.steps.find(s => s.id === 'tc-1' && s.type === 'tool') as ToolStep | undefined
    const step2 = thinkBlock.steps.find(s => s.id === 'tc-2' && s.type === 'tool') as ToolStep | undefined

    expect(step1?.streamingInput).toBe('"cmd"')
    expect(step2?.streamingInput).toBe('"path"')
  })
})
