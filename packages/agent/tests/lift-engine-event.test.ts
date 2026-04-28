/**
 * Tests for liftTurnEngineEvent — maps TurnEngineEvent → AppEvent[].
 */

import { describe, it, expect } from 'vitest'
import { liftTurnEngineEvent } from '../src/lift-engine-event'
import type { LiftEngineContext } from '../src/lift-engine-event'
import type { TurnEngineEvent, RegisteredTool } from '@magnitudedev/turn-engine'

// Registered tools with meta.defKey === 'shell' so tool_event lookups resolve.
const registeredTools: ReadonlyMap<string, RegisteredTool<unknown>> = new Map([
  ['shell', { tool: { name: 'shell' } as never, toolName: 'shell', groupName: 'default', meta: { defKey: 'shell' } } as RegisteredTool<unknown>],
])

const ctx: LiftEngineContext = {
  forkId: 'fork-1',
  turnId: 'turn-abc',
  registeredTools,
  toolCallToToolKey: new Map([
    ['tc-1', 'shell'],
    ['tc-2', 'shell'],
  ]),
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function lift(event: TurnEngineEvent) {
  return liftTurnEngineEvent(event, ctx)
}

// ─── Thought ──────────────────────────────────────────────────────────────────

describe('ThoughtStart', () => {
  it('emits thinking_start', () => {
    const events = lift({ _tag: 'ThoughtStart', kind: 'turn' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({
      type: 'thinking_start',
      forkId: 'fork-1',
      turnId: 'turn-abc',
    })
  })
})

describe('ThoughtChunk', () => {
  it('emits thinking_chunk', () => {
    const events = lift({ _tag: 'ThoughtChunk', text: 'hello' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'thinking_chunk', text: 'hello' })
  })
})

describe('ThoughtEnd', () => {
  it('emits thinking_end', () => {
    const events = lift({ _tag: 'ThoughtEnd' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'thinking_end' })
  })
})

// ─── Message ──────────────────────────────────────────────────────────────────

describe('MessageStart', () => {
  it('maps to=user → destination {kind:user}', () => {
    const events = lift({ _tag: 'MessageStart', id: 'msg-1', to: 'user' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({
      type: 'message_start',
      id: 'msg-1',
      destination: { kind: 'user' },
    })
  })

  it('maps to=parent → destination {kind:parent}', () => {
    const events = lift({ _tag: 'MessageStart', id: 'msg-2', to: 'parent' })
    expect(events[0]).toMatchObject({ destination: { kind: 'parent' } })
  })

  it('maps to=worker:task-x → destination {kind:worker, taskId}', () => {
    const events = lift({ _tag: 'MessageStart', id: 'msg-3', to: 'worker:task-x' })
    expect(events[0]).toMatchObject({ destination: { kind: 'worker', taskId: 'task-x' } })
  })

  it('unknown to defaults to user', () => {
    const events = lift({ _tag: 'MessageStart', id: 'msg-4', to: 'unknown-dest' })
    expect(events[0]).toMatchObject({ destination: { kind: 'user' } })
  })
})

describe('MessageChunk', () => {
  it('emits message_chunk with id and text', () => {
    const events = lift({ _tag: 'MessageChunk', id: 'msg-1', text: 'hello world' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'message_chunk', id: 'msg-1', text: 'hello world' })
  })
})

describe('MessageEnd', () => {
  it('emits message_end with id', () => {
    const events = lift({ _tag: 'MessageEnd', id: 'msg-1' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'message_end', id: 'msg-1' })
  })
})

// ─── Tool lifecycle ───────────────────────────────────────────────────────────

describe('ToolInputStarted → tool_event', () => {
  it('wraps in tool_event with resolved toolKey', () => {
    const events = lift({ _tag: 'ToolInputStarted', toolCallId: 'tc-1', toolName: 'shell', group: 'g1' })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolCallId: 'tc-1', toolKey: 'shell' })
  })
})

describe('ToolInputFieldChunk → tool_event', () => {
  it('uses toolCallToToolKey map', () => {
    const events = lift({ _tag: 'ToolInputFieldChunk', toolCallId: 'tc-1', field: 'command', path: ['command'], delta: 'ls' } as unknown as TurnEngineEvent)
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolCallId: 'tc-1', toolKey: 'shell' })
  })
})

describe('ToolInputFieldComplete → tool_event', () => {
  it('uses toolCallToToolKey map', () => {
    const events = lift({ _tag: 'ToolInputFieldComplete', toolCallId: 'tc-1', field: 'command', path: ['command'], value: 'ls -la' } as unknown as TurnEngineEvent)
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolCallId: 'tc-1', toolKey: 'shell' })
  })
})

describe('ToolInputReady → tool_event', () => {
  it('uses toolCallToToolKey map', () => {
    const events = lift({ _tag: 'ToolInputReady', toolCallId: 'tc-1', input: { command: 'ls' } })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolCallId: 'tc-1', toolKey: 'shell' })
  })
})

// ─── Tool execution lifecycle ─────────────────────────────────────────────────

describe('ToolExecutionStarted → tool_event', () => {
  it('wraps in tool_event with toolKey resolved from registry', () => {
    const events = lift({
      _tag: 'ToolExecutionStarted',
      toolCallId: 'tc-1',
      toolName: 'shell',
      group: 'g1',
      input: { command: 'ls' },
      cached: false,
    })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({
      type: 'tool_event',
      toolCallId: 'tc-1',
      toolKey: 'shell',
    })
    expect((events[0] as { event: TurnEngineEvent }).event._tag).toBe('ToolExecutionStarted')
  })
})

describe('ToolExecutionEnded → tool_event', () => {
  it('Success → wraps into tool_event with inner Success', () => {
    const events = lift({
      _tag: 'ToolExecutionEnded',
      toolCallId: 'tc-1',
      toolName: 'shell',
      group: 'g1',
      result: { _tag: 'Success', output: { mode: 'completed', exitCode: 0, stdout: 'hi', stderr: '' } },
    })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolKey: 'shell' })
  })

  it('Error → wraps into tool_event with inner Error', () => {
    const events = lift({
      _tag: 'ToolExecutionEnded',
      toolCallId: 'tc-2',
      toolName: 'shell',
      group: 'g1',
      result: { _tag: 'Error', error: 'command not found' },
    })
    expect(events).toHaveLength(1)
    const inner = (events[0] as { event: TurnEngineEvent }).event
    expect(inner._tag).toBe('ToolExecutionEnded')
  })
})

describe('ToolEmission → tool_event', () => {
  it('uses toolCallToToolKey map', () => {
    const events = lift({
      _tag: 'ToolEmission',
      toolCallId: 'tc-1',
      value: { progress: 50 },
    } as unknown as TurnEngineEvent)
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolCallId: 'tc-1', toolKey: 'shell' })
  })
})

describe('ToolInputDecodeFailure → tool_event', () => {
  it('wraps with inner decode failure', () => {
    const events = lift({
      _tag: 'ToolInputDecodeFailure',
      toolCallId: 'tc-1',
      toolName: 'shell',
      group: 'g1',
      detail: { msg: 'bad json' },
    })
    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ type: 'tool_event', toolKey: 'shell' })
  })
})

describe('TurnStructureDecodeFailure', () => {
  it('drops — surfaces via turn_outcome ParseFailure, not tool_event', () => {
    const events = lift({ _tag: 'TurnStructureDecodeFailure', detail: 'unexpected token' })
    expect(events).toHaveLength(0)
  })
})

// ─── TurnEnd ──────────────────────────────────────────────────────────────────

describe('TurnEnd', () => {
  it('returns empty array — cortex builds turn_outcome', () => {
    const events = lift({ _tag: 'TurnEnd', outcome: { _tag: 'Completed', toolCallsCount: 0 }, usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: null, cacheWriteTokens: null } })
    expect(events).toHaveLength(0)
  })
})

// ─── Null fork ────────────────────────────────────────────────────────────────

describe('null forkId', () => {
  it('passes through null forkId', () => {
    const events = liftTurnEngineEvent(
      { _tag: 'ThoughtStart', kind: 'turn' },
      { forkId: null, turnId: 'turn-1', registeredTools: new Map(), toolCallToToolKey: new Map() },
    )
    expect(events[0]).toMatchObject({ forkId: null })
  })
})
