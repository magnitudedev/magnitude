import { describe, expect, it } from 'vitest'
import { createHeadlessOutputRenderer, type AppEvent } from './output'

function render(events: AppEvent[]): string[] {
  const renderer = createHeadlessOutputRenderer()
  return events.flatMap((event) => [...renderer.handleEvent(event).lines])
}

function event(value: unknown): AppEvent {
  return value as AppEvent
}

describe('headless output renderer', () => {
  it('buffers assistant messages until message_end', () => {
    const renderer = createHeadlessOutputRenderer()

    expect(renderer.handleEvent(event({
      type: 'message_start',
      forkId: null,
      turnId: 't1',
      id: 'm1',
      destination: { kind: 'user' },
    })).lines).toEqual([])

    expect(renderer.handleEvent(event({
      type: 'message_chunk',
      forkId: null,
      turnId: 't1',
      id: 'm1',
      text: 'Hello',
    })).lines).toEqual([])

    expect(renderer.handleEvent(event({
      type: 'message_end',
      forkId: null,
      turnId: 't1',
      id: 'm1',
    })).lines).toEqual(['Hello'])
  })

  it('uses natural agent ids instead of fork ids', () => {
    const lines = render([event({
      type: 'agent_created',
      forkId: 'fork-random',
      parentForkId: null,
      agentId: 'impl-eng',
      name: 'Implement kanban app',
      role: 'engineer',
      context: '',
      mode: 'spawn',
      taskId: 'task-1',
      message: 'build it',
    }), event({
      type: 'turn_outcome',
      forkId: 'fork-random',
      turnId: 't1',
      chainId: 'c1',
      strategyId: 'native',
      outcome: { _tag: 'Completed', completion: { toolCallsCount: 2, finishReason: 'stop', feedback: [], yieldTarget: null } },
      inputTokens: null,
      outputTokens: null,
      cacheReadTokens: null,
      cacheWriteTokens: null,
      providerId: null,
      modelId: null,
    })])

    expect(lines.join('\n')).toContain('[impl-eng] (engineer)')
    expect(lines.join('\n')).not.toContain('[fork-random]')
  })

  it('does not print worker assistant messages', () => {
    const lines = render([event({
      type: 'message_start',
      forkId: 'fork-random',
      turnId: 't1',
      id: 'm1',
      destination: { kind: 'user' },
    }), event({
      type: 'message_chunk',
      forkId: 'fork-random',
      turnId: 't1',
      id: 'm1',
      text: 'Worker prose should not be interleaved',
    }), event({
      type: 'message_end',
      forkId: 'fork-random',
      turnId: 't1',
      id: 'm1',
    })])

    expect(lines).toEqual([])
  })

  it('prefixes worker tool output with the natural agent id', () => {
    const lines = render([event({
      type: 'agent_created',
      forkId: 'fork-random',
      parentForkId: null,
      agentId: 'impl-eng',
      name: 'Implement kanban app',
      role: 'engineer',
      context: '',
      mode: 'spawn',
      taskId: 'task-1',
      message: 'build it',
    }), event({
      type: 'tool_event',
      forkId: 'fork-random',
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'fileRead',
      event: {
        _tag: 'ToolExecutionEnded',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'fileRead',
        toolKey: 'fileRead',
        result: { _tag: 'Success', output: 'line one\nline two' },
      },
    })])

    expect(lines).toEqual([
      '▶ [impl-eng] (engineer) started · Implement kanban app',
      '  [impl-eng] (engineer) → Read (unknown) · 2 lines',
    ])
    expect(lines.join('\n')).not.toContain('[fork-random]')
  })

  it('does not duplicate worker start when spawn and agent events both arrive', () => {
    const lines = render([event({
      type: 'agent_created',
      forkId: 'fork-random',
      parentForkId: null,
      agentId: 'impl-eng',
      name: 'Implement kanban app',
      role: 'engineer',
      context: '',
      mode: 'spawn',
      taskId: 'task-1',
      message: 'build it',
    }), event({
      type: 'tool_event',
      forkId: null,
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'spawnWorker',
      event: {
        _tag: 'ToolExecutionEnded',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'spawnWorker',
        toolKey: 'spawnWorker',
        result: { _tag: 'Success', output: { agentId: 'impl-eng', title: 'Implement kanban app' } },
      },
    })])

    expect(lines).toEqual(['▶ [impl-eng] (engineer) started · Implement kanban app'])
  })

  it('renders tool parameters from accumulated lifecycle events', () => {
    const lines = render([event({
      type: 'tool_event',
      forkId: null,
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'shell',
      event: {
        _tag: 'ToolExecutionStarted',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'shell',
        toolKey: 'shell',
        input: { command: 'bun create vite kanban --template svelte' },
        cached: false,
      },
    }), event({
      type: 'tool_event',
      forkId: null,
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'shell',
      event: {
        _tag: 'ToolExecutionEnded',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'shell',
        toolKey: 'shell',
        result: { _tag: 'Success', output: { exitCode: 0, stdout: '', stderr: '' } },
      },
    })])

    expect(lines).toEqual(['$ bun create vite kanban --template svelte · exit 0'])
  })

  it('hides internal task plumbing tools', () => {
    const lines = render([event({
      type: 'tool_event',
      forkId: null,
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'createTask',
      event: {
        _tag: 'ToolExecutionEnded',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'createTask',
        toolKey: 'createTask',
        result: { _tag: 'Success', output: { taskId: 'task-1' } },
      },
    })])

    expect(lines).toEqual([])
  })

  it('formats structured errors without object stringification', () => {
    const lines = render([event({
      type: 'tool_event',
      forkId: null,
      turnId: 't1',
      toolCallId: 'tool-1',
      providerToolCallId: 'provider-1',
      toolKey: 'shell',
      event: {
        _tag: 'ToolExecutionEnded',
        toolCallId: 'tool-1',
        providerToolCallId: 'provider-1',
        toolName: 'shell',
        toolKey: 'shell',
        result: { _tag: 'Error', error: { message: 'command failed' } },
      },
    })])

    expect(lines).toEqual(['✗ shell · command failed'])
    expect(lines.join('\n')).not.toContain('[object Object]')
  })
})
