import { describe, it, expect } from 'vitest'
import { Effect } from 'effect'
import { encode, type EncodeConfig } from '../encode'
import { ToolDef } from '../../../tools/tool-def'

const config: EncodeConfig = {
  wireModelName:     'test-model',
  defaultMaxTokens:  4096,
  supportsReasoning: true,
  supportsVision:    true,
}

function run<A>(effect: Effect.Effect<A, unknown>): Promise<A> {
  return Effect.runPromise(effect as Effect.Effect<A, never>)
}

describe('encode', () => {
  it('empty memory → empty messages', async () => {
    const req = await run(encode([], [], {}, config))
    expect(req.messages).toEqual([])
    expect(req.model).toBe('test-model')
    expect(req.stream).toBe(true)
    expect(req.stream_options).toEqual({ include_usage: true })
  })

  it('unknown memory entry → skipped', async () => {
    const req = await run(encode([{ type: 'unknown_type', foo: 'bar' }], [], {}, config))
    expect(req.messages).toEqual([])
  })

  it('session_context → system message (text)', async () => {
    const mem = [{ type: 'session_context', content: [{ type: 'text', text: 'You are helpful.' }] }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages).toHaveLength(1)
    expect(req.messages[0]).toEqual({ role: 'system', content: 'You are helpful.' })
  })

  it('fork_context → system message', async () => {
    const mem = [{ type: 'fork_context', content: [{ type: 'text', text: 'Fork context.' }] }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages[0]).toEqual({ role: 'system', content: 'Fork context.' })
  })

  it('compacted → system message with wrapper', async () => {
    const mem = [{ type: 'compacted', content: [{ type: 'text', text: 'summary' }] }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages[0]).toMatchObject({ role: 'system' })
    expect((req.messages[0] as { content: string }).content).toContain('<compacted>')
    expect((req.messages[0] as { content: string }).content).toContain('summary')
  })

  it('inbox with user_message timeline → user message', async () => {
    const mem = [{
      type:     'inbox',
      results:  [],
      timeline: [{ kind: 'user_message', text: 'Hello!', timestamp: 0, attachments: [] }],
    }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages).toHaveLength(1)
    expect(req.messages[0].role).toBe('user')
    const content = (req.messages[0] as { content: string }).content
    expect(content).toContain('Hello!')
  })

  it('inbox with tool_observation → tool message', async () => {
    const mem = [{
      type: 'inbox',
      results: [{
        kind:  'turn_results',
        items: [{
          kind:       'tool_observation',
          toolCallId: 'call-1-abc',
          tagName:    'read_file',
          content:    [{ type: 'text', text: 'file content' }],
        }],
      }],
      timeline: [],
    }]
    const req = await run(encode(mem, [], {}, config))
    const toolMsg = req.messages.find(m => m.role === 'tool')
    expect(toolMsg).toBeDefined()
    expect((toolMsg as { tool_call_id: string }).tool_call_id).toBe('call-1-abc')
    expect((toolMsg as { content: string }).content).toContain('file content')
  })

  it('inbox with tool_error → tool message with error text', async () => {
    const mem = [{
      type: 'inbox',
      results: [{
        kind:  'turn_results',
        items: [{
          kind:       'tool_error',
          toolCallId: 'call-2-xyz',
          tagName:    'run_shell',
          message:    'command failed',
        }],
      }],
      timeline: [],
    }]
    const req = await run(encode(mem, [], {}, config))
    const toolMsg = req.messages.find(m => m.role === 'tool')
    expect(toolMsg).toBeDefined()
    expect((toolMsg as { tool_call_id: string }).tool_call_id).toBe('call-2-xyz')
    expect((toolMsg as { content: string }).content).toBe('command failed')
  })

  it('assistant_turn with thoughts only → reasoning_content', async () => {
    const mem = [{
      type:  'assistant_turn',
      parts: [{ type: 'thought', id: 't1', level: 'medium', text: 'I am thinking...' }],
    }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages).toHaveLength(1)
    const msg = req.messages[0] as { role: string; reasoning_content: string }
    expect(msg.role).toBe('assistant')
    expect(msg.reasoning_content).toBe('I am thinking...')
    expect((msg as Record<string,unknown>)['content']).toBeUndefined()
    expect((msg as Record<string,unknown>)['tool_calls']).toBeUndefined()
  })

  it('assistant_turn with tool calls only → tool_calls', async () => {
    const mem = [{
      type:  'assistant_turn',
      parts: [{
        type:     'tool_call',
        id:       'call-1-abc',
        toolName: 'read_file',
        input:    { path: '/foo.ts' },
      }],
    }]
    const req = await run(encode(mem, [], {}, config))
    const msg = req.messages[0] as { role: string; tool_calls: unknown[] }
    expect(msg.role).toBe('assistant')
    expect(msg.tool_calls).toHaveLength(1)
    expect((msg.tool_calls[0] as Record<string,unknown>)['id']).toBe('call-1-abc')
    const fn = (msg.tool_calls[0] as Record<string,unknown>)['function'] as Record<string,unknown>
    expect(fn['name']).toBe('read_file')
    expect(JSON.parse(fn['arguments'] as string)).toEqual({ path: '/foo.ts' })
  })

  it('assistant_turn with thoughts + tool calls → reasoning_content + tool_calls', async () => {
    const mem = [{
      type:  'assistant_turn',
      parts: [
        { type: 'thought',   id: 't1', level: 'medium', text: 'Let me think' },
        { type: 'tool_call', id: 'call-1', toolName: 'shell', input: { cmd: 'ls' } },
      ],
    }]
    const req = await run(encode(mem, [], {}, config))
    const msg = req.messages[0] as Record<string, unknown>
    expect(msg['reasoning_content']).toBe('Let me think')
    expect((msg['tool_calls'] as unknown[]).length).toBe(1)
  })

  it('assistant_turn with messages + tool calls → content + tool_calls', async () => {
    const mem = [{
      type:  'assistant_turn',
      parts: [
        { type: 'message',   id: 'm1', text: 'I will do this.' },
        { type: 'tool_call', id: 'call-1', toolName: 'shell', input: {} },
      ],
    }]
    const req = await run(encode(mem, [], {}, config))
    const msg = req.messages[0] as Record<string, unknown>
    expect(msg['content']).toBe('I will do this.')
    expect((msg['tool_calls'] as unknown[]).length).toBe(1)
  })

  it('assistant_turn with thoughts + messages + tool calls → all three fields', async () => {
    const mem = [{
      type:  'assistant_turn',
      parts: [
        { type: 'thought',   id: 't1', level: 'high', text: 'Deep thought' },
        { type: 'message',   id: 'm1', text: 'Here is my answer.' },
        { type: 'tool_call', id: 'call-1', toolName: 'read', input: { path: 'x' } },
      ],
    }]
    const req = await run(encode(mem, [], {}, config))
    const msg = req.messages[0] as Record<string, unknown>
    expect(msg['reasoning_content']).toBe('Deep thought')
    expect(msg['content']).toBe('Here is my answer.')
    expect((msg['tool_calls'] as unknown[]).length).toBe(1)
  })

  it('assistant_turn empty parts → empty content', async () => {
    const mem = [{ type: 'assistant_turn', parts: [] }]
    const req = await run(encode(mem, [], {}, config))
    expect(req.messages[0]).toEqual({ role: 'assistant', content: '' })
  })

  it('supportsReasoning=false → no reasoning_content field', async () => {
    const noReasonConfig = { ...config, supportsReasoning: false }
    const mem = [{
      type:  'assistant_turn',
      parts: [{ type: 'thought', id: 't1', level: 'medium', text: 'ignored' }],
    }]
    const req = await run(encode(mem, [], {}, noReasonConfig))
    const msg = req.messages[0] as Record<string, unknown>
    expect(msg['reasoning_content']).toBeUndefined()
  })

  it('tools array → correct function schema', async () => {
    const tools = [
      new ToolDef({ name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } } }),
    ]
    const req = await run(encode([], tools, {}, config))
    expect(req.tools).toHaveLength(1)
    const tool = req.tools![0]
    expect(tool.type).toBe('function')
    expect(tool.function.name).toBe('read_file')
    expect(tool.function.description).toBe('Read a file')
    expect(tool.function.parameters).toEqual({ type: 'object', properties: { path: { type: 'string' } } })
  })

  it('no tools → tools field absent', async () => {
    const req = await run(encode([], [], {}, config))
    expect(req.tools).toBeUndefined()
  })

  it('maxTokens option respected', async () => {
    const req = await run(encode([], [], { maxTokens: 1234 }, config))
    expect(req.max_tokens).toBe(1234)
  })

  it('defaultMaxTokens used when no maxTokens', async () => {
    const req = await run(encode([], [], {}, { ...config, defaultMaxTokens: 8192 }))
    expect(req.max_tokens).toBe(8192)
  })
})
