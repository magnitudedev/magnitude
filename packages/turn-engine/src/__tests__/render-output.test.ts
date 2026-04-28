import { describe, it, expect } from 'vitest'
import { renderToolOutput } from '../render-output'
import type { ToolResult } from '../types'

describe('renderToolOutput', () => {
  // -------------------------------------------------------------------------
  // Error / Rejected / Interrupted
  // -------------------------------------------------------------------------

  it('renders Error result', () => {
    const result: ToolResult = { _tag: 'Error', error: 'something went wrong' }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '<error>something went wrong</error>' })
  })

  it('renders Rejected result', () => {
    const result: ToolResult = { _tag: 'Rejected', rejection: 'tool not allowed' }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '<rejected>tool not allowed</rejected>' })
  })

  it('renders Interrupted result', () => {
    const result: ToolResult = { _tag: 'Interrupted' }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '<interrupted/>' })
  })

  // -------------------------------------------------------------------------
  // Success — scalar outputs
  // -------------------------------------------------------------------------

  it('renders Success with string scalar output', () => {
    const result: ToolResult = { _tag: 'Success', output: 'hello world' }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: 'hello world' })
  })

  it('renders Success with multiline string scalar', () => {
    const result: ToolResult = { _tag: 'Success', output: 'line1\nline2\nline3' }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: 'line1\nline2\nline3' })
  })

  it('renders Success with number scalar output', () => {
    const result: ToolResult = { _tag: 'Success', output: 42 }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '42' })
  })

  it('renders Success with boolean scalar output', () => {
    const result: ToolResult = { _tag: 'Success', output: true }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: 'true' })
  })

  it('renders Success with null scalar output', () => {
    const result: ToolResult = { _tag: 'Success', output: null }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: 'null' })
  })

  it('renders Success with undefined output as (no output)', () => {
    const result: ToolResult = { _tag: 'Success', output: undefined }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '(no output)' })
  })

  // -------------------------------------------------------------------------
  // Success — array output
  // -------------------------------------------------------------------------

  it('renders Success with array output as JSON', () => {
    const result: ToolResult = { _tag: 'Success', output: [1, 2, 3] }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '[1,2,3]' })
  })

  // -------------------------------------------------------------------------
  // Success — object output
  // -------------------------------------------------------------------------

  it('renders Success with flat object output', () => {
    const result: ToolResult = {
      _tag: 'Success',
      output: { mode: 'completed', exitCode: 0, stdout: 'hello\nworld', stderr: '' },
    }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({
      type: 'text',
      text: '<mode>completed</mode>\n<exitCode>0</exitCode>\n<stdout>\nhello\nworld\n</stdout>\n<stderr></stderr>',
    })
  })

  it('renders Success with object containing nested non-scalar', () => {
    const result: ToolResult = {
      _tag: 'Success',
      output: { items: [{ path: 'a.ts', name: 'a' }], totalMatches: 1 },
    }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({
      type: 'text',
      text: '<items>[{"path":"a.ts","name":"a"}]</items>\n<totalMatches>1</totalMatches>',
    })
  })

  it('renders Success with object containing null field as empty tag', () => {
    const result: ToolResult = { _tag: 'Success', output: { value: null, count: 5 } }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({
      type: 'text',
      text: '<value>null</value>\n<count>5</count>',
    })
  })

  it('omits undefined fields from object output', () => {
    const result: ToolResult = { _tag: 'Success', output: { name: 'foo', extra: undefined } }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({ type: 'text', text: '<name>foo</name>' })
  })

  // -------------------------------------------------------------------------
  // Success — image output (ContentPart image)
  // -------------------------------------------------------------------------

  it('renders Success with image-shaped output as image ContentPart', () => {
    const output = {
      type: 'image' as const,
      base64: 'abc123',
      mediaType: 'image/png' as const,
      width: 100,
      height: 100,
    }
    const result: ToolResult = { _tag: 'Success', output }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({
      type: 'image',
      base64: 'abc123',
      mediaType: 'image/png',
      width: 100,
      height: 100,
    })
  })

  // -------------------------------------------------------------------------
  // Rejected — non-string rejection
  // -------------------------------------------------------------------------

  it('renders Rejected with object rejection via JSON', () => {
    const result: ToolResult = { _tag: 'Rejected', rejection: { code: 403, reason: 'forbidden' } }
    const parts = renderToolOutput(result)
    expect(parts).toHaveLength(1)
    expect(parts[0]).toEqual({
      type: 'text',
      text: '<rejected>{"code":403,"reason":"forbidden"}</rejected>',
    })
  })
})
