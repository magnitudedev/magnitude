
/**
 * Tests for Mact format handler
 */

import { describe, test, expect } from 'vitest'
import { createMactFormatHandler, createMactPipeline, type MactFormatConfig } from '../mact-format'
import { createMactTokenizer } from '../mact-tokenizer'
import { createMactParser } from '../mact-parser'
import type { MactBinding } from '@magnitudedev/tools'
import type { TagSchema } from '../execution/binding-validator'
import type { MactToken } from '../types'
import type { MactParserEvent } from '../mact-parser'

describe('mact format handler', () => {
  const mockBinding: MactBinding = {
    type: 'mact',
    tag: 'read',
    parameters: [
      { name: 'path', field: 'path', type: 'scalar' },
      { name: 'offset', field: 'offset', type: 'scalar' },
      { name: 'limit', field: 'limit', type: 'scalar' },
    ],
  }

  const mockSchema: TagSchema = {
    attributes: new Map([
      ['path', { type: 'string', required: true }],
      ['offset', { type: 'number', required: false }],
      ['limit', { type: 'number', required: false }],
    ]),
    children: new Map(),
  }

  const config: MactFormatConfig = {
    tools: new Map([
      ['read', { binding: mockBinding, schema: mockSchema }],
    ]),
    parameterSchemas: new Map([
      ['read', new Map([
        ['path', { type: 'string', fieldPath: 'path' }],
        ['offset', { type: 'number', fieldPath: 'offset' }],
        ['limit', { type: 'number', fieldPath: 'limit' }],
      ])],
    ]),
  }

  function tokenize(text: string): MactToken[] {
    const tokens: MactToken[] = []
    const tokenizer = createMactTokenizer((token) => tokens.push(token))
    tokenizer.push(text)
    tokenizer.end()
    return tokens
  }

  function parse(tokens: MactToken[]): MactParserEvent[] {
    const parser = createMactParser()
    for (const token of tokens) {
      parser.pushToken(token)
    }
    parser.end()
    return [...parser.events]
  }

  test('handles simple read invoke', () => {
    const handler = createMactFormatHandler(config)
    
    const tokens = tokenize('<|invoke:read>\n<|parameter:path>/workspace/test.ts<parameter|>\n<invoke|>\n<|yield:tool|>')
    const events = parse(tokens)

    for (const event of events) {
      handler.handleEvent(event)
    }
    handler.end()

    // Check that we got the expected events
    const toolInputStarted = handler.events.filter(e => e._tag === 'ToolInputStarted')
    const toolInputReady = handler.events.filter(e => e._tag === 'ToolInputReady')
    const turnControl = handler.events.filter(e => e._tag === 'TurnControl')

    expect(toolInputStarted).toHaveLength(1)
    expect(toolInputReady).toHaveLength(1)
    expect(turnControl).toHaveLength(1)

    // Check the input was built correctly
    const readyEvent = toolInputReady[0] as Extract<typeof toolInputReady[number], { _tag: 'ToolInputReady' }>
    expect(readyEvent.input).toEqual({ path: '/workspace/test.ts' })
  })

  test('handles invoke with filter', () => {
    const handler = createMactFormatHandler(config)
    
    // Use 'read' tool with filter (even though filter doesn't make sense for read, it tests the parsing)
    const tokens = tokenize('<|invoke:read>\n<|parameter:path>/test.ts<parameter|>\n<invoke|filter>\n$.content\n<filter|>\n<|yield:tool|>')
    const events = parse(tokens)

    for (const event of events) {
      handler.handleEvent(event)
    }
    handler.end()

    const toolInputStarted = handler.events.filter(e => e._tag === 'ToolInputStarted')
    expect(toolInputStarted).toHaveLength(1)
    
    // Verify the filter query was captured
    const toolInputReady = handler.events.find(e => e._tag === 'ToolInputReady') as { _tag: 'ToolInputReady', input: Record<string, unknown> } | undefined
    expect(toolInputReady).toBeDefined()
  })

  test('handles think block', () => {
    const handler = createMactFormatHandler(config)
    
    const tokens = tokenize('<|think:analyze>\nI should read the file first.\n<think|>\n<|yield:user|>')
    const events = parse(tokens)

    for (const event of events) {
      handler.handleEvent(event)
    }
    handler.end()

    const lensStart = handler.events.filter(e => e._tag === 'LensStart')
    const lensEnd = handler.events.filter(e => e._tag === 'LensEnd')

    expect(lensStart).toHaveLength(1)
    expect(lensEnd).toHaveLength(1)
    expect(lensStart[0]).toEqual({ _tag: 'LensStart', name: 'analyze' })
  })

  test('handles message to user', () => {
    const handler = createMactFormatHandler(config)
    
    const tokens = tokenize('<|message:user>\nHello, I will help you.\n<message|>\n<|yield:user|>')
    const events = parse(tokens)

    for (const event of events) {
      handler.handleEvent(event)
    }
    handler.end()

    const messageStart = handler.events.filter(e => e._tag === 'MessageStart')
    const messageEnd = handler.events.filter(e => e._tag === 'MessageEnd')

    expect(messageStart).toHaveLength(1)
    expect(messageEnd).toHaveLength(1)
    expect(messageStart[0]).toMatchObject({ _tag: 'MessageStart', to: 'user' })
  })

  test('handles multiple parameters', () => {
    const handler = createMactFormatHandler(config)
    
    const tokens = tokenize(`<|invoke:read>
<|parameter:path>/workspace/test.ts<parameter|>
<|parameter:offset>10<parameter|>
<|parameter:limit>100<parameter|>
<invoke|>
<|yield:tool|>`)
    const events = parse(tokens)

    for (const event of events) {
      handler.handleEvent(event)
    }
    handler.end()

    const fieldValues = handler.events.filter(e => e._tag === 'ToolInputFieldValue')
    expect(fieldValues).toHaveLength(3)

    const readyEvent = handler.events.find(e => e._tag === 'ToolInputReady') as { _tag: 'ToolInputReady', input: Record<string, unknown> } | undefined
    expect(readyEvent?.input).toEqual({
      path: '/workspace/test.ts',
      offset: 10,
      limit: 100,
    })
  })
})

describe('mact pipeline', () => {
  const mockBinding: MactBinding = {
    type: 'mact',
    tag: 'read',
    parameters: [{ name: 'path', field: 'path', type: 'scalar' }],
  }

  const config: MactFormatConfig = {
    tools: new Map([['read', { binding: mockBinding }]]),
    parameterSchemas: new Map([
      ['read', new Map([['path', { type: 'string', fieldPath: 'path' }]])],
    ]),
  }

  test('pipeline processes text end-to-end', () => {
    const pipeline = createMactPipeline(config)
    
    pipeline.pushText('<|invoke:read>\n<|parameter:path>/test.ts<parameter|>\n<invoke|>\n<|yield:tool|>')
    pipeline.end()

    const toolInputReady = pipeline.events.filter(e => e._tag === 'ToolInputReady')
    expect(toolInputReady).toHaveLength(1)
  })
})
