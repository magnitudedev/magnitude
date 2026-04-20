
import { describe, it, expect } from 'vitest'
import { createMactTokenizer } from '../mact-tokenizer'
import { createMactParser } from '../mact-parser'

describe('Mact Parser', () => {
  function parse(input: string) {
    const parser = createMactParser()
    const tokenizer = createMactTokenizer((token) => {
      parser.pushToken(token)
    })
    
    tokenizer.push(input)
    tokenizer.end()
    parser.end()
    
    return parser.events
  }

  describe('basic prose', () => {
    it('parses plain text as prose', () => {
      const events = parse('Hello world')
      // Tokenizer coalesces content, so we get 2 events (chunk + end)
      expect(events).toHaveLength(2)
      expect(events[0]).toMatchObject({ _tag: 'ProseChunk', text: 'Hello world' })
      expect(events[1]).toMatchObject({ _tag: 'ProseEnd', content: 'Hello world' })
    })

    it('handles multiple chunks', () => {
      const events = parse('Hello\n\nWorld')
      // Content is coalesced, so we get one chunk
      expect(events.some(e => e._tag === 'ProseChunk')).toBe(true)
      expect(events.some(e => e._tag === 'ProseEnd')).toBe(true)
    })
  })

  describe('think blocks', () => {
    it('parses think with lens name', () => {
      const events = parse('<|think:analyze>\nThinking content\n<think|>')
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'LensStart',
        name: 'analyze'
      }))
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'LensChunk',
        text: '\nThinking content\n'
      }))
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'LensEnd',
        name: 'analyze',
        content: '\nThinking content\n'
      }))
    })

    it('defaults to analyze lens', () => {
      const events = parse('<|think>\nContent\n<think|>')
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'LensStart',
        name: 'analyze'
      }))
    })
  })

  describe('message blocks', () => {
    it('parses message without recipient', () => {
      const events = parse('<|message>\nHello user\n<message|>')
      
      const start = events.find(e => e._tag === 'MessageStart')
      expect(start).toMatchObject({ to: null })
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MessageChunk',
        text: '\nHello user\n'
      }))
    })

    it('parses message with recipient', () => {
      const events = parse('<|message:parent>\nHello parent\n<message|>')
      
      const start = events.find(e => e._tag === 'MessageStart')
      expect(start).toMatchObject({ to: 'parent' })
    })
  })

  describe('invoke blocks', () => {
    it('parses simple invoke', () => {
      const events = parse('<|invoke:read>\n<|parameter:path>/test.txt<parameter|>\n<invoke|>')
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactInvokeStarted',
        toolTag: 'read',
        toolName: 'read',
        group: 'default'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactParameterStarted',
        parameterName: 'path'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactParameterComplete',
        parameterName: 'path',
        value: '/test.txt'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactInvokeComplete',
        hasFilter: false
      }))
    })

    it('parses invoke with group', () => {
      const events = parse('<|invoke:fs:read>\n<invoke|>')
      
      const invoke = events.find(e => e._tag === 'MactInvokeStarted')
      expect(invoke).toMatchObject({
        toolTag: 'fs:read',
        toolName: 'read',
        group: 'fs'
      })
    })

    it('accumulates multiple parameters', () => {
      const events = parse(`<|invoke:write>
<|parameter:path>/test.txt<parameter|>
<|parameter:content>Hello World<parameter|>
<invoke|>`)
      
      const params = events.filter(e => e._tag === 'MactParameterComplete')
      expect(params).toHaveLength(2)
      expect(params[0]).toMatchObject({ parameterName: 'path', value: '/test.txt' })
      expect(params[1]).toMatchObject({ parameterName: 'content', value: 'Hello World' })
    })
  })

  describe('filter blocks', () => {
    it('parses piped filter on invoke close', () => {
      const events = parse(`<|invoke:shell>
<|parameter:command>ls -la<parameter|>
<invoke|filter>
.stdout
<filter|>`)
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactFilterStarted',
        filterType: 'filter'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactFilterChunk',
        text: '\n.stdout\n'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactFilterComplete',
        query: '\n.stdout\n'
      }))
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'MactInvokeComplete',
        hasFilter: true
      }))
    })
  })

  describe('yield', () => {
    it('parses yield:user', () => {
      const events = parse('<|yield:user|>')
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'TurnControl',
        target: 'user'
      }))
    })

    it('parses yield:tool', () => {
      const events = parse('<|yield:tool|>')
      
      expect(events).toContainEqual(expect.objectContaining({
        _tag: 'TurnControl',
        target: 'tool'
      }))
    })
  })

  describe('complex scenarios', () => {
    it('parses full turn with think, message, and invoke', () => {
      const input = `<|think:analyze>
The user wants to read a file.
<think|>

<|message>
Let me read that file.
<message|>

<|invoke:read>
<|parameter:path>/workspace/auth.js<parameter|>
<invoke|>

<|yield:tool|>`
      
      const events = parse(input)
      
      // Should have think, message, invoke, and turn control
      expect(events.some(e => e._tag === 'LensStart')).toBe(true)
      expect(events.some(e => e._tag === 'MessageStart')).toBe(true)
      expect(events.some(e => e._tag === 'MactInvokeStarted')).toBe(true)
      expect(events.some(e => e._tag === 'TurnControl')).toBe(true)
    })
  })
})
