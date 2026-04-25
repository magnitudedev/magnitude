/**
 * Parser tests for magnitude:escape tag.
 * Verifies that escape blocks strip open/close tags and pass inner content as raw text.
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { Effect } from 'effect'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { RegisteredTool, TurnEngineEvent } from '../types'

// ---------------------------------------------------------------------------
// Test tool setup
// ---------------------------------------------------------------------------

const shellTool = defineTool({
  name: 'shell',
  label: 'Shell',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: Schema.String,
  execute: (_input) => Effect.succeed('ok'),
})

const shellRegistered: RegisteredTool = {
  tool: shellTool,
  tagName: 'shell',
  groupName: 'default',
}

const tools = new Map<string, RegisteredTool>([
  ['shell', shellRegistered],
])

function parse(input: string, customTools = tools) {
  const p = createParser({ tools: customTools })
  const knownToolTags = new Set(customTools.keys())
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    knownToolTags,
  )
  tokenizer.push(input + '\n')
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd]
}

function eventsOfType<T extends TurnEngineEvent['_tag']>(
  events: TurnEngineEvent[],
  tag: T,
): Extract<TurnEngineEvent, { _tag: T }>[] {
  return events.filter(e => e._tag === tag) as any
}

describe('magnitude:escape parser tests', () => {

  describe('top-level escape as prose', () => {
    it('escape block content becomes prose', () => {
      const events = parse('<magnitude:escape>raw content</magnitude:escape>\n<magnitude:yield_user/>')
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const allProse = proseChunks.map(e => e.text).join('')
      expect(allProse).toContain('raw content')
      // Escape open/close tags should NOT appear in prose
      expect(allProse).not.toContain('<magnitude:escape>')
      expect(allProse).not.toContain('</magnitude:escape>')
    })

    it('escape tags are stripped, inner magnitude tags become literal text', () => {
      const events = parse('<magnitude:escape><magnitude:invoke tool="shell"><magnitude:parameter name="cmd">ls</magnitude:parameter></magnitude:invoke></magnitude:escape>\n<magnitude:yield_user/>')
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const allProse = proseChunks.map(e => e.text).join('')
      // Inner tags should appear as literal text in prose
      expect(allProse).toContain('<magnitude:invoke tool="shell">')
      expect(allProse).toContain('</magnitude:invoke>')
      // Should NOT trigger any invoke events
      const invokeEvents = eventsOfType(events, 'ToolInputStarted')
      expect(invokeEvents).toHaveLength(0)
    })

    it('escape block between structural elements', () => {
      const events = parse('<magnitude:message to="user">hello</magnitude:message>\n<magnitude:escape>escaped</magnitude:escape>\n<magnitude:yield_user/>')
      const msgChunks = eventsOfType(events, 'MessageChunk')
      const msgText = msgChunks.map(e => e.text).join('')
      expect(msgText).toContain('hello')
      // Escaped content should be prose, not part of message
      expect(msgText).not.toContain('escaped')
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const allProse = proseChunks.map(e => e.text).join('')
      expect(allProse).toContain('escaped')
    })

    it('empty escape block produces no additional content', () => {
      const events = parse('<magnitude:escape></magnitude:escape>\n<magnitude:yield_user/>')
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const proseText = proseChunks.map(e => e.text).join('')
      expect(proseText).not.toContain('<magnitude:escape>')
      expect(proseText).not.toContain('</magnitude:escape>')
    })

  })

  describe('escape inside message body', () => {
    it('escape content becomes part of message text', () => {
      const events = parse('<magnitude:message to="user">before <magnitude:escape>middle</magnitude:escape> after</magnitude:message>\n<magnitude:yield_user/>')
      const msgChunks = eventsOfType(events, 'MessageChunk')
      const msgText = msgChunks.map(e => e.text).join('')
      expect(msgText).toContain('before ')
      expect(msgText).toContain('middle')
      expect(msgText).toContain(' after')
      expect(msgText).not.toContain('<magnitude:escape>')
      expect(msgText).not.toContain('</magnitude:escape>')
    })

    it('escape containing parent close tag does not close message', () => {
      const events = parse('<magnitude:message to="user">text <magnitude:escape></magnitude:message></magnitude:escape> more</magnitude:message>\n<magnitude:yield_user/>')
      const msgStarts = eventsOfType(events, 'MessageStart')
      const msgEnds = eventsOfType(events, 'MessageEnd')
      expect(msgStarts).toHaveLength(1)
      expect(msgEnds).toHaveLength(1)
      const msgChunks = eventsOfType(events, 'MessageChunk')
      const msgText = msgChunks.map(e => e.text).join('')
      expect(msgText).toContain('</magnitude:message>')
      expect(msgText).toContain(' more')
    })

    it('escape containing invoke tags does not trigger invoke', () => {
      const events = parse('<magnitude:message to="user"><magnitude:escape><magnitude:invoke tool="shell"><magnitude:parameter name="command">ls</magnitude:parameter></magnitude:invoke></magnitude:escape></magnitude:message>\n<magnitude:yield_user/>')
      const invokeEvents = eventsOfType(events, 'ToolInputStarted')
      expect(invokeEvents).toHaveLength(0)
      const msgChunks = eventsOfType(events, 'MessageChunk')
      const msgText = msgChunks.map(e => e.text).join('')
      expect(msgText).toContain('<magnitude:invoke tool="shell">')
    })

    it('multiple escape blocks in message body', () => {
      const events = parse('<magnitude:message to="user"><magnitude:escape>first</magnitude:escape> mid <magnitude:escape>second</magnitude:escape></magnitude:message>\n<magnitude:yield_user/>')
      const msgChunks = eventsOfType(events, 'MessageChunk')
      const msgText = msgChunks.map(e => e.text).join('')
      expect(msgText).toContain('first')
      expect(msgText).toContain(' mid ')
      expect(msgText).toContain('second')
    })

  })

  describe('escape inside reason body', () => {
    it('escape content becomes part of reason text', () => {
      const events = parse('<magnitude:reason about="turn">before <magnitude:escape>raw</magnitude:escape> after</magnitude:reason>\n<magnitude:message to="user">hi</magnitude:message>\n<magnitude:yield_user/>')
      const lensChunks = eventsOfType(events, 'LensChunk')
      const lensText = lensChunks.map(e => e.text).join('')
      expect(lensText).toContain('before ')
      expect(lensText).toContain('raw')
      expect(lensText).toContain(' after')
    })

    it('escape containing parent close tag does not close reason', () => {
      const events = parse('<magnitude:reason about="turn">text <magnitude:escape></magnitude:reason></magnitude:escape> more</magnitude:reason>\n<magnitude:message to="user">hi</magnitude:message>\n<magnitude:yield_user/>')
      const lensStarts = eventsOfType(events, 'LensStart')
      const lensEnds = eventsOfType(events, 'LensEnd')
      expect(lensStarts).toHaveLength(1)
      expect(lensEnds).toHaveLength(1)
      const lensChunks = eventsOfType(events, 'LensChunk')
      const lensText = lensChunks.map(e => e.text).join('')
      expect(lensText).toContain('</magnitude:reason>')
    })

  })

  describe('escape inside parameter body', () => {
    it('escape content becomes part of parameter value', () => {
      const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">before <magnitude:escape>raw</magnitude:escape> after</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_invoke/>')
      const paramChunks = eventsOfType(events, 'ToolInputFieldChunk')
      const paramText = paramChunks.map((e: any) => e.delta).join('')
      expect(paramText).toContain('before ')
      expect(paramText).toContain('raw')
      expect(paramText).toContain(' after')
    })

    it('escape containing parent close tag does not close parameter', () => {
      const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">cmd <magnitude:escape></magnitude:parameter></magnitude:escape> rest</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_invoke/>')
      const paramCompletes = eventsOfType(events, 'ToolInputFieldComplete')
      expect(paramCompletes).toHaveLength(1)
      const value = (paramCompletes[0] as any).value
      expect(value).toContain('</magnitude:parameter>')
      expect(value).toContain(' rest')
    })

    it('escape containing invoke close tag does not close invoke', () => {
      const events = parse('<magnitude:invoke tool="shell">\n<magnitude:parameter name="command"><magnitude:escape></magnitude:invoke></magnitude:escape></magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_invoke/>')
      const invokeStarts = eventsOfType(events, 'ToolInputStarted')
      const invokeCompletes = events.filter(e => e._tag === 'ToolInputReady')
      expect(invokeStarts).toHaveLength(1)
      expect(invokeCompletes).toHaveLength(1)
    })

  })

  describe('no nesting', () => {
    it('inner escape open tag is treated as content', () => {
      const events = parse('<magnitude:escape><magnitude:escape>still escaped</magnitude:escape>\n<magnitude:yield_user/>')
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const proseText = proseChunks.map(e => e.text).join('')
      expect(proseText).toContain('<magnitude:escape>')
      expect(proseText).toContain('still escaped')
    })

  })

  describe('streaming behavior', () => {
    it('escape works when input is pushed character by character', () => {
      const input = '<magnitude:escape>streamed</magnitude:escape>\n<magnitude:yield_user/>'
      const p = createParser({ tools })
      const knownToolTags = new Set(tools.keys())
      const tokenizer = createTokenizer(
        (token) => p.pushToken(token),
        knownToolTags,
      )
      // Push one character at a time
      for (const ch of input) {
        tokenizer.push(ch)
      }
      tokenizer.push('\n')
      const fromPush = p.drain()
      tokenizer.end()
      p.end()
      const fromEnd = p.drain()
      const events = [...fromPush, ...fromEnd]
      const proseChunks = eventsOfType(events, 'ProseChunk')
      const proseText = proseChunks.map(e => e.text).join('')
      expect(proseText).toContain('streamed')
      expect(proseText).not.toContain('<magnitude:escape>')
      expect(proseText).not.toContain('</magnitude:escape>')
    })

  })
})
