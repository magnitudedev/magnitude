/**
 * Comprehensive close tag behavior spec tests.
 *
 * Tests every permutation of:
 * - Frame types: message, think, parameter (inside invoke)
 * - Close tag names: think, message, invoke, parameter, filter, foo, div, skill-name
 * - Close tag variants: <name|> (canonical), </name|> (slash+pipe), </name> (slash), <name> (bare)
 *
 * Expected behavior per spec:
 * - A close tag is structural ONLY when its name matches the currently open frame
 * - When it doesn't match → exact characters preserved as literal text, NO transformation
 */

import { describe, expect, it } from 'vitest'
import { createTokenizer } from '../tokenizer'
import { createParser } from '../parser/index'
import type { TurnEngineEvent, RegisteredTool } from '../types'
import { Schema } from 'effect'
import { defineTool } from '@magnitudedev/tools'

// ---------------------------------------------------------------------------
// Tool setup for parameter frame tests
// ---------------------------------------------------------------------------

const shellTool = defineTool({
  name: 'shell',
  label: 'Shell',
  group: 'fs',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: Schema.Struct({ stdout: Schema.String }),
  execute: async () => ({ stdout: '' }),
})

function makeTools(): ReadonlyMap<string, RegisteredTool> {
  return new Map([
    ['shell', { tool: shellTool, tagName: 'shell', groupName: 'fs' }],
  ])
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function parseInput(input: string): TurnEngineEvent[] {
  const p = createParser({ tools: new Map() })
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    new Set(),
  )
  tokenizer.push(input)
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd] as TurnEngineEvent[]
}

function parseInputWithTools(input: string): TurnEngineEvent[] {
  const p = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => p.pushToken(token),
    new Set(['shell']),
  )
  tokenizer.push(input)
  const fromPush = p.drain()
  tokenizer.end()
  p.end()
  const fromEnd = p.drain()
  return [...fromPush, ...fromEnd] as TurnEngineEvent[]
}

/** Collect all text from MessageChunk events */
function messageText(events: TurnEngineEvent[]): string {
  return events
    .filter((e): e is Extract<TurnEngineEvent, { _tag: 'MessageChunk' }> => e._tag === 'MessageChunk')
    .map(e => e.text)
    .join('')
}

/** Collect all text from LensChunk events */
function lensText(events: TurnEngineEvent[]): string {
  return events
    .filter((e): e is Extract<TurnEngineEvent, { _tag: 'LensChunk' }> => e._tag === 'LensChunk')
    .map(e => e.text)
    .join('')
}

/** Collect all text from ToolInputFieldChunk events (parameter content in runtime events) */
function paramText(events: TurnEngineEvent[]): string {
  return events
    .filter((e): e is Extract<TurnEngineEvent, { _tag: 'ToolInputFieldChunk' }> => e._tag === 'ToolInputFieldChunk')
    .map(e => (e as any).delta)
    .join('')
}

function hasEvent(events: TurnEngineEvent[], tag: string): boolean {
  return events.some(e => e._tag === tag)
}

/** Wrap content in a message frame */
function inMessage(content: string): string {
  return `<|message:user>\n${content}\n<message|>`
}

/** Wrap content in a think frame */
function inThink(content: string): string {
  return `<|think:test>\n${content}\n<think|>`
}

/** Wrap content in a parameter inside an invoke frame */
function inParameter(content: string): string {
  return `<|invoke:shell>\n<|parameter:command>${content}<parameter|>\n<invoke|>`
}

// ---------------------------------------------------------------------------
// SECTION 1: Made-up tags in real frames
// All 4 variants of <foo>, <div>, <skill-name> should be literal text in any frame
// ---------------------------------------------------------------------------

describe('Close tag spec: made-up tags in real frames', () => {
  const madeUpNames = ['foo', 'div', 'skill-name']
  const variants = (name: string) => [
    { variant: `<${name}|>`,   label: 'canonical' },
    { variant: `</${name}|>`,  label: 'slash+pipe' },
    { variant: `</${name}>`,   label: 'slash' },
    { variant: `<${name}>`,    label: 'bare' },
  ]

  for (const name of madeUpNames) {
    for (const { variant, label } of variants(name)) {
      it(`message frame: ${variant} (${label}) → literal text preserved exactly`, () => {
        const events = parseInput(inMessage(`before ${variant} after`))
        const text = messageText(events)
        expect(text).toContain(variant)
      })

      it(`think frame: ${variant} (${label}) → literal text preserved exactly`, () => {
        const events = parseInput(inThink(`before ${variant} after`))
        const text = lensText(events)
        expect(text).toContain(variant)
      })

      it(`parameter frame: ${variant} (${label}) → literal text preserved exactly`, () => {
        const events = parseInputWithTools(inParameter(`before ${variant} after`))
        const text = paramText(events)
        expect(text).toContain(variant)
      })
    }
  }
})

// ---------------------------------------------------------------------------
// SECTION 2: Real tags in OTHER (non-matching) real frames
// e.g., <parameter> inside a message → literal text
// ---------------------------------------------------------------------------

describe('Close tag spec: real tags in non-matching frames', () => {
  type FrameSpec = {
    name: string
    wrap: (content: string) => string
    textFn: (events: TurnEngineEvent[]) => string
  }

  const frames: FrameSpec[] = [
    { name: 'message', wrap: inMessage, textFn: messageText },
    { name: 'think', wrap: inThink, textFn: lensText },
  ]

  // Real MACT tag names (excluding the frame's own name)
  const allRealTags = ['think', 'message', 'invoke', 'parameter', 'filter']

  const variants = (name: string) => [
    { variant: `<${name}|>`,   label: 'canonical' },
    { variant: `</${name}|>`,  label: 'slash+pipe' },
    { variant: `</${name}>`,   label: 'slash' },
    { variant: `<${name}>`,    label: 'bare' },
  ]

  for (const frame of frames) {
    const otherTags = allRealTags.filter(t => t !== frame.name)
    for (const tagName of otherTags) {
      for (const { variant, label } of variants(tagName)) {
        it(`${frame.name} frame: ${variant} (${label}) → literal text preserved exactly`, () => {
          const events = parseInput(frame.wrap(`before ${variant} after`))
          const text = frame.textFn(events)
          expect(text).toContain(variant)
        })
      }
    }
  }

  // Parameter frame: all real tags except 'parameter' should be literal
  const paramOtherTags = allRealTags.filter(t => t !== 'parameter')
  for (const tagName of paramOtherTags) {
    for (const { variant, label } of variants(tagName)) {
      it(`parameter frame: ${variant} (${label}) → literal text preserved exactly`, () => {
        const events = parseInputWithTools(inParameter(`before ${variant} after`))
        const text = paramText(events)
        expect(text).toContain(variant)
      })
    }
  }
})

// ---------------------------------------------------------------------------
// SECTION 3: Real tags in MATCHING frames → structural close
// ---------------------------------------------------------------------------

describe('Close tag spec: real tags in matching frames (structural close)', () => {
  it('message frame: <message|> (canonical) → structural close (MessageEnd emitted)', () => {
    const events = parseInput('<|message:user>\nhello\n<message|>')
    expect(hasEvent(events, 'MessageEnd')).toBe(true)
    expect(messageText(events)).toContain('hello')
  })

  it('message frame: </message|> (slash+pipe) → structural close (MessageEnd emitted)', () => {
    const events = parseInput('<|message:user>\nhello\n</message|>')
    expect(hasEvent(events, 'MessageEnd')).toBe(true)
  })

  it('message frame: </message> (slash) → structural close (MessageEnd emitted)', () => {
    const events = parseInput('<|message:user>\nhello\n</message>')
    expect(hasEvent(events, 'MessageEnd')).toBe(true)
  })

  it('message frame: <message> (bare) → structural close (MessageEnd emitted)', () => {
    const events = parseInput('<|message:user>\nhello\n<message>')
    expect(hasEvent(events, 'MessageEnd')).toBe(true)
  })

  it('think frame: <think|> (canonical) → structural close (LensEnd emitted)', () => {
    const events = parseInput('<|think:test>\nhello\n<think|>')
    expect(hasEvent(events, 'LensEnd')).toBe(true)
    expect(lensText(events)).toContain('hello')
  })

  it('think frame: </think|> (slash+pipe) → structural close (LensEnd emitted)', () => {
    const events = parseInput('<|think:test>\nhello\n</think|>')
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('think frame: </think> (slash) → structural close (LensEnd emitted)', () => {
    const events = parseInput('<|think:test>\nhello\n</think>')
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('think frame: <think> (bare) → structural close (LensEnd emitted)', () => {
    const events = parseInput('<|think:test>\nhello\n<think>')
    expect(hasEvent(events, 'LensEnd')).toBe(true)
  })

  it('parameter frame: <parameter|> (canonical) → structural close (ToolInputReady emitted)', () => {
    const events = parseInputWithTools('<|invoke:shell>\n<|parameter:command>hello<parameter|>\n<invoke|>')
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
    expect(paramText(events)).toContain('hello')
  })

  it('parameter frame: </parameter|> (slash+pipe) → structural close (ToolInputReady emitted)', () => {
    const events = parseInputWithTools('<|invoke:shell>\n<|parameter:command>hello</parameter|>\n<invoke|>')
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
  })

  it('parameter frame: </parameter> (slash) → structural close (ToolInputReady emitted)', () => {
    const events = parseInputWithTools('<|invoke:shell>\n<|parameter:command>hello</parameter>\n<invoke|>')
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
  })

  it('parameter frame: <parameter> (bare) → structural close (ToolInputReady emitted)', () => {
    const events = parseInputWithTools('<|invoke:shell>\n<|parameter:command>hello<parameter>\n<invoke|>')
    expect(hasEvent(events, 'ToolInputReady')).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// SECTION 4: Real-world examples from the bug report
// ---------------------------------------------------------------------------

describe('Close tag spec: real-world examples', () => {
  it('skill-name in path inside message is preserved exactly', () => {
    const events = parseInput(inMessage(
      'See `packages/skills/builtin/<skill-name>/SKILL.md` for details.'
    ))
    const text = messageText(events)
    expect(text).toContain('<skill-name>')
    expect(text).not.toContain('<skill-name|>')
  })

  it('multiple made-up tags in message are all preserved', () => {
    const events = parseInput(inMessage('Use <div> and <span> and <skill-name> here.'))
    const text = messageText(events)
    expect(text).toContain('<div>')
    expect(text).toContain('<span>')
    expect(text).toContain('<skill-name>')
    expect(text).not.toContain('<div|>')
    expect(text).not.toContain('<span|>')
    expect(text).not.toContain('<skill-name|>')
  })

  it('<parameter> in message (not in invoke) is literal text', () => {
    const events = parseInput(inMessage('The <parameter> tag is used inside invoke.'))
    const text = messageText(events)
    expect(text).toContain('<parameter>')
    expect(text).not.toContain('<parameter|>')
  })

  it('<think> in message is literal text', () => {
    const events = parseInput(inMessage('The model uses <think> blocks for reasoning.'))
    const text = messageText(events)
    expect(text).toContain('<think>')
    expect(text).not.toContain('<think|>')
  })

  it('<invoke> in message is literal text', () => {
    const events = parseInput(inMessage('Tools are called with <invoke> syntax.'))
    const text = messageText(events)
    expect(text).toContain('<invoke>')
    expect(text).not.toContain('<invoke|>')
  })
})
