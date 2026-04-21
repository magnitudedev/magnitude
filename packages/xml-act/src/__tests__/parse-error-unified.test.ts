/**
 * Parse error tests — verifies tool and structural error variants are produced correctly.
 */
import { describe, it, expect } from 'vitest'
import { createParser } from '../parser/index'
import { createTokenizer } from '../tokenizer'
import type { TurnEngineEvent, RegisteredTool, ToolParseErrorEvent, StructuralParseErrorEvent } from '../types'
import { Schema } from 'effect'
import { defineTool } from '@magnitudedev/tools'

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

function parse(input: string): TurnEngineEvent[] {
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['shell']),
    { toolKeyword: 'invoke' },
  )
  tokenizer.push(input)
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

function getToolErrors(events: TurnEngineEvent[]): ToolParseErrorEvent[] {
  return events.filter(e => e._tag === 'ToolParseError') as ToolParseErrorEvent[]
}

function getStructuralErrors(events: TurnEngineEvent[]): StructuralParseErrorEvent[] {
  return events.filter(e => e._tag === 'StructuralParseError') as StructuralParseErrorEvent[]
}

describe('parse errors', () => {
  it('produces UnknownTool as structural error for unregistered tool', () => {
    const errors = getStructuralErrors(parse('\n<|invoke:unknown_tool>\n<invoke|>'))
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('UnknownTool')
  })

  it('produces MissingToolName as structural error for invoke without variant', () => {
    const errors = getStructuralErrors(parse('\n<|invoke>\n<invoke|>'))
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('MissingToolName')
  })

  it('produces StrayCloseTag as structural error for unmatched close', () => {
    const errors = getStructuralErrors(parse('\n<think|>\ntext'))
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0].error._tag).toBe('StrayCloseTag')
  })

  it('produces UnclosedThink as structural error at EOF', () => {
    const errors = getStructuralErrors(parse('\n<|think:reasoning>\nSome thinking'))
    expect(errors.length).toBeGreaterThan(0)
    expect(errors.some(e => e.error._tag === 'UnclosedThink')).toBe(true)
  })

  it('tool-scoped errors have toolCallId and are ToolParseError events', () => {
    // UnknownParameter is tool-scoped
    const errors = getToolErrors(parse('\n<|invoke:shell>\n<|parameter:bogus>value<parameter|>\n<invoke|>'))
    const paramError = errors.find(e => e.error._tag === 'UnknownParameter')
    expect(paramError).toBeDefined()
    expect(paramError!.toolCallId).toBeDefined()
    expect(paramError!.tagName).toBe('shell')
  })

  it('structural errors do not have toolCallId', () => {
    const errors = getStructuralErrors(parse('\n<think|>\ntext'))
    const strayError = errors.find(e => e.error._tag === 'StrayCloseTag')
    expect(strayError).toBeDefined()
    expect((strayError as any).toolCallId).toBeUndefined()
  })
})
