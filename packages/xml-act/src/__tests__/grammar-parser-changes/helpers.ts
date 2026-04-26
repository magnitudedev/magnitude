/**
 * Shared helpers for grammar-parser-changes test suite.
 */
import {
  buildValidator,
  SHELL_TOOL,
  MULTI_PARAM_TOOL,
} from '../../grammar/__tests__/helpers'
import type { GrammarToolDef } from '../../grammar/grammar-builder'
import { createParser } from '../../parser/index'
import { createTokenizer } from '../../tokenizer'
import type { TurnEngineEvent, RegisteredTool } from '../../types'
import { Schema } from 'effect'
import { defineTool } from '@magnitudedev/tools'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const YIELD_USER = '<magnitude:yield_user/>'
export const YIELD_INVOKE = '<magnitude:yield_invoke/>'

// ---------------------------------------------------------------------------
// Grammar validator
// ---------------------------------------------------------------------------

export function grammarValidator(tools: GrammarToolDef[] = [SHELL_TOOL, MULTI_PARAM_TOOL]) {
  return buildValidator(tools)
}

// ---------------------------------------------------------------------------
// Parser helpers
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

const editTool = defineTool({
  name: 'edit',
  label: 'Edit',
  group: 'fs',
  description: 'Edit a file',
  inputSchema: Schema.Struct({ path: Schema.String, old: Schema.String, new: Schema.String }),
  outputSchema: Schema.Struct({ result: Schema.String }),
  execute: async () => ({ result: '' }),
})

function makeTools(): ReadonlyMap<string, RegisteredTool> {
  return new Map([
    ['shell', { tool: shellTool, tagName: 'shell', groupName: 'fs' }],
    ['edit', { tool: editTool, tagName: 'edit', groupName: 'fs' }],
  ])
}

export function parse(input: string): TurnEngineEvent[] {
  const allEvents: TurnEngineEvent[] = []
  const parser = createParser({ tools: makeTools() })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    new Set(['shell', 'edit']),
  )
  tokenizer.push(input)
  tokenizer.end()
  parser.end()
  allEvents.push(...parser.drain())
  return allEvents
}

export function hasEvent(events: TurnEngineEvent[], tag: string): boolean {
  return events.some((e) => e._tag === tag)
}

export function countEvents(events: TurnEngineEvent[], tag: string): number {
  return events.filter((e) => e._tag === tag).length
}

export function collectLensChunks(events: TurnEngineEvent[]): string {
  return events
    .filter((e): e is any => e._tag === 'LensChunk')
    .map((e) => e.text)
    .join('')
}

export function collectMessageChunks(events: TurnEngineEvent[]): string {
  return events
    .filter((e): e is any => e._tag === 'MessageChunk')
    .map((e) => e.text)
    .join('')
}

export function getToolInput(events: TurnEngineEvent[]): Record<string, string> | undefined {
  const ready = events.find((e): e is any => e._tag === 'ToolInputReady')
  return ready?.input
}
