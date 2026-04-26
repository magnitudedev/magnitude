import { buildValidator } from '../../grammar/__tests__/helpers'
import { createTokenizer } from '../../tokenizer'
import { createParser } from '../../parser/index'
import type { Token } from '../../types'
import type { TurnEngineEvent, RegisteredTool } from '../../types'
import type { GrammarToolDef } from '../../grammar/grammar-builder'
import { Schema } from 'effect'
import { defineTool } from '@magnitudedev/tools'

// ============================================================================
// Grammar Tool Definitions
// ============================================================================

export const SHELL_TOOL_DEF: GrammarToolDef = {
  tagName: 'shell',
  parameters: [{ name: 'command', field: 'command', type: 'scalar', required: true }],
}

export const EDIT_TOOL_DEF: GrammarToolDef = {
  tagName: 'edit',
  parameters: [
    { name: 'path', field: 'path', type: 'scalar', required: true },
    { name: 'old', field: 'old', type: 'scalar', required: true },
    { name: 'new', field: 'new', type: 'scalar', required: true },
  ],
}

export const TREE_TOOL_DEF: GrammarToolDef = {
  tagName: 'tree',
  parameters: [],
}

export const GREP_TOOL_DEF: GrammarToolDef = {
  tagName: 'grep',
  parameters: [
    { name: 'pattern', field: 'pattern', type: 'scalar', required: true },
    { name: 'glob', field: 'glob', type: 'scalar', required: true },
    { name: 'path', field: 'path', type: 'scalar', required: true },
    { name: 'limit', field: 'limit', type: 'scalar', required: true },
  ],
}

export const DEFAULT_GRAMMAR_TOOLS = [SHELL_TOOL_DEF, EDIT_TOOL_DEF, TREE_TOOL_DEF]
export const YIELD = '<magnitude:yield_user/>'

// ============================================================================
// Parser Tool Definitions (real tools with schemas)
// ============================================================================

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
  inputSchema: Schema.Struct({
    path: Schema.String,
    old: Schema.String,
    new: Schema.String,
  }),
  outputSchema: Schema.Struct({ ok: Schema.Boolean }),
  execute: async () => ({ ok: true }),
})

const treeTool = defineTool({
  name: 'tree',
  label: 'Tree',
  group: 'fs',
  description: 'List directory',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.Struct({ tree: Schema.String }),
  execute: async () => ({ tree: '' }),
})

const grepTool = defineTool({
  name: 'grep',
  label: 'Grep',
  group: 'fs',
  description: 'Search files',
  inputSchema: Schema.Struct({
    pattern: Schema.String,
    glob: Schema.optional(Schema.String),
    path: Schema.optional(Schema.String),
    limit: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({ matches: Schema.String }),
  execute: async () => ({ matches: '' }),
})

function makeTools(...tools: Array<{ tool: any; tagName: string }>): ReadonlyMap<string, RegisteredTool> {
  return new Map(tools.map(t => [t.tagName, { tool: t.tool, tagName: t.tagName, groupName: 'fs' }]))
}

const DEFAULT_PARSER_TOOLS = makeTools(
  { tool: shellTool, tagName: 'shell' },
  { tool: editTool, tagName: 'edit' },
  { tool: treeTool, tagName: 'tree' },
)

const GREP_PARSER_TOOLS = makeTools(
  { tool: grepTool, tagName: 'grep' },
)

const KNOWN_TAGS = new Set(['shell', 'edit', 'tree', 'grep', 'magnitude:invoke', 'magnitude:parameter', 'magnitude:filter', 'magnitude:think', 'magnitude:message', 'magnitude:yield_user', 'magnitude:yield_invoke', 'magnitude:yield_parent', 'magnitude:yield_worker', ])

// ============================================================================
// Grammar Validator
// ============================================================================

export function grammarValidator(tools: GrammarToolDef[] = DEFAULT_GRAMMAR_TOOLS) {
  return buildValidator(tools)
}

// ============================================================================
// Parser: feed input through tokenizer → parser, return events
// ============================================================================

export function parse(input: string, tools?: ReadonlyMap<string, RegisteredTool>): TurnEngineEvent[] {
  const t = tools ?? DEFAULT_PARSER_TOOLS
  const parser = createParser({ tools: t })
  const tokenizer = createTokenizer(
    (token) => parser.pushToken(token),
    KNOWN_TAGS,
  )
  tokenizer.push(input)
  tokenizer.end()
  parser.end()
  return [...parser.drain()]
}

export function parseWithGrep(input: string): TurnEngineEvent[] {
  return parse(input, GREP_PARSER_TOOLS)
}

// ============================================================================
// Event query helpers
// ============================================================================

export function hasEvent(events: TurnEngineEvent[], tag: string): boolean {
  return events.some(e => e._tag === tag)
}

export function countEvents(events: TurnEngineEvent[], tag: string): number {
  return events.filter(e => e._tag === tag).length
}

export function getEvents(events: TurnEngineEvent[], tag: string): TurnEngineEvent[] {
  return events.filter(e => e._tag === tag)
}

export function getToolInputs(events: TurnEngineEvent[]): Array<{ toolCallId: string; input: Record<string, unknown> }> {
  return events
    .filter(e => e._tag === 'ToolInputReady')
    .map(e => ({ toolCallId: (e as any).toolCallId, input: (e as any).input }))
}

export function getLensTexts(events: TurnEngineEvent[]): string[] {
  return events
    .filter(e => e._tag === 'LensEnd')
    .map(e => (e as any).content)
}

export function getMessageTexts(events: TurnEngineEvent[]): string[] {
  return events
    .filter(e => e._tag === 'MessageEnd')
    .map(e => {
      // MessageEnd doesn't carry content — collect from MessageChunks before it
      // Actually, let's just check it exists
      return (e as any).id
    })
}

/** Collect all text from LensChunk events */
export function collectLensChunks(events: TurnEngineEvent[]): string {
  return events
    .filter(e => e._tag === 'LensChunk')
    .map(e => (e as any).text)
    .join('')
}

/** Collect all text from MessageChunk events */
export function collectMessageChunks(events: TurnEngineEvent[]): string {
  return events
    .filter(e => e._tag === 'MessageChunk')
    .map(e => (e as any).text)
    .join('')
}

/** Get tool input field values from ToolInputReady events */
export function getToolInput(events: TurnEngineEvent[], index = 0): Record<string, unknown> | undefined {
  const inputs = getToolInputs(events)
  return inputs[index]?.input
}
