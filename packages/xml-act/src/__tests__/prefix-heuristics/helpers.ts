import { expect } from 'vitest'
import { buildValidator } from '../../grammar/__tests__/helpers'
import { createTokenizer } from '../../tokenizer'
import { createParser } from '../../parser/index'
import type { TurnEngineEvent, RegisteredTool, StructuralParseErrorEvent } from '../../types'
import type { GrammarToolDef } from '../../grammar/grammar-builder'
import { Effect } from 'effect'
import { Schema } from '@effect/schema'
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
// Parser Tool Definitions
// ============================================================================

const shellTool = defineTool({
  name: 'shell',
  label: () => 'Shell',
  group: 'fs',
  description: 'Run a shell command',
  inputSchema: Schema.Struct({ command: Schema.String }),
  outputSchema: Schema.Struct({ stdout: Schema.String }),
  execute: () => Effect.succeed({ stdout: '' }),
})

const editTool = defineTool({
  name: 'edit',
  label: () => 'Edit',
  group: 'fs',
  description: 'Edit a file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    old: Schema.String,
    new: Schema.String,
  }),
  outputSchema: Schema.Struct({ ok: Schema.Boolean }),
  execute: () => Effect.succeed({ ok: true }),
})

const treeTool = defineTool({
  name: 'tree',
  label: () => 'Tree',
  group: 'fs',
  description: 'List directory',
  inputSchema: Schema.Struct({}),
  outputSchema: Schema.Struct({ tree: Schema.String }),
  execute: () => Effect.succeed({ tree: '' }),
})

const grepTool = defineTool({
  name: 'grep',
  label: () => 'Grep',
  group: 'fs',
  description: 'Search files',
  inputSchema: Schema.Struct({
    pattern: Schema.String,
    glob: Schema.optional(Schema.String),
    path: Schema.optional(Schema.String),
    limit: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.Struct({ matches: Schema.String }),
  execute: () => Effect.succeed({ matches: '' }),
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
  { tool: editTool, tagName: 'edit' },
  { tool: shellTool, tagName: 'shell' },
  { tool: treeTool, tagName: 'tree' },
)

const KNOWN_TAGS = new Set([
  'shell',
  'edit',
  'tree',
  'grep',
  'magnitude:invoke',
  'magnitude:parameter',
  'magnitude:filter',
  'magnitude:reason',
  'magnitude:message',
  'magnitude:yield_user',
  'magnitude:yield_invoke',
  'magnitude:yield_parent',
  'magnitude:yield_worker',

])

// ============================================================================
// Grammar Validator
// ============================================================================

export function grammarValidator(tools: GrammarToolDef[] = DEFAULT_GRAMMAR_TOOLS) {
  return buildValidator(tools)
}

export function grepGrammarValidator() {
  return buildValidator([GREP_TOOL_DEF])
}

// ============================================================================
// Parser
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

export function getToolInput(events: TurnEngineEvent[], index = 0): Record<string, unknown> | undefined {
  const inputs = getToolInputs(events)
  return inputs[index]?.input
}

export function collectLensChunks(events: TurnEngineEvent[]): string {
  return events
    .filter(e => e._tag === 'LensChunk')
    .map(e => (e as any).text)
    .join('')
}

export function collectMessageChunks(events: TurnEngineEvent[]): string {
  return events
    .filter(e => e._tag === 'MessageChunk')
    .map(e => (e as any).text)
    .join('')
}

export function getStructuralErrors(events: TurnEngineEvent[]): StructuralParseErrorEvent[] {
  return events.filter(e => e._tag === 'StructuralParseError') as StructuralParseErrorEvent[]
}

export function expectStructuralError(
  events: TurnEngineEvent[],
  expected: {
    variant: string
    tagName?: string
    parentTagName?: string
    expectedTagName?: string
    detailIncludes?: string[]
  },
): void {
  const errors = getStructuralErrors(events)
  expect(errors.length).toBe(1)
  const error = errors[0].error as any
  expect(error._tag).toBe(expected.variant)
  if (expected.tagName !== undefined) {
    expect(error.tagName).toBe(expected.tagName)
  }
  if (expected.parentTagName !== undefined) {
    expect(error.parentTagName).toBe(expected.parentTagName)
  }
  if (expected.expectedTagName !== undefined) {
    expect(error.expectedTagName).toBe(expected.expectedTagName)
  }
  for (const snippet of expected.detailIncludes ?? []) {
    expect(String(error.detail ?? '')).toContain(snippet)
  }
}

export function expectNoStructuralError(events: TurnEngineEvent[]): void {
  expect(getStructuralErrors(events)).toHaveLength(0)
}

export function expectPreservedInMessage(events: TurnEngineEvent[], raw: string): void {
  expect(collectMessageChunks(events)).toContain(raw)
}

export function expectPreservedInLens(events: TurnEngineEvent[], raw: string): void {
  expect(collectLensChunks(events)).toContain(raw)
}

export function normalizeToolEvents(events: TurnEngineEvent[]) {
  return events
    .filter(e => e._tag.startsWith('ToolInput') || e._tag.startsWith('Invoke'))
    .map((event: any) => {
      const { toolCallId, callId, id, invocationId, openSpan, ...rest } = event
      return rest
    })
}

export function expectToolAliasEquivalent(
  aliasInput: string,
  canonicalInput: string,
  parseFn: (input: string) => TurnEngineEvent[] = parse,
): void {
  const aliasEvents = parseFn(aliasInput)
  const canonicalEvents = parseFn(canonicalInput)
  expectNoStructuralError(aliasEvents)
  expectNoStructuralError(canonicalEvents)
  expect(normalizeToolEvents(aliasEvents)).toEqual(normalizeToolEvents(canonicalEvents))
  expect(getToolInputs(aliasEvents).map(x => x.input)).toEqual(getToolInputs(canonicalEvents).map(x => x.input))
}

export function expectParameterAliasEquivalent(
  aliasInput: string,
  canonicalInput: string,
  parseFn: (input: string) => TurnEngineEvent[] = parse,
): void {
  const aliasEvents = parseFn(aliasInput)
  const canonicalEvents = parseFn(canonicalInput)
  expectNoStructuralError(aliasEvents)
  expectNoStructuralError(canonicalEvents)
  expect(normalizeToolEvents(aliasEvents)).toEqual(normalizeToolEvents(canonicalEvents))
  expect(getToolInputs(aliasEvents).map(x => x.input)).toEqual(getToolInputs(canonicalEvents).map(x => x.input))
}
