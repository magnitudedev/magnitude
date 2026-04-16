/**
 * Tests for the dynamic GBNF grammar generator.
 *
 * Tests generate grammar from tool bindings matching real agent tool shapes:
 * - Self-closing tools with attributes (read, tree)
 * - Tools with body content (write)
 * - Tools with child tags (edit with old/new)
 * - Tools with children (create-task)
 */

import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { AST } from '@effect/schema'
import { Effect } from 'effect'
import { generateGrammar, type GrammarToolDef } from '../grammar-generator'
import { defineXmlBinding } from '../xml-binding'
import { defineTool } from '@magnitudedev/tools'

// --- Test tool definitions ---

const readTool = defineTool({
  name: 'read',
  group: 'fs',
  description: 'Read file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    offset: Schema.optional(Schema.Number),
    limit: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})

const readBinding = defineXmlBinding(readTool, {
  input: {
    attributes: [
      { field: 'path', attr: 'path' },
      { field: 'offset', attr: 'offset' },
      { field: 'limit', attr: 'limit' },
    ],
  },
  output: {},
} as const)

// Write: has body
const writeTool = defineTool({
  name: 'write',
  group: 'fs',
  description: 'Write file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    content: Schema.String,
  }),
  outputSchema: Schema.Void,
  execute: () => Effect.succeed(undefined),
})

const writeBinding = defineXmlBinding(writeTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }],
    body: 'content',
  },
  output: {},
} as const)

// Edit: has child tags (old, new)
const editTool = defineTool({
  name: 'edit',
  group: 'fs',
  description: 'Edit file',
  inputSchema: Schema.Struct({
    path: Schema.String,
    oldString: Schema.String,
    newString: Schema.String,
    replaceAll: Schema.optional(Schema.Boolean),
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})

const editBinding = defineXmlBinding(editTool, {
  input: {
    attributes: [
      { field: 'path', attr: 'path' },
      { field: 'replaceAll', attr: 'replaceAll' },
    ],
    childTags: [
      { tag: 'old', field: 'oldString' },
      { tag: 'new', field: 'newString' },
    ],
  },
  output: {},
} as const)

// Shell: has body content (no children)
const shellTool = defineTool({
  name: 'shell',
  group: 'exec',
  description: 'Run shell command',
  inputSchema: Schema.Struct({
    command: Schema.String,
    timeout: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})

const shellBinding = defineXmlBinding(shellTool, {
  input: {
    attributes: [{ field: 'timeout', attr: 'timeout' }],
    body: 'command',
  },
  output: {},
} as const)

function makeDef(binding: XmlBindingResult<any, any, any>, tool: { inputSchema: { readonly ast: AST.AST } }): GrammarToolDef {
  const tagBinding = binding.toXmlTagBinding()
  return {
    tagName: tagBinding.tag,
    binding: tagBinding,
    inputSchema: tool.inputSchema,
  }
}

describe('generateGrammar', () => {
  it('produces valid GBNF with correct root structure', () => {
    const grammar = generateGrammar([makeDef(readBinding, readTool)])

    // Must have standard protocol elements
    expect(grammar).toContain('root ::= lens* (msg | tool)*')
    expect(grammar).toContain('lens ::= "<lens name=\\"" lensname "\\">"')
    expect(grammar).toContain('msg ::= "<message to=\\""')
    expect(grammar).toContain('endturn ::=')
    expect(grammar).toContain('"<idle/>" | "<continue/>"')
  })

  it('generates tool alternation from multiple tools', () => {
    const grammar = generateGrammar([
      makeDef(readBinding, readTool),
      makeDef(writeBinding, writeTool),
      makeDef(editBinding, editTool),
    ])

    expect(grammar).toContain('tool ::= readtool | writetool | edittool')
  })

  it('generates self-closing tool with required and optional attributes', () => {
    const grammar = generateGrammar([makeDef(readBinding, readTool)])

    // read is self-closing (no body, no children)
    expect(grammar).toContain('readtool ::= "<read" ws readtool-attrs "/>"')
    // path is required string attribute
    expect(grammar).toContain('readtool-attrs ::=')
    expect(grammar).toContain('"path=\\""')
    expect(grammar).toContain('"offset=\\""')
    expect(grammar).toContain('"limit=\\""')
  })

  it('generates tool with body content', () => {
    const grammar = generateGrammar([makeDef(writeBinding, writeTool)])

    // write has body
    expect(grammar).toContain('writetool ::= "<write" ws writetool-attrs ">"')
    expect(grammar).toContain('</write>"')
  })

  it('generates tool with child tags', () => {
    const grammar = generateGrammar([makeDef(editBinding, editTool)])

    // edit has child tags (old, new)
    expect(grammar).toContain('edittool ::= "<edit" ws edittool-attrs ">"')
    expect(grammar).toContain('</edit>"')
    expect(grammar).toContain('edittool-oldtool')
    expect(grammar).toContain('edittool-newtool')
  })

  it('uses correct value patterns for attribute types', () => {
    const grammar = generateGrammar([makeDef(editBinding, editTool)])

    // replaceAll is boolean
    expect(grammar).toContain('"true" | "false"')
  })

  it('produces consistent grammar for the same inputs', () => {
    const tools = [makeDef(readBinding, readTool), makeDef(writeBinding, writeTool)]

    const grammar1 = generateGrammar(tools)
    const grammar2 = generateGrammar(tools)

    expect(grammar1).toBe(grammar2)
  })

  it('includes whitespace rule', () => {
    const grammar = generateGrammar([makeDef(readBinding, readTool)])

    expect(grammar).toContain('ws ::= [ \\t\\n]*')
  })

  it('handles empty tool list', () => {
    const grammar = generateGrammar([])
    const grammar2 = generateGrammar([makeDef(readBinding, readTool)])

    expect(grammar).toContain('root ::= lens* (msg | tool)*')
    // tool rule should handle empty case
    expect(grammar).toContain('tool ::=')
  })
})
