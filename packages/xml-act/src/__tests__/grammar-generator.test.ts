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
import { GrammarBuilder, type GrammarToolDef } from '../grammar-builder'
import { defineXmlBinding, type XmlBindingResult } from '../xml-binding'
import { defineTool } from '@magnitudedev/tools'

// --- Test tool definitions ---

const readTool = defineTool({
  name: 'read',
  group: 'fs',
  label: (input) => input.path ?? 'read',
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
  label: (input) => input.path ?? 'write',
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
  label: (input) => input.path ?? 'edit',
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
  label: (input) => input.command ?? 'shell',
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
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()

    // Must have standard protocol elements
    expect(grammar).toContain('root ::= lens* (msg | tool)*')
    expect(grammar).toContain('lens ::= "<lens name=\\"" lensname "\\">"')
    expect(grammar).toContain('msg ::= "<message to=\\""')
    expect(grammar).toContain('yield ::=')
    expect(grammar).toContain('"<yield-user/>" | "<yield-tool/>" | "<yield-worker/>"')
  })

  it('generates tool alternation from multiple tools', () => {
    const grammar = GrammarBuilder.create([
      makeDef(readBinding, readTool),
      makeDef(writeBinding, writeTool),
      makeDef(editBinding, editTool),
    ]).build()

    expect(grammar).toContain('tool ::= readtool | writetool | edittool')
  })

  it('generates self-closing tool with required and optional attributes', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()

    // read is self-closing (no body, no children)
    expect(grammar).toContain('readtool ::= "<read" ws readtool-attrs "/>"')
    // path is required string attribute
    expect(grammar).toContain('readtool-attrs ::=')
    expect(grammar).toContain('"path=\\""')
    expect(grammar).toContain('"offset=\\""')
    expect(grammar).toContain('"limit=\\""')
  })

  it('generates tool with body content', () => {
    const grammar = GrammarBuilder.create([makeDef(writeBinding, writeTool)]).build()

    // write has body
    expect(grammar).toContain('writetool ::= "<write" ws writetool-attrs ">"')
    expect(grammar).toContain('</write>"')
  })

  it('generates tool with child tags', () => {
    const grammar = GrammarBuilder.create([makeDef(editBinding, editTool)]).build()

    // edit has child tags (old, new)
    expect(grammar).toContain('edittool ::= "<edit" ws edittool-attrs ">"')
    expect(grammar).toContain('</edit>"')
    expect(grammar).toContain('edittool-oldtool')
    expect(grammar).toContain('edittool-newtool')
  })

  it('uses correct value patterns for attribute types', () => {
    const grammar = GrammarBuilder.create([makeDef(editBinding, editTool)]).build()

    // replaceAll is boolean
    expect(grammar).toContain('"true" | "false"')
  })

  it('produces consistent grammar for the same inputs', () => {
    const tools = [makeDef(readBinding, readTool), makeDef(writeBinding, writeTool)]

    const grammar1 = GrammarBuilder.create(tools).build()
    const grammar2 = GrammarBuilder.create(tools).build()

    expect(grammar1).toBe(grammar2)
  })

  it('includes whitespace rule', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()

    expect(grammar).toContain('ws ::= [ \\t\\n]*')
  })

  it('handles empty tool list', () => {
    const grammar = GrammarBuilder.create([]).build()
    const grammar2 = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()

    expect(grammar).toContain('root ::= lens* (msg | tool)*')
    // tool rule should handle empty case
    expect(grammar).toContain('tool ::=')
  })

  it('includes ws1 rule for mandatory whitespace', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()
    expect(grammar).toContain('ws1 ::= [ \\t\\n]+')
  })

  it('uses ws1 before attributes to prevent tag-attribute fusion', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()
    // Attr rules must use ws1 (one-or-more) not ws (zero-or-more) before each attribute
    expect(grammar).toContain('readtool-attrs ::= (ws1 readtool-attrs-alt)* ws')
  })

  it('uses ws1 for parent tool attr rules with children', () => {
    const grammar = GrammarBuilder.create([makeDef(editBinding, editTool)]).build()
    // Parent edit tool attrs use ws1
    expect(grammar).toContain('edittool-attrs ::= (ws1 edittool-attrs-alt)* ws')
    // Child elements (old, new) have no attrs, so their attr rule is just ws
    expect(grammar).toContain('edittool-oldtool-attrs ::= ws')
  })

  it('generates escape pattern for string attribute values', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()
    // String attrs should allow escaped quotes: ([^"] | "\\\"")* not [^"]*
    expect(grammar).toContain('([^"] | "\\\\\\"")*')
    expect(grammar).not.toMatch(/readtool-attrs-alt.*\[^\"\]\*/)
  })

  it('generates escape pattern for observe attribute', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()
    // observe attr should also use escape pattern
    expect(grammar).toContain('"observe=\\"" ([^"] | "\\\\\\"")*')
  })

  it('generates escape pattern for message to attribute', () => {
    const grammar = GrammarBuilder.create([makeDef(readBinding, readTool)]).build()
    // msg rule should use escape pattern for to value
    expect(grammar).toContain('msg ::= "<message to=\\"" ([^"] | "\\\\\\"")*')
  })

  it('generates enum pattern for literal union attributes', () => {
    const enumTool = defineTool({
      name: 'update-task',
      group: 'task',
      label: (input) => input.id ?? 'update-task',
      description: 'Update task',
      inputSchema: Schema.Struct({
        id: Schema.String,
        status: Schema.Literal('pending', 'completed', 'cancelled'),
      }),
      outputSchema: Schema.Struct({ id: Schema.String }),
      execute: () => Effect.succeed({ id: '' }),
    })
    const enumBinding = defineXmlBinding(enumTool, {
      input: {
        attributes: [
          { field: 'id', attr: 'id' },
          { field: 'status', attr: 'status' },
        ],
      },
      output: {},
    } as const)
    const grammar = GrammarBuilder.create([makeDef(enumBinding, enumTool)]).build()
    // status should be enum, not string
    expect(grammar).toContain('"status=\\"" ("pending" | "completed" | "cancelled") "\\""')
  })
})
