import { describe, it, expect } from 'vitest'
import { Schema } from '@effect/schema'
import { AST } from '@effect/schema'
import { Effect } from 'effect'
import { GrammarBuilder, type GrammarToolDef } from '../grammar-builder'
import { defineXmlBinding } from '../xml-binding'
import { defineTool } from '@magnitudedev/tools'

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

function makeDef(binding: { toXmlTagBinding: () => { tag: string } }, tool: { inputSchema: { readonly ast: AST.AST } }): GrammarToolDef {
  const tagBinding = binding.toXmlTagBinding()
  return {
    tagName: tagBinding.tag,
    binding: tagBinding,
    inputSchema: tool.inputSchema,
  }
}

const tools = [makeDef(readBinding, readTool), makeDef(writeBinding, writeTool)]

describe('GrammarBuilder', () => {
  it('GrammarBuilder.create(tools).build() produces default root rule', () => {
    const grammar = GrammarBuilder.create(tools).build()

    expect(grammar).toContain('root ::= lens* (msg | tool)* yield')
  })

  it('withMinLenses(1) produces lens+ root (at least one lens required)', () => {
    const grammar = GrammarBuilder.create(tools).withMinLenses(1).build()

    expect(grammar).toContain('root ::= lens+ (msg | tool)* yield')
  })

  it('requireMessageTo("parent") produces forced-msg with capped lenses (default 6)', () => {
    const grammar = GrammarBuilder.create(tools)
      .requireMessageTo('parent')
      .build()

    // Should have 6 optional lens-tight slots followed by forced message
    expect(grammar).toContain('root ::= lens-tight? lens-tight? lens-tight? lens-tight? lens-tight? lens-tight? forced-msg (msg | tool)* yield')
    expect(grammar).toContain('forced-msg ::= "<message to=\\"parent\\">" msg-body "</message>" ws')
  })

  it('requireMessageTo with maxLenses produces correct number of lens slots', () => {
    const grammar = GrammarBuilder.create(tools)
      .requireMessageTo('parent')
      .withMaxLenses(3)
      .build()

    // Should have exactly 3 optional lens-tight slots
    expect(grammar).toContain('root ::= lens-tight? lens-tight? lens-tight? forced-msg (msg | tool)* yield')
    expect(grammar).not.toContain('lens-tight? lens-tight? lens-tight? lens-tight?')
  })

  it('withMaxLenses is ignored when no requiredMessageTo is set', () => {
    const grammar = GrammarBuilder.create(tools)
      .withMaxLenses(3)
      .build()

    // Without forced message, should use unconstrained lens*
    expect(grammar).toContain('root ::= lens* (msg | tool)* yield')
  })

  it('is immutable: modifying builder returns a new instance and leaves original unchanged', () => {
    const original = GrammarBuilder.create(tools)
    const modified = original.withMinLenses(1)

    expect(original).not.toBe(modified)
    expect(original.build()).toContain('root ::= lens* (msg | tool)* yield')
    expect(modified.build()).toContain('root ::= lens+ (msg | tool)* yield')
  })

  it('is deterministic for the same config', () => {
    const grammar1 = GrammarBuilder.create(tools)
      .withMinLenses(1)
      .requireMessageTo('parent')
      .build()

    const grammar2 = GrammarBuilder.create(tools)
      .withMinLenses(1)
      .requireMessageTo('parent')
      .build()

    expect(grammar1).toBe(grammar2)
  })

  it('forced message root has direct transition from lenses to forced-msg', () => {
    const grammar = GrammarBuilder.create(tools)
      .requireMessageTo('parent')
      .build()

    // Lenses should flow directly into forced-msg with no intermediate rules
    expect(grammar).toContain('root ::= lens-tight? lens-tight? lens-tight? lens-tight? lens-tight? lens-tight? forced-msg (msg | tool)* yield')
  })

  it('default grammar uses unbounded ws lens; forced-msg grammar uses lens-tight with ws-bounded', () => {
    const defaultGrammar = GrammarBuilder.create(tools).build()
    const forcedGrammar = GrammarBuilder.create(tools).requireMessageTo('user').build()

    // Default: unbounded whitespace, no tight variants needed
    expect(defaultGrammar).toContain('lens ::= "<lens name=\\"" lensname "\\">" lens-body "</lens>" ws')

    // Forced: uses lens-tight with bounded whitespace to prevent ws loops
    expect(forcedGrammar).toContain('lens-tight ::= "<lens name=\\"" lensname "\\">" lens-body "</lens>" ws-bounded')
    expect(forcedGrammar).toContain('ws-bounded ::= [ \\t\\n] [ \\t\\n]? [ \\t\\n]? [ \\t\\n]?')
  })

  it('does not contain old recursive lens prefix rules', () => {
    const grammar = GrammarBuilder.create(tools)
      .requireMessageTo('parent')
      .build()

    expect(grammar).not.toContain('lensprefix')
    expect(grammar).not.toContain('lensprefix-opt')
    expect(grammar).not.toContain('lensprefix-tail')
    expect(grammar).not.toContain('lensprefix-loose')
  })

  it('includes DFA body rules for lens-body and msg-body', () => {
    const grammar = GrammarBuilder.create(tools).build()

    expect(grammar).toContain('lens-body ::= lens-body-s0')
    expect(grammar).toContain('msg-body ::= msg-body-s0')
  })

  it('includes tool rules unchanged', () => {
    const grammar = GrammarBuilder.create(tools).build()

    expect(grammar).toContain('tool ::= readtool | writetool')
    expect(grammar).toContain('readtool ::= "<read" ws readtool-attrs "/>" ws')
    expect(grammar).toContain('writetool ::= "<write" ws writetool-attrs ">" writetool-body "</write>" ws')
    expect(grammar).toContain('readtool-attrs ::= (ws1 readtool-attrs-alt)* ws')
    expect(grammar).toContain('writetool-attrs ::= (ws1 writetool-attrs-alt)* ws')
  })

  it('capped lenses prevent infinite lens loops when forced message is required', () => {
    // When maxLenses is set (via requireMessageTo), the grammar caps the number of lens slots
    // After the last lens slot, the model MUST choose between:
    //   - `<message to="parent">` (forced message) or continue with tools
    // This prevents the grammar trap where the model could infinitely loop on lenses
    const grammar = GrammarBuilder.create(tools)
      .requireMessageTo('parent')
      .withMaxLenses(2)
      .build()

    // After 2 lens slots, the model MUST proceed to forced-msg or tools
    // The grammar allows: lens? lens? forced-msg (msg | tool)* yield
    // Each lens? can be either a lens or empty, but after both slots, forced-msg is required
    expect(grammar).toContain('root ::= lens-tight? lens-tight? forced-msg')
  })
})
