/**
 * Generate a test GBNF grammar from representative tool definitions.
 * 
 * Usage: bun run packages/xml-act/scripts/generate-test-grammar.ts
 */

import { Schema } from '@effect/schema'
import { Effect } from 'effect'
import { defineTool } from '@magnitudedev/tools'
import { defineXmlBinding } from '../src/xml-binding'
import { generateGrammar, type GrammarToolDef } from '../src/grammar-generator'

// =============================================================================
// Tool definitions (matching real agent tools)
// =============================================================================

const readTool = defineTool({
  name: 'read', group: 'fs',
  description: 'Read file content',
  inputSchema: Schema.Struct({
    path: Schema.String,
    offset: Schema.optional(Schema.Number),
    limit: Schema.optional(Schema.Number),
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})
const readBinding = defineXmlBinding(readTool, {
  input: { attributes: [{ field: 'path', attr: 'path' }, { field: 'offset', attr: 'offset' }, { field: 'limit', attr: 'limit' }] },
  output: {},
} as const)

const writeTool = defineTool({
  name: 'write', group: 'fs',
  description: 'Write file content',
  inputSchema: Schema.Struct({ path: Schema.String, content: Schema.String }),
  outputSchema: Schema.Void,
  execute: () => Effect.succeed(undefined),
})
const writeBinding = defineXmlBinding(writeTool, {
  input: { attributes: [{ field: 'path', attr: 'path' }], body: 'content' },
  output: {},
} as const)

const editTool = defineTool({
  name: 'edit', group: 'fs',
  description: 'Edit file by replacing text',
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
    attributes: [{ field: 'path', attr: 'path' }, { field: 'replaceAll', attr: 'replaceAll' }],
    childTags: [{ tag: 'old', field: 'oldString' }, { tag: 'new', field: 'newString' }],
  },
  output: {},
} as const)

const shellTool = defineTool({
  name: 'shell', group: 'exec',
  description: 'Execute shell command',
  inputSchema: Schema.Struct({ command: Schema.String, timeout: Schema.optional(Schema.Number) }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})
const shellBinding = defineXmlBinding(shellTool, {
  input: { attributes: [{ field: 'timeout', attr: 'timeout' }], body: 'command' },
  output: {},
} as const)

const grepTool = defineTool({
  name: 'grep', group: 'fs',
  description: 'Search file contents',
  inputSchema: Schema.Struct({
    path: Schema.optional(Schema.String),
    limit: Schema.optional(Schema.Number),
    pattern: Schema.String,
    glob: Schema.optional(Schema.String),
  }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})
const grepBinding = defineXmlBinding(grepTool, {
  input: {
    attributes: [{ field: 'path', attr: 'path' }, { field: 'limit', attr: 'limit' }],
    childTags: [{ tag: 'pattern', field: 'pattern' }, { tag: 'glob', field: 'glob' }],
  },
  output: {},
} as const)

const treeTool = defineTool({
  name: 'tree', group: 'fs',
  description: 'List directory structure',
  inputSchema: Schema.Struct({ path: Schema.String }),
  outputSchema: Schema.String,
  execute: () => Effect.succeed(''),
})
const treeBinding = defineXmlBinding(treeTool, {
  input: { attributes: [{ field: 'path', attr: 'path' }] },
  output: {},
} as const)

// =============================================================================
// Generate and print
// =============================================================================

function makeDef(binding: any, tool: any): GrammarToolDef {
  const tagBinding = binding.toXmlTagBinding()
  return { tagName: tagBinding.tag, binding: tagBinding, inputSchema: tool.inputSchema }
}

const grammar = generateGrammar([
  makeDef(readBinding, readTool),
  makeDef(writeBinding, writeTool),
  makeDef(editBinding, editTool),
  makeDef(shellBinding, shellTool),
  makeDef(grepBinding, grepTool),
  makeDef(treeBinding, treeTool),
])

console.log(grammar)
